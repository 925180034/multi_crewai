from crewai import Agent, Task, Crew, Process
from crewai import LLM
from openai import OpenAI
import os
import litellm
import logging

# 启用LiteLLM的详细日志
litellm.set_verbose = True

# 启用详细日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置API配置
os.environ['OPENAI_API_KEY'] = 'sk-rV0rXSp2Ml9xtRvyQkbqmQvnjTkXxsrWH04ASeZGZ3XIJDHj'
os.environ['OPENAI_API_BASE'] = 'https://api.nuwaapi.com/v1'

# 测试直接使用LiteLLM
def test_litellm_directly():
    try:
        # 设置LiteLLM的配置
        litellm.api_key = "sk-rV0rXSp2Ml9xtRvyQkbqmQvnjTkXxsrWH04ASeZGZ3XIJDHj"
        litellm.api_base = "https://api.nuwaapi.com/v1"
        
        # 测试LiteLLM直接调用
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, this is a test!"}],
            api_base="https://api.nuwaapi.com/v1",
            api_key="sk-rV0rXSp2Ml9xtRvyQkbqmQvnjTkXxsrWH04ASeZGZ3XIJDHj",
            custom_llm_provider="openai"
        )
        
        print("LiteLLM直接调用成功!")
        print(f"响应: {response.choices[0].message.content}")
        return True
    
    except Exception as e:
        print(f"LiteLLM直接调用失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# 测试CrewAI功能
def test_crewai():
    try:
        # 配置LLM实例
        llm = LLM(
            model="openai/gpt-4o-mini",  # 添加前缀尝试
            api_key="sk-rV0rXSp2Ml9xtRvyQkbqmQvnjTkXxsrWH04ASeZGZ3XIJDHj",
            base_url="https://api.nuwaapi.com/v1"
        )
        
        # 创建一个简单的代理
        researcher = Agent(
            role="研究员",
            goal="收集和分析信息",
            backstory="你是一个专业的研究分析师，擅长理解和总结信息。",
            allow_delegation=False,
            verbose=True,
            llm=llm
        )
        
        # 创建一个简单的任务
        research_task = Task(
            description="简单回答：什么是CrewAI？",
            expected_output="一句话回答",
            agent=researcher
        )
        
        # 创建Crew
        crew = Crew(
            agents=[researcher],
            tasks=[research_task],
            process=Process.sequential,
            verbose=True
        )
        
        # 执行并获取结果
        result = crew.kickoff()
        
        print("\nCrewAI测试成功!")
        print(f"结果: {result}")
        return True
    
    except Exception as e:
        print(f"CrewAI测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# 替代方案：使用OpenAI客户端模拟CrewAI功能
def test_simple_agent():
    try:
        client = OpenAI(
            api_key='sk-rV0rXSp2Ml9xtRvyQkbqmQvnjTkXxsrWH04ASeZGZ3XIJDHj',
            base_url='https://api.nuwaapi.com/v1'
        )
        
        # 模拟Agent行为
        messages = [
            {"role": "system", "content": "你是一个研究员，擅长理解和总结信息。"},
            {"role": "user", "content": "简单回答：什么是CrewAI？"}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        print("简单代理测试成功!")
        print(f"响应: {response.choices[0].message.content}")
        return True
    
    except Exception as e:
        print(f"简单代理测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("开始测试API与CrewAI的兼容性...")
    print("-" * 50)

    # --- 修改开始 ---
    # 无条件执行所有测试，以收集完整信息

    # 测试1: LiteLLM直接调用
    print("\n测试1: LiteLLM直接调用")
    litellm_success = test_litellm_directly()
    print("-" * 50)

    # 测试2: CrewAI框架功能
    print("\n测试2: CrewAI框架功能")
    crewai_success = test_crewai() # 总是尝试运行
    print("-" * 50)

    # 测试3: 简单代理模拟（使用官方 OpenAI 库）
    print("\n测试3: 简单代理模拟")
    simple_agent_success = test_simple_agent() # 总是尝试运行
    print("-" * 50)

    # 测试结果总结 (可以保持不变或根据需要调整逻辑)
    print("\n测试结果总结:")
    print(f"LiteLLM Direct Call: {'✅ Succeeded' if litellm_success else '❌ Failed'}")
    print(f"CrewAI Framework Test: {'✅ Succeeded' if crewai_success else '❌ Failed'}")
    print(f"Simple Agent (OpenAI lib) Test: {'✅ Succeeded' if simple_agent_success else '❌ Failed'}")

    if litellm_success and crewai_success:
        print("\n结论: ✅ API 似乎与 LiteLLM 和 CrewAI 兼容。")
    elif simple_agent_success:
        print("\n结论: ⚠️ API 可以通过官方 OpenAI 库直接调用，但与 LiteLLM 或 CrewAI 的集成存在问题。")
        print("   -> 可能是 LiteLLM/CrewAI 对 API 响应格式的处理问题（例如非标准字段）。")
        print("   -> 尝试更新库或联系 API 提供商。")
    else:
         print("\n结论: ❌ API 调用在所有测试中都失败或 LiteLLM 基础调用失败。")
         print("   -> 请检查 API 配置、网络连接、API 服务状态，以及库版本。")
    # --- 修改结束 ---