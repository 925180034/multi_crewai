# test_nuwa_api.py - 测试Nuwa API连接
import os
import logging
from dotenv import load_dotenv
import litellm
import requests

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 启用调试日志
os.environ['LITELLM_LOG'] = 'DEBUG'

def test_direct_api_call():
    """测试直接API调用"""
    try:
        logger.info("测试直接API调用...")
        
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE", "https://api.nuwaapi.com/v1")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello, this is a test"}],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(f"{api_base}/chat/completions", headers=headers, json=data)
        
        logger.info(f"Status code: {response.status_code}")
        logger.info(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("\n✅ 直接API调用成功")
            return True
        else:
            print(f"\n❌ 直接API调用失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        return False

def test_litellm_completion():
    """测试litellm调用"""
    try:
        logger.info("测试litellm调用...")
        
        # 尝试不同的配置
        models_to_test = ["gpt-4o-mini"]
        
        for model in models_to_test:
            logger.info(f"尝试模型: {model}")
            try:
                response = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": "Hello, this is a test"}],
                    api_base="https://api.nuwaapi.com/v1",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    max_tokens=50
                )
                logger.info(f"成功！模型: {model}")
                print(f"\n✅ LiteLLM调用成功 (模型: {model})")
                print(f"响应: {response.choices[0].message.content}")
                return True
            except Exception as e:
                logger.error(f"模型 {model} 失败: {str(e)}")
                continue
        
        return False
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("开始测试Nuwa API配置...")
    print("=" * 50)
    
    # 测试环境变量
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 未设置")
        exit(1)
    
    print(f"✓ OPENAI_API_KEY: {api_key[:10]}...")
    print(f"✓ OPENAI_API_BASE: {api_base}")
    
    # 运行测试
    if test_direct_api_call():
        print("\n✅ 直接API调用测试通过")
    else:
        print("\n❌ 直接API调用测试失败")
    
    if test_litellm_completion():
        print("✅ LiteLLM测试通过")
    else:
        print("❌ LiteLLM测试失败")
        
    print("\n" + "=" * 50)