from openai import OpenAI

# 初始化客户端
client = OpenAI(api_key='sk-Tfx85QA4Fs6APoiEk7lSDrVUnWWGYgTSAR5aZlu7TxToTqGO', base_url='https://api.feidaapi.com/v1')

def test_embedding_model():
    try:
        # 输入文本
        text = "Hello, OpenAI! This is a test for the embedding model."

        # 发送请求生成嵌入向量
        response = client.embeddings.create(
            model="text-embedding-3-large",  # 使用 text-embedding-3-large 模型
            input=text
        )
        
        # 打印 API 的响应
        print("OpenAI API Response:")
        print(response)
        
        # 提取嵌入向量
        embedding = response.data[0].embedding
        print("Generated Embedding Vector:")
        print(embedding)
        
        print("Embedding model test is successful!")
    
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    test_embedding_model()