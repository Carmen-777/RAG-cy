import os
from dotenv import load_dotenv
import requests

# 1. 加载 .env 文件
load_dotenv()

# 2. 获取 Key
api_key = os.getenv("JINA_API_KEY")

if not api_key:
    print("❌ 失败：没找到 JINA_API_KEY，请检查 .env 文件路径或拼写")
else:
    print(f"✅ 成功读取到 Key (前缀): {api_key[:10]}...")
    
    # 3. 尝试调用 Jina API
    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": "中芯国际被制裁",
        "documents": ["中芯国际被列入实体清单", "今天天气真不错"]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print("🎉 验证通过！API 返回正常，重排序服务可用。")
            print(f"   测试分数: {response.json()['results'][0]['score']}")
        else:
            print(f"❌ API 报错: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ 网络请求出错: {e}")