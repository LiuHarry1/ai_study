from google.cloud import vertex_ai
# 示例输入
query = "What is the status of alert 38?"
docs = [
    "Alert 38 has been resolved, no further actions required.",
    "Alert 37 is pending review.",
    "The task related to Alert 38 is completed.",
]


# 第一步：定义评分函数
def gemini_rerank(query, docs):
    client = vertex_ai.PredictionServiceClient()
    endpoint = "your-gemini-endpoint"  # 填入你的Gemini API端点

    inputs = [{"query": query, "doc": doc} for doc in docs]

    response = client.predict(endpoint=endpoint, instances=inputs)

    scores = response.predictions  # 这里根据实际返回的格式提取评分
    return scores


# 第二步：调用 Gemini 进行 rerank
scores = gemini_rerank(query, docs)

# 第三步：根据分数重新排序
ranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]

# 输出排序后的文档
for doc in ranked_docs:
    print(doc)
