import re
from typing import List, Dict

def normalize(text: str) -> str:
    """统一格式去空格、小写，便于匹配连写形式如 alert38"""
    return text.lower().replace(" ", "").replace("-", "")

def extract_required_phrases(query: str) -> List[str]:
    """
    从 query 中提取强相关关键词，未来可以添加更多领域规则
    """
    keywords = []

    # 提取 alert/task ID，支持 alert 38, alert-38, alert38 等
    ids = re.findall(r'(alert[\s-]?\d+|task[\s-]?\d+)', query, re.IGNORECASE)
    keywords.extend([normalize(k) for k in ids])

    # 精确短语（可以继续添加）
    phrases = ["initial notification", "final report", "incident summary"]
    for phrase in phrases:
        if phrase in query.lower():
            keywords.append(phrase.lower())  # 保留空格用于精确匹配

    return keywords

def filter_docs_by_strict_topic(query: str, docs: List[Dict]) -> List[Dict]:
    """
    如果 query 中提到 alert 38 或 initial notification，doc 必须也包含这些内容。
    """
    required = extract_required_phrases(query)

    def is_valid(doc):
        content = doc['content'].lower()
        content_normalized = normalize(content)

        for kw in required:
            if " " in kw:
                # 精确短语
                if kw not in content:
                    return False
            else:
                # 连写 alert38 / task123
                if kw not in content_normalized:
                    return False
        return True

    return [doc for doc in docs if is_valid(doc)]


query = "Can I see the initial notification for alert 38?"

docs = [
    {"content": "This is the initial notification for Alert 38."},         # ✅
    {"content": "Alert38 was resolved by John."},                           # ❌ (no "initial notification")
    {"content": "Initial notification is missing for Alert 37."},          # ❌ (wrong alert)
    {"content": "The alert 38 triggered an escalation process."},          # ❌ (no "initial notification")
]

related_docs = filter_docs_by_strict_topic(query, docs)
print(related_docs)