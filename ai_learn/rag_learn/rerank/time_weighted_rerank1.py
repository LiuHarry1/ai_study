from datetime import datetime
import math


"""

recency_score(t)=e−λ⋅t

final_score= w1⋅semantic_score+w2⋅recency_score(t)

"""


jira_items = [
    {"id": "JIRA-101", "summary": "Fix login bug", "score": 0.85, "created_date": "2023-01-08"},
    {"id": "JIRA-102", "summary": "Improve search speed", "score": 0.75, "created_date": "2025-04-04"},
    {"id": "JIRA-103", "summary": "Update UI layout", "score": 0.65, "created_date": "2025-03-25"},
    {"id": "JIRA-104", "summary": "Fix regression issue", "score": 0.90, "created_date": "2021-03-10"},
]
def exponential_decay_score(created_date: str, half_life_days: float = 7.0) -> float:
    now = datetime.now()
    created = datetime.strptime(created_date, "%Y-%m-%d")
    age_days = (now - created).days
    decay = math.exp(-math.log(2) * age_days / half_life_days)
    return decay

def rerank_jira_items(jira_items, weight_score=0.8, weight_time=0.2, half_life_days=5.0):
    def final_score(item):
        recency = exponential_decay_score(item["created_date"], half_life_days)
        combined = weight_score * item["score"] + weight_time * recency
        item['final_score'] = combined
        return combined

    return sorted(jira_items, key=final_score, reverse=True)


reranked_items = rerank_jira_items(jira_items)

for item in reranked_items:
    print(f'{item["id"]}: {item["summary"]} | score={item["score"]:.2f} | date={item["created_date"]}, final_socre ={item["final_score"]:.4f} ')
