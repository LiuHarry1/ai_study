from datetime import datetime
from typing import List, Dict

def rerank_jira_data(jira_data: List[Dict], weight_score: float = 0.7, weight_recency: float = 0.3):
    # Normalize score and recency to [0,1] and apply weights
    now = datetime.now()

    # Compute max/min for normalization
    max_score = max(item['score'] for item in jira_data)
    min_score = min(item['score'] for item in jira_data)
    max_age_days = max((now - item['created_date']).days for item in jira_data)
    min_age_days = min((now - item['created_date']).days for item in jira_data)

    def normalize_score(score):
        return (score - min_score) / (max_score - min_score + 1e-5)

    def normalize_recency(created_date):
        age_days = (now - created_date).days
        return 1 - (age_days - min_age_days) / (max_age_days - min_age_days + 1e-5)

    # Compute combined score
    for item in jira_data:
        norm_score = normalize_score(item['score'])
        norm_recency = normalize_recency(item['created_date'])
        item['combined_score'] = weight_score * norm_score + weight_recency * norm_recency

    # Sort by combined_score descending
    return sorted(jira_data, key=lambda x: x['combined_score'], reverse=True)

jira_data = [
    {"id": "JIRA-1", "score": 0.8, "created_date": datetime(2024, 4, 8)},
    {"id": "JIRA-2", "score": 0.9, "created_date": datetime(2025, 3, 15)},
    {"id": "JIRA-3", "score": 0.7, "created_date": datetime(2025, 4, 7)},
]

# ranked = rerank_jira_data(jira_data)
# for item in ranked:
#     print(item["id"], item["combined_score"])

import math
def exponential_decay_rerank(jira_data: List[Dict], weight_score: float = 0.7, weight_recency: float = 0.3,
                     half_life_days: float = 7.0):
    now = datetime.now()

    # Score normalization
    max_score = max(item['score'] for item in jira_data)
    min_score = min(item['score'] for item in jira_data)

    def normalize_score(score):
        return (score - min_score) / (max_score - min_score + 1e-5)

    def exponential_decay(created_date):
        age_days = (now - created_date).days
        decay = math.exp(-math.log(2) * age_days / half_life_days)
        return decay  # Closer to 1 if newer, decays towards 0

    # Compute combined score
    for item in jira_data:
        norm_score = normalize_score(item['score'])
        decay_score = exponential_decay(item['created_date'])
        item['combined_score'] = weight_score * norm_score + weight_recency * decay_score

    return sorted(jira_data, key=lambda x: x['combined_score'], reverse=True)

jira_data = [
    {"id": "JIRA-1", "score": 0.8, "created_date": datetime(2024, 4, 8)},
    {"id": "JIRA-2", "score": 0.9, "created_date": datetime(2025, 3, 15)},
    {"id": "JIRA-3", "score": 0.7, "created_date": datetime(2025, 4, 7)},
]

ranked = exponential_decay_rerank(jira_data)
for item in ranked:
    print(item["id"], item["combined_score"])