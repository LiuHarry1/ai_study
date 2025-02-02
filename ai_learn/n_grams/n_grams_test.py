from sklearn.feature_extraction.text import CountVectorizer
from thefuzz import process
import re
from collections import defaultdict


# 生成 n-grams 和对应计数
def generate_ngrams_with_count(texts, ngram_range):
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=1,
        lowercase=True,
        token_pattern=r'\b[a-zA-Z0-9\-]+\b'
    )
    X = vectorizer.fit_transform(texts)

    # 获取 n-gram 特征名称和对应计数
    ngram_features = vectorizer.get_feature_names_out()
    ngram_counts = X.toarray().sum(axis=0)

    ngrams_with_count = dict(zip(ngram_features, ngram_counts))

    # 构建前缀索引
    prefix_index = defaultdict(list)
    for ngram in ngrams_with_count.keys():
        first_word = ngram.split()[0]
        prefix_index[first_word].append(ngram)

    return ngrams_with_count, prefix_index


# 使用 thefuzz 模糊匹配找到候选词
def get_closest_words(last_word, candidates, max_matches=3, dynamic_threshold=False):
    # 动态阈值，根据输入长度调整模糊匹配
    threshold = max(60, min(90, len(last_word) * 10)) if dynamic_threshold else 80
    closest_matches = process.extractBests(last_word, candidates, limit=max_matches, score_cutoff=threshold)
    return [match[0] for match in closest_matches]


# 根据最后一个单词预测高频后续序列
def predict_next_sequences(ngrams_with_count, prefix_index, user_input, top_k=5):
    input_prefix, last_word = split_user_input(user_input)
    # 从索引中获取候选单词
    candidates = prefix_index.keys()

    # 模糊匹配找到最接近的候选单词
    matched_words = get_closest_words(last_word, candidates)
    if not matched_words:
        return []

    # 收集匹配单词的所有 n-grams
    filtered_ngrams = {}
    for word in matched_words:
        for ngram in prefix_index[word]:
            filtered_ngrams[ngram] = ngrams_with_count[ngram]

    # 按频率排序
    sorted_ngrams = sorted(filtered_ngrams.items(), key=lambda x: x[1], reverse=True)

    # 合并模糊匹配和 n-gram 结果
    suggestions = [input_prefix+  " "+ ngram for ngram, _ in sorted_ngrams[:top_k]]
    return suggestions

def split_user_input(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()
    words = text.split()
    return " ".join(words[:-1]), words[-1].lower()



# 预处理文本
def preprocess(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).strip().lower()
    return text.split()


# 示例数据
texts = [
    "I love programming",
    "Programming is fun",
    "I love fun programming",
    "Migrate aspen-position-web component",
    "I love learning new technologies",
    "Love to learn programming",
    "Programming in Python is fun",
    "I love AI-programming",
    "I love programming languages"
]

# 生成 n-gram 计数和前缀索引
ngrams_with_count, prefix_index = generate_ngrams_with_count(texts, (2, 4))

# 用户输入
user_input = "I lo "  # 模糊输入

# 获取高频 n-grams
suggested_ngrams = predict_next_sequences(ngrams_with_count, prefix_index, user_input, top_k=5)

# 输出结果
print("User input:", user_input)
print("Suggested completions:")
for ngram in suggested_ngrams:
    print(ngram)
