from sklearn.feature_extraction.text import CountVectorizer
import re


# 生成 n-grams 和对应计数
def generate_ngrams_with_count(texts, ngram_range):
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=1,
        lowercase=False,
        token_pattern=r'\b[a-zA-Z0-9\-]+\b'
    )
    X = vectorizer.fit_transform(texts)

    # 获取 n-gram 特征名称和对应计数
    ngram_features = vectorizer.get_feature_names_out()
    ngram_counts = X.toarray().sum(axis=0)

    ngrams_with_count = dict(zip(ngram_features, ngram_counts))
    return ngrams_with_count


# 根据最后一个单词预测高频后续序列
def predict_next_sequences(ngrams_with_count, last_word, top_k=5):
    # 过滤以最后一个单词开头的 n-grams
    filtered_ngrams = {
        ngram: count
        for ngram, count in ngrams_with_count.items()
        if ngram.startswith(last_word)
    }

    # 按频率排序
    sorted_ngrams = sorted(filtered_ngrams.items(), key=lambda x: x[1], reverse=True)

    # 取出前 top_k 高频 n-grams
    suggestions = [ngram for ngram, _ in sorted_ngrams[:top_k]]
    return suggestions


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
    "Programming in Python is fun"
]

# 生成 n-gram 计数
ngrams_with_count = generate_ngrams_with_count(texts, (2, 5))

# 用户输入
user_input = "I love"
tokens = preprocess(user_input)
last_word = tokens[-1] if tokens else ""

# 获取高频 n-grams
suggested_ngrams = predict_next_sequences(ngrams_with_count, last_word, top_k=5)

# 输出结果
print("Last word:", last_word)
print("Suggested n-grams:")
for ngram in suggested_ngrams:
    print(ngram)
