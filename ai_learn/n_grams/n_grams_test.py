from sklearn.feature_extraction.text import CountVectorizer


def generate_ngrams_with_count(texts, range):
    vectorizer = CountVectorizer(ngram_range=range , min_df=1, lowercase=False, token_pattern=r'\b[a-zA-Z0-9\-]+\b')  # 指定 n-gram 范围
    X = vectorizer.fit_transform(texts)  # 转换文本到稀疏矩阵形式

    # 获取 n-gram 特征名称（每个 n-gram）和对应的计数
    ngram_features = vectorizer.get_feature_names_out()
    ngram_counts = X.toarray().sum(axis=0)  # 每个 n-gram 的总计数

    ngrams_with_count =  dict(zip(ngram_features, ngram_counts))

    sorted_ngrams = dict(sorted(ngrams_with_count.items(), key=lambda x: x[1], reverse=True))
    # print(sorted_ngrams)

    return sorted_ngrams


# 示例
texts = [" I love programming", "Programming is fun", " I love fun programming", "migrate aspen-position-web component"]
ngrams_with_count = generate_ngrams_with_count(texts, (1,2))

print(ngrams_with_count)


