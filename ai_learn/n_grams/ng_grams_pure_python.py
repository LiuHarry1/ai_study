import re
from collections import defaultdict, Counter


# 预处理文本，分词并转化为小写
def preprocess(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    return text.split()


# 构建 n-gram 模型
def build_ngram_model(data, n=2):
    ngram_model = defaultdict(Counter)
    for sentence in data:
        tokens = preprocess(sentence)
        for i in range(len(tokens) - n + 1):
            prefix = tuple(tokens[i:i + n - 1])
            next_word = tokens[i + n - 1]
            ngram_model[prefix][next_word] += 1
    return ngram_model


# 生成后续单词序列
def suggest_next_sequences(ngram_model, input_text, max_length=4):
    tokens = preprocess(input_text)
    last_word = tokens[-1] if tokens else ""
    current_prefix = (last_word,)

    suggestions = []
    for _ in range(max_length):
        if current_prefix in ngram_model:
            # 根据频率排序，选择下一个单词
            next_word = ngram_model[current_prefix].most_common(1)[0][0]
            suggestions.append(next_word)
            current_prefix = (next_word,)
        else:
            break
    return suggestions


# 示例数据
jira_subjects = [
    "Fix bug in user login",
    "Update user profile settings",
    "Add new feature to dashboard",
    "Improve performance of user search",
    "Fix issue with user authentication"
]

# 构建 bi-gram 模型
ngram_model = build_ngram_model(jira_subjects, n=3)

# 输入提示
user_input = "user"
suggested_sequence = suggest_next_sequences(ngram_model, user_input, max_length=4)
print("Suggested sequence:", suggested_sequence)
