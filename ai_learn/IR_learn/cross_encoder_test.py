# https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2

from sentence_transformers import CrossEncoder

model = CrossEncoder('C:/apps/ml_model/cross-encoder/ms-marco-MiniLM-L6-v2')
scores = model.predict([
    ("How many people live in Berlin?", "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."),
    ("How many people live in Berlin?", "Berlin is well known for its museums."),
])
print(scores)
# [ 8.607138 -4.320078]

#
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
#
# model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L6-v2')
# tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L6-v2')
#
# features = tokenizer(['How many people live in Berlin?', 'How many people live in Berlin?'], ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.'],  padding=True, truncation=True, return_tensors="pt")
#
# model.eval()
# with torch.no_grad():
#     scores = model(**features).logits
#     print(scores)
