import sentencepiece as spm

# Define a sample text corpus (usually much larger in real cases)
corpus = "sample_text.txt"

# Write some example sentences to a file for training purposes
with open(corpus, 'w') as f:
    f.write("Hello, how are you?\n")
    f.write("I am learning SentencePiece tokenization!\n")
    f.write("Tokenization is crucial in NLP.\n")
    f.write("With subword tokenization, we handle rare words effectively.\n")

# Train SentencePiece model with the BPE algorithm
spm.SentencePieceTrainer.train(input=corpus, model_prefix='spm_model', vocab_size=100, model_type='bpe')

# This will create 'spm_model.model' and 'spm_model.vocab'

import sentencepiece as spm

# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor(model_file='spm_model.model')

# Tokenize a new sentence
text = "Tokenization with SentencePiece is powerful."
tokens = sp.encode(text, out_type=str)
token_ids = sp.encode(text, out_type=int)

print("Tokens:", tokens)
print("Token IDs:", token_ids)
