import sentencepiece as spm

# Define a sample text corpus (usually much larger in real cases)
corpus = "sample_text.txt"

# Write some example sentences to a file for training purposes
with open(corpus, 'w') as f:
    f.write("Hello, how are you?\n")
    f.write("I am learning SentencePiece tokenization!\n")
    f.write("Tokenization is crucial in NLP.\n")
    f.write("With subword tokenization, we handle rare words effectively.\n")

# https://github.com/google/sentencepiece
# --input: one-sentence-per-line raw corpus file. No need to run tokenizer, normalizer or preprocessor. By default, SentencePiece normalizes the input with Unicode NFKC. You can pass a comma-separated list of files.
# --model_prefix: output model name prefix. <model_name>.model and <model_name>.vocab are generated.
# --vocab_size: vocabulary size, e.g., 8000, 16000, or 32000
# --character_coverage: amount of characters covered by the model, good defaults are: 0.9995 for languages with rich character set like Japanese or Chinese and 1.0 for other languages with small character set.
# --model_type: model type. Choose from unigram (default), bpe, char, or word. The input sentence must be pretokenized when using word type.
# Train SentencePiece model with the BPE algorithm
spm.SentencePieceTrainer.train(input=corpus, model_prefix='spm_model', vocab_size=100, model_type='bpe')

# This will create 'spm_model.model' and 'spm_model.vocab'


# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor(model_file='spm_model.model')

# Tokenize a new sentence
text = "Tokenization with SentencePiece is powerful."
tokens = sp.encode(text, out_type=str)
token_ids = sp.encode(text, out_type=int)

print("Tokens:", tokens)
print("Token IDs:", token_ids)
