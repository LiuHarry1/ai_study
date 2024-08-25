from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents
from tokenizers.trainers import WordPieceTrainer

# Define a larger sample dataset for better training
texts = [
    "Hello, how are you?",
    "I am learning WordPiece tokenization!",
    "Tokenization is crucial in NLP.",
    "Subword tokenization is a method of breaking words into smaller units.",
    "With subword tokenization, words like 'unhappy' can be split into 'un' and 'happy'.",
    "This approach helps in handling unseen words effectively."
]

# Initialize the WordPiece model with an unknown token
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

# Set normalizer and pre-tokenizer
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = Whitespace()

# Define a WordPiece trainer with a larger vocabulary size
trainer = WordPieceTrainer(vocab_size=100, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# Train the tokenizer on the provided texts
tokenizer.train_from_iterator(texts, trainer)

# Tokenize a new sentence
output = tokenizer.encode("Tokenization with WordPiece is powerful.")
print("Tokens:", output.tokens)
print("IDs:", output.ids)
