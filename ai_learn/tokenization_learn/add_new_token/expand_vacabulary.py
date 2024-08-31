from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from config import *
import os



def add_new_token(new_tokens, current_model_path, new_model_path):

    print('loading MiniLM model...')
    tokenizer = AutoTokenizer.from_pretrained(current_model_path)
    model = AutoModel.from_pretrained(current_model_path)

    # Get the current vocabulary
    vocab = tokenizer.get_vocab()

    print(len(vocab))

    # Add new tokens to the tokenizer
    tokenizer.add_tokens(new_tokens)

    # Resize the token embeddings to match the new vocabulary size
    print(len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))

    # Save the updated tokenizer
    tokenizer.save_pretrained(new_model_path)
    # Save the fine-tuned model
    model.save_pretrained(new_model_path)

def test_new_token(new_token, model_path):

    print('loading MiniLM model...')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModel.from_pretrained(model_path)

    # Get the current vocabulary
    vocab = tokenizer.get_vocab()

    print(len(vocab))

    # List of words to check
    words_to_check = new_token

    # Check if words exist in the vocabulary
    for word in words_to_check:
        if word in vocab:
            print(f"'{word}' exists in the vocabulary.")
        else:
            print(f"'{word}' does not exist in the vocabulary.")


    # Check if tokens exist in the vocabulary
    for word in new_token:
        tokens = tokenizer.tokenize(word)
        print(tokens)
        if all(token in vocab for token in tokens):
            print(f"All tokens for '{word}' exist in the vocabulary.")
        else:
            print(f"Some tokens for '{word}' do not exist in the vocabulary.")


if __name__ == '__main__':
    current_model_name = r'/Users/harry/Documents/apps/ml/all-MiniLM-L6-v2'
    new_model_path = "MiniLM"
    new_tokens = ["alert38", "alert 39", "alert 111", "default notification", "initial notification"]
    add_new_token(new_tokens, current_model_name, new_model_path)
    test_new_token(new_tokens, new_model_path)