from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments

model_name = r"/Users/harry/Documents/apps/ml/llama-2-7b-chat"

def load_data_from_json(json_file = 'test1.json'):
    # Load the dataset from the JSON file
    dataset = load_dataset('json', data_files=json_file)

    # Explore the dataset
    print(dataset)

    # Check the first entry in the dataset
    print(dataset['train'][0])
    return dataset

def tokenize_dataset(dataset):

    print("starting to load tokenizer.")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Tokenize the dataset
    def tokenize_function(examples):
        prompt = examples['input']
        response = examples['output']
        input_text = prompt + "\n" + response  # Combine the prompt and response
        # return tokenizer(input_text, truncation=True, padding="max_length", max_length=50)
        # Tokenize input text
        encodings = tokenizer(input_text, truncation=True, padding="max_length", max_length=50)

        # Set the labels for training (predict the next token)
        encodings['labels'] = encodings['input_ids'].copy()
        return encodings


    # Apply the tokenizer to the entire dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=False)


    # Check the tokenized data
    print(tokenized_dataset['train'][0])
    return tokenized_dataset

def tokenize_dict_dataset(dataset):



    print("starting to load tokenizer.")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Tokenize the dataset
    def tokenize_function1(examples):
        texts = []

        for input, output in zip(examples["input"], examples['output']):
            texts.append(input + "\n" + output)
        examples['text'] = texts

        examples['labels'] = tokenizer(texts, truncation=True, padding="max_length", max_length=50)
        return examples

    # Apply the tokenizer to the entire datasetexamples
    tokenized_dataset = dataset.map(tokenize_function1, batched=False)
    print(tokenized_dataset)
    # Check the tokenized data
    print(tokenized_dataset['train'][0])
    return tokenized_dataset
if __name__ == '__main__':
    dataset = load_data_from_json("test1.json")
    tokenize_dataset(dataset)
    # dataset = load_data_from_json("test2.json")
    # tokenize_dict_dataset(dataset)