import os
import json
from datasets import Dataset

# Define dataset directories
IMDB_DIR = "D:/train_data/aclImdb/aclImdb"
train_dir = IMDB_DIR+"/train"
test_dir = IMDB_DIR+"/test"

def load_imdb_data(directory):
    data = []
    for label in ["pos", "neg"]:
        folder = os.path.join(directory, label)
        for file_name in os.listdir(folder):
            with open(os.path.join(folder, file_name), 'r',  encoding='utf-8') as f:
                text = f.read().strip()
                data.append({"text": text, "label": 1 if label == "pos" else 0})
    return data

# Load train and test datasets
train_data = load_imdb_data(train_dir)
test_data = load_imdb_data(test_dir)

# Create Hugging Face Datasets
train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(r"C:\apps\ml_model\llama3-8b-instruction-hf\llama-3-8b-chat-hf")
model = LlamaForCausalLM.from_pretrained("huggingface/llama-3b")
tokenizer.pad_token = tokenizer.eos_token
# Configure PEFT LoRA
lora_config = LoraConfig(
    r=16, # rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], # Layer modules for LLaMA
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply the LoRA model
model = get_peft_model(model, lora_config)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=200)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

from transformers import Trainer, TrainingArguments

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Fine-tune the model
trainer.train()

# Evaluate the model
trainer.evaluate()

model.save_pretrained("fine_tuned_llama_lora")
tokenizer.save_pretrained("fine_tuned_llama_lora")




