#git clone https://github.com/huggingface/transformers.git
#cd transformers
#pip install -e .

#sudo pip install datasets

#find . -name "plain_text"
#./.cache/huggingface/datasets/Skylion007___openwebtext/plain_text
#./.cache/huggingface/hub/datasets--Skylion007--openwebtext/snapshots/b4325f019c648b1641a1784748667e8b74e5e064/plain_text


import torch

#test environment
x = torch.rand(5, 3)
print(x)

cudaA = torch.cuda.is_available()
print(f"Cuda available {cudaA}")
print("GPU count:", torch.cuda.device_count())
print("Current GPU:", torch.cuda.current_device())
print("Name:", torch.cuda.get_device_name(0))

import transformers
print(transformers.__file__)


# Run gpt-2
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # practical fix for batching

prompt = "In a distant observatory,"
inputs = tokenizer(prompt, return_tensors="pt")

out = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
)

print(tokenizer.decode(out[0], skip_special_tokens=True))





# -----------------------------
# Dataset
# -----------------------------
from datasets import load_dataset
from pathlib import Path
from datasets import load_from_disk

LM_CACHE = Path("./lm_cache")
OUTPUT_DIR = Path("./lm_output")

DATASET_NAME = "Skylion007/openwebtext"   # open WebText-style corpus
dataset = load_dataset(DATASET_NAME)

# use train split only for the tiniest reproducible example
raw_train = dataset["train"]


BLOCK_SIZE = 1024

def tokenize_fn(batch):
    return tokenizer(batch["text"])

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i:i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


tokenized = raw_train.map(
    tokenize_fn,
    batched=True,
    remove_columns=raw_train.column_names,
)

#lm_dataset = load_from_disk(LM_CACHE)
lm_dataset = tokenized.map(group_texts, batched=True)
lm_dataset.save_to_disk(LM_CACHE)


# tiny eval split carved out of train for reproducibility
split = lm_dataset.train_test_split(test_size=0.01, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]


from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer
# -----------------------------
# Collator
# -----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# -----------------------------
# Training
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    #overwrite_output_dir=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    max_steps=1000,
    logging_steps=20,
    eval_steps=100,
    save_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    report_to="none",
    #fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)