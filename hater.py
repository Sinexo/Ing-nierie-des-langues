import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
# torch.set_num_threads(torch.get_num_threads())
# torch.cuda.is_available = lambda: False
def load_data(filename):
    df = pd.read_csv(filename)
    df['name'] = df['name'].astype(str)
    df['review'] = df['review'].astype(str)
    df['input_text'] = df['name'] + " [SEP] " + df['review']
    return df['input_text'].tolist()

class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()  # Add labels for the model to calculate loss
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

tokenizer = GPT2Tokenizer.from_pretrained('distilbert/distilgpt2')  # Use the small variant of GPT-2
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('distilbert/distilgpt2')  # Use the small variant of GPT-2

texts = load_data('steam_data.csv')
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
dataset = ReviewsDataset(encodings)

# Depending on your available hardware, you might adjust this. For example, if you're using a GPU with 11GB of RAM, you might be able to go up to 8 or higher.
# Always monitor your GPU usage to find the optimal batch size without running out of memory.
 # Increase the batch size if your GPU can handle it

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,  # Updated batch size
    warmup_steps=250,
    weight_decay=0.01,
    logging_dir='./logs',
    per_device_train_batch_size=1,  # Increase if your GPU can handle it
    # gradient_accumulation_steps=8,  # Use if you need to simulate a larger batch size
    logging_steps=10,  # Log less frequently to reduce I/O overhead
    save_strategy="epoch",  # Save checkpoints less frequently
    # fp16=True,  # Enable this if your GPU supports FP
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)
trainer.train()

def generate_text(prompt):
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

generated_text = generate_text("Skyrim [SEP]")
print(generated_text)
