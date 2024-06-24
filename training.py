from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import pandas as pd
import torch

# Fonction pour charger et préparer les données
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
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialiser le tokenizer et le modèle
tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-medium',token='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')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2-medium',token='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')
model.to(device)

# Charger les données et les tokeniser
texts = load_data('steam_data.csv')
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

# Créer le dataset
dataset = ReviewsDataset(encodings)

# Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)
trainer.train()

model_path = './Farfadet'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
