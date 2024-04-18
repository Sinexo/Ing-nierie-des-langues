from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import pandas as pd
import torch

# Fonction pour charger et préparer les données
def load_data(filename):
    df = pd.read_csv(filename)
    df['name'] = df['name'].astype(str)
    df['review'] = df['review'].astype(str)
    df['input_text'] = df['name'] + " [SEP] " + df['review']
    return df['input_text'].tolist()

# Classe pour le dataset
class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

# Initialiser le tokenizer et le modèle
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Charger les données et les tokeniser
texts = load_data('cleaned_complete_reviews_data.csv')
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

# Créer le dataset
dataset = ReviewsDataset(encodings)

# Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialiser et lancer le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)
trainer.train()

def generate_text(prompt):
    tokenizer.pad_token = tokenizer.eos_token  # Assurez-vous que le pad_token est défini
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Générer un texte
generated_text = generate_text("Skyrim [SEP]")
print(generated_text)
