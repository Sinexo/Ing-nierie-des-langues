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
        # Les tenseurs sont créés sur la CPU ici
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Vérification de la disponibilité du GPU et configuration du périphérique
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialiser le tokenizer et le modèle
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
model.to(device)  # Déplacer le modèle sur le GPU si disponible

# Charger les données et les tokeniser
texts = load_data('steam_data.csv')
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

# Créer le dataset
dataset = ReviewsDataset(encodings)

# Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=24,
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


# Sauvegarder le modèle et le tokenizer
model_path = './Hater'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

def generate_text(prompt):
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(
        inputs,
        max_length=200,
        num_return_sequences=1,
        temperature=0.9,   # Ajustez pour moins de prévisibilité
        top_k=50,           # Les 50 meilleurs tokens sont pris en compte à chaque pas
        top_p=0.95,         # Utilisation du nucleus sampling
        no_repeat_ngram_size=2  # Empêche la répétition de n-grams jusqu'à une taille de 2
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


# Générer un texte
generated_text = generate_text("What do you think about Helldivers 2 ?")
print(generated_text)