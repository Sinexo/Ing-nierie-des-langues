from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch

# Chemin du modèle sauvegardé
model_path = './Hater'

# Recharger le modèle et le tokenizer
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Configurer le périphérique
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def generate_text(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        padding='max_length',
        max_length=600,  # La longueur maximale en tokens
        truncation=True,
        pad_to_max_length=True,
        padding_side='left'
    ).to(device)

    # Utilisez `do_sample=True` pour permettre la génération basée sur l'échantillonnage
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=1000,  # Définit la longueur maximale de sortie
        min_length=100,  # Définit la longueur minimale de sortie pour s'assurer d'atteindre au moins 500 caractères
        num_return_sequences=1,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        no_repeat_ngram_size=2,
        do_sample=True
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text


# Utiliser le modèle pour générer du texte
input_prompt = "Make a negative review of the game Hollow Knight"
generated_text = generate_text(input_prompt)
print('\n',generated_text)
