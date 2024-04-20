from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
import torch

model_path = './Hater'

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def generate_text(prompt):
    # Encodage simplifié
    inputs = tokenizer.encode_plus(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=200,  # Longueur maximale de l'entrée
        padding='longest'
    ).to(device)
    
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=150,  # Longueur supplémentaire du texte généré
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

input_prompt = "Helldivers 2"
generated_text = generate_text(input_prompt)
print("\n \n", generated_text)
