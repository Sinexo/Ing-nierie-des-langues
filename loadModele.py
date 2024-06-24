from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
import torch

# model_path = './Farfadet'
#Pour pouvoir déposer l'archive du projet sur moodle, je ne peux pas fournir le modèle ni le corpus avec.
model = GPT2LMHeadModel.from_pretrained('Sinexo/Farfadet-2')
tokenizer = GPT2Tokenizer.from_pretrained('Sinexo/Farfadet-2')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def generate_text(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        padding='max_length',
        max_length=200,
        truncation=True,
        pad_to_max_length=True
    ).to(device)

    
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=400, 
        min_length=20,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.8,
        no_repeat_ngram_size=1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


prompt = "What is your opinion on Safeo ?"
generated_text = generate_text(prompt)
print('\n',generated_text)


input_prompt = "What is your opinion on Safeo"
generated_text = generate_text(input_prompt)
print("\n \n", generated_text)
