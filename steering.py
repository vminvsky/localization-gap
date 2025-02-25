import gc 
import sys
import os
from pathlib import Path
import json

sys.path.append('llm-localization')
from tools.contrastiveact import contrastive_act_gen_opt
from tqdm import trange
from nnsight import LanguageModel
from transformers import AutoTokenizer
import torch as t 
from tqdm import tqdm



SAMPLE_SIZE = 50
tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/vv7118/models/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819/")
nnmodel = LanguageModel('/scratch/gpfs/vv7118/models/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819/', 
                        device_map='cuda:0', 
                        dispatch=True, 
                        torch_dtype=t.bfloat16)

alpha = 2

import pandas as pd 
df = pd.read_csv('data/all_models_eval_subset.csv')
df['swapped'] = False

if True:
    swapped = pd.read_csv('data/all_models_eval_subset_swapped.csv')
    df = pd.concat([df, swapped], ignore_index=True)

def messages_to_str(messages, tokenizer, instruction_model=True):
    if type(messages) == str:
        messages = [{"role":"user", "content":messages}]
    if instruction_model:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# prompts = df[df['country'] == 'Russia']['input']

langs = ['us', 'ru', 'fr', 'bn','tr']
lang_mapping = {
    'us': 'English',
    'ru': 'Russian',
    'fr': 'French',
    'bn': 'Bengali',
    'tr': 'Turkish'
}

country_mapping = {
    'us': 'United States',
    'ru': 'Russia',
    'fr': 'France',
    'bn': 'Bangladesh',
    'tr': 'Turkey'
}

for lang in langs:
    steering_vec = t.load(f'llm-localization/gemma2_9b_it/universal/trans_universal_{lang}_out.pt')
    # steering_vec = t.load('llm-localization/gemma2_9b_it/universal/en_universal_all_cultures.pt')

    # Create steering_outputs directory if it doesn't exist
    output_dir = Path(f'steering_outputs/translated_steering/{lang}/')
    output_dir.mkdir(exist_ok=True)


    df2 = df[df['lang'] == lang_mapping[lang]]
    df2 = df2[df2['country'] == country_mapping[lang]]
    df2 = df2[df2['hint'] == False]
    df2 = df2.drop_duplicates(subset=['prompt'])
    prompts_to_limit_to = df2.groupby(['subtask'])['key'].sample(SAMPLE_SIZE, random_state=42)
    df2 = df2[df2['key'].isin(prompts_to_limit_to.values)]
    df2 = df2.drop_duplicates(subset=['key', 'swapped'])

    print(df2.groupby(['subtask'])['key'].count())

    print("Processing", len(df2), "questions")

    for i, row in tqdm(df2.iterrows(), total=len(df2)):
        # Check if output file already exists
        output_path = output_dir / f'output_{i}.json'
        if output_path.exists():
            continue
            
        prompt = row['prompt']
        question = row['question']

        prompt = prompt.replace(',3,4', '')
        prompt = messages_to_str(prompt, tokenizer, instruction_model=True)
        out = contrastive_act_gen_opt(nnmodel, tokenizer, 
                                      alpha * steering_vec[25].unsqueeze(0), 
                                      prompt=prompt, layer=[25], 
                                      n_new_tokens=50, use_sampling=False)
        out = out[0]
        # Save the output to a json file
        output_data = {
            "prompt": prompt,
            "output": out[25][0],
            "alpha": alpha,
            "layers": [25],
            "question": question,
            "ans_local_idx": row['ans_local_idx'],
            "ans_west_idx": row['ans_west_idx'],
            'country': row['country'],
            'lang': row['lang'],
            'subtask': row['subtask'],
            'ans_local': row['ans_local'],
            'swapped': row['swapped'],
            'key': row['key'],
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        del out
        gc.collect()
        t.cuda.empty_cache()  