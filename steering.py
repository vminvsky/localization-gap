import gc 
import sys
import os
from pathlib import Path
import json
import pandas as pd
sys.path.append('llm-localization')
from tools.contrastiveact import contrastive_act_gen_opt
from tqdm import trange
from nnsight import LanguageModel
from transformers import AutoTokenizer
import torch as t 
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import Dataset


def messages_to_str(messages, tokenizer, instruction_model=True):
    if type(messages) == str:
        messages = [{"role":"user", "content":messages}]
    if instruction_model:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

def prepare_prompts(lang, run_with_suffix=False):
    with open(f'data/open_ended_generation/{lang}_{run_with_suffix}.json', 'r') as f:
        stories = json.load(f)
    return stories



SAMPLE_SIZE = 50
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", device_map='auto')
nnmodel = LanguageModel('google/gemma-2-9b-it', 
                        device_map='auto', 
                        dispatch=True, 
                        torch_dtype=t.bfloat16)

alpha = 2

def generate_prompts_chat(lang: str = 'en', run_with_suffix=False):
    fpath = f'data/open_ended_generation/{lang}_{run_with_suffix}.json'
    data = json.load(open(fpath, 'r'))
    return messages_to_str([{"role": "user", "content": prompt['translation']} for prompt in data], tokenizer)
    

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

batch_size = 64

@dataclass
class Output:
    alpha: float
    layer: int
    prompt: str 
    lang: str
    suffix: bool 

for lang in langs:
    steering_vec = t.load(f'steering/gemma2_9b_it/per_culture/{lang}_trans_avg_all_tasks.pt').unsqueeze(1)

    for suffix in suffixes:
        outputs = []

        prompts = prepare_prompts(lang, suffix)
        prompt_formats = generate_prompts_chat(lang, suffix)

        prompt_batch = [prompt_formats[i:i + batch_size] for i in range(0, len(prompt_formats), batch_size)]
        for batch in prompt_batch:
            with t.no_grad():
                out = contrastive_act_gen_opt(nnmodel, tokenizer, 
                                            alpha * steering_vec[25].unsqueeze(0), 
                                            prompt=batch, layer=[25], 
                                            n_new_tokens=256, use_sampling=False)
                for j,layer in enumerate(out[0]):
                    texts = out[0][layer]
                    probs = out[1]
                    epsilon = 1e-6
                    probs[probs < epsilon] = 0

                    for k, text in enumerate(texts):
                        out_dict = {"prompt": batch[k], "alpha": alpha, "steer_out": text, "steer_prob": probs[j,k,:,:].to_sparse(), "layer": layer}
                        out_dict.update(batch_entries[i][k])
                        outputs.append(out_dict)
                        pass
            t.cuda.empty_cache() # Clear GPU memory after each alpha 
        output_filename = f'data/open_ended_generation/steering_{alpha}/{lang}_{suffix}.json'

        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

        del out
        gc.collect()
        t.cuda.empty_cache()  