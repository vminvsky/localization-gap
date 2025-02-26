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
    return [messages_to_str([{"role": "user", "content": prompt['translation']}], tokenizer) for prompt in data]
    

langs = ['en', 'ru', 'fr', 'bn','tr']
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

batch_size = 32
num_samples = 30  # Number of samples to generate per prompt
max_new_tokens = 256

@dataclass
class Output:
    alpha: float
    layer: int
    prompt: str 
    lang: str
    suffix: bool 
    
layers = [21,22,23,24,25,26,27]

suffixes = [True, False]

# Initialize batch_entries
batch_entries = []

for lang in langs:
    if lang == 'en':
        steering_vec = t.load(f'steering/gemma2_9b_it/per_culture/us_trans_avg_all_tasks.pt').unsqueeze(1)
    else:
        steering_vec = t.load(f'steering/gemma2_9b_it/per_culture/{lang}_trans_avg_all_tasks.pt').unsqueeze(1)

    for suffix in suffixes:
        if suffix:
            alpha = 0
        else:
            alpha = 2
        outputs = []
        prompts = prepare_prompts(lang, suffix)
        prompt_formats = generate_prompts_chat(lang, suffix)
        
        # Create expanded prompt list with repeated prompts for multiple samples
        expanded_prompts = []
        for prompt in prompt_formats:
            expanded_prompts.extend([prompt] * num_samples)
        
        # Batch across both prompts and samples
        prompt_batch = [expanded_prompts[i:i + batch_size] for i in range(0, len(expanded_prompts), batch_size)]
        
        # Create base filename for outputs
        base_output_filename = f'data/open_ended_generation/steering_{alpha}/{lang}_{suffix}'
        os.makedirs(os.path.dirname(base_output_filename), exist_ok=True)
        
        for batch_idx, batch in enumerate(tqdm(prompt_batch, total=len(prompt_batch))):
            batch_outputs = []  # Store outputs for current batch
            with t.no_grad():
                out = contrastive_act_gen_opt(nnmodel, tokenizer, 
                                            alpha * steering_vec, 
                                            prompt=batch, layer=layers, 
                                            n_new_tokens=max_new_tokens, use_sampling=True)
                for j, layer in enumerate(out[0]):
                    texts = out[0][layer]
                    probs = out[1]
                    epsilon = 1e-6
                    probs[probs < epsilon] = 0

                    for k, text in enumerate(texts):
                        # Calculate original prompt index and sample index
                        global_idx = batch_idx * batch_size + k
                        prompt_idx = global_idx // num_samples
                        sample_idx = global_idx % num_samples
                        
                        # Convert sparse tensor to serializable format
                        sparse_tensor = probs[j,k,:,:].to_sparse()
                        sparse_dict = {
                            'indices': sparse_tensor.indices().cpu().numpy().tolist(),
                            'values': sparse_tensor.values().to(t.float32).cpu().numpy().tolist(),
                            'size': list(sparse_tensor.size())
                        }
                        
                        batch_entry = {
                            "generated_text": text,
                            "alpha": alpha,
                            "steer_out": text,
                            "steer_prob": sparse_dict,
                            "layer": layer,
                            "sample_idx": sample_idx,
                            "prompt_idx": prompt_idx,
                            "prompt": batch[k]
                        }
                        batch_entries.append(batch_entry)
                        out_dict = batch_entry.copy()
                        outputs.append(out_dict)
                        batch_outputs.append(out_dict)
            
            # Save intermediate results after each batch
            temp_output_filename = f'{base_output_filename}_batch_{batch_idx}.json'
            with open(temp_output_filename, 'w', encoding='utf-8') as f:
                json.dump(batch_outputs, f, indent=2, ensure_ascii=False)
            
            t.cuda.empty_cache() # Clear GPU memory after each batch
                
        # Save final complete results
        final_output_filename = f'{base_output_filename}.json'
        with open(final_output_filename, 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
            
        # Clean up temporary batch files
        for batch_idx in range(len(prompt_batch)):
            temp_file = f'{base_output_filename}_batch_{batch_idx}.json'
            if os.path.exists(temp_file):
                os.remove(temp_file)

        del out
        gc.collect()
        t.cuda.empty_cache()  