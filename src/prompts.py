import numpy as np 
from torch.utils.data import DataLoader, Dataset
import json

from utils import int2base, return_pairwise_combinations, convert_num_to_roman_numerals, convert_to_scientific



labels = list(range(1000))


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

def generate_prompts_chat(lang: str = 'en', run_with_suffix=False):
    fpath = f'data/open_ended_generation/{lang}_{run_with_suffix}.json'
    data = json.load(open(fpath, 'r'))
    return [{"role": "user", "content": prompt['translation']} for prompt in data]
    

if __name__=='__main__':
    lis, pairwise = generate_prompts_chat(1000, 'roman')
    triplets, triplet_pairs = generate_prompts_triplets(np.load('data/prompts/triplets_close_3_digits.npy'), 'number_triplets_3dig_similar')
    print(lis[0:10])
    print(pairwise[0:10])
    print(triplets[0:10])
    print(triplet_pairs[0:10])