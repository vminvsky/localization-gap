import json
import torch
from bert_score import BERTScorer
from itertools import combinations
import pandas as pd
from tqdm import tqdm
import glob
from pathlib import Path
from contextlib import contextmanager

def load_generations(data_dir="data/open_ended_generation/llama-3.1-8b-instruct-turbo"):
    """Load all generated responses from json files"""
    all_files = glob.glob(f"{data_dir}/*.json")
    generations = {}
    
    for file_path in all_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            lang = Path(file_path).stem.split('_')[0]  # Extract language from filename
            suffix = Path(file_path).stem.split('_')[1]  # Extract suffix boolean
            key = f"{lang}_{suffix}"
            generations[key] = data
            
    return generations

def calculate_bert_scores(generations, batch_size=32):
    """Calculate BERT scores between all pairs of generations for each prompt using batching."""
    # Initialize BERT Scorer with CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Increase batch size if using GPU
    if device == "cuda":
        batch_size = batch_size * 2   # Double batch size for GPU
    
    results = []
    
    for dataset_key, dataset in generations.items():
        lang, suffix = dataset_key.split('_')
        scorer = BERTScorer(lang=lang, device=device)

        # Process prompts in batches
        for prompt in tqdm(dataset, desc=f"Processing {dataset_key}"):
            if 'generation' not in prompt or len(prompt['generation']) < 2:
                continue
            
            candidates = []
            references = []
            meta = []
            
            # Create pairs of generations
            for text1, text2 in combinations(prompt['generation'], 2):
                text1_str = text1[0] if isinstance(text1, list) else text1
                text2_str = text2[0] if isinstance(text2, list) else text2
                
                candidates.append(text1_str)
                references.append(text2_str)
                meta.append({
                    'language': lang,
                    'suffix': suffix,
                    'prompt': prompt.get('prompt', ''),
                    'text1': text1_str,
                    'text2': text2_str
                })
            
            # Process in batches
            for i in range(0, len(candidates), batch_size):
                cand_batch = candidates[i:i+batch_size]
                ref_batch = references[i:i+batch_size]
                
                # Calculate scores
                with torch.cuda.amp.autocast() if device == "cuda" else nullcontext():
                    P, R, F1 = scorer.score(cand_batch, ref_batch)
                
                # Store results
                for j, (p, r, f) in enumerate(zip(P, R, F1)):
                    idx = i + j
                    res = meta[idx]
                    res['precision'] = p.item()
                    res['recall'] = r.item()
                    res['f1'] = f.item()
                    results.append(res)
                
                # Clear CUDA cache periodically
                if device == "cuda":
                    torch.cuda.empty_cache()
    
    return pd.DataFrame(results)

def analyze_diversity():
    """Main function to analyze generation diversity"""
    # Load all generations
    generations = load_generations()
    
    # Calculate BERT scores
    scores_df = calculate_bert_scores(generations)
    
    # Calculate average scores per language/suffix combination
    summary = scores_df.groupby(['language', 'suffix']).agg({
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std']
    }).round(3)
    
    print("\nDiversity Analysis Summary:")
    print(summary)
    
    # Save detailed results
    scores_df.to_csv('data/diversity_scores.csv', index=False)
    
    return scores_df, summary

# Add nullcontext for Python < 3.7
@contextmanager
def nullcontext():
    yield

if __name__ == '__main__':
    scores_df, summary = analyze_diversity()