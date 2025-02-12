import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
from pathlib import Path
from openai import OpenAI
import time
from dotenv import load_dotenv
import os

load_dotenv()

def load_generations(data_dir="data/open_ended_generation/", model_name='llama-3.1-70b-instruct-turbo'):
    """Load all generated responses from json files"""
    all_files = glob.glob(f"{data_dir}/{model_name}/*.json")
    generations = {}
    
    for file_path in all_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            lang = Path(file_path).stem.split('_')[0]  # Extract language from filename
            suffix = Path(file_path).stem.split('_')[1]  # Extract suffix boolean
            key = f"{lang}_{suffix}"
            generations[key] = data
            
    return generations

def create_embeddings_batch(texts, client, model="text-embedding-3-small", batch_size=100):
    """Create embeddings for a batch of texts using OpenAI's API"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model=model,
                input=batch,
                encoding_format="float"
            )
            embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(embeddings)
            
            # Sleep to respect rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in batch {i}: {str(e)}")
            # On error, wait longer and retry
            time.sleep(1)
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch,
                    encoding_format="float"
                )
                embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(embeddings)
            except Exception as e:
                print(f"Failed twice on batch {i}: {str(e)}")
                raise
    
    return all_embeddings

def create_embeddings(generations, batch_size=100):
    """Create embeddings for all prompts and generations using OpenAI"""
    client = OpenAI()
    all_embeddings = []
    
    for dataset_key, dataset in generations.items():
        lang, suffix = dataset_key.split('_')
        print(f"\nProcessing {dataset_key}")
        
        for prompt in tqdm(dataset):
            if 'generation' not in prompt:
                continue
            
            generations_list = [gen[0] if isinstance(gen, list) else gen 
                              for gen in prompt['generation']]
            story = prompt['story']
            
            # Combine prompt and generations for batch processing
            all_texts = generations_list
            
            # Get embeddings for the batch
            try:
                embeddings = create_embeddings_batch(all_texts, client)
                
                # Save metadata and embeddings
                for j, emb in enumerate(embeddings):
                    text = all_texts[j]
                    is_prompt = j == 0  # First text is the prompt
                    
                    all_embeddings.append({
                        'language': lang,
                        'suffix': suffix,
                        'text': text,
                        'is_prompt': is_prompt,
                        'embedding': emb,
                        'story': story
                    })
                    
            except Exception as e:
                print(f"Error processing prompt in {dataset_key}: {str(e)}")
                continue
    
    return all_embeddings

def save_embeddings(embeddings, model_name, output_dir='data/embeddings'):
    """Save embeddings and metadata separately"""
    # Create model-specific directory
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate metadata and embeddings
    metadata = []
    embedding_matrix = []
    
    for item in embeddings:
        metadata.append({
            'language': item['language'],
            'suffix': item['suffix'],
            'text': item['text'],
            'is_prompt': item['is_prompt'],
            'story': item['story']
        })
        embedding_matrix.append(item['embedding'])
    
    # Convert to appropriate formats
    metadata_df = pd.DataFrame(metadata)
    embedding_matrix = np.array(embedding_matrix)
    
    # Save files with model name in path
    metadata_df.to_csv(output_dir / 'metadata.csv', index=False)
    np.save(output_dir / 'embeddings.npy', embedding_matrix)
    
    print(f"\nSaved {len(metadata)} embeddings for {model_name}")
    print(f"Embedding shape: {embedding_matrix.shape}")
    print(f"Files saved in: {output_dir}")
    
    return metadata_df, embedding_matrix

def main(model_name='llama-3.1-70b-instruct-turbo'):
    """Main function to create and save embeddings"""
    # Load generations
    generations = load_generations(model_name=model_name)
    
    # Create embeddings
    embeddings = create_embeddings(generations)
    
    # Save results
    metadata_df, embedding_matrix = save_embeddings(embeddings, model_name)
    
    return metadata_df, embedding_matrix

if __name__ == '__main__':
    metadata_df, embedding_matrix = main()
