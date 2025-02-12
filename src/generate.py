import logging
import json
from tqdm import tqdm 
from openai import OpenAI
import pandas as pd
import os 
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from anthropic import Anthropic
import time
import random
from collections import defaultdict

load_dotenv()

from prompts import PromptDataset, generate_prompts_chat, prepare_prompts
from utils import model_paths, printv, estimate_cost
from model_configs import model_configs, return_client, model_name_mappings


def return_params(model_provider):
    temperature = model_configs[model_provider]['temperature']
    batch_size = model_configs[model_provider]['batch_size']
    num_threads = model_configs[model_provider]['num_threads']
    return temperature, batch_size, num_threads


def model_generation(batch_prompts, client, model_name, temperature, provider, max_tokens=256):
    if provider == 'anthropic':
        completion = client.messages.create(
            model=model_name_mappings[model_name],
            messages=batch_prompts,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        responses = [completion.content[0].text]
    else:
        completion = client.chat.completions.create(
            model=model_name_mappings[model_name],
            messages=[batch_prompts],
            temperature=temperature,
            max_tokens=max_tokens
        )
        responses = [choice.message.content for choice in completion.choices]
    return responses


def process_single_prompt(args):
    """
    Processes a single prompt by rerunning the generation K times.
    The prompt dict is updated with a new key 'generation' that contains
    a list of K generations.
    """
    prompt, prompt_format, client, model_name, temperature, provider, k = args
    # Optional: print the prompt format (note: output order may vary due to threading)
    print(prompt_format)
    generations = []
    for i in range(k):
        responses = model_generation(prompt_format, client, model_name, temperature, provider)
        generations.append(responses)
    prompt['generation'] = generations
    return prompt


def main(langs=['en', 'tr', 'de', 'fr', 'ru'], suffixes=[True, False], k=3):
    """
    For each model, language, and suffix combination, the same set of prompts is run k times.
    The results are saved into a file named based on the model name, language, and suffix.
    """
    run_models = [
        # {'model_name': 'DeepSeek-V3', 'provider': 'together'},
        {'model_name': 'llama-3.1-8b-instruct-turbo', 'provider': 'together'},
        {'model_name': 'llama-3.1-70b-instruct-turbo', 'provider': 'together'},
        # {'model_name': "mixtral-8x22b-instruct", 'provider': 'together'},
        {'model_name': "claude-3.5-sonnet", 'provider': 'anthropic'},
    ]
    for model in run_models:
        model_name = model['model_name']
        provider = model['provider']
        temperature, batch_size, num_threads = return_params(provider)
        client = return_client(provider)
        
        for lang in langs:
            for suffix in suffixes:
                prompts = prepare_prompts(lang, suffix)
                prompt_formats = generate_prompts_chat(lang, suffix)
                
                # Build a list of arguments to process each prompt,
                # including the number of times (k) to rerun each prompt.
                args_list = [
                    (prompt, prompt_format, client, model_name, temperature, provider, k)
                    for prompt, prompt_format in zip(prompts, prompt_formats)
                ]
                
                # Use a ThreadPoolExecutor to parallelize prompt generations.
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    processed_prompts = list(tqdm(
                        executor.map(process_single_prompt, args_list),
                        total=len(args_list),
                        desc=f"Processing {lang} (suffix={suffix}) with {model_name}"
                    ))
                
                # Save the results to a JSON file, including the model name in the filename.
                # Note: you might want to sanitize the model_name if it has spaces or special characters.
                output_filename = f'data/open_ended_generation/{model_name}/{lang}_{suffix}.json'
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(processed_prompts, f, ensure_ascii=False, indent=2)
                logging.info(f"Saved results to {output_filename}")


if __name__ == '__main__':
    main(k=30)