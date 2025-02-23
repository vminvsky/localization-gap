import asyncio
import os
import json
import aiofiles
import re
import argparse
from litellm import acompletion
from dotenv import load_dotenv
from stereo_judge_prompt import STERO_JUDGE_PROMPT
from tqdm import tqdm

load_dotenv()

# Simulated LM call using litellm with Claude 3.5 sonnet as judge.
async def query_model(prompt: str, model: str = "gpt-4o") -> str:
    async with REQUEST_SEMAPHORE:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
    return response.choices[0].message.content

async def rate_candidate(candidate_text: str, culture: str) -> int:
    judge_prompt = STERO_JUDGE_PROMPT.format(candidate_text=candidate_text, culture=culture)
    response = await query_model(judge_prompt)
    match = re.search(r'(\d+)', response)
    return int(match.group(1)) if match else 0

# new function to update progress bar after each API call
async def rate_candidate_with_progress(candidate_text: str, culture: str, progress_bar) -> int:
    rating = await rate_candidate(candidate_text, culture)
    progress_bar.update(1)
    return rating

LANG_TO_CULTURE = {
    "en": "English",
    "ru": "Russian",
    "fr": "French",
    "de": "German",
    "tr": "Turkish",
    "bn": "Bangladesh",
}

MAX_CONCURRENT_REQUESTS = 50
REQUEST_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def process_file(filepath: str, input_dir: str, output_dir: str, progress_bar):
    async with aiofiles.open(filepath, 'r') as f:
        content = await f.read()
    data = json.loads(content)  # assuming the JSON file is a list of candidate generation dicts

    for story in data:
        culture = LANG_TO_CULTURE.get(story["lang"], "Unknown")
        ratings = await asyncio.gather(*[
            rate_candidate_with_progress(candidate[0], culture, progress_bar)
            for candidate in story["generation"]
        ])
        story["stereotypicality_ratings"] = ratings
        story["stereotypicality_rating_avg"] = sum(ratings) / len(ratings)
    
    relative_path = os.path.relpath(filepath, input_dir)
    output_path = os.path.join(output_dir, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Writing to {output_path}")
    async with aiofiles.open(output_path, 'w') as f:
        await f.write(json.dumps(data, indent=2))

async def main(input_dir: str, output_dir: str):
    # Pre-calculate total candidate count for the progress bar
    total_candidates = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                for story in data:
                    total_candidates += len(story["generation"])

    progress_bar = tqdm(total=total_candidates, desc="Rating candidates")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    tasks = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                tasks.append(process_file(filepath, input_dir, output_dir, progress_bar))
                print(f"Processing {filepath}")
    await asyncio.gather(*tasks)
    progress_bar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/open_ended_generation/per_model")
    parser.add_argument("--output_dir", type=str, default="data/open_ended_generation_scored/per_model")
    args = parser.parse_args()
    asyncio.run(main(args.input_dir, args.output_dir)) 