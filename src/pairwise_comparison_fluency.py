import os
import json
import argparse
import random
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import aiofiles
from litellm import acompletion
from openai import AsyncOpenAI
from dotenv import load_dotenv
import re
from collections import defaultdict

load_dotenv()

MAX_CONCURRENT_REQUESTS = 50
REQUEST_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Initialize OpenAI client
openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

LANG_TO_CULTURE = {
    "en": "American",
    "ru": "Russian",
    "fr": "French",
    "tr": "Turkish",
    "bn": "Bengali",
}

PAIRWISE_JUDGE_PROMPT = """You will be given two text excerpts. I would like you to compare them based on their fluency. Fluency means that the text is written in natural, grammatically correct language with coherent sentence structure and smooth flow. A fluent text is easy to read and understand.

Text A:
{text_a}

Text B:
{text_b}

Which text is more fluent? Answer with:
- 'A' (if Text A is more fluent)
- 'B' (if Text B is more fluent)
- 'TIE' (if they are equally fluent)

Do only answer with the letter, no other text.
"""


async def query_model(prompt: str, model: str = "gpt-4o-2024-11-20") -> dict:
    async with REQUEST_SEMAPHORE:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
    output_text = response.choices[0].message.content
    reasoning_text = getattr(response.choices[0].message, 'reasoning', None)

    return {
        "content": output_text,
        "reasoning": reasoning_text
    }

async def compare_fluency(text_a: str, text_b: str, progress_bar, model: str = "gpt-4o-2024-11-20") -> dict:
    judge_prompt = PAIRWISE_JUDGE_PROMPT.format(text_a=text_a, text_b=text_b)
    model_response = await query_model(judge_prompt, model=model)

    # Extract content and reasoning from model response
    response_content = model_response["content"]
    response_reasoning = model_response["reasoning"]

    # Extract the judge's decision (A, B, or TIE) from content
    if 'A' in response_content and not 'B' in response_content:
        result = 'A'
    elif 'B' in response_content and not 'A' in response_content:
        result = 'B'
    else:
        # Look for A or B with some common indicators
        if re.search(r'(?i)text\s*a\s*is\s*more', response_content) or re.search(r'(?i)choose\s*a', response_content):
            result = 'A'
        elif re.search(r'(?i)text\s*b\s*is\s*more', response_content) or re.search(r'(?i)choose\s*b', response_content):
            result = 'B'
        else:
            result = 'TIE'

    progress_bar.update(1)
    return {
        "result": result,
        "raw_response": response_content,
        "reasoning": response_reasoning
    }

async def load_json_file(filepath):
    async with aiofiles.open(filepath, 'r') as f:
        content = await f.read()
    data = json.loads(content)

    # Add prompt_idx field to each item based on its position in the list
    for idx, item in enumerate(data):
        item['prompt_idx'] = idx

    return data

async def load_steering_json_file(filepath):
    async with aiofiles.open(filepath, 'r') as f:
        content = await f.read()
    data = json.loads(content)

    # Group samples by prompt_idx
    grouped_data = defaultdict(list)
    for item in data:
        if 'prompt_idx' in item:
            grouped_data[item['prompt_idx']].append(item)

    # Convert to list format with prompts as items and generations grouped
    result = []
    for prompt_idx in sorted(grouped_data.keys()):
        prompt_data = {
            'prompt_idx': prompt_idx,
            'steering_samples': grouped_data[prompt_idx]
        }

        result.append(prompt_data)

    return result

async def main(explicit_dir, steering_dir, output_dir, num_samples=5, model="gpt-4o-2024-11-20"):
    os.makedirs(output_dir, exist_ok=True)

    # Find all language files for explicit
    explicit_files = {}
    for root, _, files in os.walk(explicit_dir):
        for file in files:
            if file.endswith('.json'):
                parts = file.split('_')
                lang = parts[0]
                setting = parts[1].replace('.json', '')
                if setting == 'True':  # Only load explicit (True suffix)
                    explicit_files[lang] = os.path.join(root, file)

    # Find steering files
    steering_files = {}
    for root, _, files in os.walk(steering_dir):
        for file in files:
            if file.endswith('.json'):
                lang = file.split('_')[0]
                steering_files[lang] = os.path.join(root, file)

    print(f"Found explicit files for languages: {list(explicit_files.keys())}")
    print(f"Found steering files for languages: {list(steering_files.keys())}")

    # For each language, compare explicit vs steering
    all_comparisons = {}
    total_comparisons = 0

    for lang in explicit_files:
        if lang not in steering_files:
            print(f"Skipping {lang}: no steering data found")
            continue

        # Load explicit and steering files
        explicit_data = await load_json_file(explicit_files[lang])
        steering_data = await load_steering_json_file(steering_files[lang])

        print(f"\nProcessing {lang}:")
        print(f"  Explicit prompts: {len(explicit_data)}")
        print(f"  Steering prompts: {len(steering_data)}")

        # Count comparisons to make
        num_explicit_steering_comparisons = 0

        for explicit_prompt in explicit_data:
            # Match with steering prompt
            matching_steering_prompts = [p for p in steering_data if p['prompt_idx'] == explicit_prompt['prompt_idx']]

            if matching_steering_prompts and 'steering_samples' in matching_steering_prompts[0]:
                steering_samples = matching_steering_prompts[0]['steering_samples']
                if len(steering_samples) >= num_samples and len(explicit_prompt['generation']) >= num_samples:
                    num_explicit_steering_comparisons += num_samples

        print(f"  Will perform {num_explicit_steering_comparisons} comparisons")
        total_comparisons += num_explicit_steering_comparisons

        # Skip if no comparisons to make
        if num_explicit_steering_comparisons == 0:
            print(f"  Skipping {lang}: no valid comparisons found")
            continue

        # Create progress bar
        pbar = tqdm(total=num_explicit_steering_comparisons,
                    desc=f"Comparing {lang}")

        # Initialize comparisons for this language
        all_comparisons[lang] = {
            "explicit_vs_steering": []
        }

        # Perform comparisons for explicit vs steering
        for explicit_prompt in explicit_data:
            matching_steering_prompts = [p for p in steering_data if p['prompt_idx'] == explicit_prompt['prompt_idx']]

            if matching_steering_prompts and 'steering_samples' in matching_steering_prompts[0]:
                steering_prompt = matching_steering_prompts[0]
                steering_samples = steering_prompt['steering_samples']

                if len(steering_samples) >= num_samples and len(explicit_prompt['generation']) >= num_samples:
                    for i in range(num_samples):
                        # Randomly select generations
                        explicit_gen = random.choice(explicit_prompt['generation'])
                        steering_sample = random.choice(steering_samples)
                        steering_text = steering_sample.get('steer_out', '')

                        # Randomly assign A and B to avoid position bias
                        if random.choice([True, False]):
                            text_a = explicit_gen[0]
                            text_b = steering_text
                            assignment = "explicit_a_steering_b"
                        else:
                            text_a = steering_text
                            text_b = explicit_gen[0]
                            assignment = "steering_a_explicit_b"

                        comparison_result = await compare_fluency(
                            text_a=text_a,
                            text_b=text_b,
                            progress_bar=pbar,
                            model=model
                        )

                        # Extract result, raw_response, and reasoning from dict
                        result = comparison_result["result"]
                        raw_response = comparison_result["raw_response"]
                        reasoning = comparison_result["reasoning"]

                        # Transform result based on assignment
                        # We want 'A' = explicit wins, 'B' = steering wins
                        final_result = result
                        if assignment == "steering_a_explicit_b":
                            if result == "A":
                                final_result = "B"  # If steering (A) is more fluent, steering wins
                            elif result == "B":
                                final_result = "A"  # If explicit (B) is more fluent, explicit wins

                        # Store comparison result
                        all_comparisons[lang]["explicit_vs_steering"].append({
                            "prompt_idx": explicit_prompt['prompt_idx'],
                            "assignment": assignment,
                            "original_result": result,
                            "result": final_result,
                            "explicit_text": explicit_gen[0],
                            "steering_text": steering_text,
                            "text_a": text_a,
                            "text_b": text_b,
                            "raw_response": raw_response,
                            "reasoning": reasoning,
                            "model": model
                        })

        pbar.close()

    # Save all comparison results
    async with aiofiles.open(os.path.join(output_dir, 'raw_comparisons.json'), 'w') as f:
        await f.write(json.dumps(all_comparisons, indent=2))

    # Calculate win rates and confidence intervals
    win_rates = {}
    confidence_intervals = {}
    all_counts = {}

    for lang in all_comparisons:
        win_rates[lang] = {}
        confidence_intervals[lang] = {}
        all_counts[lang] = {}

        # Explicit vs Steering
        if all_comparisons[lang]["explicit_vs_steering"]:
            explicit_wins = sum(1 for comp in all_comparisons[lang]["explicit_vs_steering"] if comp["result"] == 'A')
            steering_wins = sum(1 for comp in all_comparisons[lang]["explicit_vs_steering"] if comp["result"] == 'B')
            ties = sum(1 for comp in all_comparisons[lang]["explicit_vs_steering"] if comp["result"] == 'TIE')

            total = len(all_comparisons[lang]["explicit_vs_steering"])
            if total > 0:
                win_rates[lang]["explicit"] = explicit_wins / total
                win_rates[lang]["steering"] = steering_wins / total

                all_counts[lang]["explicit_wins"] = explicit_wins
                all_counts[lang]["steering_wins"] = steering_wins
                all_counts[lang]["ties"] = ties

    # Calculate confidence intervals with bootstrap
    num_bootstrap = 1000

    for lang in all_comparisons:
        # Explicit vs Steering
        if all_comparisons[lang]["explicit_vs_steering"]:
            explicit_wins = [1 if comp["result"] == 'A' else 0 for comp in all_comparisons[lang]["explicit_vs_steering"]]
            steering_wins = [1 if comp["result"] == 'B' else 0 for comp in all_comparisons[lang]["explicit_vs_steering"]]

            # Bootstrap for explicit
            explicit_bootstrap = []
            for _ in range(num_bootstrap):
                indices = np.random.choice(len(explicit_wins), len(explicit_wins), replace=True)
                explicit_bootstrap.append(np.mean([explicit_wins[i] for i in indices]))

            # Bootstrap for steering
            steering_bootstrap = []
            for _ in range(num_bootstrap):
                indices = np.random.choice(len(steering_wins), len(steering_wins), replace=True)
                steering_bootstrap.append(np.mean([steering_wins[i] for i in indices]))

            confidence_intervals[lang]["explicit"] = (
                np.percentile(explicit_bootstrap, 2.5),
                np.percentile(explicit_bootstrap, 97.5)
            )

            confidence_intervals[lang]["steering"] = (
                np.percentile(steering_bootstrap, 2.5),
                np.percentile(steering_bootstrap, 97.5)
            )

    # Save win rates and confidence intervals
    async with aiofiles.open(os.path.join(output_dir, 'win_rates.json'), 'w') as f:
        await f.write(json.dumps({
            "win_rates": win_rates,
            "confidence_intervals": confidence_intervals,
            "counts": all_counts
        }, indent=2))

    print(f"\nCalculated win rates for {len(win_rates)} language(s)")
    print(f"Languages with data: {list(win_rates.keys())}")

    # Plotting
    sns.set(style="whitegrid")

    # Prepare data for plotting
    import pandas as pd
    plot_data = []

    for lang in win_rates:
        for method in ["explicit", "steering"]:
            if method in win_rates[lang]:
                error_low = 0
                error_high = 0
                if lang in confidence_intervals and method in confidence_intervals[lang]:
                    error_low = win_rates[lang][method] - confidence_intervals[lang][method][0]
                    error_high = confidence_intervals[lang][method][1] - win_rates[lang][method]

                plot_data.append({
                    'Language': lang,
                    'Method': method,
                    'Win Rate': win_rates[lang][method],
                    'Error Low': error_low,
                    'Error High': error_high
                })

    # Convert to DataFrame for plotting
    plot_df = pd.DataFrame(plot_data)

    # Check if we have data to plot
    if plot_df.empty:
        print("\nWarning: No data to plot. Skipping visualization.")
        print("Please check that comparisons were successfully completed.")
        return

    # Plot
    plt.figure(figsize=(14, 8))

    # Plot the bars
    barplot = sns.barplot(
        x='Language',
        y='Win Rate',
        hue='Method',
        data=plot_df,
        errorbar=None  # We'll add custom error bars
    )

    # Fix for properly positioning error bars
    num_languages = len(plot_df['Language'].unique())
    num_methods = len(plot_df['Method'].unique())
    bar_width = 0.8 / num_methods

    # For each language and method, add error bars
    for lang_idx, lang in enumerate(plot_df['Language'].unique()):
        for method_idx, method in enumerate(["explicit", "steering"]):
            # Get data for this specific language and method
            data = plot_df[(plot_df['Language'] == lang) & (plot_df['Method'] == method)]
            if not data.empty:
                # Calculate x position for this specific bar
                x_pos = lang_idx + (method_idx - 0.5) * bar_width

                # Add error bar
                plt.errorbar(
                    x=x_pos,
                    y=data['Win Rate'].values[0],
                    yerr=[[data['Error Low'].values[0]], [data['Error High'].values[0]]],
                    fmt='none',
                    color='black',
                    capsize=5
                )

    plt.title('Fluency Win Rates by Language and Method', fontsize=16)
    plt.xlabel('Language', fontsize=14)
    plt.ylabel('Win Rate', fontsize=14)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fluency_comparison.png'))
    plt.close()

    # Print summary
    print("Fluency Win Rates:")
    for lang in win_rates:
        print(f"\n{lang} ({LANG_TO_CULTURE.get(lang, 'Unknown')}):")
        for method, rate in win_rates[lang].items():
            print(f"  {method}: {rate:.2f}")
            if lang in confidence_intervals and method in confidence_intervals[lang]:
                print(f"    95% CI: [{confidence_intervals[lang][method][0]:.2f}, {confidence_intervals[lang][method][1]:.2f}]")

        # Print detailed counts
        total = all_counts[lang]['explicit_wins'] + all_counts[lang]['steering_wins'] + all_counts[lang]['ties']
        print(f"  Total comparisons: {total}")
        print(f"    Explicit wins: {all_counts[lang]['explicit_wins']}")
        print(f"    Steering wins: {all_counts[lang]['steering_wins']}")
        print(f"    Ties: {all_counts[lang]['ties']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--explicit_dir", type=str, default="data/open_ended_generation/per_model/Gemma 2 9b",
                        help="Directory containing explicit generations (files with '_True.json' suffix)")
    parser.add_argument("--steering_dir", type=str, default="data/open_ended_generation/steering_2",
                        help="Directory containing steering generations")
    parser.add_argument("--output_dir", type=str, default="data/fluency_comparisons")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of samples to compare per prompt")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-11-20", help="Model to use for comparisons")
    args = parser.parse_args()

    asyncio.run(main(args.explicit_dir, args.steering_dir, args.output_dir, args.num_samples, args.model))
