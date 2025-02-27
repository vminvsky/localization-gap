import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_palette("Dark2")

def load_data(results_dir):
    records = []
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.json'):
                fp = os.path.join(root, file)
                with open(fp, 'r') as f:
                    data = json.load(f)
                rel_path = os.path.relpath(fp, results_dir)
                parts = rel_path.split(os.sep)
                model = parts[0] if parts[0] else "unknown"
                # Determine localization flag from filename
                if "true" in file.lower():
                    is_localized = True
                elif "false" in file.lower():
                    is_localized = False
                else:
                    is_localized = None
                for story in data:
                    lang = story.get("lang", "unknown")
                    rating = story.get("stereotypicality_rating_avg")
                    if rating is not None:
                        records.append({
                            "model": model,
                            "lang": lang,
                            "rating": rating,
                            "localized": is_localized
                        })
    return pd.DataFrame(records)

def plot_across_models_localization(df):
    model_names = {
        "Gemma 2 27B": "Gemma 2 27B",
        "Gemma 2 9B": "Gemma 2 9B",
        "llama-3.1-70b-instruct-turbo": "Llama 3.1 70B it",
        "llama-3.1-8b-instruct-turbo": "Llama 3.1 8B it",
        "gpt-4o": "GPT-4o"
    }

    grouped = df.groupby(["model", "localized"])["rating"]
    summary = grouped.agg(['mean', 'std', 'count']).reset_index()
    summary["sem"] = summary["std"] / summary["count"]**0.5
    
    models = summary["model"].unique()
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(10,8))
    for i, loc in enumerate([True, False]):
        sub = summary[summary["localized"] == loc]
        means = []
        sems = []
        for model in models:
            row = sub[sub["model"] == model]
            if not row.empty:
                means.append(row['mean'].values[0])
                sems.append(row['sem'].values[0])
            else:
                means.append(0)
                sems.append(0)
        label = "Explicit Localization" if loc else "Implicit Localization"
        plt.bar(x + (i - 0.5)*width, means, width, yerr=sems, capsize=0, label=label)
    
    plt.xlabel("Model", fontsize=18)
    plt.ylabel("Average Stereotypicality", fontsize=18)
    plt.xticks(x, [model_names.get(m, m) for m in models], rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.ylim(0, 2.6)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig("../visuals/stereotypicality_across_models_localization.pdf")
    plt.close()

def plot_by_language_localization(df):
    model_names = {
        "Gemma 2 27B": "Gemma 2 27B",
        "Gemma 2 9B": "Gemma 2 9B",
        "Llama 3.1 70B it": "Llama 3.1 70B it",
        "Llama 3.1 8B it": "Llama 3.1 8B it",
        "gpt-4o": "gpt-4o"
    }

    languages = df["lang"].unique()
    for lang in languages:
        sub_df = df[df["lang"] == lang]
        grouped = sub_df.groupby(["model", "localized"])["rating"]
        summary = grouped.agg(['mean', 'std', 'count']).reset_index()
        summary["sem"] = summary["std"] / summary["count"]**0.5
        
        models = summary["model"].unique()
        x = np.arange(len(models))
        width = 0.35
        
        plt.figure(figsize=(10,6))
        for i, loc in enumerate([True, False]):
            sub = summary[summary["localized"] == loc]
            means = []
            sems = []
            for model in models:
                row = sub[sub["model"] == model]
                if not row.empty:
                    means.append(row['mean'].values[0])
                    sems.append(row['sem'].values[0])
                else:
                    means.append(0)
                    sems.append(0)
            plt.bar(x + (i - 0.5)*width, means, width, yerr=sems, capsize=5, label=str(loc))
            
        plt.xlabel("Model")
        plt.ylabel("Average Stereotypicality Rating")
        plt.xticks(x, [model_names.get(m, m) for m in models], rotation=45, ha='right')
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.ylim(0, 4)
        plt.legend(title="Localized")
        plt.tight_layout()
        filename = f"results_by_language_localization_{lang}.png"
        plt.savefig(filename)
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="data/open_ended_generation_scored/per_model")
    args = parser.parse_args()
    
    df = load_data(args.results_dir)
    print(df.head())
    if df.empty:
        print("No data found in the specified directory.")
        return
    
    plot_across_models_localization(df)
    # plot_by_language_localization(df)

if __name__ == "__main__":
    main() 