import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
import argparse

def load_fluency_data(data_dir):
    """Load win rates from fluency comparison directory."""
    win_rates_file = Path(data_dir) / "win_rates.json"

    if not win_rates_file.exists():
        print(f"Error: win_rates.json not found in {data_dir}")
        return None

    # Load win rates
    with open(win_rates_file, 'r') as f:
        data = json.load(f)

    # Check if counts exist
    if "counts" not in data:
        print(f"Warning: No 'counts' field in {win_rates_file}")
        return None

    # Calculate proportions for plotting
    lang_data = {}
    for lang, counts in data["counts"].items():
        if "explicit_wins" not in counts:
            print(f"Warning: No 'explicit_wins' for language {lang}, skipping")
            continue

        explicit = counts["explicit_wins"]
        steering = counts["steering_wins"]
        ties = counts["ties"]
        total = explicit + steering + ties

        if total > 0:
            lang_data[lang] = {
                "explicit": explicit / total,
                "steering": steering / total,
                "ties": ties / total
            }
        else:
            lang_data[lang] = {
                "explicit": 0,
                "steering": 0,
                "ties": 0
            }

    return lang_data

def plot_fluency_comparison(data_dir, output_file):
    """Create stacked bar plot for fluency comparison."""
    # Load data
    lang_data = load_fluency_data(data_dir)

    if not lang_data:
        print(f"No data found in {data_dir}")
        return

    # Languages (order them consistently)
    all_languages = ["bn", "ru", "fr", "en", "de", "tr"]
    languages = [lang for lang in all_languages if lang in lang_data]

    if not languages:
        print("No valid language data found")
        return

    # Colors (matching plot_agg_winrate_plots.py)
    explicit_color = "#1f77b4"  # Blue
    steering_color = "#d62728"  # Red (same as implicit in other plots)
    tie_color = "#d3d3d3"       # Light gray

    # Create figure
    fig, ax = plt.subplots(figsize=(10,7))

    x = np.arange(len(languages))

    # Prepare data arrays
    explicit_vals = []
    steering_vals = []
    ties_vals = []

    for lang in languages:
        explicit_vals.append(lang_data[lang]["explicit"])
        steering_vals.append(lang_data[lang]["steering"])
        ties_vals.append(lang_data[lang]["ties"])

    explicit_vals = np.array(explicit_vals)
    steering_vals = np.array(steering_vals)
    ties_vals = np.array(ties_vals)

    # Create stacked bars
    ax.bar(x, explicit_vals, color=explicit_color, width=0.8, zorder=2)
    ax.bar(x, steering_vals, bottom=explicit_vals, color=steering_color, width=0.8, zorder=2)
    ax.bar(x, ties_vals, bottom=explicit_vals + steering_vals, color=tie_color, width=0.8, zorder=2)

    # Formatting
    ax.set_title("Fluency", fontsize=18, pad=20)
    ax.set_xlabel("Language", fontsize=16)
    ax.set_ylabel("Proportion", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(languages, fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="y", labelsize=14)

    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=explicit_color),
        plt.Rectangle((0, 0), 1, 1, color=steering_color),
        plt.Rectangle((0, 0), 1, 1, color=tie_color),
    ]
    fig.legend(
        legend_elements,
        ["Explicit Wins", "Steering Wins", "Ties"],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, 0.002)
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    plt.close()

def plot_fluency_comparison_multiple_models(data_dirs, model_names, output_file):
    """Create stacked bar plots for multiple models side by side."""
    if len(data_dirs) != len(model_names):
        print("Error: Number of data directories must match number of model names")
        return

    # Load data for all models
    all_model_data = []
    valid_models = []
    valid_names = []

    for data_dir, model_name in zip(data_dirs, model_names):
        lang_data = load_fluency_data(data_dir)
        if lang_data:
            all_model_data.append(lang_data)
            valid_models.append(data_dir)
            valid_names.append(model_name)

    if not all_model_data:
        print("No valid data found")
        return

    # Languages (order them consistently)
    all_languages = ["bn", "ru", "fr", "en", "de", "tr"]

    # Colors (matching plot_agg_winrate_plots.py)
    explicit_color = "#1f77b4"  # Blue
    steering_color = "#d62728"  # Red (same as implicit in other plots)
    tie_color = "#d3d3d3"       # Light gray

    # Create figure with subplots
    fig, axes = plt.subplots(
        1, len(all_model_data), figsize=(4 * len(all_model_data), 5), sharey=True
    )

    # Handle single model case
    if len(all_model_data) == 1:
        axes = [axes]

    for ax, lang_data, model_name in zip(axes, all_model_data, valid_names):
        # Filter languages that have data for this model
        languages = [lang for lang in all_languages if lang in lang_data]

        if not languages:
            continue

        x = np.arange(len(languages))

        # Prepare data arrays
        explicit_vals = []
        steering_vals = []
        ties_vals = []

        for lang in languages:
            explicit_vals.append(lang_data[lang]["explicit"])
            steering_vals.append(lang_data[lang]["steering"])
            ties_vals.append(lang_data[lang]["ties"])

        explicit_vals = np.array(explicit_vals)
        steering_vals = np.array(steering_vals)
        ties_vals = np.array(ties_vals)

        # Create stacked bars
        ax.bar(x, explicit_vals, color=explicit_color, width=0.8, zorder=2)
        ax.bar(x, steering_vals, bottom=explicit_vals, color=steering_color, width=0.8, zorder=2)
        ax.bar(x, ties_vals, bottom=explicit_vals + steering_vals, color=tie_color, width=0.8, zorder=2)

        # Formatting
        ax.set_title(model_name, fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(languages, fontsize=14)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="y", labelsize=14)

    axes[0].set_ylabel("Proportion", fontsize=16)

    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=explicit_color),
        plt.Rectangle((0, 0), 1, 1, color=steering_color),
        plt.Rectangle((0, 0), 1, 1, color=tie_color),
    ]
    fig.legend(
        legend_elements,
        ["Explicit Wins", "Steering Wins", "Ties"],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, -0.05)
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot fluency comparison results")
    parser.add_argument("--data_dir", type=str, default="data/fluency_comparisons",
                        help="Directory containing win_rates.json")
    parser.add_argument("--output_file", type=str, default="visuals/fluency_comparison.pdf",
                        help="Output file path")
    parser.add_argument("--multiple_models", action="store_true",
                        help="Plot multiple models side by side")
    parser.add_argument("--data_dirs", type=str, nargs="+",
                        help="Multiple data directories (for --multiple_models)")
    parser.add_argument("--model_names", type=str, nargs="+",
                        help="Model names corresponding to data_dirs (for --multiple_models)")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)

    if args.multiple_models:
        if not args.data_dirs or not args.model_names:
            print("Error: --data_dirs and --model_names are required for --multiple_models")
        else:
            plot_fluency_comparison_multiple_models(args.data_dirs, args.model_names, args.output_file)
    else:
        plot_fluency_comparison(args.data_dir, args.output_file)
