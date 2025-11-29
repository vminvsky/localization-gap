import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

def extract_model_name(dirname):
    """Extract clean model name from directory name."""
    # Remove prefix and suffix
    if dirname.startswith("faithfulness_comparisons_"):
        name = dirname.replace("faithfulness_comparisons_", "")
    elif dirname.startswith("stereotypicality_comparisons_"):
        name = dirname.replace("stereotypicality_comparisons_", "")
    else:
        name = dirname

    # Remove _NEITHER and _NEITHER_REASONING suffixes
    name = name.replace("_NEITHER_REASONING", "").replace("_NEITHER", "")

    # Normalize capitalization
    name = name.replace("Gemma_2_27B", "gemma-2-27b")
    name = name.replace("Gemma_2_9b", "gemma-2-9b")
    name = name.replace("gemma_27b", "gemma-2-27b")
    name = name.replace("gemma_2_27b", "gemma-2-27b")
    name = name.replace("gemma_2_9b", "gemma-2-9b")

    return name

def load_data(comparison_type, reasoning_type):
    """Load win rates from all model directories for given comparison and reasoning type."""
    data_dir = Path(__file__).parent.parent / "data"
    pattern = f"{comparison_type}_comparisons_*_{reasoning_type}"

    model_data = {}

    # Find all matching directories
    for dir_path in sorted(data_dir.glob(pattern)):
        win_rates_file = dir_path / "win_rates.json"
        if not win_rates_file.exists():
            continue

        # Load win rates
        with open(win_rates_file, 'r') as f:
            data = json.load(f)

        # Extract model name
        model_name = extract_model_name(dir_path.name)

        # Check if counts exist
        if "counts" not in data:
            print(f"Warning: No 'counts' field in {win_rates_file}, skipping")
            continue

        # Calculate proportions for plotting
        model_data[model_name] = {}
        for lang, counts in data["counts"].items():
            if "explicit_wins" not in counts:
                print(f"Warning: No 'explicit_wins' in {win_rates_file} for language {lang}, skipping")
                continue

            explicit = counts["explicit_wins"]
            implicit = counts["implicit_wins"]
            ties = counts["ties_impl_expl"]
            neither = counts.get("neither_impl_expl", 0)
            total = explicit + implicit + ties + neither

            if total > 0:
                model_data[model_name][lang] = {
                    "explicit": explicit / total,
                    "implicit": implicit / total,
                    "ties": ties / total,
                    "neither": neither / total
                }
            else:
                model_data[model_name][lang] = {
                    "explicit": 0,
                    "implicit": 0,
                    "ties": 0,
                    "neither": 0
                }

    return model_data

def plot_comparison(comparison_type, reasoning_type, output_file):
    """Create stacked bar plot for given comparison and reasoning type."""
    # Load data
    model_data = load_data(comparison_type, reasoning_type)

    if not model_data:
        print(f"No data found for {comparison_type} {reasoning_type}")
        return

    # Model names and languages
    models = sorted(model_data.keys())
    languages = ["bn", "ru", "fr", "en", "de", "tr"]

    # Colors
    explicit_color = "#1f77b4"
    implicit_color = "#d62728"
    tie_color = "#d3d3d3"
    neither_color = "#f0f0f0"  # Very subtle light gray for neither

    # Create figure
    fig, axes = plt.subplots(
        1, len(models), figsize=(4 * len(models), 4), sharey=True
    )

    # Handle single model case
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        x = np.arange(len(languages))

        # Prepare data arrays
        explicit_vals = []
        implicit_vals = []
        ties_vals = []
        neither_vals = []

        for lang in languages:
            if lang in model_data[model]:
                explicit_vals.append(model_data[model][lang]["explicit"])
                implicit_vals.append(model_data[model][lang]["implicit"])
                ties_vals.append(model_data[model][lang]["ties"])
                neither_vals.append(model_data[model][lang]["neither"])
            else:
                explicit_vals.append(0)
                implicit_vals.append(0)
                ties_vals.append(0)
                neither_vals.append(0)

        explicit_vals = np.array(explicit_vals)
        implicit_vals = np.array(implicit_vals)
        ties_vals = np.array(ties_vals)
        neither_vals = np.array(neither_vals)

        # Create stacked bars
        ax.bar(x, explicit_vals, color=explicit_color, width=0.8, zorder=2)
        ax.bar(x, implicit_vals, bottom=explicit_vals, color=implicit_color, width=0.8, zorder=2)
        ax.bar(x, ties_vals, bottom=explicit_vals + implicit_vals, color=tie_color, width=0.8, zorder=2)
        ax.bar(x, neither_vals, bottom=explicit_vals + implicit_vals + ties_vals, color=neither_color, width=0.8, zorder=1)

        # Formatting
        ax.set_title(model, fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(languages, fontsize=14)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="y", labelsize=14)

    axes[0].set_ylabel("Proportion", fontsize=16)

    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=explicit_color),
        plt.Rectangle((0, 0), 1, 1, color=implicit_color),
        plt.Rectangle((0, 0), 1, 1, color=tie_color),
        plt.Rectangle((0, 0), 1, 1, color=neither_color),
    ]
    fig.legend(
        legend_elements,
        ["Explicit Wins", "Implicit Wins", "Ties", "Neither"],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, -0.008)
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    plt.close()

if __name__ == "__main__":
    # Generate all four plots
    plot_comparison("faithfulness", "NEITHER", "../visuals/faithfulness_NEITHER.pdf")
    plot_comparison("faithfulness", "NEITHER_REASONING", "../visuals/faithfulness_NEITHER_REASONING.pdf")
    plot_comparison("stereotypicality", "NEITHER", "../visuals/stereotypicality_NEITHER.pdf")
    plot_comparison("stereotypicality", "NEITHER_REASONING", "../visuals/stereotypicality_NEITHER_REASONING.pdf")
