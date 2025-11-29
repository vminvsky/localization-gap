# uv run python src/pairwise_comparison_stereo.py \
# --model gpt-4o-2024-11-20 \
# --original_dir data/open_ended_generation/per_model/gpt-4o/ \
# --output_dir data/stereotypicality_comparisons_gpt-4o_NEITHER 

uv run python src/pairwise_comparison_stereo.py \
--model gpt-4o-2024-11-20 \
--original_dir data/open_ended_generation/per_model/llama-3.1-70b-instruct-turbo/ \
--output_dir data/stereotypicality_comparisons_llama-3.1-70b-instruct-turbo_NEITHER

uv run python src/pairwise_comparison_stereo.py \
--model gpt-4o-2024-11-20 \
--original_dir data/open_ended_generation/per_model/llama-3.1-8b-instruct-turbo/ \
--output_dir data/stereotypicality_comparisons_llama-3.1-8b-instruct-turbo_NEITHER
