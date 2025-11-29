# uv run python src/pairwise_comparison_stereo.py \
# --model openai/gpt-oss-120b \
# --original_dir data/open_ended_generation/per_model/Gemma_2_27B/ \
# --output_dir data/stereotypicality_comparisons_Gemma_2_27B_NEITHER_REASONING

uv run python src/pairwise_comparison_stereo.py \
--model openai/gpt-oss-120b \
--original_dir data/open_ended_generation/per_model/Gemma_2_9b/ \
--output_dir data/stereotypicality_comparisons_Gemma_2_9b_NEITHER_REASONING

uv run python src/pairwise_comparison_stereo.py \
--model openai/gpt-oss-120b \
--original_dir data/open_ended_generation/per_model/gpt-4o/ \
--output_dir data/stereotypicality_comparisons_gpt-4o_NEITHER_REASONING

uv run python src/pairwise_comparison_stereo.py \
--model openai/gpt-oss-120b \
--original_dir data/open_ended_generation/per_model/llama-3.1-70b-instruct-turbo/ \
--output_dir data/stereotypicality_comparisons_llama-3.1-70b-instruct-turbo_NEITHER_REASONING

uv run python src/pairwise_comparison_stereo.py \
--model openai/gpt-oss-120b \
--original_dir data/open_ended_generation/per_model/llama-3.1-8b-instruct-turbo/ \
--output_dir data/stereotypicality_comparisons_llama-3.1-8b-instruct-turbo_NEITHER_REASONING
