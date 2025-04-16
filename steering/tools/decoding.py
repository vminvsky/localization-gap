import torch
import torch.nn.functional as F

def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float('Inf')
):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    Args:
        logits: logits distribution with shape (batch_size, vocab_size).
        top_k: keep only top k tokens with highest probability (0 = disabled).
        top_p: keep the smallest set of tokens whose cumulative probability >= top_p (0.0 = disabled).
        filter_value: value used to replace filtered logits.
    Returns:
        Filtered logits of the same shape as input.
    """
    # Clone so we don't mutate the original tensor
    logits = logits.clone()

    # Top-k filtering
    if top_k > 0:
        # Remove all tokens that are not in the top k
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        min_top_k = top_k_values[:, -1].unsqueeze(-1)
        # Anything below the min of these top-k values is filtered out
        logits[logits < min_top_k] = filter_value

    # Top-p (nucleus) filtering
    if top_p > 0.0:
        # Sort logits by descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Mask tokens beyond the top-p threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift so that we always include the first token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Re-map sorted indices to original
        for batch_idx in range(logits.size(0)):
            indices_to_remove = sorted_indices[batch_idx, sorted_indices_to_remove[batch_idx]]
            logits[batch_idx, indices_to_remove] = filter_value

    return logits


def decode_next_token_with_sampling(logits):
    """
    Decodes the next token using temperature, top-k, and top-p sampling.
    """
    # --- 1. Apply temperature ---
    temperature = 0.7
    logits_scaled = logits / temperature

    # --- 2. Apply top-k / top-p filtering ---
    logits_filtered = top_k_top_p_filtering(
        logits_scaled,
        top_k=50,   # Keep top 50 tokens
        top_p=0.9   # Keep tokens until cumulative prob >= 0.9
    )

    # --- 3. Convert to probabilities ---
    probs = F.softmax(logits_filtered, dim=-1)

    # --- 4. Sample the next token from the filtered distribution ---
    next_token = torch.multinomial(probs, num_samples=1)  # shape: (batch_size, 1)

    return probs, next_token