from tqdm import trange
import torch as t
import torch.nn.functional as F
from tqdm.auto import trange
from tools.decoding import top_k_top_p_filtering, decode_next_token_with_sampling


def contrastive_act_lens(nnmodel, tokenizer, intervene_vec, intervene_tok=-1,target_prompt = None, verbose=False):
    if target_prompt is None:
        id_prompt_target = "cat -> cat\n1135 -> 1135\nhello -> hello\n?"
    else:
        id_prompt_target = target_prompt

    id_prompt_tokens = tokenizer(id_prompt_target, return_tensors="pt", padding=True)["input_ids"].to(nnmodel.device)
    all_logits = []
    lrange = trange(len(nnmodel.model.layers)) if verbose else range(len(nnmodel.model.layers))
    for i in lrange:
        with nnmodel.trace(id_prompt_tokens.repeat(intervene_vec.shape[1], 1), validate=False, scan=False):
            nnmodel.model.layers[i].output[0][:,intervene_tok,:] += intervene_vec[i, :, :]
            logits = nnmodel.lm_head.output[:, -1, :].save()
        all_logits.append(logits.value.detach().cpu())
        
    all_logits = t.stack(all_logits)
    return all_logits

def contrastive_act_gen_opt(
    nnmodel, 
    tokenizer, 
    intervene_vec, 
    intervene_tok=-1, 
    verbose=False,
    prompt=None, 
    n_new_tokens=10, 
    layer=None,
    use_sampling=False
):
    """
    Generates tokens by patching the activations at a specified layer (or all layers) 
    with 'intervene_vec'. Returns a dictionary mapping layer -> [list of generated completions]
    and a probabilities tensor with shape:
        (n_layers_tested, batch_size, n_new_tokens, vocab_size).

    Parameters
    ----------
    nnmodel : YourModelClass
        Model with a 'trace' context manager, 'model.layers', 'lm_head.output', etc.
    tokenizer : YourTokenizerClass
        Tokenizer with a 'decode' method and a callable that returns 'input_ids'.
    intervene_vec : torch.Tensor
        The vector (or set of vectors) to add into the residual stream at a given layer.
        Shapes might be: (n_layers, batch_size, d_model) OR (batch_size, d_model).
    intervene_tok : int
        Index of the token at which to intervene (negative means from the end).
        [Currently not heavily used in this logic, but kept for compatibility.]
    verbose : bool
        If True, use a progress bar over layers.
    prompt : str
        The text prompt to encode.
    n_new_tokens : int
        Number of new tokens to generate.
    layer : int or list[int], optional
        The layer(s) at which to apply the intervention. If None, will test all layers.

    Returns
    -------
    completions_dict : dict
        Dict of { layer_index : [decoded_sequence_1, decoded_sequence_2, ...] }.
    probas : torch.Tensor
        Tensor of shape (n_tested_layers, batch_size, n_new_tokens, vocab_size).
    """

    device = nnmodel.device
    # Encode the prompt
    model_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, padding=True, padding_side="left").to(device)
    prompt_tokens = model_input["input_ids"].to(device)
    prompt_len = prompt_tokens.shape[1] - 1

    # Determine which layers to patch
    if layer is None:
        layers_to_test = range(len(nnmodel.model.layers))
    elif isinstance(layer, int):
        layers_to_test = [layer]
    else:
        layers_to_test = layer  # assume list of layer indices
    
    # Prepare for progress bar if verbose
    layer_iterator = trange(len(layers_to_test)) if verbose else layers_to_test

    # Container for final results
    probas_list = []
    l2toks = {}

    # We do not need gradients for generation
    with t.no_grad():
        for layer_idx in layer_iterator:
            # For each layer, we replicate the prompt for each "batch" in intervene_vec
            # If intervene_vec has shape (n_layers, batch_size, d_model),
            # we pull the slice for this layer_idx. Otherwise broadcast the single intervene_vec

            model_input_base = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, padding=True, padding_side="left").to(device)

            if intervene_vec.ndim == 3:
                # e.g. shape = [n_layers, batch_size, d_model]
                single_layer_patch = intervene_vec[layer_idx]  # shape (batch_size, d_model)
            else:
                # e.g. shape = [batch_size, d_model], broadcast for all layers
                single_layer_patch = intervene_vec
            
            # Determine batch size from the patch
            batch_size = single_layer_patch.shape[0]
            
            # Repeat the prompt tokens for each item in the batch
            #toks = prompt_tokens.repeat(batch_size, 1)  # shape (batch_size, seq_len)
            start_len = model_input_base["input_ids"].shape[1]
            
            # Probabilities for each new token (we'll stack them after generation)
            probas_for_this_layer = []

            # Generate tokens one-by-one (auto-regressive loop)
            
            model_input = {key: value.clone().detach() for key, value in model_input_base.items()}

            for _ in range(n_new_tokens): 

                # Run the forward pass under the patch
                with nnmodel.trace(model_input, validate=False, scan=False):
                    
                    # Patch only the new portion from 'prompt_len' onward
                    # Expand patch to match shape = (batch_size, seq_len_after_prompt, d_model)
                    seq_len_after_prompt = model_input["input_ids"].shape[1] - prompt_len
                    # single_layer_patch: (batch_size, d_model)
                    patch_expanded = single_layer_patch.unsqueeze(1).repeat(1, seq_len_after_prompt, 1)
                    
                    # Add the patch in place
                    # shape of nnmodel.model.layers[layer_idx].output[0] is probably: (batch_size, seq_len, d_model)
                    nnmodel.model.layers[layer_idx].output[0][:, prompt_len:, :] += patch_expanded

                    # Retrieve the logits from the final head at the last token
                    logits = nnmodel.lm_head.output[:, -1, :].save()
                
                if use_sampling:
                    logits_value = logits.value.cpu()
                    probs, next_token = decode_next_token_with_sampling(logits_value)
                    probas_for_this_layer.append(probs.cpu())
                    next_token = next_token.to(device)
                else:
                    # Convert to probabilities on CPU (to avoid holding them on GPU)
                    probs = F.softmax(logits.value, dim=-1)  # shape (batch_size, vocab_size)
                    probas_for_this_layer.append(probs.cpu())

                    # Pick the next token (greedy). Could do top-k, sampling, etc.
                    next_token = t.argmax(logits.value, dim=-1, keepdim=True)  # shape (batch_size, 1)

                # Append the new token to 'toks'
                #toks = t.cat([toks, next_token.to(device)], dim=-1)
                model_input["input_ids"] = t.cat([model_input["input_ids"], next_token], dim=-1)
                model_input["attention_mask"] = t.cat([model_input["attention_mask"], t.ones_like(next_token)], dim=-1)
                if t.all(next_token == tokenizer.eos_token_id):
                    break
            # Stack probabilities: shape => (batch_size, n_new_tokens, vocab_size)
            probas_for_this_layer = t.stack(probas_for_this_layer, dim=1)
            probas_list.append(probas_for_this_layer)

            # Save final generated tokens (just the newly generated portion)
            # shape => (batch_size, n_new_tokens)
            newly_generated_tokens = model_input["input_ids"][:, start_len:].detach().cpu()
            # Convert each row in the batch to text
            decoded_texts = [tokenizer.decode(seq) for seq in newly_generated_tokens]
            l2toks[layer_idx] = decoded_texts
    
    # Now probas_list has length == number of tested layers
    # Each element is shape (batch_size, n_new_tokens, vocab_size)
    # Stack them => (n_layers_tested, batch_size, n_new_tokens, vocab_size)
    probas = t.stack(probas_list, dim=0)

    return l2toks, probas


def contrastive_act_gen(nnmodel, tokenizer, intervene_vec, intervene_tok=-1, verbose=False,
                        prompt=None, n_new_tokens=10, layer=None):
    """
    residuals: (n_layers, batch_size, seq_len, dmodel)
    returns a list of completions when patching at different layers, and the token probabilities
    """

    prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=True)["input_ids"].to(nnmodel.device)
    probas = []
    layers = range(len(nnmodel.model.layers)) if layer is None else layer
    lrange = trange(len(layers)) if verbose else layers
    l2toks = {}
    prompt_len = prompt_tokens.shape[1]-1
    
    for i in lrange:
        toks = prompt_tokens.repeat(intervene_vec.shape[1], 1)
        start_len = toks.shape[1]
        probas_tok = []
        for idx_tok in range(n_new_tokens):
            T = toks.shape[1]
            token_index = intervene_tok-idx_tok if intervene_tok < 0 else intervene_tok

            with nnmodel.trace(toks, validate=False, scan=False):
                if len(intervene_vec.shape)>2:
                    nnmodel.model.layers[i].output[0][:, prompt_len:, :] += intervene_vec[i, :, :].repeat(toks.shape[1]-prompt_len,1,1).permute(1,0,2)
                else:
                    nnmodel.model.layers[i].output[0][:, prompt_len:, :] += intervene_vec[:, :].repeat(toks.shape[1]-prompt_len,1,1).permute(2,0,1)
                logits = nnmodel.lm_head.output[:, -1, :].save()
            probas_tok.append(logits.value.softmax(dim=-1).detach().cpu())
            pred_tok = t.argmax(logits.value, dim=-1, keepdim=True)
            toks = t.cat([toks, pred_tok.to(toks.device)], dim=-1)
            l2toks[i] = toks.detach().cpu()[:, start_len:]
        probas.append(t.stack(probas_tok))
    probas = t.stack(probas)
    probas = probas[:, :, 0]
    
    if None is not None:
        return [tokenizer.decode(t) for t in list(l2toks.values())[0]], probas
    return {k: [tokenizer.decode(t) for t in v] for k, v in l2toks.items()}, probas