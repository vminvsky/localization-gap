from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from IPython.display import IFrame, display
from sae_lens import SAE, ActivationsStore, HookedSAETransformer
from transformer_lens.hook_points import HookPoint
from functools import partial
import torch as t 

GENERATE_KWARGS = dict(temperature=0.5, freq_penalty=2.0, verbose=False)

def display_dashboard(
    sae_release="gpt2-small-res-jb",
    sae_id="blocks.7.hook_resid_pre",
    latent_idx=0,
    width=800,
    height=600,
):
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[sae_id]

    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=false&embedtest=false&height=300"

    print(url)
    display(IFrame(url, width=width, height=height))


def generate_with_steering(
    model: HookedSAETransformer,
    sae: SAE,
    prompt: str,
    latent_idx: int,
    steering_coefficient: float = 1.0,
    max_new_tokens: int = 50,
):
    """
    Generates text with steering. A multiple of the steering vector (the decoder weight for this latent) is added to
    the last sequence position before every forward pass.
    """
    _steering_hook = partial(
        steering_hook,
        sae=sae,
        latent_idx=latent_idx,
        steering_coefficient=steering_coefficient,
    )
    # Hook function to capture the logits
    logits_list = []
    def logits_hook(activations, hook):
        logits = model.unembed(activations)
        logits_list.append(logits.detach().cpu())

    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, _steering_hook), 
                                ('ln_final.hook_post', logits_hook)]):
        
        output = model.generate(prompt, max_new_tokens=max_new_tokens, **GENERATE_KWARGS)

    return output, logits_list


def gen_with_sae(
    model: HookedSAETransformer,
    input_ids: str,
    steering_latents: list,
    max_new_tokens: int = 50,
):
    fwd_hooks = []
    for (sae, latent_idx, coeff) in steering_latents:
        _steering_hook = partial(
            steering_hook,
            sae=sae,
            latent_idx=latent_idx,
            steering_coefficient=coeff,
        )
        fwd_hooks.append((sae.cfg.hook_name, _steering_hook))

    # Initialize the tokenizer and encode the prompt
    generated_ids = input_ids.clone()
    logits_list = []

    for _ in range(max_new_tokens):
        with model.hooks(fwd_hooks=fwd_hooks):
            outputs = model(generated_ids)
        # Get the logits for the last token
        next_token_logits = outputs[:, -1, :]
        logits_list.append(next_token_logits.detach().cpu())

        # Sample the next token
        next_token_id = t.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids = t.cat([generated_ids, next_token_id], dim=-1)

    # Decode the generated sequence
    output_text = model.tokenizer.decode(generated_ids[0])

    return output_text, logits_list




def steering_hook(
    activations,
    hook,
    sae,
    latent_idx: int,
    steering_coefficient: float,
):
    """
    Steers the model by returning a modified activations tensor, with some multiple of the steering vector added to all
    sequence positions.
    """
    return activations + steering_coefficient * sae.W_dec[latent_idx]

