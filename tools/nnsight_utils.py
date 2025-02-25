import torch as t
import pandas as pd
import matplotlib.colors as mcolors
from tqdm import trange
import pandas as pd
import torch as t
from IPython.display import display, HTML
import matplotlib.pyplot as plt


# some helper functions
def visualize_top_tokens(nnmodel, probs, token_index, k=10):
    n_layers = probs.shape[0]
    # Compute the top 10 tokens and corresponding probabilities
    top_tokens = probs[:, token_index].topk(k, dim=-1)
    top_indices = top_tokens.indices.detach().cpu()
    top_probs = top_tokens.values.detach().cpu().float()

    # Decode token IDs to words for each layer
    top_words = [[nnmodel.tokenizer.decode(t).strip() for t in layer] for layer in top_indices]

    # Create DataFrame to hold the results
    df = pd.DataFrame(
        [[f"{word}\n({prob:.4f})" for word, prob in zip(words, probs)]
         for words, probs in zip(top_words, top_probs)],
        columns=[f'Top {i+1}' for i in range(k)],
        index=[f'Layer {i}' for i in range(n_layers)]
    )

    # Function to color the background based on probability
    def color_background(val):
        try:
            prob = float(val.split('(')[1].split(')')[0])
            color = mcolors.to_hex(plt.cm.Blues(prob))
            return f'background-color: {color}'
        except (IndexError, ValueError):
            return 'background-color: #000000'  # Black background for errors

    # Apply styling
    styled_df = df.style.applymap(color_background)
    # change text color to black
    styled_df = styled_df.applymap(lambda x: 'color: black')
    # Display the table
    display(HTML(f"<h3>Top 10 tokens for position {token_index}</h3>"))
    display(styled_df)

def collect_residuals(nnmodel, prompt, calculate_probs=True):
    layers = nnmodel.model.layers
    probs_layers = []
    residuals_layers = []

    with nnmodel.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            for _, layer in enumerate(layers):
                # Process layer output through the model's head and layer normalization
                residuals_layers.append(layer.output[0].save())
                layer_output = nnmodel.lm_head(nnmodel.model.norm(layer.output[0]))
                if calculate_probs:
                    # Apply softmax to obtain probabilities and save the result
                    probs = t.nn.functional.softmax(layer_output, dim=-1).save()
                    probs_layers.append(probs)

    residuals = t.stack(residuals_layers)
    if calculate_probs:
        probs = t.cat([probs.value for probs in probs_layers])
    else:
        probs = None
    input_ids = invoker.inputs[0]["input_ids"][0]
    
    return {"residuals":residuals, "probs":probs, "input_ids": input_ids}

def get_gen_logits(nnmodel, prompt):
    prediction_logits = []
    with nnmodel.generate(prompt, output_scores=True, max_new_tokens=5) as tracer:
        for _ in range(5):
            prediction_logits.append(nnmodel.lm_head.output[:, -1, :].save())
            nnmodel.lm_head.next()

        output = nnmodel.generator.output.save()
    prediction_logits = t.stack(prediction_logits).squeeze(1)
    return prediction_logits

def get_text_generations(model, tokenizer, batch, device, **kwargs):
    encoding = tokenizer(batch, return_tensors='pt', add_special_tokens=False, padding=True, padding_side="left").to(device)
    with t.no_grad():
        generated_ids = model.generate(**encoding, pad_token_id=tokenizer.eos_token_id, **kwargs)
    generated_ids = generated_ids[:, encoding['input_ids'].shape[-1]:]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts
