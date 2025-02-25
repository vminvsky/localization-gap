import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as t

def logit_lens(nnmodel, model_output, limit=-1):
    probs = model_output["probs"]
    input_ids = model_output["input_ids"]

    # Find the maximum probability and corresponding tokens for each position
    max_probs, tokens = probs.max(dim=-1)

    # Decode token IDs to words for each layer
    words = [[nnmodel.tokenizer.decode(t).encode("unicode_escape").decode() for t in layer_tokens]
        for layer_tokens in tokens]

    # Access the 'input_ids' attribute of the invoker object to get the input words
    input_words = [nnmodel.tokenizer.decode(t) for t in input_ids]

    if limit > 0:
        words = [layer[-limit:] for layer in words]
        max_probs = max_probs[:, -limit:]
        input_words = input_words[-limit:]
    max_probs = max_probs.to(t.float32)

    cmap = sns.diverging_palette(255, 0, n=len(words[0]), as_cmap=True)

    plt.figure(figsize=(20, 20))
    
    ax=sns.heatmap(max_probs.cpu().detach().numpy(), annot=np.array(words), fmt='', cmap=cmap, linewidths=.5, cbar_kws={'label': 'Probability'})

    plt.title('Logit Lens Visualization')
    plt.xlabel('Input Tokens')
    plt.ylabel('Layers')

    plt.yticks(np.arange(len(words)) + 0.5, range(len(words)))

    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position("top")
    plt.xticks(np.arange(len(input_words)) + 0.5, input_words, rotation=45)

    plt.show()