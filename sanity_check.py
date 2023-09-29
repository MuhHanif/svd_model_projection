from transformers import AutoTokenizer, LlamaForCausalLM
import transformers
import torch
import re
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import re
from typing import List

def heatmap(low_rank_array:torch.tensor, full_rank_array:torch.tensor, name:str) -> None:
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom_cmap',
        [(0, 0, 1),  # Blue for values < 0
        (1, 1, 1),  # White for values = 0
        (1, 0, 0)]  # Red for values > 0
    )
    
    # Create a subplots
    fig, axs = plt.subplots(1, 3, figsize=(15*3, 15))

    # Create the heatmap in the first subplot
    axs[0].imshow(low_rank_array, cmap=cmap, interpolation='nearest')
    # Add color bar for the heatmap
    axs[0].figure.colorbar(axs[0].get_images()[0], ax=axs[0])
    axs[0].set_title('nuked weight')

    # Create the heatmap in the first subplot
    axs[1].imshow(full_rank_array, cmap=cmap, interpolation='nearest')
    # Add color bar for the heatmap
    axs[1].figure.colorbar(axs[1].get_images()[0], ax=axs[1])
    axs[1].set_title('true weight')

        # Create the heatmap in the first subplot
    axs[2].imshow(low_rank_array-full_rank_array, cmap=cmap, interpolation='nearest')
    # Add color bar for the heatmap
    axs[2].figure.colorbar(axs[2].get_images()[0], ax=axs[2])
    axs[2].set_title('delta weight')

    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.close()


intact_model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path="/home/zadmin/llama_stuff/CodeLlama-13b-Instruct-hf",
    torch_dtype=torch.float16,
    device_map="cpu",
)

lobotomized_model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path="/home/zadmin/svd_model_projection/lobotomized-CodeLlama-13b-Instruct-hf",
    torch_dtype=torch.float16,
    device_map="cpu",
)

for key, value in lobotomized_model.state_dict().items():
    lobotomized_model.state_dict()[key] = value*0

for param in lobotomized_model.parameters():

    param.data = param.data * 0
print()