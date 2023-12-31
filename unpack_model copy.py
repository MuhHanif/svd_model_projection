from transformers import AutoTokenizer, LlamaForCausalLM
import transformers
import torch
import re
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import re
from typing import List
import gc

def graph_svd(sample:np.array, name:str) -> None:
    U, sigma, Vh = np.linalg.svd(sample.astype(float))



    # Create a 1x2 grid of subplots
    fig, axs = plt.subplots(1, 5, figsize=(15*5, 15))

    # Create the heatmap in the first subplot
    axs[0].imshow(sample, cmap='viridis', interpolation='nearest')
    axs[0].set_title('Heatmap model')

    # Add color bar for the heatmap
    axs[0].figure.colorbar(axs[0].get_images()[0], ax=axs[0])
    
    # Create the heatmap in the first subplot
    axs[1].imshow(U, cmap='viridis', interpolation='nearest')
    axs[1].set_title('Heatmap U')

    # Add color bar for the heatmap
    axs[1].figure.colorbar(axs[0].get_images()[0], ax=axs[0])

    # Create the heatmap in the first subplot
    axs[2].imshow(Vh, cmap='viridis', interpolation='nearest')
    axs[2].set_title('Heatmap Vh')

    # Add color bar for the heatmap
    axs[2].figure.colorbar(axs[0].get_images()[0], ax=axs[0])

    # Create the semilog line graph in the second subplot
    x = np.linspace(0, len(sigma), len(sigma))
    axs[3].semilogy(x, sigma)
    axs[3].set_title('Semilog Line Graph')

    # create relative sigma power
    axs[4].plot(x, np.cumsum(sigma/np.sum(sigma)))
    axs[4].set_title('cumulative sum power')
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.close()

def radical_brain_surgery(sample:torch.tensor, rank_slice:int) -> torch.tensor:
    sample = sample.to(dtype=torch.float32, device="cuda:0")
    U, sigma, Vh = torch.linalg.svd(sample)


    # slice the decomposed matrix based on rank to keep
    U = U[:, :rank_slice]
    sigma = sigma[:rank_slice]
    Vh = Vh[:rank_slice, :]

    new_tensor = U @ torch.diag(sigma) @ Vh

    
    new_tensor_cpu = new_tensor.to(dtype=torch.float32, device="cpu")
    
    del U, sigma, Vh, new_tensor
    torch.cuda.empty_cache()
    gc.collect()
    return new_tensor_cpu

def compute_sigma(sample:torch.tensor) -> torch.tensor:
    sample = sample.to(dtype=torch.float32, device="cuda:0")
    sigma = torch.linalg.svdvals(sample)
    
    new_sigma = sigma.to(dtype=torch.float32, device="cpu")
    
    del sigma, sample
    torch.cuda.empty_cache()
    gc.collect()
    return new_sigma


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

def filter_list_by_patterns(input_list: List[str], patterns: List[str]) -> List[str]:
    """
    GPT generated code!
    Filter a list of strings by specified regex patterns and return the filtered list.

    Args:
        input_list (List[str]): The list of strings to be filtered.
        patterns (List[str]): The list of regex patterns to filter by.

    Returns:
        List[str]: The filtered list of strings that do not match any of the specified patterns.
    """
    # Create an empty list to store the filtered elements
    filtered_list = []

    # Iterate through each item in my_list
    for item in input_list:
        # Initialize a flag to check if any pattern matches
        match_found = False

        # Check if any of the patterns match the item
        for pattern in patterns:
            if re.search(pattern, item):
                match_found = True
                break # Exit the inner loop if any pattern matches

        # If no pattern matched, add the item to filtered_list
        if not match_found:
            filtered_list.append(item)
    
    return filtered_list

def convert_tensors_to_lists(input_dict):
    if isinstance(input_dict, dict):
        # Recursively process all the values in the dictionary
        return {key: convert_tensors_to_lists(value) for key, value in input_dict.items()}
    elif isinstance(input_dict, torch.Size):
        return list(input_dict)  # Convert PyTorch tensor to list
    else:
        # If it's neither a dictionary nor a tensor, return it as is
        return input_dict.tolist()


# init model from pipeline (easier to do)
model_dir = "/home/zadmin/llama_stuff/CodeLlama-13b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_dir)
model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_dir,
    torch_dtype=torch.float16,
    device_map="cpu",
)

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_dir,
#     torch_dtype=torch.float16,
#     device_map="cpu",
# )

# grab model layer name
layer_name = model.state_dict()

# store layer shape here
layer_shape = {}
# store gemm layer name here
gemm_layer = []

for leaf_name, leaf_params in model.state_dict().items():
    layer_shape[leaf_name] = {"shape":leaf_params.shape}
    if len(leaf_params.shape) == 2:
        gemm_layer.append(leaf_name)

# gemm layer that gonna be deranked
# gemm_layer_to_be_converted = filter_list_by_patterns(gemm_layer,[r"model.embed_tokens.weight", r"lm_head.weight"])

# derank state dict based on lut
# clone_model = model.state_dict()
# for layer in gemm_layer_to_be_converted:
#     print(f"converting {layer}")
#     clone_model[layer] = radical_brain_surgery(
#         sample = model.state_dict()[layer],
#         rank_slice=1000
#         ).to(dtype=torch.float16)


# calculate sigma for each gemm layer
for layer in gemm_layer:
    print(f"calculating sigma {layer}")
    layer_shape[layer]["sigma_array"] = compute_sigma(
        sample = model.state_dict()[layer],
        ).numpy()

# check if the model has bias
# is_bias_exist = [leaf_node if bool(re.search(r"bias", leaf_node)) else None for leaf_node in layer_name]


# grab leaf node dimension for each layer 
# model_shape = {}
# gemm_layer = {}
# for leaf_name, leaf_params in pipeline.model.state_dict().items():
#     model_shape[leaf_name] = leaf_params.shape
#     if len(leaf_params.shape) == 2:
#         gemm_layer[leaf_name] = leaf_params.numpy()


# for leaf_name, leaf_params in gemm_layer.items():
#     graph_svd(name=leaf_name, sample=leaf_params)

# sampled layer
# sample = pipeline.model.state_dict()["model.layers.39.self_attn.q_proj.weight"].numpy()
# model.load_state_dict(clone_model)
# model.save_pretrained("lobotomized-CodeLlama-13b-Instruct-hf-1000")
# model = model.load_state_dict(clone_model)

layer_shape_python = convert_tensors_to_lists(layer_shape)

# Serialize the ordered dictionary to MessagePack format
packed_weight = msgpack.packb(layer_shape_python)

# Store the packed data in a file (binary mode)
with open('CodeLlama-13b-Instruct-hf-sigma.msgpack', 'wb') as file:
    file.write(packed_weight)

print()