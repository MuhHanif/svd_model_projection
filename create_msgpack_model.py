from transformers import LlamaForCausalLM
import torch
import re
from typing import List
import msgpack
from collections import OrderedDict

def radical_brain_surgery_projected(sample: torch.tensor, sigma_threshold: int) -> torch.tensor:
    U, sigma, Vh = torch.linalg.svd(sample)

    def _sigma_threshold(sigma: torch.tensor, sigma_threshold: int) -> int:
        # Initialize max_sigma_value with the value of the first element of S
        max_sigma_value = sigma[0]

        # Calculate min_sigma_value by dividing max_sigma_value by the target value
        min_sigma_value = max_sigma_value / sigma_threshold

        # Count the number of elements in S that are greater than min_sv
        index = int(torch.sum(sigma > min_sigma_value).item())

        # Ensure index is at least 1 and at mostne len(S) - 1
        index = max(1, min(index, len(sigma) - 1))

        return index

    rank_slice = _sigma_threshold(sigma, sigma_threshold)

    # slice the decomposed matrix based on rank to keep
    U = U[:, :rank_slice]
    sigma = sigma[:rank_slice]
    Vh = Vh[:rank_slice, :]

    new_tensor = U @ torch.diag(sigma) @ Vh

    return new_tensor

def radical_brain_surgery(sample: torch.tensor, sigma_threshold: int) -> List[torch.tensor]:
    U, sigma, Vh = torch.linalg.svd(sample)

    def _sigma_threshold(sigma: torch.tensor, sigma_threshold: int) -> int:
        # Initialize max_sigma_value with the value of the first element of S
        max_sigma_value = sigma[0]

        # Calculate min_sigma_value by dividing max_sigma_value by the target value
        min_sigma_value = max_sigma_value / sigma_threshold

        # Count the number of elements in S that are greater than min_sv
        index = int(torch.sum(sigma > min_sigma_value).item())

        # Ensure index is at least 1 and at mostne len(S) - 1
        index = max(1, min(index, len(sigma) - 1))

        return index

    rank_slice = _sigma_threshold(sigma, sigma_threshold)

    # slice the decomposed matrix based on rank to keep
    U = U[:, :rank_slice]
    sigma = sigma[:rank_slice]
    Vh = Vh[:rank_slice, :]

    down_projection = U @ torch.sqrt(torch.diag(sigma))
    up_projection =  torch.sqrt(torch.diag(sigma)) @ Vh
    
    return down_projection, up_projection

def get_gemm_layer_name(model_state_dict: dict) -> List[str]:

    # store gemm layer name here
    gemm_layer_name = []

    for leaf_name, leaf_params in model_state_dict.items():
        if len(leaf_params.shape) == 2:
            gemm_layer_name.append(leaf_name)
    
    return gemm_layer_name

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

model_dir = "/home/zadmin/llama_stuff/CodeLlama-13b-Instruct-hf"

model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_dir,
    torch_dtype=torch.float16,
    device_map="cpu",
)

gemm_layer = get_gemm_layer_name(model.state_dict())

# gemm layer that gonna be deranked
gemm_layer_to_be_converted = filter_list_by_patterns(gemm_layer,[r"model.embed_tokens.weight", r"lm_head.weight"])

# derank state dict based on lut
clone_model = model.state_dict()

# convert to low rank decomposition
for layer in gemm_layer_to_be_converted:
    print(f"converting {layer}")
    decompositions = radical_brain_surgery(
        sample = model.state_dict()[layer].to(dtype=torch.float32),
        sigma_threshold=20
        )
    # split the name to prepend projection name
    # down layer
    down_layer_name = layer.split(".")
    down_layer_name[-2] = f"{down_layer_name[-2]}_down"
    down_layer_name = ".".join(down_layer_name)
    clone_model[down_layer_name] = decompositions[0].to(dtype=torch.float16)
    # up layer
    up_layer_name = layer.split(".")
    up_layer_name[-2] = f"{up_layer_name[-2]}_up"
    up_layer_name = ".".join(up_layer_name)
    clone_model[up_layer_name] = decompositions[1].to(dtype=torch.float16)

    
    del clone_model[layer]

# Serialize the ordered dictionary to MessagePack format
packed_weight = msgpack.packb(clone_model, use_ordered_dict=True)

# Store the packed data in a file (binary mode)
with open('CodeLlama-13b-Instruct-hf-RBS-20.msgpack', 'wb') as file:
    file.write(packed_data)

print()