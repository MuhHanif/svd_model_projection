import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from llama import LlamaForCausalLM
from typing import Any
import gc
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import optree
from safetensors import safe_open
from safetensors.torch import save_file
import gc
import json

def save_to_json(data, file_path):
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to be saved as JSON.
        file_path (str): The file path where the JSON data will be saved.

    Returns:
        bool: True if the data was successfully saved, False otherwise.
    """
    try:
        with open(file_path, "w") as json_file:
            json.dump(data, json_file)
        return True
    except Exception as e:
        print(f"An error occurred while saving to JSON: {str(e)}")
        return False

def load_from_json(file_path):
    """
    Load a dictionary from a JSON file.

    Args:
        file_path (str): The file path from which to load the JSON data.

    Returns:
        dict: The dictionary loaded from the JSON file, or an empty dictionary if the file does not exist.
    """
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"The file '{file_path}' does not exist.")
        return {}
    except Exception as e:
        print(f"An error occurred while loading from JSON: {str(e)}")
        return {}

def load_all_tensors(path: str):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def get_dict_slice(original_dict: dict, list_of_key: list):
    sliced_dict = {}

    # Iterate through the keys in your list
    for key in list_of_key:
        if key in original_dict:
            sliced_dict[key] = original_dict[key]

    return sliced_dict


def merge_heads(tensor: torch.tensor):
    batch_dim, head, sequence_dim, head_dim = tensor.shape
    tensor = tensor.transpose(1, 2).contiguous()
    return tensor.reshape(batch_dim, sequence_dim, head_dim * head)


def graph_weight(dict_of_tensors: list, name: str = None) -> None:
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap",
        [
            (1, 1, 1),  # White for values = 0
            # (0, 0, 1),  # Blue for values < 0
            (1, 0, 0),
        ],  # Red for values > 0
    )

    # Create a 1x2 grid of subplots
    fig, axs = plt.subplots(
        1, len(dict_of_tensors), figsize=(30 * len(dict_of_tensors), 10)
    )
    # Create a meshgrid for the x and y coordinates

    for count, item in enumerate(dict_of_tensors.items()):
        key, sample = item
        sample = sample[0]
        x, y = np.meshgrid(np.arange(sample.shape[1]), np.arange(sample.shape[0]))

        # Create the heatmap in the first subplot
        axs[count].imshow(sample, cmap=cmap, interpolation="nearest")
        axs[count].set_title(f"Heatmap {key}")

        # Add color bar for the heatmap
        axs[count].figure.colorbar(axs[count].get_images()[0], ax=axs[count])

    # Adjust spacing between subplots
    plt.tight_layout()
    if name == None:
        plt.show()
    else:
        plt.savefig(f"{name}.png")
    plt.close()

def graph_weight_2d(dict_of_tensors: list, name: str = None) -> None:

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap",
        [
            (1, 1, 1),  # White for values = 0
            # (0, 0, 1),  # Blue for values < 0
            (1, 0, 0),
        ],  # Red for values > 0
    )

    # Create a 1x2 grid of subplots
    fig, axs = plt.subplots(
        1, len(dict_of_tensors), figsize=(30 * len(dict_of_tensors), 10)
    )
    # Create a meshgrid for the x and y coordinates

    for count, item in enumerate(dict_of_tensors.items()):
        key, sample = item
        sample = sample.to(device="cpu").numpy()
        x, y = np.meshgrid(np.arange(sample.shape[1]), np.arange(sample.shape[0]))

        # Create the heatmap in the first subplot
        axs[count].imshow(sample, cmap=cmap, interpolation="nearest")
        axs[count].set_title(f"Heatmap {key}")

        # Add color bar for the heatmap
        axs[count].figure.colorbar(axs[count].get_images()[0], ax=axs[count])

    # Adjust spacing between subplots
    plt.tight_layout()
    if name == None:
        plt.show()
    else:
        plt.savefig(f"{name}.png")
    plt.close()

def cum_sum_percentage(S, percentage_target):
    # Convert S to a NumPy array
    S = np.array(S)

    # get cumulative sum of the array
    cumsum_S = np.cumsum(S)

    max_value = np.max(cumsum_S)

    target = max_value * percentage_target

    # Calculate absolute difference from the target value to decide near zero point
    absolute_div = np.abs(cumsum_S - target)

    # grab the index of near zero value
    index = np.argmin(absolute_div)

    return index


def index_sv_ratio(S, target):
    # Convert S to a NumPy array
    S = np.array(S)

    # Initialize max_sv with the value of the first element of S
    max_sv = S[0]

    # Calculate min_sv by dividing max_sv by the target value
    min_sv = max_sv / target

    # Count the number of elements in S that are greater than min_sv
    index = np.sum(S > min_sv)

    # Ensure index is at least 1 and at most len(S) - 1
    index = max(1, min(index, len(S) - 1))

    return index


def index_sv_ratio_torch(S, target):
    # Initialize max_sv with the value of the first element of S
    max_sv = S[0]

    # Calculate min_sv by dividing max_sv by the target value
    min_sv = max_sv / target

    # Count the number of elements in S that are greater than min_sv
    index = torch.sum(S > min_sv)

    # Ensure index is at least 1 and at most len(S) - 1
    index = max(1, min(index, len(S) - 1))

    return index


def minmax_scale(xs):
    return (xs - xs.min()) / (xs.max() - xs.min())


def stacked_plot(
    sigma_data: dict,
    name: str,
    percentage: float = None,
    max_mag_decay_ratio: float = None,
) -> None:
    # Create a subplots
    fig, axs = plt.subplots(len(sigma_data), 2, figsize=(20 * 3, 10 * len(sigma_data)))

    for row, items in enumerate(sigma_data.items()):
        chart_name, data = items

        # create rainbow opacity
        gradient_color = (1 - 0.1) / len(data)
        for count, array in enumerate(data):
            if percentage:
                array_treshold = cum_sum_percentage(array, percentage)
            elif max_mag_decay_ratio:
                array_treshold = index_sv_ratio(array, max_mag_decay_ratio)
            elif max_mag_decay_ratio and percentage:
                raise "cannot use both thresholding at the same time!"
            else:
                NotImplementedError
            orig_array = array
            array = array[:array_treshold]
            # print(0.1 + gradient_color * count)
            x = np.linspace(0, len(array), len(array))
            axs[row, 0].semilogy(
                x,
                array,
                label=f"{chart_name} layer {count}",
                alpha=0.1 + gradient_color * count,
                color=(0.1 + gradient_color * count, 0.0, 0.0),
            )
            axs[row, 1].plot(
                x,
                (np.cumsum(array / np.sum(orig_array))),
                label=f"{chart_name} layer {count}",
                alpha=0.1 + gradient_color * count,
                color=(0.1 + gradient_color * count, 0.0, 0.0),
            )

        axs[row, 0].legend()
        axs[row, 0].set_title(f"{chart_name} sigma progression", fontsize=16)

        axs[row, 1].legend()
        axs[row, 1].set_title(f"{chart_name} sigma cumulative", fontsize=16)

    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.close()

def replace_values_below_threshold(tensor, threshold, replacement_value):
    # Create a boolean mask where values are lower than the threshold
    mask = tensor < threshold

    # Replace values in the tensor using the mask
    tensor[mask] = replacement_value

    return tensor

def replace_values_above_threshold(tensor, threshold, replacement_value):
    # Create a boolean mask where values are lower than the threshold
    mask = tensor > threshold

    # Replace values in the tensor using the mask
    tensor[mask] = replacement_value

    return tensor

def create_activation_mask(xs, max_mag_decay_ratio=None, percentage=None):
    def _min_max_scale(xs):
        return (xs - xs.min()) / (xs.max() - xs.min())

    def _cum_sum_percentage(S, percentage_target):
        # get cumulative sum of the array
        cumsum_S = toch.cumsum(S)

        max_value = torch.max(cumsum_S, dim=0)

        target = max_value * percentage_target

        # Calculate absolute difference from the target value to decide near zero point
        absolute_div = torch.abs(cumsum_S - target)

        # grab the index of near zero value
        index = torch.argmin(absolute_div)

        return index


    def _index_sv_ratio(S, target):
        # Initialize max_sv with the value of the first element of S
        max_sv = S.max()

        # Calculate min_sv by dividing max_sv by the target value
        min_sv = max_sv / target

        # Count the number of elements in S that are greater than min_sv
        index = torch.sum(S > min_sv)

        # Ensure index is at least 1 and at most len(S) - 1
        index = max(1, min(index, len(S) - 1))

        return index

    # collapse the channel dimension by summing it 
    # to determine the activation contribution
    # then rescale to 0 to 1 range
    sum_and_scale = _min_max_scale(xs.view(2048, -1).to(dtype=torch.float32).sum(axis=0))

    # sort the tensor descending so lower activation value can be culled / removed
    sort_act_importance = sum_and_scale.sort(axis=0, descending=True).values


    if percentage:
        threshold = sort_act_importance[cum_sum_percentage(sort_act_importance, percentage)]
    elif max_mag_decay_ratio:
        threshold = sort_act_importance[_index_sv_ratio(sort_act_importance, max_mag_decay_ratio)]
    elif max_mag_decay_ratio and percentage:
        raise "cannot use both thresholding at the same time!"
    else:
        NotImplementedError

    # thresholding by cutting the sorted tensor by decay magnitude
    # threshold = sorted_tensor[_index_sv_ratio(sorted_tensor, max_mag_decay_ratio)]
    # print(threshold)
    # Replace values below threshold with zeros and the rest with 1
    sum_and_scale[sum_and_scale < threshold] = 0
    sum_and_scale[sum_and_scale > 0] = 1


    # def _prune(tensor, sorted_tensor, max_mag_decay_ratio=None, percentage=None):
    #     if percentage:
    #         threshold = sorted_tensor[cum_sum_percentage(sorted_tensor, percentage)]
    #     elif max_mag_decay_ratio:
    #         threshold = sorted_tensor[_index_sv_ratio(sorted_tensor, max_mag_decay_ratio)]
    #     elif max_mag_decay_ratio and percentage:
    #         raise "cannot use both thresholding at the same time!"
    #     else:
    #         NotImplementedError

    #     # thresholding by cutting the sorted tensor by decay magnitude
    #     # threshold = sorted_tensor[_index_sv_ratio(sorted_tensor, max_mag_decay_ratio)]
    #     # print(threshold)
    #     # Replace values below threshold with zeros and the rest with 1
    #     tensor[tensor < threshold] = 0
    #     tensor[tensor > 0] = 1

    #     return tensor

    # return _prune(sum_and_scale, sort_act_importance, max_mag_decay_ratio)
    return sum_and_scale
    
def modify_weights(model, mask:dict):

    # TODO: 
    # rebuild mask in here
    # do svd on salient weights and non salient weights separately
    # then recombine it after truncation by preserving the salient but sacrifice the non salient one
    # put the low rank mask in here!
   
    # Modify the weight parameter
    for name, param in model.named_parameters():

        for layer_name, mask_tensor in mask.items():
            print()
            if name == layer_name:

                gpu_mask = mask_tensor.to(param.data.device)
                param.data *= gpu_mask
                del gpu_mask
                gc.collect()
                break # break the inner loop
            
def create_mask(layer_count:int, mask_1d:dict):

    x = layer_count
    mask_dict = {}
    for x in range(layer_count):
        
        mask_dict[f"model.layers.{x}.self_attn.q_proj.weight"] = torch.outer(mask_1d[f"layer_{x}.query_output"], mask_1d[f"layer_{x}.qkv_input"])
        mask_dict[f"model.layers.{x}.self_attn.k_proj.weight"] = torch.outer(mask_1d[f"layer_{x}.key_output"], mask_1d[f"layer_{x}.qkv_input"])
        mask_dict[f"model.layers.{x}.self_attn.v_proj.weight"] = torch.outer(mask_1d[f"layer_{x}.value_output"], mask_1d[f"layer_{x}.qkv_input"])
        mask_dict[f"model.layers.{x}.self_attn.o_proj.weight"] = torch.outer(mask_1d[f"layer_{x}.out_input"], mask_1d[f"layer_{x}.out_input"])
        mask_dict[f"model.layers.{x}.mlp.gate_proj.weight"] = torch.outer(mask_1d[f"layer_{x}.gate_proj_output"], mask_1d[f"layer_{x}.mlp_gate_up_input"])
        mask_dict[f"model.layers.{x}.mlp.up_proj.weight"] = torch.outer(mask_1d[f"layer_{x}.up_proj_output"], mask_1d[f"layer_{x}.mlp_gate_up_input"])
        mask_dict[f"model.layers.{x}.mlp.down_proj.weight"] = torch.outer(mask_1d[f"layer_{x}.down_proj_output"], mask_1d[f"layer_{x}.down_proj_input"])
    return mask_dict

def group_1d_mask(layer_count:int, mask_1d:dict):

    x = layer_count
    mask_dict = {}
    for x in range(layer_count):
        
        mask_dict[f"model.layers.{x}.self_attn.q_proj.weight"] = (mask_1d[f"layer_{x}.query_output"], mask_1d[f"layer_{x}.qkv_input"])
        mask_dict[f"model.layers.{x}.self_attn.k_proj.weight"] = (mask_1d[f"layer_{x}.key_output"], mask_1d[f"layer_{x}.qkv_input"])
        mask_dict[f"model.layers.{x}.self_attn.v_proj.weight"] = (mask_1d[f"layer_{x}.value_output"], mask_1d[f"layer_{x}.qkv_input"])
        mask_dict[f"model.layers.{x}.self_attn.o_proj.weight"] = (mask_1d[f"layer_{x}.out_input"], mask_1d[f"layer_{x}.out_input"])
        mask_dict[f"model.layers.{x}.mlp.gate_proj.weight"] = (mask_1d[f"layer_{x}.gate_proj_output"], mask_1d[f"layer_{x}.mlp_gate_up_input"])
        mask_dict[f"model.layers.{x}.mlp.up_proj.weight"] = (mask_1d[f"layer_{x}.up_proj_output"], mask_1d[f"layer_{x}.mlp_gate_up_input"])
        mask_dict[f"model.layers.{x}.mlp.down_proj.weight"] = (mask_1d[f"layer_{x}.down_proj_output"], mask_1d[f"layer_{x}.down_proj_input"])
    return mask_dict

def modify_weights_with_1d_mask(model, mask:dict):

    # TODO: 
    # rebuild mask in here
    # do svd on salient weights and non salient weights separately
    # then recombine it after truncation by preserving the salient but sacrifice the non salient one
    # put the low rank mask in here!
   
    # Modify the weight parameter
    for name, param in model.named_parameters():

        for layer_name, mask_tensor in mask.items():

            if name == layer_name:
                # rebuild mask
                mask_tensor_2d = torch.outer(mask_tensor[0], mask_tensor[1])
                gpu_mask = mask_tensor_2d.to(param.data.device)
                param.data *= gpu_mask
                del gpu_mask, mask_tensor_2d
                gc.collect()
                break # break the inner loop


def reload_from_cpu(model, model_state_dict_cpu:dict):
   
    # Modify the weight parameter
    for in_gpu_layer_name, in_gpu_tensor in model.named_parameters():
        
        for in_cpu_layer_name, in_cpu_tensor in model_state_dict_cpu.items():

            if in_gpu_layer_name == in_cpu_layer_name:

                in_gpu_tensor.data = in_cpu_tensor.to(device=in_gpu_tensor.data.device)
                gc.collect()
                break # break the inner loop

def store_state_dict(model):

    cpu_params = {}
    # Modify the weight parameter
    for in_gpu_layer_name, in_gpu_tensor in model.named_parameters():
        cpu_params[in_gpu_layer_name] = in_gpu_tensor.to(device="cpu")
    
    return cpu_params


def load_cached_activation(safetensors_path:str):

    cached_activation = load_all_tensors(safetensors_path)
    # there's inf value so i flip it to neg inf
    cached_activation = optree.tree_map(
        lambda x: torch.nan_to_num(
            x.to(device="cpu", dtype=torch.float32), posinf=-float("inf")
        ),
        cached_activation,
    )
    # then get the max and replace it
    cached_activation = optree.tree_map(
        lambda x: torch.nan_to_num(
            x.to(device="cpu", dtype=torch.float32), neginf=x.max()
        ),
        cached_activation,
    )
    gc.collect()
    # merge the head because i forgot to slain the hydra
    for layer in range(40):
        cached_activation[f"layer_{layer}.out_input"] = merge_heads(
            cached_activation[f"layer_{layer}.out_input"]
        )
    return cached_activation

def main():
    cached_activation = load_cached_activation("activation.safetensors")


    

    model_dir = "/home/zadmin/llama_stuff/CodeLlama-13b-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_dir)
    # you must use hf loader to load it as pipeline parallel (how annoying)
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        torch_dtype=torch.float16,
        # device_map="cpu"
        device_map="sequential",
        max_memory={0: "13GiB", 1: "24GiB"},
    )

    cpu_params = store_state_dict(model)


    prompt = "[INST] hello, can you explain what dark matter is?[/INST]"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

    # weight mask by activation
    # collapse the channel dimension by summing it 
    mask = optree.tree_map(lambda x: create_activation_mask(x, percentage=0.95),cached_activation)
    weight_mask = group_1d_mask(40, mask)
    modify_weights_with_1d_mask(model, weight_mask)
    # del weight_mask
    # gc.collect()
    # then mask must be loaded and unloaded manually
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

    # revert modification
    reload_from_cpu(model, cpu_params)
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    
    print()
    # layer = 0
    # plot_dict={
    #     f"model.layers.{layer}.self_attn.q_proj.weight":model.state_dict()[f"model.layers.{layer}.self_attn.q_proj.weight"] ,
    #     f"model.layers.{layer}.self_attn.k_proj.weight":model.state_dict()[f"model.layers.{layer}.self_attn.k_proj.weight"] ,
    #     f"model.layers.{layer}.self_attn.v_proj.weight":model.state_dict()[f"model.layers.{layer}.self_attn.v_proj.weight"] ,
    #     f"model.layers.{layer}.self_attn.o_proj.weight":model.state_dict()[f"model.layers.{layer}.self_attn.o_proj.weight"] ,
    #     f"model.layers.{layer}.mlp.gate_proj.weight":model.state_dict()[f"model.layers.{layer}.mlp.gate_proj.weight"] ,
    #     f"model.layers.{layer}.mlp.up_proj.weight":model.state_dict()[f"model.layers.{layer}.mlp.up_proj.weight"] ,
    #     f"model.layers.{layer}.mlp.down_proj.weight":model.state_dict()[f"model.layers.{layer}.mlp.down_proj.weight"] ,
    # }
    # plot_mask_dict={
    #     f"model.layers.{layer}.self_attn.q_proj.weight":weight_mask[f"model.layers.{layer}.self_attn.q_proj.weight"] ,
    #     f"model.layers.{layer}.self_attn.k_proj.weight":weight_mask[f"model.layers.{layer}.self_attn.k_proj.weight"] ,
    #     f"model.layers.{layer}.self_attn.v_proj.weight":weight_mask[f"model.layers.{layer}.self_attn.v_proj.weight"] ,
    #     f"model.layers.{layer}.self_attn.o_proj.weight":weight_mask[f"model.layers.{layer}.self_attn.o_proj.weight"] ,
    #     f"model.layers.{layer}.mlp.gate_proj.weight":weight_mask[f"model.layers.{layer}.mlp.gate_proj.weight"] ,
    #     f"model.layers.{layer}.mlp.up_proj.weight":weight_mask[f"model.layers.{layer}.mlp.up_proj.weight"] ,
    #     f"model.layers.{layer}.mlp.down_proj.weight":weight_mask[f"model.layers.{layer}.mlp.down_proj.weight"] ,
    # }
    # print()

    # # to determine the activation contribution
    # # then rescale to 0 to 1 range
    # cached_activation_scaled = optree.tree_map(
    #     lambda x: minmax_scale(x.view(2048, -1).to(dtype=torch.float32).sum(axis=0)),
    #     cached_activation,
    # )
    # # sort the tensor descending so lower activation value can be culled / removed
    # cached_activation_sorted = optree.tree_map(
    #     lambda x: x.sort(axis=0, descending=True).values, cached_activation_scaled
    # )
    # sigma_data = {
    #     "down_proj_input": [
    #         cached_activation_sorted[f"layer_{x}.down_proj_input"].numpy()
    #         for x in range(40)
    #     ],
    #     "down_proj_output": [
    #         cached_activation_sorted[f"layer_{x}.down_proj_output"].numpy()
    #         for x in range(40)
    #     ],
    #     "gate_proj_output": [
    #         cached_activation_sorted[f"layer_{x}.gate_proj_output"].numpy()
    #         for x in range(40)
    #     ],
    #     "key_output": [
    #         cached_activation_sorted[f"layer_{x}.key_output"].numpy() for x in range(40)
    #     ],
    #     "mlp_gate_up_input": [
    #         cached_activation_sorted[f"layer_{x}.mlp_gate_up_input"].numpy()
    #         for x in range(40)
    #     ],
    #     "out_input": [
    #         cached_activation_sorted[f"layer_{x}.out_input"].numpy() for x in range(40)
    #     ],
    #     "out_output": [
    #         cached_activation_sorted[f"layer_{x}.out_output"].numpy() for x in range(40)
    #     ],
    #     "qkv_input": [
    #         cached_activation_sorted[f"layer_{x}.qkv_input"].numpy() for x in range(40)
    #     ],
    #     "query_output": [
    #         cached_activation_sorted[f"layer_{x}.query_output"].numpy()
    #         for x in range(40)
    #     ],
    #     "up_proj_output": [
    #         cached_activation_sorted[f"layer_{x}.up_proj_output"].numpy()
    #         for x in range(40)
    #     ],
    #     "value_output": [
    #         cached_activation_sorted[f"layer_{x}.value_output"].numpy()
    #         for x in range(40)
    #     ],
    # }
    # stacked_plot(sigma_data, "percentage_0.8", percentage=0.8)
    # stacked_plot(sigma_data, "decay_2x", max_mag_decay_ratio=20)
    print()
    # for x in range(40):
    #     layer=x
    #     layer_of_interest = [
    #         f"layer_{layer}.down_proj_input",
    #         f"layer_{layer}.down_proj_output",
    #         f"layer_{layer}.gate_proj_output",
    #         f"layer_{layer}.key_output",
    #         f"layer_{layer}.mlp_gate_up_input",
    #         f"layer_{layer}.out_input",
    #         f"layer_{layer}.out_output",
    #         f"layer_{layer}.qkv_input",
    #         f"layer_{layer}.query_output",
    #         f"layer_{layer}.up_proj_output",
    #         f"layer_{layer}.value_output",
    #     ]
    #     observed_layer = get_dict_slice(cached_activation, layer_of_interest)
    #     # observed_layer[f"layer_{layer}.out_input"] = merge_heads(observed_layer[f"layer_{layer}.out_input"])

    #     graph_weight(observed_layer, f"layer_{x}_mag")


if __name__ == "__main__":
    main()
