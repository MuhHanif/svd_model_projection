from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import msgpack
import pandas as pd
from typing import List

# Open the MsgPack file in binary read mode
with open('CodeLlama-13b-Instruct-hf-sigma.msgpack', 'rb') as file:
    # Read the contents of the file
    msgpack_data = file.read()

# Unpack the MsgPack data into a dictionary
unpacked_data = msgpack.unpackb(msgpack_data, raw=False)

def convert_leaf_to_np_array(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = convert_leaf_to_np_array(value)
    elif isinstance(data, list):
        return np.array(data)
    return data

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

def gavish_donoho_magic_constant_square_matrix(S):
    # Convert S to a NumPy array
    S = np.array(S)
    
    # Initialize max_sv with the value of the first element of S
    index_truncation = 2.858 * np.median(S)
    
    return index_truncation

def stacked_plot(
    treshold:float,
    query:List[np.array], 
    key:List[np.array], 
    value:List[np.array], 
    out:List[np.array], 
    gate:List[np.array], 
    up:List[np.array], 
    down:List[np.array], 
    hidden_dim_size:int,
    mlp_proj_dim_size:int,
    name:str) -> None:


    # Create a subplots

    sigma_data = {
        "query":query, 
        "key":key, 
        "value":value,
        "out":out,
        "gate":gate,
        "up":up,
        "down":down,
        }

    
    fig, axs = plt.subplots(len(sigma_data), 4, figsize=(20*3, 10*len(sigma_data)))

    for row, items in enumerate(sigma_data.items()):
        chart_name, data = items


        # create rainbow opacity
        gradient_color = (1-0.1) / len(data)
        for count, array in enumerate(data):
            array_treshold = index_sv_ratio(array, treshold)
            orig_array = array
            array = array[:array_treshold]
            # print(0.1 + gradient_color * count)
            x = np.linspace(0, len(array), len(array))
            axs[row,0].semilogy(x, array, label=f'{chart_name} layer {count}', alpha=0.1 + gradient_color * count, color=(0.1 + gradient_color * count,0.,0.))
            axs[row,1].plot(x,np.cumsum(array/np.sum(orig_array)), label=f'{chart_name} layer {count}', alpha=0.1 + gradient_color * count, color=(0.1 + gradient_color * count,0.,0.))

        axs[row,0].legend()
        axs[row,0].set_title(f'{chart_name} sigma progression', fontsize=16)

        axs[row,1].legend()
        axs[row,1].set_title(f'{chart_name} sigma cumulative', fontsize=16)


        rank = [index_sv_ratio(x, treshold) for x in data]

        x = np.linspace(0, len(rank), len(rank))
        axs[row,2].bar(x, rank, label=f'layer rank theshold max/{treshold}', color=(1.,0.,0.))
        axs[row,2].legend()
        axs[row,2].set_title(f'{chart_name} rank', fontsize=16)


        # this is wrong, there's tall and wide matrix that i should accounted for
        tensor_size = np.array(rank) * hidden_dim_size * 2
        if chart_name in ["up", "down", "gate"]:
            hidden_dim_size_array = np.array([hidden_dim_size * mlp_proj_dim_size]*len(rank))
        else:
            hidden_dim_size_array = np.array([hidden_dim_size ** 2]*len(rank))
        print(hidden_dim_size_array)
        reduction_ratio = tensor_size/hidden_dim_size_array

        x = np.linspace(0, len(rank), len(rank))
        
        axs[row,3].bar(x, 1-reduction_ratio, label=f'layer rank theshold max/{treshold}', color=(1.,0.,0.))
        axs[row,3].legend()
        axs[row,3].set_title(f'{chart_name} reduction', fontsize=16)


    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.close()

unpacked_data = convert_leaf_to_np_array(unpacked_data)
query = [unpacked_data[f"model.layers.{x}.self_attn.q_proj.weight"]["sigma_array"] for x in range(40)]
key = [unpacked_data[f"model.layers.{x}.self_attn.k_proj.weight"]["sigma_array"] for x in range(40)]
value = [unpacked_data[f"model.layers.{x}.self_attn.v_proj.weight"]["sigma_array"] for x in range(40)]
out = [unpacked_data[f"model.layers.{x}.self_attn.o_proj.weight"]["sigma_array"] for x in range(40)]
gate = [unpacked_data[f"model.layers.{x}.mlp.gate_proj.weight"]["sigma_array"] for x in range(40)]
up = [unpacked_data[f"model.layers.{x}.mlp.up_proj.weight"]["sigma_array"] for x in range(40)]
down = [unpacked_data[f"model.layers.{x}.mlp.down_proj.weight"]["sigma_array"] for x in range(40)]
stacked_plot(10,query, key, value, out, gate, up, down, 5120,1231231, "50")
print()