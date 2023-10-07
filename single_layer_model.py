import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from llama import LlamaForCausalLM, LlamaConfig
from typing import Any
from torch import nn
import gc
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from transformers.activations import ACT2FN
import optree
import safetensors
from safetensors.torch import save_file

def dict_to_dot_notation(input_dict, parent_key='', separator='.'):
    """
    Convert a dictionary to dot notation.
    
    Args:
        input_dict (dict): The input dictionary to be converted.
        parent_key (str): The parent key for recursive calls (used internally).
        separator (str): The separator to use between keys in dot notation.

    Returns:
        dict: A new dictionary with dot notation keys.
    """
    output_dict = {}
    for key, value in input_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            output_dict.update(dict_to_dot_notation(value, new_key, separator))
        else:
            output_dict[new_key] = value
    return output_dict

def dot_notation_to_dict(input_dict, separator='.'):
    """
    Convert a dictionary in dot notation back to a nested dictionary.
    
    Args:
        input_dict (dict): The input dictionary with dot notation keys.
        separator (str): The separator used between keys in dot notation.

    Returns:
        dict: A new nested dictionary.
    """
    output_dict = {}
    for key, value in input_dict.items():
        keys = key.split(separator)
        current_dict = output_dict
        for k in keys[:-1]:
            current_dict = current_dict.setdefault(k, {})
        current_dict[keys[-1]] = value
    return output_dict

def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=4096):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]


class LlamaEmbedding(nn.Module):
    """
    Just a Llama Embedding layer
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    
    def forward(self, input_ids):
        return self.embed_tokens(input_ids)



def main():
    model_dir = "/home/zadmin/llama_stuff/CodeLlama-13b-Instruct-hf"

    config = LlamaConfig.from_json_file("dummy_config.json")
    embbeding_model = LlamaEmbedding(config)

    # load model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_dir)
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        torch_dtype=torch.float16,
        device_map="sequential",
        max_memory={0: "13GiB", 1: "24GiB"},
    )

    # get dataset
    token = get_calib_dataset(tokenizer=tokenizer, n_samples=5000*4, block_size=2048)

    with torch.no_grad():
        # compute and stash gemm activation
        outputs = model(token[0], output_gemm_activation=True)
        activation = optree.tree_map(lambda gpu_leaf: gpu_leaf.to(device="cpu"), outputs[2])
        gc.collect()

        for x in range(1, len(token)):
            outputs = model(token[x], output_gemm_activation=True)
            activation = optree.tree_map(lambda cpu_leaf, gpu_leaf: cpu_leaf + gpu_leaf.to(device="cpu"), activation, outputs[2])
            gc.collect()
        
        # absolute average activation magnitude
        activation = optree.tree_map(lambda cpu_leaf: torch.abs(cpu_leaf / len(token)), activation)

        # flatten the dict for saving it as safetensors
        activation = dict_to_dot_notation(activation)
        save_file(activation, "activation.safetensors")

    print()

if __name__ == "__main__":
    main()