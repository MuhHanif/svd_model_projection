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

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_dir)
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        torch_dtype=torch.float16,
        device_map="sequential",
        max_memory={0: "13GiB", 1: "24GiB"},
    )

    
    token = get_calib_dataset(tokenizer=tokenizer, n_samples=500, block_size=2048)

    with torch.no_grad():
        outputs = model(token[0], output_gemm_activation=True)
        activation = optree.tree_map(lambda gpu_leaf: gpu_leaf.to(device="cpu"), outputs[2])
        gc.collect()

        for x in range(1, len(token)):
            outputs = model(token[x], output_gemm_activation=True)
            activation = optree.tree_map(lambda cpu_leaf, gpu_leaf: cpu_leaf + gpu_leaf.to(device="cpu"), activation, outputs[2])
            gc.collect()
        
        activation = optree.tree_map(lambda cpu_leaf: cpu_leaf / len(token), activation)
    print()

if __name__ == "__main__":
    main()