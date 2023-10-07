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

# type hints stuff
LLM_models = Any

model_dir = "/home/zadmin/llama_stuff/CodeLlama-13b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_dir)
model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_dir,
    torch_dtype=torch.float16,
    device_map="sequential",
    max_memory={0: "13GiB", 1: "24GiB"},
)

def graph_weight(sample:np.array, new_tensor:np.array, name:str=None) -> None:
  
  cmap = mcolors.LinearSegmentedColormap.from_list(
    'custom_cmap',
    [(1, 1, 1),
    (0, 0, 1),  # Blue for values < 0
    (1, 0, 0),  # Red for values > 0
    ]  # White for values = 0
    
  )
    
  # Create a 3x2 grid of subplots
  fig, axs = plt.subplots(2, 3, figsize=(15*3, 15*2))
  # Create a meshgrid for the x and y coordinates
  x, y = np.meshgrid(np.arange(sample.shape[1]), np.arange(sample.shape[0]))

  # Create conture map original
  axs[1,0] = plt.subplot(2,3,4, projection='3d')
  axs[1,0].plot_surface(x, y, sample, cmap=cmap)
  axs[1,0].set_title('Heatmap model')

  # Create conture map reconstruction
  axs[1,1] = plt.subplot(2,3,5, projection='3d')
  axs[1,1].plot_surface(x, y, new_tensor, cmap=cmap)
  axs[1,1].set_title('reconstructed model')

  # Create conture map delta
  axs[1,2] = plt.subplot(2,3,6, projection='3d')
  axs[1,2].plot_surface(x, y, new_tensor - sample, cmap=cmap)
  axs[1,2].set_title('delta model')


  # Create the heatmap original
  axs[0,0].imshow(sample, cmap=cmap, interpolation='nearest')
  axs[0,0].set_title('Heatmap model')
  axs[0,0].figure.colorbar(axs[0,0].get_images()[0], ax=axs[0,0])
  
  # Create the heatmap reconstruction
  axs[0,1].imshow(new_tensor, cmap=cmap, interpolation='nearest')
  axs[0,1].set_title('reconstructed model')
  axs[0,1].figure.colorbar(axs[0,1].get_images()[0], ax=axs[0,1])

  # Create the heatmap delta
  axs[0,2].imshow(new_tensor - sample, cmap=cmap, interpolation='nearest')
  axs[0,2].set_title('delta model')
  axs[0,2].figure.colorbar(axs[0,2].get_images()[0], ax=axs[0,2])

  # Adjust spacing between subplots
  plt.tight_layout()
  # plt.show()
  if name != None:
    plt.savefig(f"{name}.png")
  plt.close()

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


def move_tensors_to_cpu(nested_list):
    if isinstance(nested_list, torch.Tensor):
        nested_list = torch.abs(nested_list.mean(axis=0))
        cpu_nested_list = nested_list.to('cpu')
        del nested_list
        torch.cuda.empty_cache()
        gc.collect()
        return cpu_nested_list
    elif isinstance(nested_list, list):
        return [move_tensors_to_cpu(item) for item in nested_list]
    elif isinstance(nested_list, tuple):
        return tuple(move_tensors_to_cpu(item) for item in nested_list)
    elif isinstance(nested_list, dict):
        return {key: move_tensors_to_cpu(value) for key, value in nested_list.items()}
    else:
        return nested_list  # Return unchanged if not a tensor or container type

class BrainProbe:
    def __init__(self, model: LLM_models, cache_tensor_location: str = "cpu", accumulation_scale:int = 1):
        # use accumulation scale to do averaging since the weight can only be added one by one
        self.cache_tensor_location = cache_tensor_location
        self.activations = {}
        self.model = model
        self.scale = accumulation_scale
  
    def measurement_probe(self, name: str):
        def _measure(
            module, input_tensor: torch.tensor, output_tensor: torch.tensor
        ) -> None:
            """
            intercept and store the activation
            """

            def _move_tensor_to_x(tensor):
                cpu_tensor = tensor.to(self.cache_tensor_location)
                del tensor
                torch.cuda.empty_cache()
                gc.collect()
                return cpu_tensor
                
            self.activations[name] = {
                "input_activation": optree.tree_map(_move_tensor_to_x, input_tensor),
                "output_activation": optree.tree_map(_move_tensor_to_x, output_tensor),
            }
            del input_tensor, output_tensor
            torch.cuda.empty_cache()
            gc.collect()

        return _measure

    def attach_probe(self):
        # Iterate through all modules in the model and register hooks
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Module):
                # skips embedding layer
                if name == "model.embed_tokens":
                    continue
                module.register_forward_hook(self.measurement_probe(name))
        pass


token = get_calib_dataset(tokenizer=tokenizer, n_samples=100, block_size=4096)

probe = BrainProbe(model=model)
probe.attach_probe()

with torch.no_grad():

    outputs = model(token[0])
    cache = probe.activations
    for x in range(1, len(token)):
        outputs = model(token[x])
        cache = optree.tree_map(lambda x, y: x + y, cache, probe.activation)
print()
