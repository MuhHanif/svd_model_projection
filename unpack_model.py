from transformers import AutoTokenizer
import transformers
import torch
import re
from matplotlib import pyplot as plt
import numpy as np

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

# init model from pipeline (easier to do)
model_dir = "/home/zadmin/llama_stuff/CodeLlama-13b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_dir,
    torch_dtype=torch.float16,
    device_map="cpu",
)

# grab model layer name
layer_name = list(pipeline.model.state_dict().keys())

# check if the model has bias
is_bias_exist = [leaf_node if bool(re.search(r"bias", leaf_node)) else None for leaf_node in layer_name]


# grab leaf node dimension for each layer 
model_shape = {}
gemm_layer = {}
for leaf_name, leaf_params in pipeline.model.state_dict().items():
    model_shape[leaf_name] = leaf_params.shape
    if len(leaf_params.shape) == 2:
        gemm_layer[leaf_name] = leaf_params.numpy()


for leaf_name, leaf_params in gemm_layer.items():
    graph_svd(name=leaf_name, sample=leaf_params)

# sampled layer
# sample = pipeline.model.state_dict()["model.layers.39.self_attn.q_proj.weight"].numpy()



print()