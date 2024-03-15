import glob
import os
import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama


model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
# GPU
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2, # CPU cores
    n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU. Higher value for professional github runners. 
    n_gpu_layers=32, # Change this value based on your model and your GPU VRAM pool. High value tolerated for professional github runners.
    n_ctx = 3072 # Max context length
    )

# Get the path of the GitHub workspace
github_workspace_path = os.getenv("GITHUB_WORKSPACE")

# Open and read the "difference_hunk.txt" file
with open(f"{github_workspace_path}/difference_hunk.txt", "r") as diff_handle:
    diff = diff_handle.read()

prompt = ("Here is the code difference" + diff)
prompt_template = prompt_template = f"""
Given the code difference below, provide a concise one-line summary that includes the nature of the change, its primary effect on the project, and any quick recommendations. Use the format: 
"CHANGE: [long description];
EFFECT: [impact in points];
RECOMMENDATIONS: [if any]."

CODE DIFFERENCE:
{diff}
"""
    
response=lcpp_llm(prompt=prompt_template, max_tokens=1536, temperature=0.5, top_p=0.95, repeat_penalty=1.2, top_k=75, echo=False)
response = response["choices"][0]["text"]

# Write the comment to the output file
with open("src/files/output.txt", "a") as f:
  f.write(f"{response}")
