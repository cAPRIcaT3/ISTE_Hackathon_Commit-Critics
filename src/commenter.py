import glob
import os
import torch
import re
import json
import subprocess
from huggingface_hub import hf_hub_download
from llama_cpp import Llama


def run_lizard(directory):
    # Run Lizard and return the complexity report as JSON
    result = subprocess.run(['lizard', directory, '-l', 'cpp', '-l', 'python', '-o', 'json'], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception("Lizard failed: " + result.stderr)
    
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise Exception("Failed to parse JSON output: " + str(e))


model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
# GPU
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2, # CPU cores
    n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=32, # Change this value based on your model and your GPU VRAM pool.
    n_ctx = 3072
    )

# Get the path of the GitHub workspace
github_workspace_path = os.getenv("GITHUB_WORKSPACE")

# Open and read the "difference_hunk.txt" file
with open(f"{github_workspace_path}/difference_hunk.txt", "r") as diff_handle:
    diff = diff_handle.read()

prompt = ("just print:-comment will be shown here")
prompt_template=f'''SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully. 

USER: {prompt}

ASSISTANT:
'''
    
response=lcpp_llm(prompt=prompt_template, max_tokens=1536, temperature=0.5, top_p=0.95, repeat_penalty=1.2, top_k=50, echo=False)
response = response["choices"][0]["text"]

complexity_report = run_lizard(github_workspace_path)

# Write the comment to the output file
with open("src/files/output.txt", "a") as f:
  f.write(f"{response} + \n + \n + {complexity_report} ")
