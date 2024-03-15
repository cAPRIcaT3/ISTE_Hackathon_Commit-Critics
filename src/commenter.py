import glob
import os
import torch
import re
from huggingface_hub import hf_hub_download
from llama_cpp import Llama


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

prompt = ("Here is the code difference" + diff)
prompt_template = """
SYSTEM: You are a highly knowledgeable and concise code review assistant with expertise in various programming languages and development practices. Your task is to provide clear, insightful, and concise summaries of code changes to help developers understand the implications of these changes quickly. Focus on summarizing the intent behind the changes, any potential impact on the project, and suggestions for improvement if necessary.

CODE DIFFERENCE:
{diff}

Based on the above code difference, write a one-line summary that includes:
1. The nature of the change (e.g., bug fix, feature addition, code refactoring).
2. The primary effect of this change on the project (e.g., improves performance, fixes a user-reported issue, enhances readability of code).
3. Any recommendations for further improvements or considerations.

Format your response as follows:
CHANGE: [Nature of the change]. EFFECT: [Primary effect of this change]. RECOMMENDATIONS: [Any recommendations].

Remember, your response should be concise, informative, and directly related to the code difference provided.
"""
    
response=lcpp_llm(prompt=prompt_template, max_tokens=1536, temperature=0.5, top_p=0.95, repeat_penalty=1.2, top_k=35, echo=False)
response = response["choices"][0]["text"]

# Write the comment to the output file
with open("src/files/output.txt", "a") as f:
  f.write(f"{response}")
