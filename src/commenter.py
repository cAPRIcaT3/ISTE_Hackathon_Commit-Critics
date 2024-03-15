import glob
import os
import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import os
from Knowledge_Based_RAG import generate_comment

# Assuming the rest of your setup and GitHub Actions context handling is unchanged

# Get the path of the GitHub workspace
github_workspace_path = os.getenv("GITHUB_WORKSPACE")

# Open and read the "difference_hunk.txt" file
with open(f"{github_workspace_path}/difference_hunk.txt", "r") as diff_handle:
    diff = diff_handle.read()

prompt = ("Here is the code difference" + diff)
# Generate a query based on the diff or the context of the PR
query = "Generate a concise summary based on the following code changes: " + prompt

# Use the RAG pipeline to generate a response
response = generate_comment(query)

# Write the comment to the output file
with open("src/files/output.txt", "w") as f:
  f.write(response)
