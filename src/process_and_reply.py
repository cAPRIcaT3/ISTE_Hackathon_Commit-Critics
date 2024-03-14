import os
import sys
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Initialize Llama CPP with the model
model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2,
    n_batch=512,
    n_gpu_layers=32,
    n_ctx=3072
)

def generate_reply(comment):
    prompt = f"""SYSTEM: You are a helpful, respectful, and honest assistant.

USER: {comment}

ASSISTANT:
"""
    response = lcpp_llm(prompt=prompt, max_tokens=1536, temperature=0.5, top_p=0.95, repeat_penalty=1.2, top_k=50, echo=False)
    return response["choices"][0]["text"]

if __name__ == "__main__":
    comment = os.getenv('COMMENT_BODY')
    if not comment:
        sys.exit("No comment provided.")

    reply = generate_reply(comment)

    # Write the reply to a file for the GitHub Action to use
    with open("src/files/output.txt", "w") as f:
        f.write(reply)
