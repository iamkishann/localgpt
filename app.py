# app.py
import torch
import os
import signal
import psutil
from fastmcp import FastMCP, StaticFile
from pydantic import BaseModel
from llama_cpp import Llama
from huggingface_hub import try_to_load_from_cache
from typing import List, Dict, Any

# Define the server and static/template resources
mcp = FastMCP("llama-service")
mcp.add_resource(StaticFile("static"), name="static")
mcp.add_resource(StaticFile("templates"), name="templates")

# Model details
REPO_ID = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
FILENAME = "Meta-Llama-3-8B-Instruct.Q8_0.gguf"

# Function to load the model
def load_llm_model():
    """Checks for cached model and loads it, or downloads it if not found."""
    try:
        model_path = try_to_load_from_cache(repo_id=REPO_ID, filename=FILENAME)
        if model_path and os.path.exists(model_path):
            print(f"Loading model from cache: {model_path}")
            return Llama(model_path=str(model_path), n_gpu_layers=40, n_ctx=4096, verbose=True)
        else:
            print("Model not found in cache. Downloading...")
            return Llama.from_pretrained(repo_id=REPO_ID, filename=FILENAME, n_gpu_layers=40, n_ctx=4096, verbose=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Pydantic model for the request body
class PromptRequest(BaseModel):
    prompt: str
    chat_history: List[Dict[str, str]] = []

# Define the LLM tool endpoint
@mcp.tool()
async def generate_response(request_data: PromptRequest):
    """Generates a text completion from the Llama model, considering chat history."""
    if not llm:
        return {"error": "Model failed to load."}, 500
    
    messages = request_data.chat_history + [{"role": "user", "content": request_data.prompt}]
    
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        stop=["<|end_of_text|>", "<|eot|>"]
    )
    generated_text = output['choices']['message']['content']
    return {"response": generated_text}

# Define the webpage endpoint
@mcp.tool()
async def serve_webpage():
    """Serves the main chat webpage."""
    return await mcp.resources.templates["index.html"]

# Helper function to kill existing processes
def kill_process_on_port(port: int):
    # ... (same kill_process_on_port function as before) ...

if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = 8000
    print(f"Starting server on http://{HOST}:{PORT}")
    kill_process_on_port(PORT)
    mcp.run(transport="http", host=HOST, port=PORT)
