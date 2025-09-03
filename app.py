# app.py
import torch
import os
import signal
import psutil
from fastmcp import FastMCP
from pydantic import BaseModel
from llama_cpp import Llama
from huggingface_hub import try_to_load_from_cache
from typing import List, Dict, Any
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path

# Define the server
mcp = FastMCP("llama-service")

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

# Load the model using the custom function
llm = load_llm_model()

# Pydantic model for the request body
class PromptRequest(BaseModel):
    prompt: str
    chat_history: List[Dict[str, str]] = []

# --- Standard HTTP Routes (for the web interface) ---
# Serve the main chat webpage at the root URL
@mcp.app.get("/", response_class=HTMLResponse)
async def serve_webpage_http():
    return Path("templates/index.html").read_text()

# Serve static CSS files
@mcp.app.get("/static/css/{filename:str}")
async def serve_static_css(filename: str):
    return FileResponse(f"static/css/{filename}")

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

# Helper function to kill existing processes
def kill_process_on_port(port: int):
    """
    Finds and kills any process using the specified port.
    This function is cross-platform, using the psutil library.
    """
    try:
        for proc in psutil.process_iter(['connections']):
            for conn in proc.info['connections']:
                if conn.laddr.port == port:
                    print(f"Killing process {proc.pid} on port {port}")
                    proc.send_signal(signal.SIGTERM)
    except Exception as e:
        print(f"Error while killing process on port {port}: {e}")

if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = 8000
    print(f"Starting server on http://{HOST}:{PORT}")
    kill_process_on_port(PORT)
    mcp.run(transport="http", host=HOST, port=PORT)
