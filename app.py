# app.py
import torch
import os
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_cpp import Llama
from fastmcp import FastMCP
from huggingface_hub import try_to_load_from_cache
from typing import List, Dict, Any

# Initialize FastAPI app
app = FastAPI()

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Jinja2 templates
templates = Jinja2Templates(directory="templates")

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

# Pydantic model for the request body, now including chat history
class PromptRequest(BaseModel):
    prompt: str
    chat_history: List[Dict[str, str]] = []

# Define the LLM tool endpoint
@app.post("/api/generate")
async def generate_response(request_data: PromptRequest):
    """Generates a text completion from the Llama model, considering chat history."""
    if not llm:
        return {"error": "Model failed to load."}, 500
    
    # Construct the messages list for the LLM from the provided history and current prompt
    messages = request_data.chat_history + [{"role": "user", "content": request_data.prompt}]
    
    # Use the create_chat_completion method with the full message history
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        stop=["<|end_of_text|>", "<|eot|>"]
    )
    generated_text = output['choices']['message']['content']
    return {"response": generated_text}

# Endpoint to serve the main webpage
@app.get("/")
async def serve_webpage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Convert the existing FastAPI app into a FastMCP server
mcp = FastMCP.from_fastapi(app=app, name="llama-service")

if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = 8000
    print(f"Starting server on http://{HOST}:{PORT}")
    mcp.run(transport="http", host=HOST, port=PORT)
