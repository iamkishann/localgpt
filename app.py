import json
import torch
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# IMPORTANT: Ensure these libraries are installed
# pip install llama-cpp-python[server]
# pip install huggingface_hub

from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Initialize FastAPI app
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Define the Hugging Face repository and the specific GGUF filename
repo_id = "unsloth/DeepSeek-V3-0324-GGUF"
filename = "DeepSeek-V3-0324-Q2_K_XL.gguf"

# This will be automatically cached, so it's only downloaded once.
try:
    print(f"Downloading model from {repo_id}...")
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"Model downloaded to: {model_path}")
except Exception as e:
    print(f"Error downloading model: {e}")
    model_path = None

# A higher number uses more VRAM but is faster.
n_gpu_layers = 40

# Create the Llama instance for the GGUF model
try:
    if model_path:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=4096,
            verbose=True
        )
    else:
        llm = None
except Exception as e:
    print(f"Error loading model: {e}")
    llm = None

# Pydantic model for the request body
class PromptRequest(BaseModel):
    prompt: str

# API endpoint for generating responses
@app.post("/api/generate")
def generate_response(request: PromptRequest):
    """Generates a response from the GGUF model."""
    if not llm:
        return {"error": "Model failed to load."}, 500
    
    # Perform inference
    output = llm(
        request.prompt,
        max_tokens=150,
        stop=["<｜end of sentence｜>", "<|im_end|>"]
    )
    
    generated_text = output['choices'][0]['text']
    
    return {"response": generated_text}

# Endpoint to serve the main webpage
@app.get("/")
def serve_webpage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
