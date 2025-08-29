import json
import torch
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Initialize FastAPI app
app = FastAPI()

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Use Llama.from_pretrained() for simpler model loading
# It automatically handles the download, and authentication
try:
    llm = Llama.from_pretrained(
        repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        filename="Meta-Llama-3-8B-Instruct.Q8_0.gguf",
        n_gpu_layers=40,
        n_ctx=4096,  # Set the context size
        verbose=True
    )
except Exception as e:
    print(f"Error loading model: {e}")
    llm = None

# Pydantic model for the request body
class PromptRequest(BaseModel):
    prompt: str

# API endpoint for generating responses
@app.post("/api/generate")
def generate_response(request: PromptRequest):
    """Generates a response from the Llama model."""
    if not llm:
        return {"error": "Model failed to load."}, 500
    
    # Format the prompt for Llama 3 instruction models
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{request.prompt}<|eot|><|start_header_id|>assistant<|end_header_id|>\n"
    
    # Perform inference
    output = llm(
        formatted_prompt,
        max_tokens=256, # Increased max tokens for more complete answers
        stop=["<|end_of_text|>", "<|eot|>"]
    )
    
    generated_text = output['choices'][0]['text']
    
    return {"response": generated_text}

# Endpoint to serve the main webpage
@app.get("/")
def serve_webpage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
