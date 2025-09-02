import torch
import os
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_cpp import Llama
from fastmcp import FastMCP
from huggingface_hub import hf_hub_download, try_to_load_from_cache

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
    llm_instance = None
    try:
        # Check for the cached file path
        model_path = try_to_load_from_cache(
            repo_id=REPO_ID,
            filename=FILENAME
        )
        
        if model_path and os.path.exists(model_path):
            print(f"Loading model from cache: {model_path}")
            llm_instance = Llama(
                model_path=str(model_path),
                n_gpu_layers=40,
                n_ctx=4096,
                verbose=True
            )
        else:
            print("Model not found in cache. Downloading...")
            # Fallback to from_pretrained which will download and cache
            llm_instance = Llama.from_pretrained(
                repo_id=REPO_ID,
                filename=FILENAME,
                n_gpu_layers=40,
                n_ctx=4096,
                verbose=True
            )
    except Exception as e:
        print(f"Error loading model: {e}")
        llm_instance = None
    return llm_instance

# Load the model using the custom function
llm = load_llm_model()

# Pydantic model for the request body
class PromptRequest(BaseModel):
    prompt: str

# API endpoint
@app.post("/api/generate")
def generate_response(request_data: PromptRequest):
    """Generates a text completion from the Llama model based on a user prompt."""
    if not llm:
        return {"error": "Model failed to load."}, 500
    
    # Use the create_chat_completion method
    output = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": request_data.prompt}
        ],
        max_tokens=512,
        stop=["<|end_of_text|>", "<|eot|>"]
    )
    
    generated_text = output['choices']['message']['content']
    
    return {"response": generated_text}

# Endpoint to serve the main webpage
@app.get("/")
def serve_webpage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Convert the existing FastAPI app into a FastMCP server
mcp = FastMCP.from_fastapi(app=app, name="llama-service")

if __name__ == "__main__":
    mcp.run()
