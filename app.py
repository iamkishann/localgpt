import json
import torch
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Define the model ID from Hugging Face
model_id = 'MBZUAI/LaMini-Flan-T5-783M'

# Check for GPU availability and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on device: {device}")

# Create a text-to-text generation pipeline, loading the model once
try:
    text_pipeline = pipeline(
        "text2text-generation",
        model=model_id,
        device=device,
    )
except Exception as e:
    print(f"Error loading model: {e}")
    text_pipeline = None

# Pydantic model for the request body
class PromptRequest(BaseModel):
    prompt: str

# API endpoint for generating responses
@app.post("/api/generate")
def generate_response(request: PromptRequest):
    """Generates a response from the text-to-text model."""
    if not text_pipeline:
        return {"error": "Model failed to load."}, 500
    
    # Perform inference using the pipeline
    output = text_pipeline(request.prompt, max_length=150)
    generated_text = output[0]['generated_text']
    
    return {"response": generated_text}

# Endpoint to serve the main webpage
@app.get("/")
def serve_webpage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
