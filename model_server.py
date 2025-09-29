import torch
import os
import signal
import psutil
import asyncio
from fastmcp import FastMCP
from pydantic import BaseModel
from llama_cpp import Llama
from huggingface_hub import try_to_load_from_cache
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

REPO_ID = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
FILENAME = "Meta-Llama-3-8B-Instruct.Q8_0.gguf"
llm = None  # Will be loaded in a startup event

def load_llm_model():
    """Checks for cached model and loads it, or downloads it if not found."""
    try:
        model_path = try_to_load_from_cache(repo_id=REPO_ID, filename=FILENAME)
        if model_path and os.path.exists(model_path):
            print(f"Loading model from cache: {model_path}")
        else:
            print("Model not found in cache. Attempting to download...")
            # Use a more robust download method
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
            print(f"Model downloaded to: {model_path}")
            
        print("Initializing Llama model...")
        llm = Llama(
            model_path=str(model_path), 
            n_gpu_layers=15, # Use a conservative value
            n_ctx=4096, 
            verbose=True
        )
        print("Llama model initialized successfully.")
        return llm
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Add a check to see if the model file is missing or corrupted
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
        return None

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global llm
    llm = await asyncio.to_thread(load_llm_model)

class GenerateRequest(BaseModel):
    prompt: str
    chat_history: List[Dict[str, str]] = []

@app.post("/api/generate", operation_id="generate_response")
async def generate_response_endpoint(data: GenerateRequest):
    """Generates a text completion from the Llama model, considering chat history."""
    if not llm:
        return JSONResponse({"error": "Model failed to load."}, status_code=500)
    
    messages = data.chat_history + [{"role": "user", "content": data.prompt}]
    
    output = await asyncio.to_thread(llm.create_chat_completion,
            messages=messages,
            max_tokens=512,
            stop=["<|end_of_text|>", "<|eot|>"])
            
    generated_text = output['choices'][0]['message']['content']
    return JSONResponse({"generated_text": generated_text})

class CodeReviewRequest(BaseModel):
    code_diff: str

@app.post('/api/review-code', operation_id="review_code_with_llm")
async def review_code_endpoint(data: CodeReviewRequest):
    """Analyzes a code diff and provides review comments using the LLM."""
    if not data.code_diff:
        return JSONResponse({'error': 'No code diff provided'}, status_code=400)

    if not llm:
        return JSONResponse({"error": "Model failed to load."}, status_code=500)
    
    system_prompt = (
        "You are an AI code reviewer. Analyze the following code changes and "
        "provide suggestions, identify potential bugs, or offer improvements. "
        "Address code style, best practices, and efficiency. Focus on the provided code diff."
    )
    
    try:
        response = await asyncio.to_thread(llm.create_chat_completion,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': f"Review the following code:\n\n{data.code_diff}"}
                    ],
                    max_tokens=512,
                    stop=["<|end_of_text|>", "<|eot|>"])
        
        return JSONResponse({'comment': response['choices'][0]['message']['content']})
    except Exception as e:
        print(f"Error calling LLM for code review: {e}")
        return JSONResponse({'error': 'Internal server error'}, status_code=500)

if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 8001
    print(f"Starting FastAPI server with custom routes on http://{HOST}:{PORT}")
    
    # FastMCP now correctly introspects the FastAPI app to find both endpoints.
    mcp = FastMCP.from_fastapi(app=app, name="llama-service")
    mcp.run(transport="http", host=HOST, port=PORT)
