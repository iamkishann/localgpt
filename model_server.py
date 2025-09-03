# model_server.py
import torch
import os
import signal
import psutil
from fastmcp import FastMCP
from pydantic import BaseModel
from llama_cpp import Llama
from huggingface_hub import try_to_load_from_cache
from typing import List, Dict, Any
from starlette.requests import Request
from starlette.responses import JSONResponse

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

# Define the LLM tool endpoint
@mcp.tool()
async def generate_response(prompt: str, chat_history: List[Dict[str, str]] = []):
    """Generates a text completion from the Llama model, considering chat history."""
    print("Inside generate_response tool!", flush=True)
    if not llm:
        return {"error": "Model failed to load."}, 500
    
    messages = chat_history + [{"role": "user", "content": prompt}]
    
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        stop=["<|end_of_text|>", "<|eot|>"]
    )
    generated_text = output['choices']['message']['content']
    return generated_text

# --- CRUCIAL: Add the custom route to handle MCP messages ---
@mcp.custom_route("/mcp/messages", methods=["POST"])
async def mcp_messages_handler(request: Request) -> JSONResponse:
    """Handles JSON-RPC requests for MCP tools."""
    try:
        payload = await request.json()
        tool_name = payload.get("method")
        params = payload.get("params", {})
        
        # Dispatch the tool call, unpacking the parameters
        result = await mcp.dispatch_tool(tool_name, **params)
        
        # Return the JSON-RPC compliant response
        response_payload = {
            "jsonrpc": "2.0",
            "result": result
        }
        return JSONResponse(response_payload)
    except Exception as e:
        # Return an error in JSON-RPC format
        print(f"Error in mcp_messages_handler: {e}", flush=True)
        return JSONResponse({"jsonrpc": "2.0", "error": {"message": str(e)}}, status_code=500)

if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 8001
    print(f"Starting MCP Server on http://{HOST}:{PORT}")
    mcp.run(transport="http", host=HOST, port=PORT)
