import subprocess
import time
import sys
from fastmcp import FastMCP
from typing import List, Dict, Any
from openai import OpenAI
import sys
from huggingface_hub import try_to_load_from_cache
from flask import Flask, request, jsonify

# Define the server. FastMCP will listen on this port.
mcp = FastMCP("openai-gpt-oss-20b-service")

# --- vLLM Server Configuration ---
MODEL_ID = "openai/gpt-oss-20b"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8002  # Use a different port for vLLM
VLLM_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
vllm_process = None

# OpenAI client setup to talk to the vLLM server
openai_client = OpenAI(
    api_key="sk-not-required",
    base_url=VLLM_URL,
)

def start_vllm_server():
    """Starts the vLLM server in a subprocess."""
    global vllm_process
    # 1. Pre-check if the model is already in cache
    #    This prevents unnecessary download attempts if network issues exist
    print(f"Starting vLLM server for model: {MODEL_ID} on {VLLM_URL}")
    
    model_cache_path = try_to_load_from_cache(repo_id=MODEL_ID, filename="config.json")
    if not model_cache_path:
        print(f"Model {MODEL_ID} not found in cache. vLLM will attempt to download it.")
    else:
        print(f"Model {MODEL_ID} found in cache at: {model_cache_path}")
        
    command = [
        "vllm", "serve",
        MODEL_ID,
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--served-model-name", MODEL_ID,
        "--max-model-len", "128000"
    ]
    
    # We detach the child process from the parent's terminal session
    vllm_process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stdout and stderr for easier logging
        start_new_session=True,  # This detaches the process
        text=True, # Ensure output is treated as text
        bufsize=1 # Line-buffered output
    )

    print("Waiting for vLLM server to start and model to download...")
    
    # Read the output line by line to monitor progress and prevent blocking
    while True:
        line = vllm_process.stdout.readline()
        if line == '' and vllm_process.poll() is not None:
            break
        if line:
            print(f"vLLM: {line.strip()}")
            # Add a more robust readiness check (e.g., waiting for a specific log line)
            if "Uvicorn running on" in line:
                print("vLLM server confirmed running.")
                break
    
    # Check if the process exited prematurely with an error
    if vllm_process.returncode is not None:
        raise RuntimeError(f"vLLM server exited unexpectedly with code {vllm_process.returncode}")

@mcp.tool()
async def generate_response(prompt: str, chat_history: List[Dict[str, str]] = []):
    """Generates a text completion using the gpt-oss-20b model via the vLLM server."""
    messages = chat_history + [{"role": "user", "content": prompt}]
    
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=512,
            stop=["<|end_of_text|>", "<|eot|>"]
        )
        generated_text = response.choices.message.content
        return generated_text
    except Exception as e:
        print(f"Error generating response from vLLM: {e}")
        return {"error": "Failed to generate response."}, 500

@mcp.tool()
async def review_code_with_llm(code_diff: str):
    """Analyzes a code diff and provides review comments using the LLM."""
    # Define the system prompt for code review
    system_prompt = (
        "You are an AI code reviewer. Analyze the following code changes and "
        "provide suggestions, identify potential bugs, or offer improvements. "
        "Address code style, best practices, and efficiency. Focus on the provided code diff."
    )
    
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Review the following code:\n\n{code_diff}"}
            ],
            max_tokens=512,
            stop=["<|end_of_text|>", "<|eot|>"]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating code review from vLLM: {e}")
        return f"Error: Failed to generate code review. {str(e)}"

app = Flask(__name__)
@app.route('/api/review-code', methods=['POST'])
async def review_code_endpoint():
    data = request.get_json()
    code_diff = data.get('code_diff', '')
    
    if not code_diff:
        return jsonify({'error': 'No code diff provided'}), 400

    try:
        # Call the dedicated mcp.tool() from the custom Flask endpoint
        review_comment = await review_code_with_llm(code_diff)
        return jsonify({'comment': review_comment})
    except Exception as e:
        print(f"Error calling review_code_with_llm: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    start_vllm_server()
    
    HOST = "127.0.0.1"
    PORT = 8001
    print(f"Starting MCP Server with custom routes on http://{HOST}:{PORT}")
    try:
        mcp.add_api_routes(app)
        mcp.run(transport="http", host=HOST, port=PORT)
    except Exception as e:
        print(f"MCP Server shut down with an error: {e}")
