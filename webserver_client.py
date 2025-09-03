# webserver_client.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
from flask import Flask, render_template, jsonify, request, send_from_directory
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any
from fastmcp import Client # Correct import for the client class

# Define the client service
app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

# --- Pydantic model for request validation ---
class PromptRequest(BaseModel):
    prompt: str
    chat_history: List[Dict[str, str]] = []

# Initialize MCP client
mcp_client = Client("http://localhost:8001/mcp")

# Use a ThreadPoolExecutor to run async code from a sync context
executor = ThreadPoolExecutor(max_workers=1)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def run_async_in_thread(coro):
    return loop.run_until_complete(coro)

# --- Standard HTTP Routes (for the web interface) ---
@app.route("/")
def serve_webpage_http():
    """Serves the main chat webpage."""
    return render_template("index.html")

@app.route("/static/css/<path:filename>")
def serve_static_css(filename):
    """Serves static CSS files from the static/css directory."""
    return send_from_directory("static/css", filename)

@app.route("/api/generate", methods=["POST"])
def generate_response_http():
    """Receives a request from the web page and forwards it to the MCP server."""
    try:
        request_data = request.get_json()
        prompt_request = PromptRequest(**request_data)
        
        # --- Run the async MCP call in the executor thread ---
        payload = {
            "prompt": prompt_request.prompt,
            "chat_history": prompt_request.chat_history
        }
        
        async def call_mcp():
            async with mcp_client:
                result = await mcp_client.call_tool("generate_response", payload)
                return result.text

        generated_response = executor.submit(run_async_in_thread, call_mcp()).result()

        if generated_response:
            return jsonify({"response": generated_response})
        else:
            return jsonify({"error": "Unexpected format in MCP server response"}), 500

    except Exception as e:
        app.logger.error("Error in generate_response_http: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
