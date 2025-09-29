import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
from flask import Flask, render_template, jsonify, request, send_from_directory
from pydantic import BaseModel
from typing import List, Dict, Any
from fastmcp import Client # Correct import for the client class
from threading import Thread

class PromptRequest(BaseModel):
    prompt: str
    chat_history: List[Dict[str, str]] = []

# --- Standard HTTP Routes (for the web interface) ---
app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

asyncio_loop = asyncio.new_event_loop()
asyncio_thread = Thread(target=asyncio_loop.run_forever, daemon=True)
asyncio_thread.start()

# Initialize MCP client once
mcp_client = Client("http://localhost:8001/mcp")

# Asynchronous function to call the MCP tool
async def call_mcp_tool(prompt: str, chat_history: List[Dict[str, str]]):
    async with mcp_client:
        payload = {"prompt": prompt, "chat_history": chat_history}
        result = await mcp_client.call_tool("generate_response", payload)
        response_json = result.json()
        gen_text = response_json.get("generated_test")
        if not gen_test:
            return "Model server returned no output"
        return gen_text
        #return result.text

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

        # Submit the async function to our shared asyncio loop
        future = asyncio.run_coroutine_threadsafe(
            call_mcp_tool(prompt_request.prompt, prompt_request.chat_history),
            asyncio_loop
        )
        
        generated_response = future.result(timeout=60) # Add a timeout

        if generated_response:
            return jsonify({"response": generated_response})
        else:
            return jsonify({"error": "Failed to get a valid response from the MCP server"}), 500

    except Exception as e:
        app.logger.error("Error in generate_response_http: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
