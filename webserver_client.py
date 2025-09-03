# webserver_client.py
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
# This client will connect to the MCP server you ran on port 8001.
# The URL must point to the correct MCP endpoint, which is /mcp
mcp_client = Client("http://localhost:8001/mcp")

# --- Standard HTTP Routes (for the web interface) ---
@app.route("/")
def serve_webpage_http():
    """Serves the main chat webpage."""
    return render_template("index.html")

@app.route("/static/css/<path:filename>")
def serve_static_css(filename):
    """Serves static CSS files from the static/css directory."""
    return send_from_directory("static/css", filename)

# A standard HTTP POST endpoint for the web page
@app.route("/api/generate", methods=["POST"])
def generate_response_http():
    """Receives a request from the web page and forwards it to the MCP server."""
    try:
        request_data = request.get_json()
        prompt_request = PromptRequest(**request_data)
        
        # --- Make the MCP tool call using the client ---
        # The client handles the JSON-RPC details, including the /mcp/messages path
        payload = {
            "request_data": {
                "prompt": prompt_request.prompt,
                "chat_history": prompt_request.chat_history
            }
        }
        
        result = mcp_client.call_tool("generate_response", payload)

        if result:
            return jsonify({"response": result.text})
        else:
            return jsonify({"error": "Failed to get a valid response from the MCP server"}), 500

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to connect to MCP server: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask web server on port 8000
    app.run(host="0.0.0.0", port=8000, debug=True)
