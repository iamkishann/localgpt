# webserver_client.py
import requests
from flask import Flask, render_template, jsonify, request, send_from_directory
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any

# Define the client service
app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

# --- Pydantic model for request validation ---
class PromptRequest(BaseModel):
    prompt: str
    chat_history: List[Dict[str, str]] = []

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
        
        # --- Make the JSON-RPC call to the MCP server ---
        mcp_server_url = "http://localhost:8001/mcp/messages"
        payload = {
            "jsonrpc": "2.0",
            "method": "generate_response",
            "params": {
                "request_data": {
                    "prompt": prompt_request.prompt,
                    "chat_history": prompt_request.chat_history
                }
            }
        }
        
        response = requests.post(mcp_server_url, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes
        
        mcp_response = response.json()
        result = mcp_response.get('result')

        if result:
            generated_response = result.get('response')
            if generated_response:
                return jsonify({"response": generated_response})
            else:
                return jsonify({"error": "Unexpected format in MCP server response"}), 500
        else:
            # If a JSON-RPC error occurred on the server
            error_data = mcp_response.get('error', {})
            return jsonify({"error": f"MCP server error: {error_data.get('message', 'Unknown error')}"}), 500

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to connect to MCP server: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask web server on port 8000
    app.run(host="0.0.0.0", port=8000, debug=True)
