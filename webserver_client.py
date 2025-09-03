# client_service.py
import requests
from flask import Flask, render_template, jsonify, request, send_from_directory
from pathlib import Path
from typing import List, Dict, Any

# Define the client service
app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

# Pydantic model for the request body (used for validation, not in Flask directly)
class PromptRequest:
    def __init__(self, prompt: str, chat_history: List[Dict[str, str]] = []):
        self.prompt = prompt
        self.chat_history = chat_history

# --- Standard HTTP Routes (for the web interface) ---
@app.route("/")
def serve_webpage_http():
    return render_template("index.html")

@app.route("/static/css/<path:filename>")
def serve_static_css(filename):
    return send_from_directory("static/css", filename)

# A standard HTTP POST endpoint for the web page
@app.route("/api/generate", methods=["POST"])
def generate_response_http():
    try:
        request_data = request.json
        prompt_request = PromptRequest(**request_data)
        
        # Call the generate_response tool on the MCP server using requests
        # Note: This is a synchronous call.
        mcp_server_url = "http://localhost:8000/mcp/messages"
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
        result = mcp_response['result']
        
        return jsonify({"response": result['response']})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to connect to MCP server: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
