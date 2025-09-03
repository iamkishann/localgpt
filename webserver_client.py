# webserver_client.py (FastAPI version)
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastmcp import Client
from pydantic import BaseModel
from typing import List, Dict, Any

# Define the client service
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic model for the request body
class PromptRequest(BaseModel):
    prompt: str
    chat_history: List[Dict[str, str]] = []

# Initialize MCP client
mcp_client = Client("http://localhost:8001/mcp")

@app.get("/", response_class=HTMLResponse)
async def serve_webpage_http(request: Request):
    return Path("templates/index.html").read_text()

@app.get("/static/css/{filename:str}")
async def serve_static_css(filename: str):
    return FileResponse(Path("static/css") / filename)

@app.post("/api/generate")
async def generate_response_http(prompt_request: PromptRequest) -> JSONResponse:
    async with mcp_client:
        try:
            result = await mcp_client.call_tool("generate_response", {"request_data": prompt_request.dict()})
            return JSONResponse({"response": result.text})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
