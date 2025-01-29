from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
from MitratLangV1 import create_workflow, State, get_connection
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from fastapi.responses import StreamingResponse
import mysql.connector
from mysql.connector import errorcode

app = FastAPI(
    title="Mitrat LangGraph API",
    description="API for processing provider queries using LangGraph",
    version="1.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    thread_id: str

# Initialize workflow globally
workflow = None

@app.on_event("startup")
async def startup_event():
    """Initialize workflow on startup"""
    try:
        # Initialize the LangGraph workflow with memory
        global workflow
        memory = MemorySaver()
        workflow = create_workflow().compile(checkpointer=memory)
        print("LangGraph workflow initialized successfully with memory")
    except Exception as e:
        print(f"Startup error: {str(e)}")
        # Don't raise the exception here, just log it

@app.get("/")
async def root():
    try:
        # Test database connection
        connection = get_connection()
        if connection:
            connection.close()
        return {"message": "Mitrat LangGraph API v1.2 is running"}
    except Exception as e:
        if "Server IP not whitelisted" in str(e):
            raise HTTPException(
                status_code=403,
                detail="Database connection failed: Server IP not whitelisted. Please add this server's IP to the database whitelist."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Database connection failed: {str(e)}"
            )

@app.post("/chat")
async def chat_endpoint_post(chat_request: ChatRequest):
    """POST endpoint for production use with thread_id"""
    try:
        # Start timing for the entire request
        start_time = time.time()
        
        # Get the current state from memory (if exists)
        state_result = workflow.get_state(
            config={"configurable": {"thread_id": chat_request.thread_id}}
        )
        
        # Extract messages from state if it exists
        messages = state_result[0]["messages"] if state_result and state_result[0] else []
        messages.append(HumanMessage(content=chat_request.query))
        
        initial_state = State(
            messages=messages
        )
        
        # Process query through LangGraph workflow with thread_id
        result = workflow.invoke(
            initial_state,
            config={"configurable": {"thread_id": chat_request.thread_id}}
        )
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Prepare response
        response = {
            "response": result["messages"][-1].content,
            "performance_metrics": {
                "total_time": f"{total_time:.2f}s",
                **result.get("performance_metrics", {})
            }
        }
        
        return response
        
    except Exception as e:
        return {
            "error": "Query processing failed",
            "details": str(e)
        }

@app.post("/chat/stream")
async def chat_endpoint_stream(chat_request: ChatRequest):
    """Streaming endpoint for chat responses"""
    try:
        # Prepare initial state
        messages = []
        if chat_request.thread_id:
            state_result = workflow.get_state(
                config={"configurable": {"thread_id": chat_request.thread_id}}
            )
            messages = state_result[0]["messages"] if state_result and state_result[0] else []
        
        messages.append(HumanMessage(content=chat_request.query))
        initial_state = State(messages=messages)

        # Create async generator for streaming
        async def generate():
            async for event in workflow.astream_events(
                initial_state,
                config={"configurable": {"thread_id": chat_request.thread_id}} if chat_request.thread_id else None,
                version="v2",
                stream_mode="messages-tuple"
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        yield f"data: {chunk.content}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return {
            "error": "Streaming failed",
            "details": str(e)
        }

def start_server():
    """Function to start the server"""
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    start_server() 
