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
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize workflow on startup"""
    try:
        # Initialize the LangGraph workflow with memory
        global workflow
        memory = MemorySaver()
        workflow = create_workflow().compile(checkpointer=memory)
        print("LangGraph workflow initialized successfully with memory")
        yield
    except Exception as e:
        print(f"Startup error: {str(e)}")
        # Don't raise the exception here, just log it

app = FastAPI(
    title="Mitrat LangGraph API",
    description="API for processing provider queries using LangGraph",
    version="1.5",
    lifespan=lifespan
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

@app.get("/")
async def root():
    try:
        # Test database connection
        connection = get_connection()
        if connection:
            connection.close()
        return {"message": "Mitrat LangGraph API v1.5 is running"}
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
        start_time = time.time()
        perf_metrics = {}
        
        # Workflow creation time
        workflow_start = time.time()
        global workflow
        if workflow is None:
            memory = MemorySaver()
            workflow = create_workflow().compile(checkpointer=memory)
        perf_metrics["workflow_creation_time"] = f"{time.time() - workflow_start:.2f}s"

        # Get current state with all fields
        state_start = time.time()
        state_result = workflow.get_state(
            config={"configurable": {"thread_id": chat_request.thread_id}}
        )
        current_state = state_result[0] if state_result and state_result[0] else {}
        
        # Preserve existing performance metrics between requests
        initial_state = {
            "messages": current_state.get("messages", []) + [HumanMessage(content=chat_request.query)],
            "performance_metrics": current_state.get("performance_metrics", {})
        }

        # Workflow execution
        workflow_start = time.time()
        result = workflow.invoke(
            initial_state,  # Now includes performance_metrics
            config={"configurable": {"thread_id": chat_request.thread_id}}
        )
        perf_metrics["workflow_execution"] = f"{time.time() - workflow_start:.2f}s"

        # Collect metrics directly from the final state
        final_metrics = result.get("performance_metrics", {})
        perf_metrics.update(final_metrics)
        
        # Ensure we include all timing metrics even if some are missing
        expected_metrics = [
            "intent_detection_time", "get_schema_time", "refine_query_time",
            "sql_generation_time", "execute_query_time", "explain_results_time",
            "conversational_response_time"
        ]
        for metric in expected_metrics:
            if metric not in perf_metrics:
                perf_metrics[metric] = "N/A"

        # Aggregate all metrics
        total_time = time.time() - start_time
        perf_metrics["total_time"] = f"{total_time:.2f}s"

        return {
            "response": result["messages"][-1].content,
            "sql_query": result.get("sql_query"),
            "performance_metrics": perf_metrics
        }
        
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
