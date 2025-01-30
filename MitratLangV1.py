print("Starting script...")

import os
import io
from PIL import Image as PILImage
from typing import Annotated, TypedDict, Any, List, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain.tools import tool
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error, errorcode
import sys
import time
from openai import OpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv(override=True)
print("Environment variables loaded")

# Initialize the client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Remove the connection_pool initialization from the global scope
connection_pool = None

def get_connection():
    """Get a connection from the pool (lazy initialization)"""
    global connection_pool
    try:
        if connection_pool is None:
            connection_pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="mitrat_pool",
                pool_size=5,
                host=os.getenv('DB_HOST'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                database=os.getenv('DB_NAME'),
                port=3306,
                connect_timeout=30,
                raise_on_warnings=True,
                use_pure=True,
                auth_plugin='mysql_native_password',
                ssl_disabled=True
            )
        return connection_pool.get_connection()
    except mysql.connector.Error as e:
        if e.errno == errorcode.CR_CONN_HOST_ERROR:
            raise Exception("Server IP not whitelisted. Please add this server's IP to the database whitelist.")
        else:
            raise Exception(f"Database connection failed: {str(e)}")

# Cache schemas after the first retrieval
schema_cache = {}

@tool
def get_schema_tool(table_name: str) -> str:
    """Get the schema and sample data for a specific table"""
    if table_name in schema_cache:
        return schema_cache[table_name]
    
    print(f"[Tool Log] get_schema_tool called with table_name={table_name}")
    
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Get schema
        cursor.execute(f"SHOW CREATE TABLE {table_name}")
        schema_result = cursor.fetchone()
        schema = schema_result['Create Table']
        
        # Get sample data (first 5 rows)
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        sample_data = cursor.fetchall()
        
        # Format the response
        response = f"Table: {table_name}\n\nSCHEMA:\n{schema}\n\nSAMPLE DATA:"
        for row in sample_data:
            response += f"\n{row}"
            
        cursor.close()
        connection.close()
        
        # Cache the schema
        schema_cache[table_name] = response
        return response
        
    except Exception as e:
        return f"Error retrieving schema and data: {str(e)}"

@tool
def execute_query_tool(query: str) -> List[Dict[str, Any]]:
    """Execute a SQL query using a connection from the pool"""
    connection = None
    try:
        print(f"[Tool Log] Attempting to get a connection from the pool...")
        connection = get_connection()
        if not connection:
            print("[Tool Log] Failed to get a connection from the pool")
            return []
        
        print(f"[Tool Log] Executing query: {query}")
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"[Tool Log] Query executed successfully. Results: {results}")
        
        return results
    except Exception as e:
        print(f"[Tool Log] Error executing query: {str(e)}")
        return {
            "error": str(e)
        }
    finally:
        if connection:
            print("[Tool Log] Closing connection...")
            connection.close()

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    intent: str | None
    schema: str | None  
    chosen_table: str | None
    sql_query: str | None
    query_results: List[Dict] | None
    performance_metrics: Dict[str, str]

def print_performance_metrics(state: Dict) -> None:
    """Helper function to print performance metrics"""
    if "performance_metrics" in state:
        print("\nPerformance Metrics:")
        for metric, value in state["performance_metrics"].items():
            print(f"{metric.replace('_', ' ').title()}: {value}")
        print()

def detect_intent_node(state: State) -> Dict[str, Any]:
    """Detect user intent from the last message using LLM"""
    start_time = time.time()
    print("NODE: detect_intent_node - RUNNING")
    
    try:
        # Get the last HUMAN message (not AI or Tool messages)
        user_msg = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), "No user message")
        
        # Optimized LLM prompt for intent classification
        prompt = f"""CLASSIFY THIS QUERY IN 1 WORD: "{user_msg}"
        
        [RULES]
        Reply "query" ONLY if it contains:
        - Provider/service search terms: find, list, show, search, providers, services, "near me", location names
        - Specific service types: plan management, support coordination, therapy
        - Field-specific terms: phone, email, location, profile
        
        Reply "conversational" for:
        - Greetings (hi, hello, bye)
        - NDIS/process questions
        - General help/FAQ
        - Bot identity questions
        
        [EXAMPLES]
        "NDIS providers in Sydney" → query
        "How do I register?" → conversational
        "Find assistance animals" → query
        "What's your name?" → conversational
        
        ANSWER:"""
        
        # Call the LLM for intent classification
        intent_response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "system", "content": prompt}]
        )
        intent = intent_response.choices[0].message.content.strip().lower()
        print(f"Intent detected: {intent}")
        
        elapsed = time.time() - start_time
        result = {
            "messages": [AIMessage(content=f"(Debug) Intent detected: {intent}")],
            "intent": intent,
            "performance_metrics": {
                **state.get("performance_metrics", {}),
                "intent_detection_time": f"{elapsed:.2f}s"
            }
        }
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "messages": [AIMessage(content="Sorry, I had trouble understanding your intent.")],
            "intent": None,
            "performance_metrics": {
                **state.get("performance_metrics", {}),
                "intent_detection_time": f"{elapsed:.2f}s"
            }
        }

def get_schema_node(state: State) -> Dict[str, List[ToolMessage]]:
    """Get schemas for relevant tables based on user query"""
    start_time = time.time()
    print("\nNODE: get_schema_node - RUNNING")
    
    try:
        # Get schema for the tables we need
        relevant_tables = ["users_data", "list_professions", "users_reviews"]
        all_schemas = []
        
        for table in relevant_tables:
            print(f"[Tool Log] Getting schema for {table}")
            schema = get_schema_tool.invoke(table)
            all_schemas.append(schema)
        
        # Combine all schemas with clear separation
        combined_schemas = "\n\n=== TABLE SCHEMAS ===\n\n".join(all_schemas)
        
        # Store in state - This is the key fix!
        state["schema"] = combined_schemas
        #print(f"Schema stored in state: {combined_schemas[:200]}...")  
        
        elapsed = time.time() - start_time
        return {
            "messages": [
                ToolMessage(
                    content=f"Retrieved schemas for tables: {', '.join(relevant_tables)}",
                    tool_name="get_schema_tool",
                    tool_call_id="schema_lookup",
                    additional_kwargs={"schemas": combined_schemas}
                )
            ],
            "schema": combined_schemas,
            "performance_metrics": {
                **state.get("performance_metrics", {}),
                "get_schema_time": f"{elapsed:.2f}s"
            }
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "messages": [
                ToolMessage(
                    content="Failed to retrieve schemas",
                    tool_name="get_schema_tool",
                    tool_call_id="schema_lookup"
                )
            ],
            "schema": None,
            "performance_metrics": {
                **state.get("performance_metrics", {}),
                "get_schema_time": f"{elapsed:.2f}s"
            }
        }

def should_continue_after_intent(state: State) -> str:
    """
    Decide next node based on recognized intent.
    """
    print("NODE: should_continue_after_intent - RUNNING")
    intent = state.get("intent")
    print(f"Current intent: {intent}")
    
    if intent == "conversational":
        print("Routing to conversational_agent_node")
        return "conversational_agent_node"
    elif intent == "query":
        print("Routing to get_schema_node")
        return "get_schema_node"
    else:
        print("No recognized intent, defaulting to conversational_agent_node")
        return "conversational_agent_node"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_model_with_retry(messages):
    print("Retrying to get the response")
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=500,
    )

def conversational_agent_node(state: State) -> Dict[str, List[AIMessage]]:
    start_time = time.time()
    print("NODE: conversational_agent_node - RUNNING")
    
    try:
        # Get only the last HUMAN message
        user_msg = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), "")
        
        # Simplified message structure
        messages = [
            {
                "role": "system",
                "content": """You are Mitrat Chatbot specializing in NDIS services. Respond concisely:
1. Keep responses under 100 words
2. Use simple, direct language
3. Provide specific, actionable information
4. Include links to Mitrat website when relevant

Common Response Templates:
- For general NDIS questions: Brief explanation + suggestion to visit Mitrat website
- For service inquiries: List 2-3 relevant services + how to search
- For provider questions: Direct to search functionality
- For process questions: Step-by-step brief points

Format in HTML with these tags only: <p>, <br>, <strong>, <a>, <ul>, <li>

Example responses:
Q: "What is NDIS?"
A: "<p>NDIS (National Disability Insurance Scheme) provides funding for disability supports. Visit <a href='https://mitrat.com.au/about-ndis'>Mitrat's NDIS Guide</a> for details.</p>"

Q: "How do I find providers?"
A: "<p>You can find providers by:<br>1. Using our search bar above<br>2. Filtering by location<br>3. Choosing specific services</p>"
"""},  
        ]

        # Add retry with specific error handling
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
        def safe_chat_completion():
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
                max_tokens=300
            )
            
        response = safe_chat_completion()
        content = response.choices[0].message.content.strip()
        
        # Print performance metrics
        elapsed = time.time() - start_time
        print(f"Conversational agent response time: {elapsed:.2f}s")
        
        return {
            "messages": [AIMessage(content=content)],
            "performance_metrics": {
                **state.get("performance_metrics", {}),
                "conversational_response_time": f"{elapsed:.2f}s"
            }
        }
        
    except Exception as e:
        print(f"Error in conversational_agent_node: {str(e)}")
        print(f"Messages: {messages}")  # Log the messages being sent
        elapsed = time.time() - start_time
        return {
            "messages": [AIMessage(content="I apologize, but I encountered an error. Please try again.")],
            "performance_metrics": {
                **state.get("performance_metrics", {}),
                "response_time": f"{elapsed:.2f}s",
                "error": str(e)
            }
        }

def refine_query_node(state: State) -> Dict[str, Any]:
    """Refine the user query into a SQL query"""
    start_time = time.time()
    print("NODE: refine_query_node - RUNNING")
    try:
        # Get the last HUMAN message
        user_msg = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), "")
        
        print("[Refine Query Node] Refining query based on schemas...")
        schemas = state.get("schema", "")
        #print(schemas)
        
        # Combined prompt for both refinement and SQL generation
        combined_prompt = f"""As an SQL expert, directly convert this user request into a valid SQL query using EXACT field names from these schemas:

USER REQUEST: "{user_msg}"

SCHEMAS:
{schemas}

RULES:
1. Use proper SQL syntax with exact field names, the following words are not keywords (NDIS,provider,assistance)
2. WHERE clauses MUST include:
   - active = 2
   - (address1 LIKE '%<location>%' OR city LIKE '%<location>%') for locations
   - (first_name LIKE '%<keyword>%' OR last_name LIKE '%<keyword>%' OR company LIKE '%<keyword>%') for names
3. Always SELECT: company, phone_number, email, address1, filename
5. LIMIT 5
6. Use concise SQL without comments

Always considers EXAMPLES:
User: "NDIS providers in Sydney"
SQL: SELECT company, phone_number, email, address1, filename FROM users_data WHERE (address1 LIKE '%Sydney%' OR city LIKE '%Sydney%') AND active = 2 LIMIT 5;

User: "Where can I find assistance animals providers?"
SQL: "SELECT company, phone_number, email, address1, filename FROM users_data WHERE (first_name LIKE '%animal%' OR last_name LIKE '%animal%' OR company LIKE '%animal%') AND active = 2 LIMIT 5;

User:What are the newest NDIS providers listed? OR Can you show me the latest NDIS providers?
SQL: SELECT company, phone_number, email, address1, filename FROM users_data WHERE active = 2 ORDER BY modtime DESC LIMIT 5;

DIRECTLY OUTPUT THE SQL QUERY:"""
        
        # Single LLM call with temperature=0
        start = time.time()
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Consider gpt-3.5-turbo for faster response
            temperature=0,
            messages=[{"role": "system", "content": combined_prompt}],
            max_tokens=200  # Limit response length
        )
        sql_query = response.choices[0].message.content.strip()
        
        # Simple cleanup instead of separate parsing
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        print(f"[Refine Query Node] Generated SQL: {sql_query}")
        elapsed = time.time() - start_time
        
        return {
            "messages": [AIMessage(content=f"Generated SQL: {sql_query}")],
            "sql_query": sql_query,
            "performance_metrics": {
                **state.get("performance_metrics", {}),
                "refine_query_time": f"{(time.time()-start):.2f}s",
                "sql_generation_time": "0.00s"  # Combined metric
            }
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[Refine Query Node] Error: {str(e)}")
        return {
            "messages": [AIMessage(content="Sorry, I had trouble processing your query.")],
            "performance_metrics": {
                **state.get("performance_metrics", {}),
                "refine_query_time": "N/A",
                "sql_generation_time": "N/A"
            }
        }

def execute_query_node(state: State) -> Dict[str, Any]:
    """Execute the SQL query using the tool and return results"""
    start_time = time.time()
    print("NODE: execute_query_node - RUNNING")
    
    sql_query = state.get("sql_query")
    if not sql_query:
        elapsed = time.time() - start_time
        return {
            "messages": [ToolMessage(
                content="No SQL query found to execute.",
                tool_name="execute_query_tool",
                tool_call_id="query_execution"
            )],
            "performance_metrics": {
                **state.get("performance_metrics", {}),
                "execute_query_time": f"{elapsed:.2f}s"
            }
        }
    
    try:
        # Use the tool to execute the query
        results = execute_query_tool.invoke(sql_query)
        
        elapsed = time.time() - start_time
        result = {
            "messages": [ToolMessage(
                content=results,
                tool_name="execute_query_tool",
                tool_call_id="query_execution"
            )],
            "query_results": results,
            "performance_metrics": {
                **state.get("performance_metrics", {}),
                "execute_query_time": f"{elapsed:.2f}s"
            }
        }
        print_performance_metrics(result)
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "messages": [ToolMessage(
                content=f"Error executing query: {str(e)}",
                tool_name="execute_query_tool",
                tool_call_id="query_execution"
            )],
            "performance_metrics": {
                **state.get("performance_metrics", {}),
                "execute_query_time": f"{elapsed:.2f}s"
            }
        }

def explain_results_node(state: State) -> Dict[str, Any]:
    """Generate a natural language explanation of the SQL results"""
    start_time = time.time()
    print("NODE: explain_results_node - RUNNING")
    
    query_results = state.get("query_results", [])
    if not query_results:
        return {
            "messages": [AIMessage(content="No results found!!!.")]
        }
    
    # Generate the explanation with all required fields
    explanation = "Here are the providers I found:\n\n"
    for result in query_results:
        explanation += f"<h2>{result.get('company', 'Unknown')}</h2>\n"
        explanation += f"<p><strong>Phone:</strong> {result.get('phone_number', 'N/A')}<br>\n"
        explanation += f"<strong>Email:</strong> {result.get('email', 'N/A')}<br>\n"
        explanation += f"<strong>Location:</strong> {result.get('address1', 'N/A')}, {result.get('city', '')}, {result.get('state_code', '')}<br>\n"
        explanation += f"<strong>Profile:</strong> <a href=\"https://mitrat.com.au/{result['filename']}\" target=\"_blank\">View Profile</a></p><br>\n"
    
    elapsed = time.time() - start_time
    result = {
        "messages": [AIMessage(content=explanation)],
        "performance_metrics": {
            **state.get("performance_metrics", {}),
            "explain_results_time": f"{elapsed:.2f}s"
        }
    }
    print_performance_metrics(result)
    return result

workflow = None

def create_workflow():
    """Creates and returns the workflow with proper visualization"""
    global workflow
    if workflow is not None:
        return workflow

    workflow_start = time.time()
    
    # Create the workflow with MessagesState for memory
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("detect_intent_node", detect_intent_node)
    workflow.add_node("get_schema_node", get_schema_node)
    workflow.add_node("conversational_agent_node", conversational_agent_node)
    workflow.add_node("refine_query_node", refine_query_node)
    workflow.add_node("execute_query_node", execute_query_node)
    workflow.add_node("explain_results_node", explain_results_node)

    # Add edges
    workflow.add_edge(START, "detect_intent_node")
    
    workflow.add_conditional_edges(
        "detect_intent_node",
        should_continue_after_intent,
        {
            "conversational_agent_node": "conversational_agent_node",
            "get_schema_node": "get_schema_node",
        }
    )
    
    workflow.add_edge("conversational_agent_node", END)
    
    workflow.add_edge("get_schema_node", "refine_query_node")
    workflow.add_edge("refine_query_node", "execute_query_node")
    workflow.add_edge("execute_query_node", "explain_results_node")
    workflow.add_edge("explain_results_node", END)

    elapsed = time.time() - workflow_start
    print(f"\nPerformance Metrics:")
    print(f"Total workflow creation time: {elapsed:.2f}s")
    
    return workflow

def main():
    print("Starting script...")
    print("Environment variables loaded")
    
    # Initialize the workflow
    workflow = create_workflow()
    
    while True:
        # Get user input
        user_query = input("\nEnter your query (or type 'exit' to quit): ")
        
        if user_query.lower() == 'exit':
            print("Exiting...")
            break
            
        if not user_query.strip():
            print("Please enter a valid query.")
            continue
            
        print(f"\nProcessing query: {user_query}")
        
        # Start timing for the entire query
        start_time = time.time()
        
        # Run the workflow with the user's query
        result = workflow.invoke({"messages": [HumanMessage(content=user_query)]})
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Print the final messages
        print("\nFinal messages:")
        for message in result["messages"]:
            if isinstance(message, AIMessage):
                print(f"ai: {message.content}")
            elif isinstance(message, ToolMessage):
                print(f"tool: {message.content}")
            elif isinstance(message, HumanMessage):
                print(f"human: {message.content}")
        
        # Print performance metrics
        print(f"\nTotal time for query: {total_time:.2f}s")
        
        # Print individual performance metrics if available
        if "performance_metrics" in result:
            print("Performance Metrics:")
            for metric, value in result["performance_metrics"].items():
                print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
