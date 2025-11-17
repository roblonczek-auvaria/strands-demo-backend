"""
FastAPI server for the RAG Agent compatible with Bedrock AgentCore.

This server provides the required endpoints for Bedrock AgentCore:
- POST /invocations: Main agent interaction endpoint
- GET /ping: Health check endpoint

The server integrates with the RAG agent that has access to a documentation Knowledgebase.

Key Features:
- Multi-model support with configurable foundation models
- Session-based agent caching for conversation continuity
- Streaming and non-streaming response modes
- Structured JSON response parsing with fallback to raw text
- Knowledge base document integration
- CORS-enabled for cross-origin requests
- Health monitoring and reset capabilities

Architecture:
- Agent instances are cached per model and session to maintain context
- Supports both AgentCore format and simple prompt format
- Knowledge base documents are retrieved and included in responses
- Comprehensive error handling and logging throughout
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Union, Optional, AsyncIterator
from datetime import datetime, timezone
import json
import re
import logging
import os
from rag_agent import create_rag_agent, AVAILABLE_MODELS, DEFAULT_MODEL_ID
from ask_knowledgebase import get_last_kb_documents_and_clear

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG Agent",
    description="RAG Agent with knowledge base access",
    version="1.0.0"
)

# Allow all origins (adjust if you need to restrict)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single agent instance for this microVM
# AgentCore provides session isolation via dedicated microVMs,
# so we only need one agent instance per container/microVM
_current_agent: Optional[Any] = None
_current_model_id: Optional[str] = None

def _get_or_create_agent(model_id: Optional[str], session_id: Optional[str]) -> Any:
    """
    Get or create a single RAG agent instance for this microVM.
    
    AgentCore Runtime provides session isolation via dedicated microVMs,
    so we maintain only one agent instance per container. Model switching
    will recreate the agent as needed.
    
    Args:
        model_id: The foundation model identifier (e.g., 'nova-pro')
                 Falls back to DEFAULT_MODEL_ID if None or invalid
        session_id: Session identifier (logged but not used for caching
                   since AgentCore handles session isolation)
    
    Returns:
        RAG agent instance configured for the specified model
        
    Raises:
        Exception: If agent creation fails for the requested model
    """
    global _current_agent, _current_model_id
    
    chosen = model_id or DEFAULT_MODEL_ID
    if chosen not in AVAILABLE_MODELS:
        logger.warning("Unknown model_id '%s' requested; falling back to default '%s'", chosen, DEFAULT_MODEL_ID)
        chosen = DEFAULT_MODEL_ID
    
    # Log session_id for debugging but don't use for caching (AgentCore handles isolation)
    if session_id:
        logger.debug("Request for session_id=%s", session_id)
    
    # Recreate agent only if model changed or not yet created
    if _current_agent is None or _current_model_id != chosen:
        try:
            logger.info("Creating agent for model_id=%s (previous=%s)", chosen, _current_model_id)
            _current_agent = create_rag_agent(chosen)
            _current_model_id = chosen
            logger.info("Agent created successfully for model_id=%s", chosen)
        except Exception as e:
            logger.error("Failed to create agent for model_id=%s: %s", chosen, e)
            _current_agent = None
            _current_model_id = None
            raise
    
    return _current_agent

try:
    # Initialize default agent
    _current_agent = create_rag_agent(DEFAULT_MODEL_ID)
    _current_model_id = DEFAULT_MODEL_ID
    logger.info("Default RAG agent initialized successfully (model=%s)", DEFAULT_MODEL_ID)
except Exception as e:
    logger.error(f"Failed to initialize default RAG agent: {e}")
    _current_agent = None
    _current_model_id = None

# Reset bookkeeping for monitoring and debugging
# Tracks when the agent was last reset and how many times
last_reset_iso: Optional[str] = None  # ISO timestamp of last reset operation
reset_count: int = 0  # Total number of resets performed since server start


def _normalize_bool(value: Any) -> Optional[bool]:
    """
    Convert various input types to boolean values with flexible parsing.
    
    Supports common boolean representations:
    - Python booleans: True/False
    - Strings: "true"/"false", "yes"/"no", "on"/"off", "1"/"0" (case-insensitive)
    - Numbers: truthy/falsy values
    
    Args:
        value: The value to convert to boolean
        
    Returns:
        bool: Converted boolean value
        None: If value cannot be interpreted as boolean
        
    Examples:
        _normalize_bool("true") -> True
        _normalize_bool("FALSE") -> False
        _normalize_bool(1) -> True
        _normalize_bool("maybe") -> None
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _extract_response_text(result: Any) -> str:
    """
    Extract text content from various agent response formats.
    
    Handles multiple response structures commonly returned by different agent types:
    1. Objects with 'message' attribute containing structured content
    2. Objects with 'content' attribute (direct or nested)
    3. Direct string conversion as fallback
    
    The function navigates nested structures to find the actual text content,
    particularly looking for the pattern: content -> list -> dict -> "text" key
    
    Args:
        result: Agent response object of any type
        
    Returns:
        str: Extracted text content, empty string if no content found
        
    Response Structure Examples:
        - result.message.content[0].text
        - result.content.content[0].text
        - str(result) as fallback
    """
    if result is None:
        return ""

    message_obj = getattr(result, "message", None)
    if message_obj is not None:
        if isinstance(message_obj, dict):
            content = message_obj.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    return first.get("text", "")
            return str(message_obj)
        return str(message_obj)

    content_attr = getattr(result, "content", None)
    if content_attr is not None:
        if isinstance(content_attr, dict):
            nested = content_attr.get("content")
            if isinstance(nested, list) and nested:
                first = nested[0]
                if isinstance(first, dict) and "text" in first:
                    return first.get("text", "")
            return str(content_attr)
        return str(content_attr)

    return str(result)


def _json_candidates(text: str) -> list[str]:
    """
    Extract potential JSON objects from text using multiple extraction strategies.
    
    Searches for JSON content in common formats:
    1. XML-style tags: <json>...</json>
    2. Markdown code fences: ```json...```
    3. Raw JSON objects containing required fields ("answer" and "sources")
    
    Args:
        text: Input text that may contain JSON objects
        
    Returns:
        list[str]: Unique JSON candidate strings in order of discovery
        
    Extraction Patterns:
        - <json>content</json> (case-insensitive, multiline)
        - ```json content ``` (case-insensitive, multiline)
        - {..."answer"..."sources"...} (any JSON object with required fields)
    """
    candidates: list[str] = []
    json_tag_match = re.findall(r'<json>\s*(.*?)\s*</json>', text, re.DOTALL | re.IGNORECASE)
    candidates.extend(m.strip() for m in json_tag_match if m.strip())
    code_fence = re.findall(r'```json\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
    candidates.extend(m.strip() for m in code_fence if m.strip())
    object_matches = re.findall(r'\{[\s\S]*?\}', text)
    filtered = [o for o in object_matches if '"answer"' in o and '"sources"' in o]
    candidates.extend(filtered)
    unique: list[str] = []
    seen = set()
    for c in candidates:
        if c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


def _parse_structured_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse and validate structured JSON responses from agent text.
    
    Attempts to parse JSON candidates in reverse order (most recent first)
    and validates that they contain the required RAG response structure
    with 'answer' and 'sources' fields.
    
    Args:
        text: Raw text response from the agent
        
    Returns:
        dict: Parsed JSON object if valid structure found
        None: If no valid structured JSON response found
        
    Required JSON Structure:
        {
            "answer": "response text",
            "sources": [...],  # source references
            ...  # additional fields allowed
        }
    """
    for candidate in reversed(_json_candidates(text)):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and 'answer' in parsed and 'sources' in parsed:
                return parsed
        except Exception:
            continue
    return None


def _build_response_payload(response_str: str) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Build comprehensive response payloads for different client formats.
    
    Creates multiple response formats from agent output:
    1. Unified format: Complete response with all metadata
    2. AgentCore format: Bedrock AgentCore-compatible structure
    3. Simple format: Minimal client-friendly response
    4. Parsed JSON: Extracted structured data (if available)
    
    Integrates knowledge base documents and handles both structured
    and unstructured agent responses gracefully.
    
    Args:
        response_str: Raw text response from the agent
        
    Returns:
        tuple containing:
        - unified (dict): Complete response with metadata
        - agentcore_response (dict): Bedrock AgentCore format
        - simple_response (dict): Client-friendly format
        - parsed_json (dict|None): Structured data if available
        
    Response Integration:
        - Merges knowledge base documents from last query
        - Preserves structured JSON if available
        - Falls back to raw text for unstructured responses
        - Adds timestamps and agent identification
    """
    parsed_json = _parse_structured_json(response_str)
    
    # Capture knowledge base documents from the last query
    kb_documents = get_last_kb_documents_and_clear()
    
    unified = {
        # Fallback: if the model did not return structured JSON with an 'answer', use the raw text as the answer
        "answer": parsed_json.get("answer") if parsed_json else response_str,
        "sources": parsed_json.get("sources") if parsed_json else None,
        "documents": kb_documents,  # Add the knowledge base documents
        "raw": response_str,
        "has_structured": parsed_json is not None,
    }
    agentcore_response = {
        "output": {
            "message": unified,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "rag-agent",
            "agent_type": "strands_rag_agent",
        }
    }
    if parsed_json:
        # Include documents in the simple response as well
        enhanced_json = {**parsed_json, "documents": kb_documents}
        simple_response = {"response": json.dumps(enhanced_json, ensure_ascii=False)}
    else:
        simple_response = {"response": response_str, "documents": kb_documents}
    return unified, agentcore_response, simple_response, parsed_json

# Pydantic models for request/response validation and documentation

class InvocationRequest(BaseModel):
    """
    Request model for agent invocation endpoint.
    
    Supports multiple input formats for flexibility:
    - Bedrock AgentCore format: {"input": {"prompt": "question"}}
    - Simple format: {"prompt": "question"}
    - Model selection: {"model_id": "amazon-nova-pro", ...}
    
    Attributes:
        input: Nested input object (AgentCore format)
        prompt: Direct prompt string (simple format)
        model_id: Foundation model identifier for this request
    """
    input: Optional[Dict[str, Any]] = None
    prompt: Optional[str] = None  # Support both formats
    model_id: Optional[str] = None  # Allow top-level model_id as convenience

class InvocationResponse(BaseModel):
    """
    Response model for agent invocation endpoint.
    
    Provides flexible response format depending on client needs:
    - AgentCore format: {"output": {...}}
    - Simple format: {"response": "..."}
    
    Attributes:
        output: Structured output object (AgentCore format)
        response: Direct response string (simple format)
    """
    output: Optional[Dict[str, Any]] = None
    response: Optional[str] = None  # Support both formats

@app.post("/invocations", response_model=Union[InvocationResponse, Dict])
async def invoke_agent(request: Union[InvocationRequest, Dict[str, Any]], raw_request: Request):
    """
    Main agent invocation endpoint required by Bedrock AgentCore.
    
    This endpoint handles RAG agent interactions with comprehensive support for:
    - Multiple input formats (AgentCore and simple)
    - Model selection and session management
    - Streaming and non-streaming responses
    - Knowledge base integration
    - Structured JSON response parsing
    
    Input Formats Supported:
        AgentCore: {"input": {"prompt": "question", "model_id": "amazon-nova-pro"}}
        Simple: {"prompt": "question", "model_id": "amazon-nova-pro"}
        Session: Headers or body can include session_id for continuity
    
    Model Selection Priority:
        1. request.input.model_id (nested)
        2. request.model_id (top-level)
        3. DEFAULT_MODEL_ID (fallback)
    
    Session Management:
        - Header: x-amzn-bedrock-agentcore-runtime-session-id
        - Body: session_id field (useful for testing)
        - Agents cached per model+session combination
    
    Streaming Support:
        - Query param: ?stream=true
        - Header: Accept: text/event-stream
        - Body: {"stream": true} or {"input": {"stream": true}}
        - Returns Server-Sent Events (SSE) format
    
    Response Formats:
        Simple requests return: {"response": "answer", "documents": [...]}
        AgentCore requests return: {"output": {...}, "response": "...", "model_id": "..."}
        
    Error Handling:
        - 400: Missing or invalid prompt
        - 500: Agent creation or processing failure
        
    Knowledge Base Integration:
        - Automatically retrieves relevant documents
        - Includes document metadata in response
        - Clears document cache after each request
    """
    try:
        # STEP 1: Extract model selection from multiple possible locations
        # Priority: input.model_id > top-level model_id > DEFAULT_MODEL_ID
        requested_model: Optional[str] = None
        if isinstance(request, dict):
            requested_model = request.get("model_id")
            if "input" in request and isinstance(request["input"], dict):
                requested_model = request["input"].get("model_id") or requested_model
        else:
            requested_model = getattr(request, "model_id", None)
            if request.input and isinstance(request.input, dict):
                requested_model = request.input.get("model_id") or requested_model

        # STEP 2: Extract session identifier for conversation continuity
        # Priority: body session_id > header session_id > no session (global)
        session_identifier: Optional[str] = raw_request.headers.get("x-amzn-bedrock-agentcore-runtime-session-id")

        # Allow session_id to be specified in body as well (useful for local testing tools)
        body_session_id: Optional[str] = None
        if isinstance(request, dict):
            body_session_id = request.get("session_id")
            body_input = request.get("input")
            if not body_session_id and isinstance(body_input, dict):
                candidate = body_input.get("session_id")
                if isinstance(candidate, str):
                    body_session_id = candidate
        else:
            body_session_id = getattr(request, "session_id", None)
            body_input = getattr(request, "input", None)
            if not body_session_id and isinstance(body_input, dict):
                candidate = body_input.get("session_id")  # type: ignore[call-arg]
                if isinstance(candidate, str):
                    body_session_id = candidate

        session_identifier = body_session_id or session_identifier
        if session_identifier:
            logger.info("Using session_id=%s", session_identifier)

        # STEP 3: Get or create agent instance with caching
        # Agents are cached by (model_id, session_id) to maintain context
        try:
            agent = _get_or_create_agent(requested_model, session_identifier)
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to create agent for requested model")

        # STEP 4: Extract user prompt from flexible input formats
        # Supports both nested (AgentCore) and flat (simple) formats
        user_message = None
        body_stream_preference: Optional[bool] = None

        if isinstance(request, dict):
            inner = request.get("input")
            if isinstance(inner, dict):
                # Prefer nested prompt if provided
                nested_prompt = inner.get("prompt")
                if isinstance(nested_prompt, str) and nested_prompt.strip():
                    user_message = nested_prompt
                candidate_stream = _normalize_bool(inner.get("stream"))
                if candidate_stream is not None:
                    body_stream_preference = candidate_stream
            # Fallback to top-level prompt if none extracted yet
            if user_message is None and isinstance(request.get("prompt"), str):
                user_message = request.get("prompt")
            if body_stream_preference is None:
                top_stream = _normalize_bool(request.get("stream"))
                if top_stream is not None:
                    body_stream_preference = top_stream
        else:
            inner = getattr(request, "input", None)
            if isinstance(inner, dict):
                nested_prompt = inner.get("prompt")
                if isinstance(nested_prompt, str) and nested_prompt.strip():
                    user_message = nested_prompt
                candidate_stream = _normalize_bool(inner.get("stream"))
                if candidate_stream is not None:
                    body_stream_preference = candidate_stream
            if user_message is None and isinstance(getattr(request, "prompt", None), str):
                user_message = request.prompt  # type: ignore[attr-defined]
            if body_stream_preference is None:
                top_stream = _normalize_bool(getattr(request, "stream", None))
                if top_stream is not None:
                    body_stream_preference = top_stream

        if not user_message:
            raise HTTPException(
                status_code=400,
                detail="No prompt found. Please provide either {'prompt': 'your question'} or {'input': {'prompt': 'your question'}}"
            )

        logger.info(f"Processing query: {user_message[:100]}...")

        # STEP 5: Determine streaming preference from multiple sources
        # Priority: body stream param > query param > Accept header
        query_stream_preference = _normalize_bool(raw_request.query_params.get("stream"))
        accept_header = raw_request.headers.get("accept", "")
        header_stream_preference = "text/event-stream" in accept_header.lower()
        stream_requested = (
            body_stream_preference
            if body_stream_preference is not None
            else query_stream_preference
            if query_stream_preference is not None
            else header_stream_preference
        )

        if stream_requested and not hasattr(agent, "stream_async"):
            logger.warning("Streaming requested but rag_agent has no stream_async method; falling back to non-streaming response")
            stream_requested = False

        # STEP 6A: Handle streaming response path
        if stream_requested:
            logger.info("Streaming response requested (model_id=%s)", getattr(agent.model, 'model_id', 'unknown'))

            async def event_stream() -> AsyncIterator[str]:
                """
                Generate Server-Sent Events (SSE) stream for real-time response.
                
                Event Types:
                - :stream-start: Initial connection marker
                - chunk: Incremental response content
                - final: Complete response with metadata
                - error: Error information
                - [DONE]: Stream termination marker
                """
                full_chunks: list[str] = []
                final_event_result: Any = None

                def format_sse(payload: Dict[str, Any]) -> str:
                    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

                try:
                    yield ":stream-start\n\n"
                    async for event in agent.stream_async(user_message):
                        if not isinstance(event, dict):
                            continue
                        chunk = event.get("data")
                        if chunk:
                            chunk_text = str(chunk)
                            full_chunks.append(chunk_text)
                            yield format_sse({"type": "chunk", "content": chunk_text})
                        if event.get("result") is not None:
                            final_event_result = event["result"]

                    response_text = "".join(full_chunks)
                    if final_event_result is not None:
                        final_text = _extract_response_text(final_event_result)
                        if final_text:
                            response_text = final_text

                    unified, agentcore_response, simple_response, parsed_json = _build_response_payload(response_text)
                    agentcore_response["output"]["model_id"] = getattr(agent.model, 'model_id', None)
                    final_payload = {
                        "type": "final",
                        "response": simple_response["response"],
                        "unified": unified,
                        "agentcore": agentcore_response["output"],
                        "structured": parsed_json is not None,
                    }
                    yield format_sse(final_payload)
                    yield "data: [DONE]\n\n"
                except Exception as stream_err:
                    logger.error("Agent streaming failed: %s", stream_err)
                    error_payload = {"type": "error", "message": str(stream_err)}
                    yield format_sse(error_payload)

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # STEP 6B: Handle non-streaming response path
        logger.info("Using model_id=%s", getattr(agent.model, 'model_id', 'unknown'))
        result = agent(user_message)
        response_str = _extract_response_text(result)
        unified, agentcore_response, simple_response, parsed_json = _build_response_payload(response_str)
        agentcore_response["output"]["model_id"] = getattr(agent.model, 'model_id', None)

        logger.info(f"Final raw content (first 200 chars): {response_str[:200]}")
        logger.info(f"Structured parsed: {parsed_json is not None}")
        logger.info("Query processed successfully")

        # STEP 7: Determine response format based on request type
        # Simple format: {"prompt": "..."} -> returns simple response
        # AgentCore format: {"input": {...}} -> returns AgentCore response
        is_simple = False
        if isinstance(request, dict):
            is_simple = "input" not in request and "prompt" in request
        else:
            try:
                has_prompt = bool(getattr(request, 'prompt', None))
                has_input = bool(getattr(request, 'input', None))
                is_simple = has_prompt and not has_input
            except Exception:
                pass

        logger.info(f"Request classified as simple={is_simple} (type={type(request)})")

        if is_simple:
            return simple_response
        agentcore_response_with_alias = {**agentcore_response, "response": simple_response["response"], "model_id": agentcore_response["output"].get("model_id")}
        return agentcore_response_with_alias
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")

@app.get("/ping")
async def ping():
    """
    Health check endpoint required by Bedrock AgentCore.
    
    Provides basic health status and agent initialization state.
    Used by load balancers and monitoring systems to verify service health.
    
    Returns:
        dict: Health status information
            - status: "healthy" (always)
            - agent_initialized: bool indicating if agent is ready
            - timestamp: ISO timestamp of the health check
    """
    return {
        "status": "healthy",
        "agent_initialized": _current_agent is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/")
async def root():
    """
    Root endpoint providing service information and API documentation.
    
    Serves as a discovery endpoint for clients to understand:
    - Available endpoints and their purposes
    - Current service status and configuration
    - Available models and reset statistics
    - API capabilities and formats supported
    
    Returns:
        dict: Service information including:
            - service: Service name and description
            - endpoints: Available API endpoints with descriptions
            - agent_status: Current agent initialization state
            - reset metrics: Last reset time and count
            - model info: Available models and default selection
    """
    return {
        "service": "RAG Agent",
        "description": "RAG Agent with knowledge base access",
        "endpoints": {
            "POST /invocations": "Main agent interaction endpoint",
            "POST /reset": "Reset and recreate in-memory agent (clears conversational context)",
            "POST /resetStatus": "Return current reset bookkeeping info (alias GET /resetStatus)",
            "GET /resetStatus": "Return current reset bookkeeping info",
            "GET /ping": "Health check endpoint"
        },
        "agent_status": "initialized" if _current_agent is not None else "not_initialized",
        "last_reset": last_reset_iso,
        "reset_count": reset_count,
        "available_models": AVAILABLE_MODELS,
        "default_model": DEFAULT_MODEL_ID
    }

@app.post("/reset")
async def reset_agent():
    """
    Reset the agent instance and clear state.
    
    This endpoint provides a way to:
    - Clear the current agent instance
    - Reset any internal state
    - Reinitialize with the default model
    - Update reset tracking metrics
    
    Note: In AgentCore Runtime, sessions are isolated via microVMs,
    so this primarily handles model switching and development/testing scenarios.
    
    Returns:
        dict: Reset operation results including:
            - status: "reset" confirmation
            - agent_status: Post-reset initialization state
            - reset metrics: Updated timestamp and count
            - model info: Available models and default
            
    Raises:
        HTTPException(500): If agent reset or recreation fails
    """
    global _current_agent, _current_model_id, last_reset_iso, reset_count
    try:
        # Clear current agent and recreate with default model
        _current_agent = None
        _current_model_id = None
        _current_agent = create_rag_agent(DEFAULT_MODEL_ID)
        _current_model_id = DEFAULT_MODEL_ID
        reset_count += 1
        last_reset_iso = datetime.now(timezone.utc).isoformat()
        logger.info("RAG agent reset successfully (count=%s)", reset_count)
        return {
            "status": "reset",
            "agent_status": "initialized" if _current_agent else "not_initialized",
            "last_reset": last_reset_iso,
            "reset_count": reset_count,
            "available_models": AVAILABLE_MODELS,
            "default_model": DEFAULT_MODEL_ID
        }
    except Exception as e:
        logger.error(f"Failed to reset agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset agent: {str(e)}")

@app.get("/resetStatus")
@app.post("/resetStatus")
async def reset_status():
    """
    Get current reset status and metrics without performing a reset.
    
    Accepts both GET and POST methods for client convenience.
    Provides monitoring and debugging information about agent resets.
    
    Returns:
        dict: Current reset status including:
            - agent_status: Whether default agent is initialized
            - last_reset: ISO timestamp of most recent reset (null if never reset)
            - reset_count: Total number of resets since server startup
            - timestamp: Current timestamp
            - model info: Available models and default selection
    """
    return {
        "agent_status": "initialized" if _current_agent is not None else "not_initialized",
        "last_reset": last_reset_iso,
        "reset_count": reset_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "available_models": AVAILABLE_MODELS,
        "default_model": DEFAULT_MODEL_ID
    }

@app.get("/models")
async def list_models():
    """
    Get available foundation models for agent configuration.
    
    Provides clients with information about supported models
    for dynamic model selection in requests.
    
    Returns:
        dict: Model information including:
            - available_models: List of supported model identifiers
            - default_model: Default model used when none specified
            - count: Total number of available models
            
    Example Response:
        {
            "available_models": ["amazon-nova-pro", "amazon-nova-lite", ...],
            "default_model": "amazon-nova-pro",
            "count": 5
        }
    """
    return {
        "available_models": AVAILABLE_MODELS,
        "default_model": DEFAULT_MODEL_ID,
        "count": len(AVAILABLE_MODELS)
    }

if __name__ == "__main__":
    """
    Direct execution entry point for development and testing.
    
    Starts the FastAPI server using uvicorn with:
    - Host: 0.0.0.0 (accepts connections from any IP)
    - Port: 8080 (standard port for this service)
    - Auto-reload disabled (production-like behavior)
    
    For production deployment, use a proper ASGI server configuration
    with appropriate workers, logging, and monitoring setup.
    """
    import uvicorn
    logger.info("Starting RAG Agent server...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
