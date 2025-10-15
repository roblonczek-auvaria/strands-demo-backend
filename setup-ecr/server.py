"""
FastAPI server for the RAG Agent compatible with Bedrock AgentCore.

This server provides the required endpoints f        # Debug: Log the raw response content
        logger.info(f"Raw response content (first 200 chars): {response_str[:200]}")
        logger.info(f"Response content type: {type(result)}") Bedrock AgentCore:
- POST /invocations: Main agent interaction endpoint
- GET /ping: Health check endpoint

The server integrates with the RAG agent that has access to Romanian legal documents.
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
from rag_agent import create_rag_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Romanian Legal RAG Agent",
    description="RAG Agent with access to Romanian legal documents, treaties, and international agreements",
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

# Initialize the RAG agent
try:
    rag_agent = create_rag_agent()
    logger.info("RAG agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG agent: {e}")
    rag_agent = None

# Reset bookkeeping
last_reset_iso: Optional[str] = None
reset_count: int = 0


def _normalize_bool(value: Any) -> Optional[bool]:
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
    for candidate in reversed(_json_candidates(text)):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and 'answer' in parsed and 'sources' in parsed:
                return parsed
        except Exception:
            continue
    return None


def _build_response_payload(response_str: str) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    parsed_json = _parse_structured_json(response_str)
    unified = {
        "answer": parsed_json.get("answer") if parsed_json else None,
        "sources": parsed_json.get("sources") if parsed_json else None,
        "raw": response_str,
        "has_structured": parsed_json is not None,
    }
    agentcore_response = {
        "output": {
            "message": unified,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "romanian-legal-rag-agent",
            "agent_type": "strands_rag_agent",
        }
    }
    if parsed_json:
        simple_response = {"response": json.dumps(parsed_json, ensure_ascii=False)}
    else:
        simple_response = {"response": response_str}
    return unified, agentcore_response, simple_response, parsed_json

# Pydantic models for request/response
class InvocationRequest(BaseModel):
    input: Optional[Dict[str, Any]] = None
    prompt: Optional[str] = None  # Support both formats
    topic: Optional[str] = None   # Allow passing topic at top-level
    active_only: Optional[Any] = None  # bool or string; we'll normalize later

class InvocationResponse(BaseModel):
    output: Optional[Dict[str, Any]] = None
    response: Optional[str] = None  # Support both formats

@app.post("/invocations", response_model=Union[InvocationResponse, Dict])
async def invoke_agent(request: Union[InvocationRequest, Dict[str, Any]], raw_request: Request):
    """
    Main agent invocation endpoint required by Bedrock AgentCore.
    
    Supports multiple input formats:
    - Bedrock AgentCore format: {"input": {"prompt": "question"}}
    - Simple format: {"prompt": "question"}
    """
    try:
        if rag_agent is None:
            raise HTTPException(
                status_code=500, 
                detail="RAG agent not initialized. Check server logs for details."
            )
        agent = rag_agent
        # Capture raw body for diagnostics (helps when fields missing like topic)
        raw_body_bytes = b""
        try:
            raw_body_bytes = await raw_request.body()
            trimmed = raw_body_bytes[:1000]
            logger.info("Raw request body (truncated 1000 chars): %r", trimmed)
        except Exception as rb_err:
            logger.warning("Could not read raw request body: %s", rb_err)

        # Extract user message and optional filter controls from different possible formats
        user_message = None
        topic = None
        active_only = None
        body_stream_preference: Optional[bool] = None

        if isinstance(request, dict):
            # Handle direct dict input
            if "input" in request and isinstance(request["input"], dict):
                inner = request["input"]
                user_message = inner.get("prompt")
                topic = inner.get("topic") if isinstance(inner, dict) else None
                active_only = inner.get("active_only") if isinstance(inner, dict) else None
                candidate = _normalize_bool(inner.get("stream"))
                if candidate is not None:
                    body_stream_preference = candidate
            elif "prompt" in request:
                user_message = request.get("prompt")
                topic = request.get("topic")
                active_only = request.get("active_only")
                candidate = _normalize_bool(request.get("stream"))
                if candidate is not None:
                    body_stream_preference = candidate
        else:
            # Handle Pydantic model input
            if request.input and isinstance(request.input, dict):
                user_message = request.input.get("prompt")
                topic = request.input.get("topic")
                active_only = request.input.get("active_only")
                candidate = _normalize_bool(request.input.get("stream"))
                if candidate is not None:
                    body_stream_preference = candidate
            elif request.prompt:
                user_message = request.prompt
                # Pull direct fields if present on model
                if getattr(request, 'topic', None) and not topic:
                    topic = request.topic
                if getattr(request, 'active_only', None) is not None and active_only is None:
                    active_only = request.active_only
                candidate = _normalize_bool(getattr(request, "stream", None))
                if candidate is not None:
                    body_stream_preference = candidate
        
        # Fallback: if topic still None attempt manual JSON parse of raw body (in case FastAPI model coercion dropped field)
        if topic is None or active_only is None:
            try:
                import json as _json
                rb_json = _json.loads(raw_body_bytes.decode('utf-8')) if raw_body_bytes else {}
                if isinstance(rb_json, dict):
                    # Mirror earlier extraction logic
                    if topic is None and 'topic' in rb_json and rb_json.get('topic') not in (None, ""):
                        topic = rb_json.get('topic')
                    if active_only is None and 'active_only' in rb_json:
                        active_only = rb_json.get('active_only')
                    elif 'input' in rb_json and isinstance(rb_json['input'], dict) and rb_json['input'].get('topic'):
                        topic = rb_json['input'].get('topic')
                        if active_only is None and 'active_only' in rb_json['input']:
                            active_only = rb_json['input'].get('active_only')
                    if body_stream_preference is None:
                        if 'stream' in rb_json:
                            body_stream_preference = _normalize_bool(rb_json.get('stream'))
                        elif 'input' in rb_json and isinstance(rb_json['input'], dict):
                            body_stream_preference = _normalize_bool(rb_json['input'].get('stream'))
            except Exception as _fallback_err:
                logger.debug("Fallback topic parse failed: %s", _fallback_err)

        if not user_message:
            raise HTTPException(
                status_code=400,
                detail="No prompt found. Please provide either {'prompt': 'your question'} or {'input': {'prompt': 'your question'}}"
            )

        logger.info(f"Processing query: {user_message[:100]}...")

        # Propagate topic / active_only into environment for dynamic filter (ask_knowledgebase uses env vars)
        if topic:
            os.environ["KB_TOPIC"] = str(topic)
        else:
            os.environ.pop("KB_TOPIC", None)
        if isinstance(active_only, bool):
            os.environ["KB_ACTIVE_ONLY"] = "1" if active_only else "0"
        elif isinstance(active_only, str):
            os.environ["KB_ACTIVE_ONLY"] = "1" if active_only.lower() in {"1", "true", "yes"} else "0"
        elif active_only is None:
            os.environ.setdefault("KB_ACTIVE_ONLY", "0")

        logger.info(
            "Applied retrieval filter context: topic=%s active_only=%s (KB_TOPIC=%s, KB_ACTIVE_ONLY=%s)",
            topic, active_only, os.getenv("KB_TOPIC"), os.getenv("KB_ACTIVE_ONLY")
        )

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

        if stream_requested:
            logger.info("Streaming response requested")

            async def event_stream() -> AsyncIterator[str]:
                full_chunks: list[str] = []
                final_event_result: Any = None

                def format_sse(payload: Dict[str, Any]) -> str:
                    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

                try:
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

        result = agent(user_message)
        response_str = _extract_response_text(result)
        unified, agentcore_response, simple_response, parsed_json = _build_response_payload(response_str)

        logger.info(f"Final raw content (first 200 chars): {response_str[:200]}")
        logger.info(f"Structured parsed: {parsed_json is not None}")

        logger.info("Query processed successfully")

        # Determine if original request was the simple format (just {"prompt": ...})
        is_simple = False
        if isinstance(request, dict):
            is_simple = "input" not in request and "prompt" in request
        else:
            # Pydantic model path
            try:
                has_prompt = bool(getattr(request, 'prompt', None))
                has_input = bool(getattr(request, 'input', None))
                is_simple = has_prompt and not has_input
            except Exception:
                pass

        logger.info(f"Request classified as simple={is_simple} (type={type(request)})")

        if is_simple:
            return simple_response
        # Ensure AgentCore response ALSO includes a top-level 'response' for frontend resiliency
        agentcore_response_with_alias = {**agentcore_response, "response": simple_response["response"]}
        return agentcore_response_with_alias
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing failed: {str(e)}"
        )

@app.get("/ping")
async def ping():
    """
    Health check endpoint required by Bedrock AgentCore.
    """
    return {
        "status": "healthy",
        "agent_initialized": rag_agent is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/")
async def root():
    """
    Root endpoint with basic information.
    """
    return {
        "service": "Romanian Legal RAG Agent",
        "description": "RAG Agent with access to Romanian legal documents, treaties, and international agreements",
        "endpoints": {
            "POST /invocations": "Main agent interaction endpoint",
            "POST /reset": "Reset and recreate in-memory agent (clears conversational context)",
            "POST /resetStatus": "Return current reset bookkeeping info (alias GET /resetStatus)",
            "GET /resetStatus": "Return current reset bookkeeping info",
            "GET /ping": "Health check endpoint"
        },
        "agent_status": "initialized" if rag_agent is not None else "not_initialized",
        "last_reset": last_reset_iso,
        "reset_count": reset_count
    }

@app.post("/reset")
async def reset_agent():
    """Reset the in-memory agent object (drops prior conversation state)."""
    global rag_agent, last_reset_iso, reset_count
    try:
        rag_agent = create_rag_agent()
        reset_count += 1
        last_reset_iso = datetime.now(timezone.utc).isoformat()
        logger.info("RAG agent reset successfully (count=%s)", reset_count)
        return {
            "status": "reset",
            "agent_status": "initialized" if rag_agent else "not_initialized",
            "last_reset": last_reset_iso,
            "reset_count": reset_count
        }
    except Exception as e:
        logger.error(f"Failed to reset agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset agent: {str(e)}")

@app.get("/resetStatus")
@app.post("/resetStatus")
async def reset_status():
    """Return reset bookkeeping without performing a reset (GET or POST)."""
    return {
        "agent_status": "initialized" if rag_agent is not None else "not_initialized",
        "last_reset": last_reset_iso,
        "reset_count": reset_count,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Romanian Legal RAG Agent server...")
    uvicorn.run(app, host="0.0.0.0", port=8080)