"""RAG agent factory with dynamic Bedrock foundation model selection.

Enhanced base agent with RAG knowledge base capabilities. Demonstrates how to integrate
the ask_knowledgebase tool with a Strands agent while allowing runtime LLM choice.
"""
from strands import Agent
from strands.models import BedrockModel
from strands_tools import current_time
from ask_knowledgebase import ask_knowledgebase
import os

# List of allowed foundation models for dynamic selection.
AVAILABLE_MODELS = [
    "eu.amazon.nova-lite-v1:0",
    "eu.amazon.nova-pro-v1:0",
    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "eu.anthropic.claude-sonnet-4-20250514-v1:0",
]

DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", AVAILABLE_MODELS[1])  # Default to nova-pro unless overridden.


def create_rag_agent(model_id: str | None = None):
    """Create a Strands agent with RAG capabilities.

    Args:
        model_id: Optional Bedrock model identifier chosen by the client. If not supplied or invalid,
                  the DEFAULT_MODEL_ID is used.
    """

    chosen = model_id or DEFAULT_MODEL_ID
    if chosen not in AVAILABLE_MODELS:
        # Fallback silently to default if an unknown model is requested.
        chosen = DEFAULT_MODEL_ID

    bedrock_model = BedrockModel(
        region_name="eu-central-1",
        model_id=chosen,
    )

    system_prompt = (
        """You have access to a knowledge base of atp documentation. Use the tool ask_knowledgebase to answer questions about atp.
        whenever you are unsure how to answer a question reliably, use the knowledgebase."""
    )

    agent = Agent(
        model=bedrock_model,
        tools=[ask_knowledgebase, current_time],
        name="RAG Knowledge Assistant",
        system_prompt=system_prompt,
    )

    return agent