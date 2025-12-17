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
    "eu.amazon.nova-2-lite-v1:0",
    "eu.amazon.nova-pro-v1:0",
    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
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
        """You are a specialized assistant with access to the Auvaria Webpage content knowledge base. Your primary purpose is to provide accurate, well-cited information about Auvaria.

## Tools Available
- **ask_knowledgebase**: Use this tool to query Auvaria Webpage content when answering user questions.

## Instructions
1. When a user asks a question, use the ask_knowledgebase tool to retrieve relevant information.
2. Always use the knowledge base when you are unsure about any technical details or specifications.
3. Provide clear, concise answers based on the retrieved documents.
4. CRUCIAL: Use citations sparingly and strategically - only cite when introducing new concepts, key facts, or technical specifications.
5. Avoid redundant citations for the same information repeated multiple times.
6. Group related information together and cite once per concept rather than per sentence.
7. Focus on citing the most authoritative or comprehensive source when multiple documents contain similar information.
8. Always render your responses in proper markdown format with appropriate headers, lists, code blocks, and formatting.

## Citation Guidelines
- Use markers like %[1]%, %[2]%, %[3]% only for significant claims or technical details
- Place citations at the end of paragraphs or sections rather than after every sentence
- Example: "Auvaria is a cloud consulting company %[1]%."
- Prioritize readability over comprehensive citation coverage
        """
    )

    agent = Agent(
        model=bedrock_model,
        tools=[ask_knowledgebase, current_time],
        name="RAG Knowledge Assistant",
        system_prompt=system_prompt,
    )

    return agent