"""
Enhanced base agent with RAG knowledge base capabilities.
This demonstrates how to integrate the ask_knowledgebase tool with a Strands agent.
"""
from strands import Agent
from strands.models import BedrockModel
from strands_tools import current_time
from ask_knowledgebase import ask_knowledgebase


def create_rag_agent():
    """Create a Strands agent with RAG capabilities."""
    
    # Create a Bedrock model instance
    bedrock_model = BedrockModel(
        region_name="eu-central-1",
        model_id="eu.amazon.nova-pro-v1:0",
        temperature=0.1,
        top_p=0.8,
    )
    
    # Create an agent with the ask_knowledgebase tool
    agent = Agent(
        model=bedrock_model,
        tools=[ask_knowledgebase, current_time],
        name="RAG Knowledge Assistant",
        system_prompt="""
       you have access to a knowledge base of atp documentation. Use the tool ask_knowledgebase to answer questions about atp.
"""
    )
#You can also suggest topic filters like "ACORDURI, TRATATE, CONVENTII INTERNAIONALE" to narrow searches when appropriate.    
    return agent