"""
Simple Bedrock AgentCore Deployment Script.

This script deploys the RAG agent to Amazon Bedrock AgentCore with Cognito JWT authentication.

Usage:
    python deploy_agent.py
"""

import boto3
import json
import time

# ============================================================================
# CONFIGURATION - Update these values
# ============================================================================

AGENT_NAME = 'demo_rag_streaming_agent'  # Use underscores, not hyphens
CONTAINER_URI = '081302066317.dkr.ecr.eu-central-1.amazonaws.com/streaming-agent:latest'
ROLE_ARN = 'arn:aws:iam::081302066317:role/strands-test-agent'
REGION = 'eu-central-1'

# Cognito Authentication (from Amplify)
COGNITO_USER_POOL_ID = 'eu-central-1_OGkb7HbRv'
COGNITO_CLIENT_ID = '2b10v9vo7lu63usc59g53ruaol'
COGNITO_REGION = 'eu-central-1'

# Network Configuration
NETWORK_MODE = 'PUBLIC'

# ============================================================================
# Deployment
# ============================================================================

def deploy_agent():
    """Deploy the agent to Bedrock AgentCore."""
    
    print("="*70)
    print("Deploying Streaming RAG Agent to Bedrock AgentCore")
    print("="*70)
    print(f"\nAgent Name: {AGENT_NAME}")
    print(f"Container URI: {CONTAINER_URI}")
    print(f"Role ARN: {ROLE_ARN}")
    print(f"Region: {REGION}")
    print(f"Network Mode: {NETWORK_MODE}")
    print(f"\nCognito User Pool ID: {COGNITO_USER_POOL_ID}")
    print(f"Cognito Client ID: {COGNITO_CLIENT_ID}")
    
    # Create the client
    client = boto3.client('bedrock-agentcore-control', region_name=REGION)
    
    # Construct Cognito discovery URL (must end with /.well-known/openid-configuration)
    discovery_url = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}/.well-known/openid-configuration"
    print(f"Discovery URL: {discovery_url}")
    
    print("\n" + "-"*70)
    print("Creating agent runtime...")
    print("-"*70 + "\n")
    
    try:
        # Call the CreateAgentRuntime operation
        response = client.create_agent_runtime(
            agentRuntimeName=AGENT_NAME,
            agentRuntimeArtifact={
                'containerConfiguration': {
                    'containerUri': CONTAINER_URI
                }
            },
            authorizerConfiguration={
                'customJWTAuthorizer': {
                    'discoveryUrl': discovery_url,
                    'allowedClients': [COGNITO_CLIENT_ID]
                }
            },
            networkConfiguration={'networkMode': NETWORK_MODE},
            roleArn=ROLE_ARN
        )
        
        agent_runtime_arn = response['agentRuntimeArn']
        status = response['status']
        
        print("✓ Agent Runtime created successfully!")
        print(f"\n{'='*70}")
        print("DEPLOYMENT SUCCESSFUL")
        print("="*70)
        print(f"\nAgent Runtime ARN:")
        print(f"  {agent_runtime_arn}")
        print(f"\nStatus: {status}")
        print(f"Created: {response.get('createdAt', 'N/A')}")
        
        # Wait for agent to be ready
        if status != 'READY':
            print(f"\n⏳ Waiting for agent to be ready...")
            wait_for_ready(client, agent_runtime_arn)
        
        print("\n" + "="*70)
        print("Agent is ready for use!")
        print("="*70)
        
        return agent_runtime_arn
        
    except Exception as e:
        print(f"\n✗ Error creating agent runtime: {e}")
        raise


def wait_for_ready(client, agent_runtime_arn, max_wait=300):
    """Wait for agent to reach READY status."""
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = client.get_agent_runtime(
                agentRuntimeArn=agent_runtime_arn
            )
            status = response['status']
            
            if status == 'READY':
                print("✓ Agent is READY")
                return
            elif status in ['FAILED', 'STOPPED']:
                raise Exception(f"Agent deployment failed with status: {status}")
            else:
                print(f"  Status: {status} (waiting...)")
                time.sleep(10)
        except Exception as e:
            if 'READY' not in str(e):
                print(f"  Checking status... ({int(time.time() - start_time)}s elapsed)")
            time.sleep(10)
    
    raise TimeoutError(f"Agent did not become ready within {max_wait} seconds")


if __name__ == "__main__":
    try:
        agent_arn = deploy_agent()
        print(f"\n✓ Deployment complete!")
        print(f"\nTo invoke the agent, use this ARN:")
        print(f"  {agent_arn}")
    except Exception as e:
        print(f"\n✗ Deployment failed: {e}")
        exit(1)
