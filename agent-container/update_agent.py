"""
Update AgentCore agent with new ECR image.

Usage:
    python update_agent.py
"""

import boto3
import time

# ============================================================================
# CONFIGURATION - Updated to match deploy_agent-prod.py
# ============================================================================

# TODO: Replace with your actual agent ARN after initial deployment
AGENT_RUNTIME_ARN = 'arn:aws:bedrock-agentcore:eu-central-1:081302066317:runtime/demo_streaming_rag_agent_prod-6wk6lmHdeK'
CONTAINER_URI = '081302066317.dkr.ecr.eu-central-1.amazonaws.com/streaming-agent:latest'
ROLE_ARN = 'arn:aws:iam::081302066317:role/strands-test-agent'
REGION = 'eu-central-1'

# Cognito Authentication (from deploy_agent-prod.py)
COGNITO_USER_POOL_ID = 'eu-central-1_2ZryMv0qs'
COGNITO_CLIENT_ID = '5aqpc5nkm2cinrfcdiiiks6kgc'
COGNITO_REGION = 'eu-central-1'

# ============================================================================
# Update
# ============================================================================

def update_agent():
    """Update the agent with new container image."""
    
    print("="*70)
    print("Updating AgentCore Agent")
    print("="*70)
    print(f"\nAgent ARN: {AGENT_RUNTIME_ARN}")
    print(f"New Container URI: {CONTAINER_URI}")
    print(f"Region: {REGION}")
    
    # Extract agent runtime ID from ARN
    agent_runtime_id = AGENT_RUNTIME_ARN.split('/')[-1]
    
    client = boto3.client('bedrock-agentcore-control', region_name=REGION)
    
    # Construct Cognito discovery URL
    discovery_url = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}/.well-known/openid-configuration"
    
    print("\n" + "-"*70)
    print("Updating agent runtime (preserving all configs)...")
    print("-"*70)
    print(f"JWT Auth: {discovery_url}")
    print()
    
    try:
        response = client.update_agent_runtime(
            agentRuntimeId=agent_runtime_id,
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
            roleArn=ROLE_ARN,
            networkConfiguration={'networkMode': 'PUBLIC'}
        )
        
        status = response['status']
        
        print("✓ Agent update initiated!")
        print(f"Status: {status}")
        
        # Wait for update to complete
        if status != 'READY':
            print(f"\n⏳ Waiting for agent to be ready...")
            wait_for_ready(client, agent_runtime_id)
        
        print("\n" + "="*70)
        print("✓ Agent updated successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error updating agent: {e}")
        raise


def wait_for_ready(client, agent_runtime_id, max_wait=300):
    """Wait for agent to reach READY status."""
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = client.get_agent_runtime(
                agentRuntimeId=agent_runtime_id
            )
            status = response['status']
            
            if status == 'READY':
                print("✓ Agent is READY")
                return
            elif status in ['FAILED', 'STOPPED']:
                raise Exception(f"Agent update failed with status: {status}")
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
        update_agent()
    except Exception as e:
        print(f"\n✗ Update failed: {e}")
        exit(1)
