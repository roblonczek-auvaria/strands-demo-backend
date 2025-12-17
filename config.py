"""
Deployment configuration for Bedrock AgentCore.

This module provides environment-specific configuration for deploying
RAG agents to staging and production environments.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class DeploymentConfig:
    """Configuration for deploying an agent to Bedrock AgentCore."""
    
    env: Literal['staging', 'prod']
    agent_name: str
    region: str
    account_id: str
    cognito_user_pool_id: str
    cognito_client_id: str
    cognito_region: str
    role_arn: str
    runtime_version: str = 'PYTHON_3_11'  # Default for this project, matching Dockerfile
    network_mode: str = 'PUBLIC'
    
    @property
    def s3_bucket(self) -> str:
        """S3 bucket for deployment packages."""
        return f"bedrock-agentcore-code-{self.account_id}-{self.region}"
    
    @property
    def s3_prefix(self) -> str:
        """S3 prefix for this agent's deployment package."""
        return f"{self.agent_name}/deployment_package.zip"
    
    @property
    def discovery_url(self) -> str:
        """Cognito OpenID discovery URL for JWT authentication."""
        return f"https://cognito-idp.{self.cognito_region}.amazonaws.com/{self.cognito_user_pool_id}/.well-known/openid-configuration"


# Staging Configuration
STAGING = DeploymentConfig(
    env='staging',
    agent_name='demo_rag_streaming_agent_staging',
    region='eu-central-1',
    account_id='081302066317',
    cognito_user_pool_id='eu-central-1_OGkb7HbRv',
    cognito_client_id='2b10v9vo7lu63usc59g53ruaol',
    cognito_region='eu-central-1',
    role_arn='arn:aws:iam::081302066317:role/strands-test-agent',
)

# Production Configuration
PROD = DeploymentConfig(
    env='prod',
    agent_name='demo_streaming_rag_agent_prod',
    region='eu-central-1',
    account_id='081302066317',
    cognito_user_pool_id='eu-central-1_2ZryMv0qs',
    cognito_client_id='5aqpc5nkm2cinrfcdiiiks6kgc',
    cognito_region='eu-central-1',
    role_arn='arn:aws:iam::081302066317:role/strands-test-agent',
)

# Map for easy access
CONFIGS = {
    'staging': STAGING,
    'prod': PROD,
}
