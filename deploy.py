"""
Unified deployment script for Bedrock AgentCore with direct code deployment.

This script handles:
- Creating deployment packages
- Uploading to S3
- Creating new agent runtimes
- Updating existing agent runtimes
- Health monitoring and validation

Usage:
    python deploy.py --env staging                    # Create new agent in staging
    python deploy.py --env prod                       # Create new agent in production
    python deploy.py --env staging --update           # Update existing staging agent
    python deploy.py --env staging --arn <ARN>        # Update specific agent ARN
    python deploy.py --env staging --skip-build       # Skip building, use existing ZIP
"""

import argparse
import boto3
import json
import time
import os
import subprocess
from pathlib import Path
from typing import Optional
from config import CONFIGS, DeploymentConfig


class AgentDeployer:
    """Handles deployment of Bedrock AgentCore agents using direct code deployment."""
    
    def __init__(self, config: DeploymentConfig, verbose: bool = True):
        """Initialize deployer with configuration.
        
        Args:
            config: Deployment configuration for the target environment
            verbose: Enable detailed logging
        """
        self.config = config
        self.verbose = verbose
        self.s3_client = boto3.client('s3', region_name=config.region)
        self.agentcore_client = boto3.client('bedrock-agentcore-control', region_name=config.region)
    
    def log(self, message: str, prefix: str = "→"):
        """Print log message if verbose is enabled."""
        if self.verbose:
            print(f"{prefix} {message}")
    
    def log_section(self, title: str):
        """Print section header."""
        if self.verbose:
            print("\n" + "="*70)
            print(title)
            print("="*70)
    
    def build_package(self) -> Path:
        """Build the deployment package using the build script.
        
        Returns:
            Path to the built ZIP file
            
        Raises:
            Exception: If build fails
        """
        self.log_section("Building Deployment Package")
        
        script_path = Path(__file__).parent / "build_package.sh"
        if not script_path.exists():
            raise Exception(f"Build script not found: {script_path}")
        
        try:
            # Ensure script is executable
            os.chmod(script_path, 0o755)
            
            result = subprocess.run(
                [str(script_path)],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                check=True
            )
            
            if self.verbose:
                print(result.stdout)
            
            zip_path = Path(__file__).parent / "deployment_package.zip"
            if not zip_path.exists():
                raise Exception("Build script succeeded but ZIP file not found")
            
            # Get file size
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            self.log(f"Package size: {size_mb:.2f} MB", "✓")
            
            if size_mb > 250:
                raise Exception(f"Package size ({size_mb:.2f} MB) exceeds 250 MB limit")
            
            return zip_path
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Build failed:\n{e.stderr}")
            raise Exception("Package build failed")
    
    def ensure_s3_bucket(self):
        """Ensure S3 bucket exists, create if needed.
        
        Raises:
            Exception: If bucket creation fails
        """
        bucket_name = self.config.s3_bucket
        
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            self.log(f"S3 bucket exists: {bucket_name}", "✓")
        except:
            self.log(f"Creating S3 bucket: {bucket_name}")
            try:
                if self.config.region == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.config.region}
                    )
                self.log(f"S3 bucket created: {bucket_name}", "✓")
            except Exception as e:
                raise Exception(f"Failed to create S3 bucket: {e}")
    
    def upload_to_s3(self, zip_path: Path) -> str:
        """Upload deployment package to S3.
        
        Args:
            zip_path: Path to the ZIP file to upload
            
        Returns:
            S3 URI of the uploaded file
            
        Raises:
            Exception: If upload fails
        """
        self.log_section("Uploading to S3")
        
        bucket = self.config.s3_bucket
        key = self.config.s3_prefix
        
        self.log(f"Uploading to s3://{bucket}/{key}")
        
        try:
            self.s3_client.upload_file(
                str(zip_path),
                bucket,
                key,
                ExtraArgs={'ExpectedBucketOwner': self.config.account_id}
            )
            
            s3_uri = f"s3://{bucket}/{key}"
            self.log(f"Upload complete: {s3_uri}", "✓")
            return s3_uri
            
        except Exception as e:
            raise Exception(f"Failed to upload to S3: {e}")
    
    def create_agent_runtime(self) -> str:
        """Create a new agent runtime.
        
        Returns:
            Agent runtime ARN
            
        Raises:
            Exception: If creation fails
        """
        self.log_section(f"Creating Agent Runtime ({self.config.env})")
        
        self.log(f"Agent Name:    {self.config.agent_name}")
        self.log(f"Region:        {self.config.region}")
        self.log(f"Runtime:       {self.config.runtime_version}")
        self.log(f"Network Mode:  {self.config.network_mode}")
        self.log(f"Cognito Pool:  {self.config.cognito_user_pool_id}")
        self.log(f"Discovery URL: {self.config.discovery_url}")
        
        try:
            response = self.agentcore_client.create_agent_runtime(
                agentRuntimeName=self.config.agent_name,
                agentRuntimeArtifact={
                    'codeConfiguration': {
                        'code': {
                            's3': {
                                'bucket': self.config.s3_bucket,
                                'prefix': self.config.s3_prefix
                            }
                        },
                        'runtime': self.config.runtime_version,
                        'entryPoint': ['opentelemetry-instrument', "server.py"]
                    }
                },
                authorizerConfiguration={
                    'customJWTAuthorizer': {
                        'discoveryUrl': self.config.discovery_url,
                        'allowedClients': [self.config.cognito_client_id]
                    }
                },
                networkConfiguration={'networkMode': self.config.network_mode},
                roleArn=self.config.role_arn
            )
            
            agent_runtime_arn = response['agentRuntimeArn']
            status = response['status']
            
            self.log(f"Agent created: {agent_runtime_arn}", "✓")
            self.log(f"Initial status: {status}")
            
            # Wait for agent to be ready
            if status != 'READY':
                self.log("Waiting for agent to be ready...")
                self.wait_for_ready(agent_runtime_arn)
            
            return agent_runtime_arn
            
        except Exception as e:
            raise Exception(f"Failed to create agent runtime: {e}")
    
    def update_agent_runtime(self, agent_runtime_arn: str) -> str:
        """Update an existing agent runtime.
        
        Args:
            agent_runtime_arn: ARN of the agent to update
            
        Returns:
            Agent runtime ARN
            
        Raises:
            Exception: If update fails
        """
        self.log_section(f"Updating Agent Runtime ({self.config.env})")
        
        # Extract runtime ID from ARN
        agent_runtime_id = agent_runtime_arn.split('/')[-1]
        
        self.log(f"Agent ARN: {agent_runtime_arn}")
        self.log(f"Updating with new deployment package...")
        
        try:
            # First, check if the agent runtime currently uses container or code
            try:
                current_config = self.agentcore_client.get_agent_runtime(agentRuntimeId=agent_runtime_id)
                # If we need to switch from container to code, we might need special handling
                # But typically update_agent_runtime should handle the switch if we provide the new config
                self.log(f"Current status: {current_config['status']}")
            except Exception as e:
                self.log(f"Warning: Could not fetch current agent config: {e}", "⚠")

            response = self.agentcore_client.update_agent_runtime(
                agentRuntimeId=agent_runtime_id,
                agentRuntimeArtifact={
                    'codeConfiguration': {
                        'code': {
                            's3': {
                                'bucket': self.config.s3_bucket,
                                'prefix': self.config.s3_prefix
                            }
                        },
                        'runtime': self.config.runtime_version,
                        'entryPoint': ["opentelemetry-instrument", "server.py"]
                    }
                },
                authorizerConfiguration={
                    'customJWTAuthorizer': {
                        'discoveryUrl': self.config.discovery_url,
                        'allowedClients': [self.config.cognito_client_id]
                    }
                },
                networkConfiguration={'networkMode': self.config.network_mode},
                roleArn=self.config.role_arn
            )
            
            status = response['status']
            self.log(f"Update initiated, status: {status}", "✓")
            
            # Wait for agent to be ready
            if status != 'READY':
                self.log("Waiting for agent to be ready...")
                self.wait_for_ready(agent_runtime_arn)
            
            return agent_runtime_arn
            
        except Exception as e:
            raise Exception(f"Failed to update agent runtime: {e}")
    
    def wait_for_ready(self, agent_runtime_arn: str, max_wait: int = 300):
        """Wait for agent to reach READY status.
        
        Args:
            agent_runtime_arn: ARN of the agent to monitor
            max_wait: Maximum time to wait in seconds
            
        Raises:
            TimeoutError: If agent doesn't become ready in time
            Exception: If agent deployment fails
        """
        agent_runtime_id = agent_runtime_arn.split('/')[-1]
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = self.agentcore_client.get_agent_runtime(
                    agentRuntimeId=agent_runtime_id
                )
                status = response['status']
                
                if status == 'READY':
                    self.log("Agent is READY", "✓")
                    return
                elif status in ['FAILED', 'STOPPED', 'CREATE_FAILED']:
                    # Extract failure reason if available
                    failure_reason = response.get('failureReason', 'No failure reason provided')
                    self.log(f"Agent deployment failed!", "✗")
                    self.log(f"Status: {status}")
                    self.log(f"Failure Reason: {failure_reason}")
                    raise Exception(f"Agent deployment failed with status: {status}. Reason: {failure_reason}")
                else:
                    elapsed = int(time.time() - start_time)
                    self.log(f"Status: {status} ({elapsed}s elapsed)")
                    time.sleep(10)
            except Exception as e:
                if 'READY' not in str(e):
                    elapsed = int(time.time() - start_time)
                    self.log(f"Checking status... ({elapsed}s elapsed)")
                time.sleep(10)
        
        raise TimeoutError(f"Agent did not become ready within {max_wait} seconds")
    
    def get_agent_arn_by_name(self) -> Optional[str]:
        """Find existing agent runtime ARN by name.
        
        Returns:
            Agent ARN if found, None otherwise
        """
        try:
            response = self.agentcore_client.list_agent_runtimes()
            
            for agent in response.get('agentRuntimes', []):
                if agent.get('agentRuntimeName') == self.config.agent_name:
                    return agent.get('agentRuntimeArn')
            
            return None
        except Exception as e:
            self.log(f"Warning: Could not list agents: {e}", "⚠")
            return None


def main():
    """Main entry point for deployment script."""
    parser = argparse.ArgumentParser(
        description='Deploy Bedrock AgentCore agents with direct code deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create new agent in staging
  uv run deploy.py --env staging
  
  # Update existing agent in staging
  uv run deploy.py --env staging --update
  
  # Update specific agent ARN
  uv run deploy.py --env staging --arn arn:aws:bedrock-agentcore:...
  
  # Skip build step (use existing ZIP)
  uv run deploy.py --env staging --skip-build
        """
    )
    
    parser.add_argument(
        '--env',
        choices=['staging', 'prod'],
        required=True,
        help='Target environment (staging or prod)'
    )
    
    parser.add_argument(
        '--update',
        action='store_true',
        help='Update existing agent instead of creating new one'
    )
    
    parser.add_argument(
        '--arn',
        type=str,
        help='Specific agent ARN to update (overrides auto-detection)'
    )
    
    parser.add_argument(
        '--skip-build',
        action='store_true',
        help='Skip building package and upload (use existing package on S3)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = CONFIGS[args.env]
    deployer = AgentDeployer(config, verbose=not args.quiet)
    
    try:
        # Step 1: Build package (unless skipped)
        if args.skip_build:
            deployer.log("Skipping build, using existing deployment_package.zip", "→")
            zip_path = Path(__file__).parent / "deployment_package.zip"
            if not zip_path.exists():
                raise Exception("deployment_package.zip not found. Run without --skip-build first.")
        else:
            zip_path = deployer.build_package()
        
        # Step 2: Ensure S3 bucket exists
        deployer.ensure_s3_bucket()
        
        # Step 3: Upload to S3
        if args.skip_build:
            deployer.log("Skipping upload as build was skipped (assuming package exists on S3)", "→")
            s3_uri = f"s3://{config.s3_bucket}/{config.s3_prefix}"
        else:
            s3_uri = deployer.upload_to_s3(zip_path)
        
        # Step 4: Create or update agent
        if args.update or args.arn:
            # Update mode
            agent_arn = args.arn
            
            if not agent_arn:
                # Auto-detect ARN by name
                deployer.log("Looking for existing agent by name...")
                agent_arn = deployer.get_agent_arn_by_name()
                
                if not agent_arn:
                    raise Exception(
                        f"Agent '{config.agent_name}' not found. "
                        "Create it first (without --update) or provide --arn explicitly."
                    )
            
            agent_arn = deployer.update_agent_runtime(agent_arn)
        else:
            # Create mode
            agent_arn = deployer.create_agent_runtime()
        
        # Success!
        deployer.log_section("✓ Deployment Successful!")
        print(f"\nAgent ARN:")
        print(f"  {agent_arn}")
        print(f"\nEnvironment: {config.env}")
        print(f"Region:      {config.region}")
        print(f"S3 Package:  {s3_uri}")
        
        print(f"\nNext steps:")
        print(f"  • Test endpoint in AWS Console")
        print(f"  • Monitor CloudWatch logs")
        print(f"  • Update frontend with ARN if needed")
        
    except KeyboardInterrupt:
        print("\n\n✗ Deployment cancelled by user")
        exit(1)
    except Exception as e:
        print(f"\n✗ Deployment failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
