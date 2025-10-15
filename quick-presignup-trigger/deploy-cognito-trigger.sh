#!/bin/bash

# Deploy Cognito Pre-Sign Up Trigger Lambda Function
# This script packages and deploys the Lambda function to AWS

set -e

# Configuration
FUNCTION_NAME="cognito-pre-signup-trigger"
RUNTIME="python3.13"
HANDLER="cognito_pre_signup_trigger.lambda_handler"
ROLE_NAME="cognito-presignup-trigger-role"
REGION="eu-central-1" # Change as needed

# Get role ARN dynamically
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting deployment of Cognito Pre-Sign Up Trigger Lambda...${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if user is logged in to AWS
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}Not authenticated with AWS. Please run 'aws configure' first.${NC}"
    exit 1
fi

# Create a temporary directory for packaging
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"

# Copy Lambda function to temp directory
cp cognito_pre_signup_trigger.py "$TEMP_DIR/"

# Create deployment package
cd "$TEMP_DIR"
zip -r "../${FUNCTION_NAME}.zip" .
cd -

# Move zip file to current directory
mv "$TEMP_DIR/../${FUNCTION_NAME}.zip" "./"

# Clean up temp directory
rm -rf "$TEMP_DIR"

echo -e "${GREEN}Deployment package created: ${FUNCTION_NAME}.zip${NC}"

# Check if Lambda function exists
if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" &> /dev/null; then
    echo -e "${YELLOW}Function exists. Updating code...${NC}"
    
    # Update existing function
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --zip-file "fileb://${FUNCTION_NAME}.zip" \
        --region "$REGION"
        
    echo -e "${GREEN}Function code updated successfully!${NC}"
else
    echo -e "${YELLOW}Function does not exist. Creating new function...${NC}"
    
    # Verify role exists
    if ! aws iam get-role --role-name "$ROLE_NAME" &> /dev/null; then
        echo -e "${RED}Error: IAM role '$ROLE_NAME' does not exist.${NC}"
        echo "Please run './setup-cognito-permissions.sh' first to create the required IAM role."
        exit 1
    fi
    
    # Create new function
    aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --runtime "$RUNTIME" \
        --role "$ROLE_ARN" \
        --handler "$HANDLER" \
        --zip-file "fileb://${FUNCTION_NAME}.zip" \
        --timeout 30 \
        --memory-size 128 \
        --region "$REGION" \
        --description "Cognito pre-sign up trigger that only allows @auvaria.com email addresses"
        
    echo -e "${GREEN}Function created successfully!${NC}"
fi

# Get function ARN for Cognito configuration
FUNCTION_ARN=$(aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" --query 'Configuration.FunctionArn' --output text)

echo -e "${GREEN}Deployment completed!${NC}"
echo -e "${YELLOW}Function ARN: ${FUNCTION_ARN}${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Configure this Lambda function as a pre-sign up trigger in your Cognito User Pool"
echo "2. Make sure the Lambda function has permission to be invoked by Cognito"
echo "3. Test the function with different email domains to verify it works correctly"
echo ""
echo -e "${YELLOW}To add Cognito invoke permission:${NC}"
echo "aws lambda add-permission \\"
echo "  --function-name $FUNCTION_NAME \\"
echo "  --statement-id cognito-trigger \\"
echo "  --action lambda:InvokeFunction \\"
echo "  --principal cognito-idp.amazonaws.com \\"
echo "  --source-arn arn:aws:cognito-idp:$REGION:ACCOUNT_ID:userpool/USER_POOL_ID"

# Clean up deployment package
rm "${FUNCTION_NAME}.zip"
echo "Deployment package cleaned up."