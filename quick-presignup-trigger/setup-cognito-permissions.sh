#!/bin/bash

# Setup IAM Role and Permissions for Cognito Pre-Sign Up Trigger Lambda
# This script creates the necessary IAM role and policies

set -e

# Configuration
ROLE_NAME="cognito-presignup-trigger-role"
POLICY_NAME="cognito-presignup-trigger-policy"
LAMBDA_FUNCTION_NAME="cognito-pre-signup-trigger"
USER_POOL_ID="eu-central-1_2ZryMv0qs" # You'll need to provide this
REGION="eu-central-1" # Change as needed

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up IAM permissions for Cognito Pre-Sign Up Trigger Lambda...${NC}"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account ID: $ACCOUNT_ID"

# Create trust policy for Lambda
cat > lambda-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create IAM role if it doesn't exist
if aws iam get-role --role-name "$ROLE_NAME" &> /dev/null; then
    echo -e "${YELLOW}Role $ROLE_NAME already exists${NC}"
else
    echo -e "${YELLOW}Creating IAM role: $ROLE_NAME${NC}"
    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document file://lambda-trust-policy.json \
        --description "IAM role for Cognito pre-sign up trigger Lambda function"
    
    echo -e "${GREEN}IAM role created successfully${NC}"
fi

# Attach AWS managed policy for basic Lambda execution
echo -e "${YELLOW}Attaching basic Lambda execution policy...${NC}"
aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"

# Create custom policy if it doesn't exist
if aws iam get-policy --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}" &> /dev/null; then
    echo -e "${YELLOW}Policy $POLICY_NAME already exists${NC}"
else
    echo -e "${YELLOW}Creating custom IAM policy: $POLICY_NAME${NC}"
    aws iam create-policy \
        --policy-name "$POLICY_NAME" \
        --policy-document file://cognito-trigger-iam-policy.json \
        --description "Custom policy for Cognito pre-sign up trigger Lambda"
    
    echo -e "${GREEN}Custom IAM policy created successfully${NC}"
fi

# Attach custom policy to role
echo -e "${YELLOW}Attaching custom policy to role...${NC}"
aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}"

# Get role ARN
ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)
echo -e "${GREEN}Role ARN: $ROLE_ARN${NC}"

# Wait for role to be available (IAM eventual consistency)
echo -e "${YELLOW}Waiting for IAM role to be available...${NC}"
sleep 10

echo -e "${GREEN}IAM setup completed!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Use this role ARN when creating the Lambda function: $ROLE_ARN"
echo "2. After deploying the Lambda function, you need to give Cognito permission to invoke it:"
echo ""
echo -e "${YELLOW}Command to add Cognito invoke permission:${NC}"

if [ -n "$USER_POOL_ID" ]; then
    echo "aws lambda add-permission \\"
    echo "  --function-name $LAMBDA_FUNCTION_NAME \\"
    echo "  --statement-id cognito-trigger-permission \\"
    echo "  --action lambda:InvokeFunction \\"
    echo "  --principal cognito-idp.amazonaws.com \\"
    echo "  --source-arn arn:aws:cognito-idp:$REGION:$ACCOUNT_ID:userpool/$USER_POOL_ID"
else
    echo "aws lambda add-permission \\"
    echo "  --function-name $LAMBDA_FUNCTION_NAME \\"
    echo "  --statement-id cognito-trigger-permission \\"
    echo "  --action lambda:InvokeFunction \\"
    echo "  --principal cognito-idp.amazonaws.com \\"
    echo "  --source-arn arn:aws:cognito-idp:$REGION:$ACCOUNT_ID:userpool/YOUR_USER_POOL_ID"
    echo ""
    echo -e "${RED}Note: Replace YOUR_USER_POOL_ID with your actual Cognito User Pool ID${NC}"
fi

echo ""
echo -e "${YELLOW}To configure the trigger in Cognito User Pool:${NC}"
echo "1. Go to AWS Cognito Console"
echo "2. Select your User Pool"
echo "3. Go to 'User pool properties' → 'Lambda triggers'"
echo "4. Set 'Pre sign-up' trigger to your Lambda function ARN"

# Clean up temporary files
rm -f lambda-trust-policy.json

echo -e "${GREEN}Setup script completed!${NC}"