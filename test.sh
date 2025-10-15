#!/bin/bash

# Configuration - Update these values
EMAIL="r.oblonczek@auvaria.com"
PASSWORD="Gladbach1."
CLIENT_ID="5aqpc5nkm2cinrfcdiiiks6kgc"
USER_POOL_ID="eu-central-1_2ZryMv0qs"
REGION="eu-central-1"
AGENT_ARN="arn:aws:bedrock-agentcore:eu-central-1:081302066317:runtime/demo_streaming_rag_agent_prod-6wk6lmHdeK"

echo "🔐 Authenticating with Cognito..."

# Use admin-initiate-auth which works better for server-to-server auth
# This requires the AWS credentials to have cognito-idp:AdminInitiateAuth permission
AUTH_RESPONSE=$(aws cognito-idp initiate-auth \
    --client-id "$CLIENT_ID" \
    --region "$REGION" \
    --auth-flow USER_PASSWORD_AUTH \
    --auth-parameters USERNAME="$EMAIL",PASSWORD="$PASSWORD" \
    --output json)

if [ $? -ne 0 ]; then
    echo "❌ Authentication failed"
    echo "$AUTH_RESPONSE"
    exit 1
fi

# Extract the access token (matching your frontend's usage)
JWT_TOKEN=$(echo "$AUTH_RESPONSE" | jq -r '.AuthenticationResult.AccessToken')

if [ "$JWT_TOKEN" = "null" ] || [ -z "$JWT_TOKEN" ]; then
    echo "❌ Failed to extract JWT token"
    echo "Response: $AUTH_RESPONSE"
    exit 1
fi

echo "✅ Authentication successful"
echo "🚀 Calling agent..."

# Step 2: Call the agent with the JWT token
# URL encode the ARN exactly like the frontend does with encodeURIComponent
ENCODED_ARN=$(printf '%s' "$AGENT_ARN" | python3 -c "import sys, urllib.parse; print(urllib.parse.quote(sys.stdin.read().strip(), safe=''))")
ENDPOINT="https://bedrock-agentcore.$REGION.amazonaws.com/runtimes/$ENCODED_ARN/invocations?qualifier=DEFAULT"

echo "📍 Endpoint: $ENDPOINT"

curl -X POST "$ENDPOINT" \
-H "Authorization: Bearer $JWT_TOKEN" \
-H "Content-Type: application/json" \
-H "Accept: text/event-stream" \
-H "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: test_session_$(date +%s)_$(date +%N)_extra" \
-d '{
"prompt": "what is hibernation state?"
}'