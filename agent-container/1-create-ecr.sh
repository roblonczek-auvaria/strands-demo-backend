# 1. Create ECR repository (if it doesn't exist)
aws ecr create-repository \
    --repository-name streaming-agent \
    --region eu-central-1 \
    --image-scanning-configuration scanOnPush=true
