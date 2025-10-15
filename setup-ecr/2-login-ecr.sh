# 2. Login to ECR
aws ecr get-login-password --region eu-central-1 | \
    docker login --username AWS --password-stdin 081302066317.dkr.ecr.eu-central-1.amazonaws.com
