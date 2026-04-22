# ML Intern - AWS CloudFormation Deployment

Deploy ML Intern with Bedrock support on AWS using CloudFormation.

## Templates

| Template | Description | Best For |
|----------|-------------|----------|
| `ml-intern-stack.yaml` | EC2-based deployment | Simple setup, SSH access, development |
| `ml-intern-fargate.yaml` | Fargate + ALB deployment | Production, auto-scaling, high availability |

---

## Option 1: EC2 Deployment (Simple)

### Prerequisites
- AWS CLI configured
- Hugging Face token

### Deploy

```bash
aws cloudformation create-stack \
  --stack-name ml-intern \
  --template-body file://ml-intern-stack.yaml \
  --parameters \
    ParameterKey=HFToken,ParameterValue=hf_xxxxx \
    ParameterKey=GitHubToken,ParameterValue=ghp_xxxxx \
    ParameterKey=BedrockModel,ParameterValue=bedrock/anthropic.claude-sonnet-4-6 \
    ParameterKey=InstanceType,ParameterValue=t3.medium \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1
```

### Get Web UI URL

```bash
aws cloudformation describe-stacks \
  --stack-name ml-intern \
  --query 'Stacks[0].Outputs[?OutputKey==`WebUIURL`].OutputValue' \
  --output text
```

---

## Option 2: Fargate Deployment (Production)

### Step 1: Deploy Infrastructure

```bash
aws cloudformation create-stack \
  --stack-name ml-intern-fargate \
  --template-body file://ml-intern-fargate.yaml \
  --parameters \
    ParameterKey=HFToken,ParameterValue=hf_xxxxx \
    ParameterKey=GitHubToken,ParameterValue=ghp_xxxxx \
    ParameterKey=BedrockModel,ParameterValue=bedrock/anthropic.claude-sonnet-4-6 \
    ParameterKey=ContainerCpu,ParameterValue=1024 \
    ParameterKey=ContainerMemory,ParameterValue=2048 \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1
```

### Step 2: Build and Push Docker Image

```bash
# Get ECR login
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Clone and build
git clone https://github.com/huggingface/ml-intern.git
cd ml-intern
docker build -t ml-intern .

# Tag and push
docker tag ml-intern:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-intern:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-intern:latest

# Force deployment
aws ecs update-service \
  --cluster ml-intern-fargate-cluster \
  --service ml-intern-fargate-service \
  --force-new-deployment
```

### Step 3: Get ALB URL

```bash
aws cloudformation describe-stacks \
  --stack-name ml-intern-fargate \
  --query 'Stacks[0].Outputs[?OutputKey==`WebUIURL`].OutputValue' \
  --output text
```

---

## Bedrock Models

### Available at Deploy Time (CloudFormation Parameter)

| Model | Parameter Value |
|-------|-----------------|
| Claude Opus 4.7 | `bedrock/anthropic.claude-opus-4-7` |
| Claude Sonnet 4.6 | `bedrock/anthropic.claude-sonnet-4-6` |
| Claude Opus 4.6 | `bedrock/anthropic.claude-opus-4-6` |
| Claude Sonnet 4.5 | `bedrock/anthropic.claude-sonnet-4-5` |
| Claude Haiku 4.5 | `bedrock/anthropic.claude-haiku-4-5` |
| Llama 3.1 70B | `bedrock/meta.llama3-1-70b-instruct-v1:0` |
| Llama 3.1 8B | `bedrock/meta.llama3-1-8b-instruct-v1:0` |

### Runtime Model Switching (Web UI)

Users can switch models at runtime via the Web UI without redeploying:

1. Open the ML Intern Web UI
2. Click the model selector dropdown
3. Choose from available models:
   - **Bedrock**: Claude Opus 4.7, Sonnet 4.6, Haiku 4.5, Llama 3.1
   - **Anthropic Direct**: Claude Opus 4.6
   - **HuggingFace**: MiniMax M2.7, Kimi K2.6, GLM 5.1

> **Note**: Ensure the model is enabled in your AWS Bedrock console before using.

---

## Enable Bedrock Models

1. Go to **AWS Console** → **Amazon Bedrock** → **Model access**
2. Click **Manage model access**
3. Enable the models you want to use
4. Wait for access to be granted (usually instant for Anthropic models)

---

## Costs (Estimated)

### EC2 Deployment
| Resource | Monthly Cost |
|----------|--------------|
| t3.medium | ~$30 |
| EBS 30GB | ~$3 |
| Elastic IP | ~$4 |
| **Total** | **~$37/month** + Bedrock usage |

### Fargate Deployment
| Resource | Monthly Cost |
|----------|--------------|
| Fargate (1 vCPU, 2GB) | ~$35 |
| ALB | ~$20 |
| **Total** | **~$55/month** + Bedrock usage |

### Bedrock Pricing (Claude Sonnet 4.6)
- Input: $3.00 / 1M tokens
- Output: $15.00 / 1M tokens

### Bedrock Pricing (Claude Haiku 4.5)
- Input: $0.25 / 1M tokens
- Output: $1.25 / 1M tokens

---

## Cleanup

```bash
# EC2 deployment
aws cloudformation delete-stack --stack-name ml-intern

# Fargate deployment
aws cloudformation delete-stack --stack-name ml-intern-fargate
```

---

## Troubleshooting

### Check EC2 Logs
```bash
# SSH into instance
ssh -i your-key.pem ec2-user@<public-ip>

# View setup logs
sudo cat /var/log/user-data.log

# View service logs
sudo journalctl -u ml-intern -f
```

### Check Fargate Logs
```bash
aws logs tail /ecs/ml-intern-fargate --follow
```

### Common Issues

1. **Bedrock Access Denied**: Enable the model in Bedrock console
2. **Container fails to start**: Check CloudWatch logs for errors
3. **Health check failing**: Ensure port 7860 is accessible
