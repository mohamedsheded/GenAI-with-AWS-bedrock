# AWS Bedrock Setup Guide

## 1. Configure Amazon Bedrock Environment

### Step 1: Create a Virtual Environment
```bash
# Create a virtual environment
python3 -m venv bedrock-env

# Activate the virtual environment
source bedrock-env/bin/activate   # On Linux/MacOS
bedrock-env\Scripts\activate     # On Windows
```

### Step 2: Install Required Packages
```bash
pip install boto3
pip install awscli
```

---

## 2. Create an IAM User
1. Go to the **AWS Management Console**.
2. Navigate to **IAM** (Identity and Access Management).
3. Click on **Users** > **Add users**.
4. Provide a **username** and select **Programmatic access**.
5. Attach the following policies:
   - **AmazonBedrockFullAccess**
   - **AmazonS3FullAccess** (if using S3 for storage).
6. Download the **Access Key ID** and **Secret Access Key**.

---

## 3. Configure AWS CLI
```bash
# Initialize AWS CLI configuration
aws configure
```
You will be prompted to enter:
- **AWS Access Key ID**: Enter the key obtained during IAM user creation.
- **AWS Secret Access Key**: Enter the secret key obtained.
- **Default region name**: Choose your AWS region (e.g., `us-east-1`).
- **Default output format**: Use `json`.

---

## 4. Model Access and Region Management
1. Go to **Amazon Bedrock** in AWS Management Console.
2. Select **Model Access**.
3. Enable access to the required models such as:
   - AI21 Labs (e.g., Jurassic-2)
   - Anthropic (e.g., Claude)
   - Meta (e.g., LLaMA)
4. Choose the region that supports the required models, e.g., **us-east-1** or **us-west-2**.

---

## 5. Default Region and Output Format
Ensure the following are set:
```bash
aws configure
```
Example values:
```
AWS Access Key ID [None]: YOUR_ACCESS_KEY
AWS Secret Access Key [None]: YOUR_SECRET_KEY
Default region name [None]: us-east-1
Default output format [None]: json
```

---

## 6. Run the Provided API Scripts
### Example Script 1: claude.py

### Code Explanation:
```python
import boto3
import json
```
- **boto3**: AWS SDK for Python to interact with AWS services.
- **json**: Used for handling JSON data formats for input and output.

```python
prompt_data="""
Act as a Shakespeare and write a poem on Generative AI
"""
```
- **prompt_data**: Specifies the input text prompt for the AI model to generate a response.
- In this example, it instructs the model to act like Shakespeare and compose a poem about Generative AI.

```python
bedrock=boto3.client(service_name="bedrock-runtime")
```
- **boto3.client**: Creates a client to connect to Amazon Bedrock's runtime service for inference.

```python
payload={
    "prompt":prompt_data,
    "maxTokens":512,
    "temperature":0.8,
    "topP":0.8
}
```
- **payload**: Contains parameters sent to the model.
  - **prompt**: Text input for the AI.
  - **maxTokens**: Limits the response length to 512 tokens.
  - **temperature**: Controls creativity; higher values lead to more varied outputs.
  - **topP**: Restricts token selection to the top 80% probability mass for diverse outputs.

```python
body = json.dumps(payload)
```
- Converts the payload into a JSON string format.

```python
model_id = "ai21.j2-mid-v1"
```
- Specifies the model ID to be used, here AI21's Jurassic-2 Mid.

```python
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)
```
- **invoke_model**: Calls the specified AI model with the given payload.
  - **body**: JSON-encoded input data.
  - **modelId**: ID of the selected model.
  - **accept**: Expected response format.
  - **contentType**: Input data format.

```python
response_body = json.loads(response.get("body").read())
```
- Reads and parses the JSON response body.

```python
response_text = response_body.get("completions")[0].get("data").get("text")
```
- Extracts the generated text from the response JSON.

```python
print(response_text)
```
- Displays the generated output in the console.

---


This completes the setup for using Amazon Bedrock with the provided scripts.

