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

## 2. Integration of AWS bedrock with Langchain and streamlit

### Code Explanation:

1. **Imports and Setup**
```python
import json
import os
import boto3
import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
```
- Includes required libraries for AWS Bedrock integration, embeddings, text processing, and vector storage.

2. **Bedrock Client and Embeddings**
```python
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)
```
- Initializes the Bedrock client and sets up embeddings using Amazon Titan.

3. **Data Ingestion**
```python
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs=text_splitter.split_documents(documents)
    return docs
```
- Loads and splits PDF data into smaller chunks for processing.

4. **Vector Storage**
```python
def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")
```
- Generates vector embeddings and saves them in a FAISS vector store.

5. **LLM Setup**
```python
def get_claude_llm():
    llm=Bedrock(model_id="ai21.j2-mid-v1",client=bedrock, model_kwargs={'maxTokens':512})
    return llm

def get_llama2_llm():
    llm=Bedrock(model_id="meta.llama2-70b-chat-v1",client=bedrock, model_kwargs={'max_gen_len':512})
    return llm
```
- Configures Claude and LLaMA2 models.

6. **Prompt Template**
```python
prompt_template = """
Human: Use the following pieces of context to provide a detailed 250-word answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
```
- Defines a structured template for the model's response.

7. **Response Generation**
```python
def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']
```
- Generates answers using the model and vector store.

8. **Streamlit Application**
```python
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        if st.button("Vectors Update"):
            docs = data_ingestion()
            get_vector_store(docs)
            st.success("Done")

    if st.button("Claude Output"):
        faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
        llm=get_claude_llm()
        st.write(get_response_llm(llm,faiss_index,user_question))

    if st.button("Llama2 Output"):
        faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
        llm=get_llama2_llm()
        st.write(get_response_llm(llm,faiss_index,user_question))

if __name__ == "__main__":
    main()
```
- Implements a Streamlit web app for interacting with PDFs using Claude and LLaMA2.

---

This completes the setup for AWS Bedrock and integration with Streamlit and LangChain.


