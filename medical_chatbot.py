# Importing Libraries
import pinecone
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embedding_model
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
from src.prompt import *  # Import everything from prompt file

import os
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# Setting up the Pinecone Client and Groq-Api-Key
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "medical-chatbot-implementation"
pc = Pinecone(api_key=pinecone_api_key)

# Initialize Flask App
app = Flask(__name__)

# Download Embeddings Model
embeddings = download_huggingface_embedding_model()

# Check if the index exists and connect to it
if index_name in pc.list_indexes().names():
    print(f"Connecting to index: {index_name}")
    index = pc.Index(index_name)
    print(f"Successfully connected to index: {index_name}")
else:
    print(f"Index '{index_name}' does not exist. Please create the index before running this code.")

# Create Pinecone Vector Store
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Define a Prompt Template to Answer Questions
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = ChatGroq(temperature=0.3, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

# Initialize the RetrievalQA chain 
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Initialize Flask App Routes
@app.route("/")
def index():
    return render_template("index.html")

# Process the user query and return the answer
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    input = msg.strip()  # Stripping white spaces from user input
    print(f"User Input: {input}")

    if not input:
        return jsonify({
            'response': """
            **Insufficient Information**
            =================================

            Unfortunately, the provided context does not contain enough information to accurately answer your question. To provide a helpful response, I would need more details about your inquiry.

            **Recommendation:**
            Please provide more context or clarify your question, and I will do my best to assist you. If you have a medical concern, I suggest consulting a qualified healthcare professional for personalized advice.
            """
        }), 200, {'ContentType': 'application/json'}

    try:
        result = qa.invoke({"query": input})
        response = result['result']
    except Exception as e:
        response = f"An error occurred: {str(e)}"

    print(f"Response: {response}")
    return jsonify({'response': response}), 200, {'ContentType': 'application/json'}

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
