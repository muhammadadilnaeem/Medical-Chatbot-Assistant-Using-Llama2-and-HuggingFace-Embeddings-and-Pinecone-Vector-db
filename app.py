# Importing Libraries
import pinecone
from langchain_pinecone import PineconeVectorStore
from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embedding_model
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
from src.prompt import * # mean import everything in prompt file

import os
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# Setting up the Pinecone Client
api_key = os.getenv("PINECONE_API_KEY")
index_name = "medical-chatbot-implementation"
pc = Pinecone(api_key=api_key)

# Initialize Flask App
app = Flask(__name__)

# Let's Download Embeddings Model
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

# Let's Define a Prompt Template to Answer Questions
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Let's Initialize LLM Model using CTransformers
llm = CTransformers(model = r"E:\Practice python\Krish Naik\End to end Medical Chatbot Implementation\models\llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens':512,
                            'temperature':0.8})

# Let's Initialize the RetrievalQA chain 
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Initialize Flask App Routes by setting up the homepage
@app.route("/")
def index():
    return render_template("index.html")

# This will process the user query and return the answer
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


# This will be called when the user submits the question
if __name__ == "__main__":
    app.run(debug=True)
