# Importing Libraries
import pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from src.helper import load_pdf, text_split, download_huggingface_embedding_model, batch_upsert

import os
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# Setting up the Pinecone Client
api_key = os.getenv("PINECONE_API_KEY")
index_name = "medical-chatbot-implementation"
pc = Pinecone(api_key=api_key)

# Create a new index if it doesn't exist
# Check if the index exists
if index_name in pc.list_indexes().names():
    print(f"Index '{index_name}' already exists. Connecting to the existing index.")
    index = pc.Index(index_name)
else:
    print(f"Index '{index_name}' does not exist. Creating a new index.")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    index = pc.Index(index_name)

print(f"Successfully connected to index: {index_name}")

# Extract and Load the Data from PDF
extracted_data = load_pdf(r"E:\Practice python\Krish Naik\End to end Medical Chatbot Implementation\data")

# Let's Split Text into Chunks/Documents
text_chunks = text_split(extracted_data)

# Let's Download Embeddings Model
embeddings = download_huggingface_embedding_model()

# Let's Create Embeddings for the Chunks
model = embeddings.client
embeddings_list = model.encode([t.page_content for t in text_chunks]).tolist()

# Upsert documents into Pinecone index
index = pc.Index(index_name)
batch_upsert(index, text_chunks, embeddings_list)

# Create Pinecone Vector Store
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)