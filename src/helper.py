import os
import warnings
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.simplefilter("ignore", FutureWarning)
load_dotenv()


# Extract and Load the Data from PDF
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


# Let's Split Text into Chunks/Documents
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Let's Download Embeddings Model
def download_huggingface_embedding_model():
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


# Let's Create Embeddings for the Chunks
def batch_upsert(index, text_chunks, embeddings, batch_size=100):
    for i in tqdm(range(0, len(embeddings), batch_size)):
        i_end = min(i+batch_size, len(embeddings))
        batch = list(zip(
            [str(j) for j in range(i, i_end)],
            embeddings[i:i_end],
            [{"text": chunk.page_content} for chunk in text_chunks[i:i_end]]
        ))
        index.upsert(vectors=batch)

