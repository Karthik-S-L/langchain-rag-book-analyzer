from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter  
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil
#OPENAPI
#from langchain.embeddings import OpenAIEmbedding

#Using hugging face instead of openapi
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

# Set OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = CharacterTextSplitter(  
        chunk_size=300,  
        chunk_overlap=100  
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Print a sample chunk
    document = chunks[0]  # Get first chunk
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    # db = Chroma.from_documents(
    #     chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    # )

    db = Chroma.from_documents(
    chunks, HuggingFaceEmbeddings(), persist_directory=CHROMA_PATH
)
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
