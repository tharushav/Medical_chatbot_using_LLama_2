from src.helper import extract_data_from_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

data_dir = "data/"
if not os.path.exists(data_dir) or not os.listdir(data_dir):
    raise FileNotFoundError(f"No PDF files found in {data_dir}")

extracted_data = extract_data_from_pdf(data_dir)
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


# Initialize Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)
index_name = "medicalchatbot"

# Create the LangChain Pinecone vector store from the text chunks
docsearch = PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)