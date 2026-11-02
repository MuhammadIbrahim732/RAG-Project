from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Document Loaders
loader = PyPDFLoader("document_loaders/Attention Is All You Need.pdf")
docs = loader.load()

# Text Splitters
splitter = RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap= 200
)

chunks = splitter.split_documents(docs)

# Generate Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-2"
)

# Vector Stores
vector_store = Chroma.from_documents(
    documents= chunks,
    embedding= embedding_model,
    persist_directory= "chroma-db"

)