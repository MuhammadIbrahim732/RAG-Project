import shutil
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma-db"
STATIC_DIR = BASE_DIR / "static"
MAX_UPLOAD_MB = 25

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are a helpful RAG assistant. Answer questions using only the provided context.
If the answer is not found in the context, say:
"I don't know based on the provided documents."
Keep answers clear, accurate, and concise. Do not make up information.
"""),
        ("human", """
Context: {context}
Question: {question}
"""),
    ]
)

app = FastAPI(title="Ask your PDF")

# One vector store per uploaded PDF, cached after first use
stores: dict[str, Chroma] = {}


def valid_doc_id(doc_id: str) -> str:
    try:
        return uuid.UUID(doc_id).hex
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document id.")


def get_store(doc_id: str) -> Chroma:
    doc_id = valid_doc_id(doc_id)
    if doc_id not in stores:
        path = DB_DIR / doc_id
        if not path.exists():
            raise HTTPException(status_code=404, detail="Document not found. Upload it again.")
        stores[doc_id] = Chroma(
            persist_directory=str(path),
            embedding_function=embedding_model,
        )
    return stores[doc_id]


def to_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part) for part in content
    )


class ChatRequest(BaseModel):
    doc_id: str
    question: str


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        if tmp_path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_MB} MB.")

        try:
            docs = PyPDFLoader(str(tmp_path)).load()
        except Exception:
            raise HTTPException(status_code=422, detail="This file could not be read as a PDF.")

        if not any(d.page_content.strip() for d in docs):
            raise HTTPException(
                status_code=422,
                detail="No text found. Scanned PDFs need OCR before they can be used.",
            )

        chunks = splitter.split_documents(docs)
        doc_id = uuid.uuid4().hex

        stores[doc_id] = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=str(DB_DIR / doc_id),
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "pages": len(docs),
        "chunks": len(chunks),
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Type a question first.")

    store = get_store(req.doc_id)
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.invoke({"context": context, "question": question})
    response = llm.invoke(final_prompt)

    pages = sorted({int(d.metadata["page"]) + 1 for d in docs if "page" in d.metadata})
    return {"answer": to_text(response.content), "pages": pages}


@app.delete("/api/document/{doc_id}")
def delete_document(doc_id: str):
    doc_id = valid_doc_id(doc_id)
    stores.pop(doc_id, None)
    shutil.rmtree(DB_DIR / doc_id, ignore_errors=True)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
