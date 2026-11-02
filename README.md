# 📄 Ask Your PDF: RAG Chatbot

A Retrieval-Augmented Generation (RAG) app that lets you upload a PDF and ask questions about it. Answers are generated **only from the document's content**, with page references. Built with **LangChain**, **Google Gemini**, **ChromaDB**, and **FastAPI**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C)
![Gemini](https://img.shields.io/badge/Google-Gemini-4285F4)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)

---

## ✨ Features

- Upload any text-based PDF (up to 25 MB) through a web interface
- Semantic search over the document using Gemini embeddings
- MMR retrieval for relevant and diverse context
- Answers grounded strictly in the document. If the answer isn't there, the bot says: *"I don't know based on the provided documents."*
- Page numbers returned with every answer
- Separate vector store per uploaded PDF, persisted on disk
- Delete a document and its embeddings through the API
- Command-line version for quick testing

---

## 🧠 How It Works

```
PDF upload
   │
   ▼
PyPDFLoader  ──►  RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
   │
   ▼
Gemini embeddings  ──►  ChromaDB (persisted per document)
   │
   ▼
User question ──► MMR retriever (k=4, fetch_k=10, λ=0.5)
   │
   ▼
Prompt (context + question) ──► Gemini LLM ──► Answer + source pages
```

| Setting          | Value                    |
| ---------------- | ------------------------ |
| Embedding model  | `gemini-embedding-2`     |
| LLM              | `gemini-3.5-flash-lite`  |
| Chunk size       | 1000 characters          |
| Chunk overlap    | 200 characters           |
| Retrieval        | MMR, `k=4`, `fetch_k=10` |
| Max upload size  | 25 MB                    |

---

## 📁 Project Structure

```
.
├── app.py             # FastAPI web app (upload + chat + delete)
├── file.py            # Script: index a PDF into ChromaDB (CLI workflow)
├── main.py            # Script: terminal chat over the indexed PDF
├── static/
│   └── index.html     # Web UI
├── chroma-db/         # Vector stores (created automatically)
├── requirements.txt
└── .env               # API key (not committed)
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Gemini API key

Get a key from [Google AI Studio](https://aistudio.google.com/apikey) and create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here
```

---

## ▶️ Usage

### Option A: Web app

```bash
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000**, upload a PDF, and start asking questions.
Interactive API docs are at **http://127.0.0.1:8000/docs**.

### Option B: Command line

1. Place your PDF in `document_loaders/` and set its filename in `file.py`.
2. Index the document:
   ```bash
   python file.py
   ```
3. Start the chat:
   ```bash
   python main.py
   ```
   Type your questions, and enter `0` to exit.

---

## 🔌 API Reference

| Method | Endpoint                  | Description                                      |
| ------ | ------------------------- | ------------------------------------------------ |
| GET    | `/`                       | Web interface                                    |
| POST   | `/api/upload`             | Upload a PDF (multipart `file`), returns `doc_id` |
| POST   | `/api/chat`               | Ask a question about an uploaded document        |
| DELETE | `/api/document/{doc_id}`  | Delete a document and its vector store          |

**Upload**

```bash
curl -X POST http://127.0.0.1:8000/api/upload -F "file=@paper.pdf"
```

```json
{
  "doc_id": "3f2a9c...",
  "filename": "paper.pdf",
  "pages": 15,
  "chunks": 42
}
```

**Chat**

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "3f2a9c...", "question": "What is the main idea of the paper?"}'
```

```json
{
  "answer": "The paper introduces ...",
  "pages": [1, 2, 3]
}
```

### Error responses

| Status | Meaning                                                  |
| ------ | -------------------------------------------------------- |
| `400`  | Not a PDF, invalid document id, or empty question        |
| `404`  | Document not found (upload it again)                     |
| `413`  | File larger than 25 MB                                   |
| `422`  | Unreadable PDF, or no text found (scanned PDFs need OCR) |

---

## ⚠️ Limitations

- Scanned or image-only PDFs are not supported (no OCR)
- Answers depend on retrieval quality, and only the top 4 chunks are sent to the model
- No chat memory, so each question is answered independently

---

## 🛠️ Tech Stack

- **Python**
- **LangChain**: orchestration, loaders, text splitting, prompts
- **Google Gemini**: embeddings and LLM (`langchain-google-genai`)
- **ChromaDB**: vector database
- **FastAPI + Uvicorn**: backend API
- **PyPDF**: PDF parsing

---

## 🔮 Future Improvements

- Conversation memory for follow-up questions
- Streaming responses
- OCR support for scanned PDFs
- Multi-PDF chat
- Show source text snippets alongside page numbers
- Docker deployment

---

## 👤 Author

**Ibrahim**
BS Software Engineering, National Textile University (NTU), Faisalabad

🔗 [LinkedIn](https://linkedin.com/in/muhammad-ibrahim-python)

---

## 📄 License

This project is licensed under the MIT License.
