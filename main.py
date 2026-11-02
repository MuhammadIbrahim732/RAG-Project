from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-2"
)

vector_store = Chroma(
    persist_directory="chroma-db",
    embedding_function= embedding_model
)

# Retrievers
retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

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
""")
    ]
)

print("-----RAG System-----")
print("Press 0 to exit")

while True:
    query = input("You: ")
    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query 
        }
    )

    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")
