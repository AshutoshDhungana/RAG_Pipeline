import os, pathlib
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Load key from a local .env file (e.g. GEMINI_API_KEY=... or GOOGLE_API_KEY=...)
try:
    from dotenv import load_dotenv
    load_dotenv() 
except Exception:
    pass  

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
if not GEMINI_API_KEY:
    raise RuntimeError(
        "Gemini API key not found. Add GEMINI_API_KEY=your-key to a .env file "
        "next to app.py (or set it in the environment) before running app.py."
    )
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY  # what langchain-google-genai reads

# same models as notebook
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, google_api_key=GEMINI_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=GEMINI_API_KEY)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
PROMPT = ChatPromptTemplate.from_template("Context: {context}\nQuestion: {question}\nAnswer:")

app = FastAPI(title="RAG Chatbot - Level 6")
vectorstore = None
UPLOAD_DIR = pathlib.Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

class Q(BaseModel):
    question: str

def _extract_content(content):
    """Gemini may return content as str or a list of blocks; always return plain str."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(p.get("text", "") or p.get("content", "") or "")
            else:
                parts.append(getattr(p, "text", "") or getattr(p, "content", "") or "")
        return "".join(parts)
    return str(content)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global vectorstore
    try:
        # keep just the base filename to avoid path-traversal issues
        safe_name = pathlib.Path(file.filename).name
        p = UPLOAD_DIR / safe_name
        p.write_bytes(await file.read())
        docs = PyPDFLoader(str(p)).load()
        chunks = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        return {"pages": len(docs), "chunks": len(chunks), "file": safe_name}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "detail": traceback.format_exc()[-1500:]})

@app.post("/ask")
async def ask(q: Q):
    if vectorstore is None:
        return {"error": "No document uploaded. POST /upload first."}
    docs = vectorstore.similarity_search(q.question, k=3)
    ctx = "\n".join(d.page_content for d in docs)
    ans = llm.invoke(PROMPT.format(context=ctx, question=q.question)).content
    return {"answer": _extract_content(ans)}

@app.get("/health")
def health():
    return {"ok": True, "loaded": vectorstore is not None}
