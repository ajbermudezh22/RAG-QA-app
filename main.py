import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile

# Load environment variables
load_dotenv()

# --- Global State & RAG Logic ---
app_state = {"sessions": {}}

def process_document(file_path: str, page_limit: int = 200):
    """Loads, validates, and creates a RAG chain from a PDF."""
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    
    if len(pages) > page_limit:
        raise ValueError(f"Document exceeds the {page_limit}-page limit.")
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # Create prompt template using ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(
        "Context: {context}\nQuestion: {question}\nAnswer:"
    )
    
    # Create the question-answering chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the retrieval chain
    chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Wrap it to maintain compatibility with the old RetrievalQA interface
    class RetrievalQAWrapper:
        def __init__(self, chain, retriever):
            self.chain = chain
            self.retriever = retriever
        
        def __call__(self, query_dict):
            # The new chain expects {"input": ...} and returns {"answer": ..., "context": ...}
            result = self.chain.invoke({"input": query_dict["query"]})
            
            # Get source documents from retriever
            source_docs = self.retriever.invoke(query_dict["query"])
            
            # Return in the old format
            return {
                "result": result["answer"],
                "source_documents": source_docs
            }
    
    return RetrievalQAWrapper(chain, retriever)

# --- FastAPI App Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # No model loading on startup anymore
    print("API is starting up.")
    yield
    # Cleanup on shutdown
    app_state.clear()
    print("API is shutting down.")

app = FastAPI(
    title="Dynamic RAG Q&A API",
    lifespan=lifespan
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Pydantic Models ---
class UploadResponse(BaseModel):
    session_id: str
    message: str

class QuestionRequest(BaseModel):
    session_id: str
    question: str

class SourceDocument(BaseModel):
    page_content: str
    source: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]

# --- API Endpoints ---
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    try:
        # Save uploaded file to a temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Process the document and create the RAG chain
        rag_chain = process_document(tmp_file_path)
        
        # Clean up the temporary file
        os.remove(tmp_file_path)
        
        # Create a unique session ID and store the chain
        session_id = str(uuid.uuid4())
        app_state["sessions"][session_id] = rag_chain
        
        return UploadResponse(session_id=session_id, message="Document processed successfully.")

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if request.session_id not in app_state["sessions"]:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")
    
    rag_chain = app_state["sessions"][request.session_id]
    
    try:
        result = rag_chain({"query": request.question})
        
        sources = [
            SourceDocument(
                page_content=doc.page_content,
                source=doc.metadata.get("source")
            ) for doc in result["source_documents"]
        ]
        
        return AnswerResponse(answer=result["result"], sources=sources)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during query: {str(e)}")