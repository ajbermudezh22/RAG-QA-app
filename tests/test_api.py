import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
from main import app

client = TestClient(app)

# --- Mock RAG Chain ---
class MockRAGChain:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, query_dict):
        class MockDoc:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = metadata

        return {
            "result": f"Mock answer to '{query_dict['query']}'",
            "source_documents": [
                MockDoc(page_content="Mock page content", metadata={"source": "mock_source.pdf"})
            ]
        }

# --- Tests ---

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@patch('main.process_document', return_value=MockRAGChain())
def test_upload_and_ask(mock_process_document):
    # 1. Test Upload
    dummy_pdf_content = b"%PDF-1.5 test file"
    files = {'file': ('test.pdf', dummy_pdf_content, 'application/pdf')}
    response_upload = client.post("/upload", files=files)
    
    assert response_upload.status_code == 200
    upload_data = response_upload.json()
    assert "session_id" in upload_data
    assert upload_data["message"] == "Document processed successfully."
    
    session_id = upload_data["session_id"]
    
    # 2. Test Ask with valid session
    question = "What is the capital of France?"
    response_ask = client.post("/ask", json={"session_id": session_id, "question": question})
    
    assert response_ask.status_code == 200
    ask_data = response_ask.json()
    assert ask_data["answer"] == f"Mock answer to '{question}'"
    assert len(ask_data["sources"]) == 1
    assert ask_data["sources"][0]["source"] == "mock_source.pdf"

def test_upload_not_pdf():
    dummy_txt_content = b"this is not a pdf"
    files = {'file': ('test.txt', dummy_txt_content, 'text/plain')}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert response.json() == {"detail": "Only PDF files are allowed."}

def test_ask_invalid_session():
    response = client.post("/ask", json={"session_id": "invalid-session-id", "question": "test"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found. Please upload a document first."}
