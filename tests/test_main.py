# backend/tests/test_main.py
import os
import sys
from fastapi.testclient import TestClient
import pytest

# Adjust the path so that main.py can be imported.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from main import app

client = TestClient(app)

def test_query_endpoint():
    payload = {"query": "What are the emerging market trends?"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert "sentiment" in data
    assert "topics" in data
