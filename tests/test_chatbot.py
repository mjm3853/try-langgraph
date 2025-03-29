import pytest
from unittest.mock import MagicMock
from src.chatbot import chatbot, create_chatbot_func
from langchain_google_genai import ChatGoogleGenerativeAI

@pytest.fixture
def mock_llm():
    """Fixture for a mocked ChatGoogleGenerativeAI instance."""
    mock = MagicMock(spec=ChatGoogleGenerativeAI)
    mock.invoke.return_value = {"content": "Hello, world!"}
    return mock

@pytest.fixture
def sample_state():
    """Fixture for a sample state."""
    return {"messages": [{"role": "user", "content": "Hi"}]}

def test_chatbot(mock_llm, sample_state):
    """Test the chatbot function."""
    result = chatbot(sample_state, mock_llm)
    mock_llm.invoke.assert_called_once_with(sample_state["messages"])
    assert "messages" in result
    assert result["messages"][0]["content"] == "Hello, world!"

def test_create_chatbot_func(mock_llm, sample_state):
    """Test the create_chatbot_func function."""
    chatbot_func = create_chatbot_func(mock_llm)
    result = chatbot_func(sample_state)
    mock_llm.invoke.assert_called_once_with(sample_state["messages"])
    assert "messages" in result
    assert result["messages"][0]["content"] == "Hello, world!"
