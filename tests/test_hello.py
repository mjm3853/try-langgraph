import pytest
from unittest.mock import MagicMock, patch, mock_open
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import warnings  # Import warnings module
from hello import chatbot, save_graph_to_markdown, setup_graph, State
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

# Suppress specific DeprecationWarning from pydantic
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r".*pydantic\.v1\.typing",
)

@pytest.fixture
def mock_llm():
    """Fixture for a mocked ChatGoogleGenerativeAI instance."""
    mock = MagicMock(spec=ChatGoogleGenerativeAI)
    mock.invoke.return_value = {"content": "Hello, world!"}
    return mock

@pytest.fixture
def graph_builder():
    """Fixture for a StateGraph instance."""
    return StateGraph(State)

def test_chatbot(mock_llm):
    """Test the chatbot function."""
    state = {"messages": [{"role": "user", "content": "Hi"}]}
    result = chatbot(state, mock_llm)
    mock_llm.invoke.assert_called_once_with(state["messages"])
    assert "messages" in result
    assert result["messages"][0]["content"] == "Hello, world!"

@patch("builtins.open", new_callable=mock_open)
def test_save_graph_to_markdown(mock_open):
    """Test saving the graph to a Markdown file."""
    mock_graph = MagicMock(spec=CompiledStateGraph)
    mock_graph.get_graph().draw_mermaid.return_value = "graph TD; A-->B;"
    save_graph_to_markdown(mock_graph, "test_output.md")
    mock_open.assert_called_once_with("test_output.md", "w")
    mock_open().write.assert_any_call("```mermaid\n")
    mock_open().write.assert_any_call("graph TD; A-->B;")
    mock_open().write.assert_any_call("\n```")

def test_setup_graph(graph_builder, mock_llm):
    """Test the setup_graph function."""
    graph = setup_graph(graph_builder, mock_llm)
    assert isinstance(graph, CompiledStateGraph)
    nodes = graph.get_graph().nodes
    assert "chatbot" in nodes
