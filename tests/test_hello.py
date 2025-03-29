import pytest
from unittest.mock import MagicMock
from src.chatbot import create_chatbot_func
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from src.graph import State

@pytest.fixture
def mock_llm():
    """Fixture for a mocked ChatGoogleGenerativeAI instance."""
    mock = MagicMock(spec=ChatGoogleGenerativeAI)
    mock.invoke.return_value = {"content": "Hello, world!"}
    return mock

@pytest.fixture
def fresh_graph_builder():
    """Fixture for a fresh StateGraph instance."""
    return StateGraph(state_schema=State)

def test_setup_graph(mock_llm, fresh_graph_builder):
    """Test the setup_graph function."""
    def setup_graph(builder, chatbot_func):
        builder.add_node("chatbot", chatbot_func)
        builder.add_edge(START, "chatbot")
        builder.add_edge("chatbot", END)

    chatbot_func = create_chatbot_func(mock_llm)
    setup_graph(fresh_graph_builder, chatbot_func)
    graph = fresh_graph_builder.compile()
    assert isinstance(graph, CompiledStateGraph)
    assert "chatbot" in graph.get_graph().nodes
    edges = [(edge.source, edge.target) for edge in graph.get_graph().edges]
    assert edges == [(START, "chatbot"), ("chatbot", END)]
