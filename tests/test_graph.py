import pytest
from unittest.mock import MagicMock, patch, mock_open
from src.graph import save_graph_to_markdown, State
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

@pytest.fixture
def fresh_graph_builder():
    """Fixture for a fresh StateGraph instance."""
    return StateGraph(state_schema=State)  # Provide the state_schema explicitly

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

def test_setup_graph(fresh_graph_builder):
    """Test the setup of the StateGraph."""
    # Define a helper function for adding nodes and edges
    def setup_graph(builder):
        builder.add_node("chatbot", lambda state: {"messages": [{"content": "test"}]})
        builder.add_edge(START, "chatbot")
        builder.add_edge("chatbot", END)

    setup_graph(fresh_graph_builder)
    graph = fresh_graph_builder.compile()
    assert isinstance(graph, CompiledStateGraph)
    assert "chatbot" in graph.get_graph().nodes
