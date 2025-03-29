import pytest
from unittest.mock import MagicMock
from src.tools import BasicToolNode, ToolMessage

@pytest.fixture
def mock_tool():
    """Fixture for a mock tool."""
    tool = MagicMock()
    tool.name = "mock_tool"
    tool.invoke.return_value = {"result": "mock_result"}
    return tool

@pytest.fixture
def tool_node(mock_tool):
    """Fixture for the BasicToolNode."""
    return BasicToolNode(tools=[mock_tool])

def test_tool_node_with_valid_input(tool_node, mock_tool):
    """Test tool_node with valid input."""
    mock_message = MagicMock()
    mock_message.tool_calls = [{"name": "mock_tool", "args": {}, "id": "123"}]
    inputs = {"messages": [mock_message]}

    result = tool_node(inputs)

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0].name == "mock_tool"
    assert result["messages"][0].tool_call_id == "123"
    assert result["messages"][0].content == '{"result": "mock_result"}'
    mock_tool.invoke.assert_called_once_with({})

def test_tool_node_with_no_messages(tool_node):
    """Test tool_node with no messages in input."""
    inputs = {"messages": []}
    with pytest.raises(ValueError, match="No message found in input"):
        tool_node(inputs)
