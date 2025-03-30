from langgraph.prebuilt import ToolNode

from src.tools.search import search_tool
from src.tools.human import human_assistance

tools = [search_tool, human_assistance]
tool_node = ToolNode(tools=[search_tool, human_assistance])