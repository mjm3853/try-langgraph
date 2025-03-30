from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition

from src.llm import llm
from src.memory import memory
from src.tools.toolbox import tools, tool_node
from src.chatbot import create_chatbot_func
from src.graph import (
    stream_graph_updates,
    save_graph_to_markdown,
    GRAPH_OUTPUT_FILE,
    graph_builder
)


def setup_graph(chatbot_func) -> CompiledStateGraph:
    """Set up the state graph with nodes and edges."""
    graph_builder.add_node("chatbot", chatbot_func)
    # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
    # it is fine directly responding. This conditional routing defines the main agent loop.
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges("chatbot",tools_condition)
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    return graph_builder.compile(checkpointer=memory)


def main(chatbot_func):
    """Main function to run the application."""
    graph = setup_graph(chatbot_func)
    save_graph_to_markdown(graph, GRAPH_OUTPUT_FILE)

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input, graph)
        except Exception as e:  # noqa: E722
            # Print detailed exception information
            print(f"An error occurred: {e}")
            print(f"Exception type: {type(e).__name__}")
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("Something went wrong!")
            break


if __name__ == "__main__":
    llm_with_tools = llm.bind_tools(tools)
    chatbot_func = create_chatbot_func(llm_with_tools)
    main(chatbot_func)
