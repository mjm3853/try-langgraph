from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv

from src.chatbot import create_chatbot_func
from src.graph import (
    stream_graph_updates,
    save_graph_to_markdown,
    GRAPH_OUTPUT_FILE,
    graph_builder,
)

load_dotenv()


def setup_graph(chatbot_func) -> CompiledStateGraph:
    """Set up the state graph with nodes and edges."""
    graph_builder.add_node("chatbot", chatbot_func)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile()


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
        except:  # noqa: E722
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input, graph)
            break


if __name__ == "__main__":
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
    chatbot_func = create_chatbot_func(llm)
    main(chatbot_func)
