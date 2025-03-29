from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


def stream_graph_updates(user_input: str, graph: CompiledStateGraph):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


GRAPH_OUTPUT_FILE = "graph_output.md"

def save_graph_to_markdown(graph: CompiledStateGraph, file_name: str):
    """Save the Mermaid syntax of the graph to a Markdown file."""
    try:
        mermaid_syntax = graph.get_graph().draw_mermaid()
        with open(file_name, "w") as file:
            file.write("```mermaid\n")
            file.write(mermaid_syntax)
            file.write("\n```")
        print(f"Graph saved to '{file_name}'. You can open it in a Markdown viewer that supports Mermaid.")
    except Exception:
        print("Failed to save the graph. Ensure all dependencies are installed.")

def setup_graph() -> CompiledStateGraph:
    """Set up the state graph with nodes and edges."""
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile()

def main():
    graph = setup_graph()
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
    main()
