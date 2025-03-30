from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command



class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


def stream_graph_updates(user_input: str, graph: CompiledStateGraph):
    """Stream updates from the graph based on user input."""
    config = {"configurable": {"thread_id": "1"}}
    events = graph.stream({"messages": [{"role": "user", "content": user_input}]}, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    state = graph.get_state(config)
    if state.tasks:
        if (state.tasks[0].interrupts):
            print(state.tasks[0].interrupts)
            human_response = input("Tell the AI something... ")
            human_command = Command(resume={"data": human_response})

            events = graph.stream(human_command, config, stream_mode="values")
            for event in events:
                if "messages" in event:
                    event["messages"][-1].pretty_print() 


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

graph_builder = StateGraph(State)