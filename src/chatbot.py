from langchain_google_genai import ChatGoogleGenerativeAI
from src.graph import State


def chatbot(state: State, llm: ChatGoogleGenerativeAI):
    """Chatbot node function that uses the provided LLM."""
    message = llm.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


def create_chatbot_func(llm: ChatGoogleGenerativeAI):
    """Create a chatbot function with the given LLM."""
    return lambda state: chatbot(state, llm)
