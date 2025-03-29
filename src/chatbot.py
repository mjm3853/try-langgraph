from langchain_google_genai import ChatGoogleGenerativeAI
from src.graph import State


def chatbot(state: State, llm: ChatGoogleGenerativeAI):
    """Chatbot node function that uses the provided LLM."""
    return {"messages": [llm.invoke(state["messages"])]}


def create_chatbot_func(llm: ChatGoogleGenerativeAI):
    """Create a chatbot function with the given LLM."""
    return lambda state: chatbot(state, llm)
