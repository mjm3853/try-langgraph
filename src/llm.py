from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from src.settings import GEMINI_MODEL

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)