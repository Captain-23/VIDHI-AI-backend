from langchain_groq import ChatGroq
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import BadRequestError, RateLimitError

class GroqLLM:
    """
    Wrapper so nodes can call llm.invoke(prompt)
    """

    def __init__(self):
        load_dotenv()
        project_dir = Path(__file__).resolve().parents[2]
        for candidate in (project_dir / ".env", project_dir.parent / ".env"):
            if candidate.exists():
                load_dotenv(candidate)

        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing from environment variables.")

        # Create underlying Groq LLM
        self.llm = ChatGroq(
            api_key=self.api_key,
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "1200"))
        )

    def invoke(self, prompt: str):
        """
        Make Groq behave like other LangChain LLMs.
        """
        for attempt in range(3):
            try:
                return self.llm.invoke(prompt)
            except (RateLimitError, BadRequestError) as exc:
                if attempt == 2:
                    raise
                if "Request too large" in str(exc):
                    raise
                time.sleep(20)

    def get_llm(self):
        """
        Still keep this method for compatibility.
        """
        return self.llm
