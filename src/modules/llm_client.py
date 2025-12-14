import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

class LLMClient:
    def __init__(self, provider: str = "ollama", model_name: str = "llama3"):
        """
        :param provider: 'ollama' or 'openai'
        :param model_name: e.g., 'llama3', 'mistral', 'gpt-4o'
        """
        self.provider = provider
        self.model_name = model_name
        self._llm = None

    def get_llm(self) -> BaseChatModel:
        """Returns the initialized LangChain ChatModel"""
        if self._llm:
            return self._llm

        if self.provider == "ollama":
            print(f"🔌 Connecting to local Ollama ({self.model_name})...")
            self._llm = ChatOllama(
                model=self.model_name,
                temperature=0  # Deterministic for explainability experiments
            )
        
        elif self.provider == "openai":
            print(f"Connecting to OpenAI ({self.model_name})...")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            self._llm = ChatOpenAI(
                model=self.model_name,
                temperature=0,
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return self._llm