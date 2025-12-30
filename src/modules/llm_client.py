import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq



class LLMClient:
    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "llama3",
        structured_output = None,
    ):
        """
        :param provider: 'ollama' or 'openai'
        :param model_name: e.g., 'llama3', 'mistral', 'gpt-4o'
        :param structured_output: Return parsed Pydantic output when invoking the LLM.
        """
        self.provider = provider
        self.model_name = model_name
        self.structured_output = structured_output
        self._base_llm: Optional[BaseChatModel] = None
        self._structured_llm: Optional[Runnable] = None

    def _init_base_llm(self) -> BaseChatModel:
        """Lazily initialize the underlying chat model."""
        if self._base_llm:
            return self._base_llm

        if self.provider == "ollama":
            print(f"Connecting to local Ollama ({self.model_name})...")
            self._base_llm = ChatOllama(
                model=self.model_name,
                reasoning=False,
                temperature=0
            )
        elif self.provider == "groq":
            print(f"Connecting to Groq ({self.model_name})...")
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set.")

            self._llm = ChatGroq(
                model=self.model_name,
                temperature=0,
                groq_api_key=api_key,
            )

        elif self.provider == "openai":
            print(f"Connecting to OpenAI ({self.model_name})...")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            self._base_llm = ChatOpenAI(
                model=self.model_name,
                temperature=0,
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return self._base_llm

    def get_llm(self) -> Runnable | BaseChatModel:
        """
        Returns the LangChain ChatModel (optionally wrapped to enforce structured output).
        By default, structured output is enabled to match the JSON format defined in the prompt.
        """
        wants_structured = self.structured_output is not None
        base_llm = self._init_base_llm()

        if wants_structured:
            if self._structured_llm:
                return self._structured_llm
            self._structured_llm = base_llm.with_structured_output(self.structured_output, include_raw=True)
            return self._structured_llm

        return base_llm

    def _create_next_query_prompt(self, initial_query: str, context: str) -> str:
        """Creates the prompt to generate the next search query."""

        return f"""
            You are a research assistant. Your goal is to break down a complex question into a series of search queries.
            Based on the original question and the context gathered so far, generate the next specific search query to find the missing information.
            Do not repeat queries. Generate only the new search query and nothing else.

            Original Question: {initial_query}

            Context Gathered So Far:
            {context}

            Next Search Query:
            """
    
    def _create_final_answer_prompt(self, initial_query: str, context: str, extra: str = '') -> str:
        """Creates the prompt to generate the final answer from the accumulated context."""
        
        return f"""
            You are a helpful assistant. Using all the provided context from multiple search hops, you must answer the original question.
            If the context is not sufficient, state that you cannot answer the question with the given information.

            Original Question: {initial_query}

            Full Context from all search hops:
            {context}

            {extra}
            Final Answer:
            """