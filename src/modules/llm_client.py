import os
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


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
            self._llm = ChatOpenAI(
                model=self.model_name,
                temperature=0,
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return self._llm

    def build_explainable_messages(
        self,
        question: str,
        retrieval_trace: List[Dict[str, Any]],
        answer_guidance: Optional[str] = "Antworte prägnant und auf Deutsch.",
    ) -> List[BaseMessage]:
        """
        Build System/User messages that force the LLM to explain which chunks it used.
        
        Args:
            question: Original user question.
            retrieval_trace: Output of RAGEngine.retrieve_with_scores.
            answer_guidance: Optional stylistic guidance for the answer.
        """
        system_content = (
            "Du bist ein erklärbarer RAG-Assistent. Nutze ausschließlich die "
            "bereitgestellten Kontext-Chunks. Gib zuerst eine kurze Antwort. "
            "Führe danach eine Begründung auf, die auf Chunk-IDs und Titel verweist. "
            "Wenn kein Kontext vorliegt, sage das explizit."
        )

        if not retrieval_trace:
            context_block = "Keine Chunks vorhanden."
        else:
            lines = []
            for item in retrieval_trace:
                chunk_id = item.get("id", "chunk")
                title = item.get("title") or "Ohne Titel"
                score = item.get("score")
                score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
                preview = item.get("preview") or item.get("content", "")
                preview = preview.replace("\n", " ")
                lines.append(f"[{chunk_id}] (score={score_str}) {title} :: {preview}")
            context_block = "\n".join(lines)

        user_content = (
            f"Frage: {question}\n\n"
            "Kontext-Chunks:\n"
            f"{context_block}\n\n"
            "Aufgabe:\n"
            "- Antworte knapp basierend auf den Chunks.\n"
            "- Liste danach die genutzten Chunk-IDs und warum sie relevant sind.\n"
            "- Formatiere so:\n"
            "Antwort: ...\n"
            "Begründung:\n"
            "- [chunk-1] Titel: Grund\n"
            "- [chunk-2] Titel: Grund\n"
        )
        if answer_guidance:
            user_content += f"\nHinweis: {answer_guidance}"

        return [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ]
