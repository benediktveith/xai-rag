from __future__ import annotations
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


class StrictLLMFormatter:
    def build_messages(
        self,
        question: str,
        retrieval_trace: List[Dict[str, Any]],
        answer_guidance: Optional[str] = "Antworte prägnant und auf Deutsch.",
    ) -> List[BaseMessage]:
        system_content = (
            "Du bist ein erklärbarer RAG Assistent. Du darfst ausschließlich Informationen aus den Kontext Chunks verwenden. "
            "Du darfst nur Chunk IDs zitieren, die im Kontextblock exakt vorkommen. "
            "Wenn du keine Evidenz findest, antworte genau mit: Nicht genügend Evidenz."
        )

        if not retrieval_trace:
            context_block = "Keine Chunks vorhanden."
        else:
            lines = []
            for item in retrieval_trace:
                cid = str(item.get("id", "chunk"))
                title = (item.get("title") or "Ohne Titel").strip()
                score = item.get("score", None)
                score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
                content = item.get("content", "") or ""
                content = content.replace("\n", " ").strip()
                lines.append(f"[{cid}] score={score_str} title={title} :: {content}")
            context_block = "\n".join(lines)

        user_content = (
            f"Frage:\n{question}\n\n"
            f"Kontext Chunks:\n{context_block}\n\n"
            "Format strikt einhalten:\n"
            "Antwort: <kurz, oder Nicht genügend Evidenz>\n"
            "Begründung:\n"
            "- [chunk-id] <kurzer Grund>\n"
        )

        if answer_guidance:
            user_content += f"\nHinweis: {answer_guidance}\n"

        return [SystemMessage(content=system_content), HumanMessage(content=user_content)]
