from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, Sequence


class ExplainableModule(ABC):
    def __init__(self) -> None:
        self._metrics: Dict[str, Any] = {
            "duration_seconds": 0.0,
            "steps": 0,
        }

    def _reset_metrics(self) -> None:
        self._metrics["duration_seconds"] = 0.0
        self._metrics["steps"] = 0

    def _increment_steps(self, count: int = 1) -> None:
        self._metrics["steps"] += count

    def _llm_call(self, llm: Any, *args: Any, **kwargs: Any) -> Any:
        self._increment_steps()
        return llm.invoke(*args, **kwargs)

    def explain(self, *args: Any, **kwargs: Any) -> Any:
        self._reset_metrics()
        started = time.perf_counter()
        try:
            return self._explain(*args, **kwargs)
        finally:
            self._metrics["duration_seconds"] = time.perf_counter() - started

    @abstractmethod
    def _explain(self, query: str, answer: str, documents: Sequence[Any]) -> Any:
        raise NotImplementedError

    def metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)
