from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class ExplainableModule(ABC):
    @abstractmethod
    def explain(self, query: str, answer: str, documents: Sequence[Any]):
        raise NotImplementedError
