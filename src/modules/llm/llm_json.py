from __future__ import annotations
import json
import re
from typing import Any, Dict, Optional


class LLMJSON:
    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        text = text.strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None

        blob = m.group(0)
        try:
            return json.loads(blob)
        except Exception:
            return None
