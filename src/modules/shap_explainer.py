import shap
import numpy

from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from base_explainer import BaseExplainer

class ShapExplainer(BaseExplainer):
    """
    We need to distinguish between cases:
    1. Explaining a simple cos-similarity retrieval (Focus on this!!!)
    2. Explaining a NRM


    Function for Evaluating the Explanation by performing pertubation.

    """

    def __init__(self):
        super().__init__()


    def calculate_similarity(self, query: str, document: str):
        pass

    
    def explain(self):

        pass

    def score(self, retriever, queries: List[str]):
        """
        Caculates the AOPC Metric for the Shap Explainer.
        ...
        """

        score = 0

        for query in queries:

            documents = retriever.invoke(query)
            top_document = retriever

        return score