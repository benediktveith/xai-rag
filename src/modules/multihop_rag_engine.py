from typing import Any, Dict, List
from .rag_engine import RAGEngine
from .llm_client import LLMClient

class MultiHopRAGEngine:
    """
    A wrapper that orchestrates a multi-hop RAG process, inspired by the DSPy multi-hop tutorial.
    It uses an LLM to generate new queries at each hop to "dig deeper" for the answer.
    """

    def __init__(self, rag_engine: RAGEngine, llm_client: LLMClient, num_hops: int = 2):
        """
        Initializes the multi-hop engine. 
        
        :param rag_engine: An initialized instance of your RAGEngine for retrieval.
        :param llm_client: An initialized instance of your LLMClient for generation.
        :param num_hops: The number of retrieval-generation hops to perform. Defaults to 2.
        """
        if not rag_engine._retriever:
            raise RuntimeError("The provided RAGEngine must be setup() before use.")
            
        self.rag_engine = rag_engine
        self.llm = llm_client.get_llm()
        self.num_hops = num_hops

    def run_and_trace(self, initial_query: str) -> Dict[str, Any]:
        """
        Executes the multi-hop RAG process for a given query and returns a detailed trace.

        :param initial_query: The user's starting question.
        :return: A dictionary containing the detailed execution trace.
        """
        trace = {
            "initial_query": initial_query,
            "hops": [],
            "final_answer": None
        }

        current_query = initial_query
        context_so_far = ""

        print(f"--- Starting Multi-Hop Search for: '{initial_query}' ---")

        for i in range(self.num_hops):
            hop_number = i + 1
            print(f"\n[ Hop {hop_number} ]")
            print(f"Executing search with query: '{current_query}'")

            # 1. Retrieve documents for the current query
            retrieved_docs = self.rag_engine.retrieve_documents(current_query)
            if not retrieved_docs:
                print("No documents found for this hop. Stopping.")
                break
            
            # For simplicity, we'll focus on the top document for the trace
            top_doc = retrieved_docs[0]

            # 2. Store the results of this hop in our trace
            hop_info = {
                "hop_number": hop_number,
                "query_for_this_hop": current_query,
                "retrieved_document_for_this_hop": top_doc
            }
            trace["hops"].append(hop_info)
            
            # 3. Accumulate context for the next steps
            context_so_far += f"\n\n--- Context from Hop {hop_number} (Query: {current_query}) ---\n{top_doc.page_content}"

            # 4. If not the last hop, generate the next query
            if i < self.num_hops - 1:
                prompt = self._create_next_query_prompt(initial_query, context_so_far)
                print("Generating next query...")
                
                llm_response = self.llm.invoke(prompt)
                next_query = llm_response.content.strip()
                
                current_query = next_query

        # 5. After all hops, generate the final answer using all gathered context
        print("\nGenerating final answer...")
        answer_prompt = self._create_final_answer_prompt(initial_query, context_so_far)
        
        final_answer_response = self.llm.invoke(answer_prompt)
        final_answer = final_answer_response.content.strip()

        trace["final_answer"] = final_answer
        print(f"--- Multi-Hop Search Complete. Final Answer: {final_answer} ---")

        return trace

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

    def _create_final_answer_prompt(self, initial_query: str, context: str) -> str:
        """Creates the prompt to generate the final answer from the accumulated context."""
        
        return f"""
            You are a helpful assistant. Using all the provided context from multiple search hops, you must answer the original question.
            If the context is not sufficient, state that you cannot answer the question with the given information.

            Original Question: {initial_query}

            Full Context from all search hops:
            {context}

            Final Answer:
            """
