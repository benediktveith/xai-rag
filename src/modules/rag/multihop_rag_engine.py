from typing import Any, Dict, List
from .rag_engine import RAGEngine
from ..llm.llm_client import LLMClient
from ..explainers.cot_explainable import _format_documents

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
        self._llm_client = llm_client
        self.num_hops = num_hops

    def run_and_trace(self, initial_query: str, extra: str = '') -> Dict[str, Any]:
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

        all_documents = []
        retrieved_document_ids = []
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
            lowest_doc = retrieved_docs[-1] # also track the lowest ranked document
            context_docs = retrieved_docs[:min(4,len(retrieved_docs))]

            # Deduplicate documents
            new_docs = [doc for doc in retrieved_docs if doc.id not in retrieved_document_ids]
            all_documents.extend(new_docs)
            retrieved_document_ids.extend([doc.id for doc in new_docs])

            # 2. Store the results of this hop in our trace
            hop_info = {
                "hop_number": hop_number,
                "query_for_this_hop": current_query,
                "highest_ranked_document": top_doc,
                "lowest_ranked_document" : lowest_doc,       # added for comparison to low ranked documents
                "context_documents" : context_docs
            }
            trace["hops"].append(hop_info)

            # 3. Accumulate context for the next steps
            context_so_far += f"\n\n {_format_documents([context_docs], current_query, hop_number)}"

            # 4. If not the last hop, generate the next query
            if i < self.num_hops - 1:
                print("Generating next query...")
                prompt = self._llm_client._create_next_query_prompt(initial_query, context_so_far)
                
                # 5. Check if the LLM considered the context as sufficient, then STOP hop process
                if prompt == "STOP":
                    print(f"Context sufficient at Hop: {hop_number} of {self.num_hops}")
                    break
                
                llm_response = self.llm.invoke(prompt)
                next_query = llm_response.content.strip()
                
                current_query = next_query

        # 6. After all hops, generate the final answer using all gathered context
        print("\nGenerating final answer...")
        answer_prompt = self._llm_client._create_final_answer_prompt(initial_query, context_so_far, extra)
        
        final_answer_response = self.llm.invoke(answer_prompt)
        final_answer = final_answer_response.content.strip()

        trace["final_answer"] = final_answer
        print('')
        print(f"--- Multi-Hop Search Complete. Final Answer: {final_answer} ---")
        print(f"--- Multi-Hop Context: {context_so_far} ---")

        return trace, all_documents
