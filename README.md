# XAI-RAG

This project is a Python-based framework for building and evaluating explainable Retrieval-Augmented Generation (RAG) systems. It is designed to work with the Statspearl and MedMCQA dataset and provides tools for explaining the behavior of the retrieval. The different explaination strategies are done in separate python notebooks.

## Features

- **Retrieval-Augmented Generation (RAG):** Implements a Multi-Hop RAG pipeline using `langchain` for document retrieval and generation.
- **Explainable Information Retrieval (XIR):** Uses SHAP to explain the ranking of retrieved documents, providing insights into the model's behavior.
- **Explainable AI (XAI):** Uses various methods to explain the Conversational QA System. Including
    - Knowledge Graph based method
    - Textual rationales
    - Evidence highlighting
- **Vector Store:** Utilizes ChromaDB for efficient vector storage and retrieval.
- **Data Handling:** Includes a data loader for the Dataset, which automatically downloads and processes the data.
    - Additionally to the MedMCQA and StatsPearl dataset we also have dataloader for the BoolQ dataset
- **Modular Structure:** The project is organized into modules for data loading, LLM interaction, and the RAG engine.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd xai-rag
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .
    source bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    If you plan to use the OpenAI API, you need to set your API key as an environment variable. Create a `.env` file in the root of the project and add the following line:
    ```
    OPENAI_API_KEY="your-api-key"
    ```

## Usage

The `notebooks/` directory contains Jupyter notebooks for exploring the explainability features of the project. Results can be directly extracted from the notebooks.

If for some reason one wants to use the modules directly please consider following short RAGEx example:

```python
from src.modules.explainers.rag_ex_explainable import RAGExExplainable, RAGExConfig
from src.modules.rag.rag_engine import RAGEngine
from src.modules.rag.multihop_rag_engine import MultiHopRAGEngine, _format_documents
from src.modules.llm.llm_client import LLMClient
from src.modules.loader.medmcqa_data_loader import MedMCQADataLoader, format_medmcqa_question
from src.modules.loader.statspearls_data_loader import StatPearlsDataLoader
from src.evaluation.evaluator import Evaluator

import tomllib

# Configuration step
config_path = project_root / "config.toml"
config = {}

if config_path.exists():
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

medmcqa_config = config.get("medmcqa") or {}
rag_config = config.get("rag") or {}
llm_config = config.get("llm") or {}

llm_model = llm_config.get("model", "gemma3:4b")
llm_provider = llm_config.get("provider", "ollama")

client = LLMClient(provider=llm_provider, model_name=llm_model)

LIMIT = medmcqa_config.get("n_qa_questions", 10)
SPLIT = medmcqa_config.get("split", "val")
PERSIST_DIR = project_root / "data" / "vector_db_statpearls"
NUM_HOPS = rag_config.get('n_hops', 2)
kg_capable_ids = medmcqa_config.get("kg_capable", [])

# Data Loading
stat_loader = StatPearlsDataLoader(root_dir=str(project_root / "data"))
documents, stats = stat_loader.setup()

rag_engine = RAGEngine(persist_dir=str(PERSIST_DIR))
rag_engine.setup(documents=documents)

multi_hop = MultiHopRAGEngine(rag_engine=rag_engine, llm_client=client, num_hops=NUM_HOPS)
evaluator = Evaluator()

# Executing
med_loader = MedMCQADataLoader()
questions = med_loader.setup(split=SPLIT, as_documents=False, limit=LIMIT, ids=kg_capable_ids)

if not questions:
    raise RuntimeError("No MedMCQA questions loaded.")

results = []
for item in questions:
    question_text = format_medmcqa_question(item)
    if not question_text:
        continue

    trace, all_documents = multi_hop.run_and_trace(question_text, extra='Only answer based on your context not your knowledge. Do not include any explanations, reasoning, or extra fields.\n Example: Final Answer: B: Housing')
    final_answer = (trace.get("final_answer") or "").strip()

    documents_for_explanation = all_documents
    context_blocks = []
    for doc in documents_for_explanation:
        content = getattr(doc, "page_content", None)
        if content is None:
            content = str(doc)
        context_blocks.append(str(content).strip())
    
    context = "\n\n".join([c for c in context_blocks if c])

    config = RAGExConfig()
    config.pertubation_depth = 1
    config.pertubation_mode = 'sentences'
    explainer = RAGExExplainable(llm_client=client, config=config)
    explanation = explainer.explain(query=question_text, answer=final_answer, context=context)
    metrics = explainer.metrics()

    perturbed_answers = []
    for result_item in explanation.get("results", []):
        for detail in result_item.get("details", []):
            perturbed_answer = detail.get("perturbed_answer")
            if perturbed_answer:
                perturbed_answers.append(perturbed_answer)

    answer_scores = evaluator.evaluate(perturbed_answers, baseline_answer=final_answer)

    results.append(
        {
            "question": question_text,
            "final_answer": final_answer,
            "trace": trace,
            "explanation": explanation,
            "metrics": metrics,
            "answer_scores": answer_scores,
            "documents": all_documents,
        }
    )
```