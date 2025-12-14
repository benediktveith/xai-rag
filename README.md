# XAI-RAG

This project is a Python-based framework for building and evaluating explainable Retrieval-Augmented Generation (RAG) systems. It is designed to work with the HotpotQA dataset and provides tools for explaining the behavior of the retrieval model using SHAP.

## Features

- **Retrieval-Augmented Generation (RAG):** Implements a RAG pipeline using `langchain` for document retrieval and generation.
- **Explainable AI (XAI):** Uses SHAP to explain the ranking of retrieved documents, providing insights into the model's behavior.
- **Flexible LLM Support:** Supports both local models via Ollama (e.g., Llama 3, Mistral) and proprietary models via OpenAI API (e.g., GPT-4o).
- **Vector Store:** Utilizes ChromaDB for efficient vector storage and retrieval.
- **Data Handling:** Includes a data loader for the HotpotQA dataset, which automatically downloads and processes the data.
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

The main entry point for the project is `main.py`. You can run it from the command line:

```bash
python main.py
```

The project can be configured to use different LLMs and retrieval settings. See the `src/modules/` directory for more details on the available options.

The `notebooks/` directory contains a Jupyter notebook for exploring the explainability features of the project.

## Project Structure

```
.
├── data/                    # Data files
├── notebooks/               # Jupyter notebooks
│   └── DeepShapNRM.ipynb    # Notebook for SHAP explanations
├── src/                     # Source code
│   ├── modules/
│   │   ├── data_loader.py   # Data loader for HotpotQA
│   │   ├── llm_client.py    # Client for LLM interaction
│   │   └── rag_engine.py    # RAG engine implementation
│   └── evaluation/          # Evaluation scripts
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
└── setup.py                 # Setup script
```

## Dependencies

The project uses the following major libraries:

- `langchain`
- `chromadb`
- `shap`
- `torch`
- `transformers`
- `openai`
- `ollama`

For a full list of dependencies, see the `requirements.txt` file.