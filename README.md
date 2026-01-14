# XAI-RAG

This project is a Python-based framework for building and evaluating explainable Retrieval-Augmented Generation (RAG) systems. It is designed to work with the Statspearl and MedMCQA dataset and provides tools for explaining the behavior of the retrieval.

## Features

- **Retrieval-Augmented Generation (RAG):** Implements a Multi-Hop RAG pipeline using `langchain` for document retrieval and generation.
- **Explainable Information Retrieval (XIR):** Uses SHAP to explain the ranking of retrieved documents, providing insights into the model's behavior.
- **Explainable AI (XAI):** Uses various methods to explain the Conversational QA System.
- **Vector Store:** Utilizes ChromaDB for efficient vector storage and retrieval.
- **Data Handling:** Includes a data loader for the Dataset, which automatically downloads and processes the data.
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

The `notebooks/` directory contains Jupyter notebooks for exploring the explainability features of the project.