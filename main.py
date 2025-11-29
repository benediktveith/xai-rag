import os
from dotenv import load_dotenv
import json
import requests
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# Configuration
DATA_DIR = Path("data")
HOTPOTQA_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json"
DATASET_FILE = DATA_DIR / "hotpot_test_fullwiki_v1.json"


def get_faiss_index_path(provider: str) -> str:
    """Get provider-specific FAISS index path"""
    return f"faiss_hotpotqa_index_{provider}"


def setup_api_keys(provider: str):
    """Setup API keys based on the selected provider"""
    if provider == "ollama":
        return

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it using: export OPENAI_API_KEY='your-key-here'"
            )
        return api_key
    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please set it using: export GOOGLE_API_KEY='your-key-here'"
            )
        return api_key
    else:
        raise ValueError(f"Unknown provider: {provider}")


def initialize_models(provider: str) -> Tuple[Any, Any]:
    """
    Initialize embeddings and LLM based on the selected provider
    
    Args:
        provider: Either 'openai', 'gemini', or 'ollama'
    
    Returns:
        Tuple of (embeddings, llm)
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        print(f"✓ Initialized OpenAI models (GPT-3.5-turbo + text-embedding-3-small)")
        
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, convert_system_message_to_human=True)
        print(f"✓ Initialized Google Gemini models (gemini-pro + embedding-001)")
        
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        
        llm = ChatOllama(model="llama3.2", temperature=0)
        print(f"✓ Initialized Ollama models (llama3.2 + nomic-embed-text)")
        
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose 'openai', 'gemini', or 'ollama'.")
    
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="llama3.2")

    return embeddings, llm


def download_hotpotqa():
    """Download HotPotQA test dataset if not already downloaded"""
    DATA_DIR.mkdir(exist_ok=True)
    
    if DATASET_FILE.exists():
        print(f"✓ HotPotQA dataset already exists at {DATASET_FILE}")
        return
    
    print(f"Downloading HotPotQA test dataset from {HOTPOTQA_URL}...")
    print("This may take a few minutes (~50MB)...")
    
    response = requests.get(HOTPOTQA_URL, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(DATASET_FILE, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\rProgress: {progress:.1f}%", end='')
    
    print(f"\n✓ Dataset downloaded successfully to {DATASET_FILE}")


def load_hotpotqa_data() -> List[Dict[str, Any]]:
    """Load HotPotQA dataset from JSON file"""
    print(f"Loading HotPotQA data from {DATASET_FILE}...")
    
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} questions from HotPotQA")
    return data


def create_documents_from_hotpotqa(data: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert HotPotQA data to LangChain Document objects
    
    HotPotQA format:
    {
        "_id": "question_id",
        "question": "...",
        "answer": "...",
        "supporting_facts": [["title", sent_id], ...],
        "context": [
            ["title", ["sent1", "sent2", ...]],
            ...
        ]
    }
    """
    documents = []
    
    print("Converting HotPotQA contexts to documents...")
    
    for item in data:
        question_id = item.get("_id", "unknown")
        question = item.get("question", "")
        answer = item.get("answer", "")
        context = item.get("context", [])
        supporting_facts = item.get("supporting_facts", [])
        
        # Convert supporting facts to a dict for quick lookup
        supporting_dict = {}
        for title, sent_id in supporting_facts:
            if title not in supporting_dict:
                supporting_dict[title] = set()
            supporting_dict[title].add(sent_id)
        
        # Process each context article
        for title, sentences in context:
            # Join sentences to create document content
            content = " ".join(sentences)
            
            # Check if this article contains supporting facts
            is_supporting = title in supporting_dict
            
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "title": title,
                    "question_id": question_id,
                    "question": question,
                    "answer": answer,
                    "is_supporting": is_supporting,
                    "source": "hotpotqa"
                }
            )
            documents.append(doc)
    
    #documents = documents[:1000]
    print(f"✓ Created {len(documents)} documents from HotPotQA contexts")
    return documents


def create_or_load_vectorstore(documents: List[Document], embeddings: Any, faiss_index_path: str) -> FAISS:
    """
    Create FAISS vector store or load from disk if it exists
    This ensures we only need to create embeddings once
    
    Args:
        documents: List of documents to index
        embeddings: Embedding model (OpenAI or Gemini)
        faiss_index_path: Path to save/load the FAISS index
    """
    if Path(faiss_index_path).exists():
        print(f"✓ Loading existing FAISS index from {faiss_index_path}...")
        vectorstore = FAISS.load_local(
            faiss_index_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✓ Loaded FAISS index with {vectorstore.index.ntotal} vectors")
        return vectorstore
    
    print("Creating new FAISS index...")
    print("This will take several minutes as we need to generate embeddings for all documents...")
    
    # Split documents into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400,
        add_start_index=True
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"✓ Split {len(documents)} documents into {len(splits)} chunks")
    
    # Create FAISS index with chunked processing to avoid overwhelming Ollama
    print("\nGenerating embeddings with chunked processing...")
    print("(This prevents server overload and provides progress feedback)\n")
    
    vectorstore = create_vectorstore_chunked(splits, embeddings, chunk_size=500)
    
    # Save for future use
    print(f"\nSaving FAISS index to {faiss_index_path}...")
    vectorstore.save_local(faiss_index_path)
    print(f"✓ FAISS index created and saved with {vectorstore.index.ntotal} vectors")
    
    return vectorstore


def create_vectorstore_chunked(documents: List[Document], embeddings: Any, chunk_size: int = 500) -> FAISS:
    """
    Create FAISS vectorstore in chunks to avoid overwhelming the embedding server
    
    This is especially important for Ollama which can crash when processing
    too many embeddings at once.
    
    Args:
        documents: List of document chunks to embed
        embeddings: Embedding model
        chunk_size: Number of documents to process per batch
    
    Returns:
        FAISS vectorstore with all documents
    """
    try:
        from tqdm import tqdm
    except ImportError:
        print("Warning: tqdm not installed. Install with 'pip install tqdm' for progress bar.")
        tqdm = None
    
    total_docs = len(documents)
    print(f"Processing {total_docs} documents in batches of {chunk_size}...")
    
    # Create initial vectorstore with first batch
    print(f"\nProcessing initial batch (0-{min(chunk_size, total_docs)})...")
    vectorstore = FAISS.from_documents(
        documents[:chunk_size], 
        embeddings,
        normalize_L2=True
    )
    print(f"✓ Initial batch complete ({vectorstore.index.ntotal} vectors)")
    
    # Process remaining documents in batches
    if total_docs > chunk_size:
        batches = range(chunk_size, total_docs, chunk_size)
        iterator = tqdm(batches, desc="Embedding batches") if tqdm else batches
        
        for i in iterator:
            batch_end = min(i + chunk_size, total_docs)
            batch = documents[i:batch_end]
            
            try:
                # Create temporary vectorstore for this batch
                temp_store = FAISS.from_documents(batch, embeddings, normalize_L2=True)
                
                # Merge into main vectorstore
                vectorstore.merge_from(temp_store)
                
                if not tqdm:
                    print(f"✓ Processed batch {i//chunk_size + 1}: {i}-{batch_end} ({vectorstore.index.ntotal} total vectors)")
                    
            except Exception as e:
                print(f"\n⚠️  Error processing batch {i}-{batch_end}: {e}")
                print("Continuing with remaining batches...")
                continue
    
    print(f"\n✓ Chunked processing complete: {vectorstore.index.ntotal} total vectors")
    return vectorstore


def create_conversational_chain(vectorstore: FAISS, llm: Any) -> ConversationalRetrievalChain:
    """
    Create ConversationalRetrievalChain with memory
    
    This chain:
    1. Reformulates the current question based on chat history
    2. Retrieves relevant documents
    3. Generates an answer using the documents and chat history
    """
    # Setup memory to track conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    
    # Create retriever from vectorstore
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
    )
    
    # Create conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    return chain


def run_conversation(chain: ConversationalRetrievalChain, provider: str):
    """
    Interactive conversation loop
    """
    print("\n" + "="*70)
    print(f"Conversational RAG with HotPotQA - Ready! (Provider: {provider.upper()})")
    print("="*70)
    print("\nYou can ask questions about the HotPotQA dataset.")
    print("The system will remember previous questions and context.")
    print("\nCommands:")
    print("  - Type your question and press Enter")
    print("  - Type 'quit' or 'exit' to end the conversation")
    print("  - Type 'clear' to clear conversation history")
    print("="*70 + "\n")
    
    while True:
        # Get user input
        question = input("\n💭 You: ").strip()
        
        if not question:
            continue
        
        # Check for commands
        if question.lower() in ['quit', 'exit']:
            print("\n👋 Goodbye!")
            break
        
        if question.lower() == 'clear':
            chain.memory.clear()
            print("🔄 Conversation history cleared!")
            continue
        
        # Get response from chain
        try:
            print("\n🤔 Thinking...\n")
            result = chain({"question": question})
            
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            
            # Display answer
            print(f"🤖 Assistant: {answer}\n")
            
            # Display sources
            if source_docs:
                print(f"📚 Sources ({len(source_docs)} documents):")
                for i, doc in enumerate(source_docs[:2], 1):  # Show top 2 sources
                    title = doc.metadata.get("title", "Unknown")
                    content_preview = doc.page_content[:150] + "..."
                    print(f"  {i}. {title}")
                    print(f"     {content_preview}\n")
        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please try again with a different question.\n")


def main():
    """Main execution function"""

    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Conversational RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Use Ollama (default)
  python main.py --provider openai  # Use OpenAI explicitly
  python main.py --provider gemini  # Use Google Gemini
        """
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "gemini", "ollama"],
        default="ollama",
        help="LLM and embedding provider (default: ollama)"
    )
    args = parser.parse_args()
    
    provider = args.provider
    
    print("\n" + "="*70)
    print(f"Conversational RAG System with HotPotQA")
    print(f"Provider: {provider.upper()}")
    print("="*70 + "\n")
    
    # 1. Setup API keys
    print(f"1. Checking {provider.upper()} API key...")
    try:
        setup_api_keys(provider)
        print(f"✓ {provider.upper()} API key found\n")
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # 2. Download HotPotQA dataset
    print("2. Setting up HotPotQA dataset...")
    try:
        download_hotpotqa()
        print()
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return
    
    # 3. Load HotPotQA data
    print("3. Loading HotPotQA data...")
    try:
        data = load_hotpotqa_data()
        print()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # 4. Create documents
    print("4. Creating documents from HotPotQA...")
    documents = create_documents_from_hotpotqa(data)
    print()
    
    # 5. Setup embeddings and LLM
    print(f"5. Initializing {provider.upper()} models...")
    try:
        embeddings, llm = initialize_models(provider)
        print()
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        return
    
    # 6. Create or load FAISS vector store
    print("6. Setting up FAISS vector store...")
    faiss_index_path = get_faiss_index_path(provider)
    vectorstore = create_or_load_vectorstore(documents, embeddings, faiss_index_path)
    print()
    
    # 7. Create conversational chain
    print("7. Creating conversational retrieval chain...")
    chain = create_conversational_chain(vectorstore, llm)
    print("✓ Chain created\n")
    
    # 8. Run interactive conversation
    run_conversation(chain, provider)


if __name__ == "__main__":
    main()
