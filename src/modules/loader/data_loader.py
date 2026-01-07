import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from tqdm import tqdm  # Recommended for progress bars

class DataLoader:

    def __init__(self):
        try:
            src_dir = Path(__file__).resolve().parent 
            self.PROJECT_ROOT = src_dir.parent.parent
            
        except NameError:
            print("Warning: Could not determine module path. Assuming current directory is project root.")
            self.PROJECT_ROOT = Path.cwd()

        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DIR = self.DATA_DIR / "raw"
        self.DATASET_FILE = self.RAW_DIR / "hotpot_test_fullwiki_v1.json"
        self.HOTPOTQA_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json"
        
        self._data = None

    def setup(self, as_documents: bool = True, limit: Optional[int] = None) -> List[Any]:
        """
        Orchestrator: Checks paths, downloads if needed, loads JSON, 
        and optionally converts to LangChain Documents.
        
        Args:
            as_documents: If True, returns List[Document]. If False, returns raw List[Dict].
            limit: If set, only processes the first N items (good for testing).
        """
        # 1. Ensure directory structure
        self.RAW_DIR.mkdir(parents=True, exist_ok=True)
        
        # 2. Download if missing
        if not self.DATASET_FILE.exists():
            self._download_hotpotqa()
        else:
            print(f"✓ Dataset found at {self.DATASET_FILE}")

        # 3. Load Raw Data
        raw_data = self._load_hotpotqa_data()
        
        if limit:
            raw_data = raw_data[:limit]
            print(f"Limiting data to first {limit} entries.")

        # 4. Return requested format
        if as_documents:
            return self.create_documents(raw_data)
        return raw_data

    def _download_hotpotqa(self):
        """Internal download logic"""
        print(f"Downloading HotPotQA from {self.HOTPOTQA_URL}...")
        try:
            response = requests.get(self.HOTPOTQA_URL, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.DATASET_FILE, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc="Downloading"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            print(f"\n✓ Download complete.")
            
        except Exception as e:
            if self.DATASET_FILE.exists():
                self.DATASET_FILE.unlink()
            raise RuntimeError(f"Failed to download dataset: {e}")

    def _load_hotpotqa_data(self) -> List[Dict[str, Any]]:
        """Internal JSON loader"""
        if self._data is not None:
            return self._data

        print(f"Loading data into memory...")
        try:
            with open(self.DATASET_FILE, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
            print(f"✓ Loaded {len(self._data)} questions.")
            return self._data
        except json.JSONDecodeError:
            print("Error: The dataset file seems corrupted.")
            return []

    def create_documents(self, data: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert HotPotQA data to LangChain Document objects.
        
        Structure:
        Creates 1 Document per Context Paragraph (Title).
        Metadata includes 'gold_sentence_indices' for evaluation.
        """
        documents = []
        print("Converting HotPotQA contexts to documents...")
        
        for item in tqdm(data, desc="Processing Articles"):
            question_id = item.get("_id", "unknown")
            question = item.get("question", "")
            answer = item.get("answer", "")
            context = item.get("context", [])
            supporting_facts = item.get("supporting_facts", [])
            
            # Map: Title -> Set of Sentence Indices that are supporting facts
            # e.g. {"Eiffel Tower": {0, 2}}
            supporting_map = {}
            for title, sent_id in supporting_facts:
                if title not in supporting_map:
                    supporting_map[title] = set()
                supporting_map[title].add(sent_id)
            
            # Process each context article
            for title, sentences in context:
                # Join sentences to create the full chunk text
                content = " ".join(sentences)
                
                # Identify which sentences in THIS paragraph are "gold"
                gold_indices = list(supporting_map.get(title, set()))
                is_supporting = len(gold_indices) > 0
                
                # Create document with RICH metadata for Explainability
                doc = Document(
                    page_content=content,
                    metadata={
                        "title": title,
                        "question_id": question_id,
                        "question": question,
                        "answer": answer,
                        # Crucial for Plausibility Metric:
                        "is_supporting": is_supporting, 
                        "gold_sentence_indices": ",".join(map(str, gold_indices)), 
                        "source": "hotpotqa"
                    }
                )
                documents.append(doc)
        
        print(f"✓ Created {len(documents)} context chunks.")
        return documents