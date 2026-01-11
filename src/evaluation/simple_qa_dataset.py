"""
Simple German QA Dataset (N=10)
================================
A small dataset with simple question-answer pairs and short contexts
for testing and evaluation purposes.
"""

import re
from typing import List

try:
    from langchain_core.documents import Document
except ImportError:
    Document = None


SIMPLE_QA_DATASET = [
    {
        "question": "Welche Autofarbe hat Tom?",
        "context": "Max hat ein rotes Auto. Tom hat ein blaues Auto. Lisa fährt einen grünen Wagen.",
        "answer": "blau",
        "evidence": "Tom hat ein blaues Auto."
    },
    {
        "question": "Welches Haustier hat Anna?",
        "context": "Peter hat einen Hund namens Bello. Anna besitzt eine Katze. Michael hat ein Kaninchen.",
        "answer": "Katze",
        "evidence": "Anna besitzt eine Katze."
    },
    {
        "question": "Was ist der Beruf von Sarah?",
        "context": "Sarah arbeitet als Lehrerin an einer Grundschule. Ihr Bruder ist Arzt. Die Mutter ist Ingenieurin.",
        "answer": "Lehrerin",
        "evidence": "Sarah arbeitet als Lehrerin an einer Grundschule."
    },
    {
        "question": "Wo wohnt Klaus?",
        "context": "Klaus wohnt in Berlin. Seine Schwester lebt in München. Die Eltern sind nach Hamburg gezogen.",
        "answer": "Berlin",
        "evidence": "Klaus wohnt in Berlin."
    },
    {
        "question": "Welches Hobby hat Julia?",
        "context": "Julia spielt gerne Tennis am Wochenende. Ihr Freund fotografiert gern. Ihre Schwester malt Aquarelle.",
        "answer": "Tennis",
        "evidence": "Julia spielt gerne Tennis am Wochenende."
    },
    {
        "question": "Was ist Marks Lieblingsessen?",
        "context": "Mark isst am liebsten Pizza. Seine Frau bevorzugt Sushi. Der Sohn mag Burger am meisten.",
        "answer": "Pizza",
        "evidence": "Mark isst am liebsten Pizza."
    },
    {
        "question": "Wie alt ist Emma?",
        "context": "Emma ist 25 Jahre alt. Ihr Bruder Felix ist 30. Die jüngste Schwester Sophie ist erst 18.",
        "answer": "25 Jahre",
        "evidence": "Emma ist 25 Jahre alt."
    },
    {
        "question": "Hat David Kinder?",
        "context": "David hat zwei Kinder, einen Sohn und eine Tochter. Sein Kollege Stefan ist kinderlos. Die Nachbarin hat drei Kinder.",
        "answer": "Ja, zwei Kinder",
        "evidence": "David hat zwei Kinder, einen Sohn und eine Tochter."
    },
    {
        "question": "Wohin reist Maria im Urlaub?",
        "context": "Maria fliegt dieses Jahr nach Spanien. Letztes Jahr war sie in Italien. Ihre Freundin fährt nach Frankreich.",
        "answer": "Spanien",
        "evidence": "Maria fliegt dieses Jahr nach Spanien."
    },
    {
        "question": "Welche Sportart betreibt Tim?",
        "context": "Tim trainiert regelmäßig Fußball im Verein. Seine Schwester schwimmt gerne. Der Vater spielt Golf.",
        "answer": "Fußball",
        "evidence": "Tim trainiert regelmäßig Fußball im Verein."
    }
]


def _split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences based on sentence boundaries (., !, ?).
    Returns a list of sentences with leading/trailing whitespace stripped.
    """
    # Split on sentence boundaries while keeping the delimiter
    sentences = re.split(r'([.!?])', text)
    
    # Reconstruct sentences by pairing content with delimiter
    result = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        if i + 1 < len(sentences):
            delimiter = sentences[i + 1]
            if sentence:
                result.append(sentence + delimiter)
    
    # Handle last item if it doesn't have a delimiter
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())
    
    return [s for s in result if s]


def get_dataset():
    """Returns the simple QA dataset."""
    return SIMPLE_QA_DATASET


def get_questions():
    """Returns only the questions from the dataset."""
    return [item["question"] for item in SIMPLE_QA_DATASET]


def get_contexts():
    """Returns only the contexts from the dataset."""
    return [item["context"] for item in SIMPLE_QA_DATASET]


def get_answers():
    """Returns only the answers from the dataset."""
    return [item["answer"] for item in SIMPLE_QA_DATASET]


def get_item(index: int):
    """Returns a specific item from the dataset by index."""
    if 0 <= index < len(SIMPLE_QA_DATASET):
        return SIMPLE_QA_DATASET[index]
    raise IndexError(f"Index {index} out of range for dataset of size {len(SIMPLE_QA_DATASET)}")


def get_context_as_documents(index: int):
    """
    Returns the context of a specific dataset item as a list of Document objects,
    with each Document representing one sentence.
    
    Args:
        index: The index of the dataset item
        
    Returns:
        List of Document objects, one per sentence
        
    Raises:
        ImportError: If langchain_core is not installed
        IndexError: If index is out of range
    """
    if Document is None:
        raise ImportError("langchain_core is not installed. Please install it to use this function.")
    
    item = get_item(index)
    context = item["context"]
    question = item["question"]
    answer = item["answer"]
    
    sentences = _split_into_sentences(context)
    
    documents = []
    for i, sentence in enumerate(sentences):
        doc = Document(
            page_content=sentence,
            metadata={
                "sentence_id": i,
                "question_id": index,
                "question": question,
                "answer": answer,
                "full_context": context
            }
        )
        documents.append(doc)
    
    return documents


def get_all_contexts_as_documents():
    """
    Returns all contexts from the dataset as Document objects,
    with each Document representing one sentence.
    
    Returns:
        List of tuples (question_id, List[Document]) where each tuple contains
        the dataset index and the corresponding list of sentence Documents
        
    Raises:
        ImportError: If langchain_core is not installed
    """
    if Document is None:
        raise ImportError("langchain_core is not installed. Please install it to use this function.")
    
    all_documents = []
    for i in range(len(SIMPLE_QA_DATASET)):
        docs = get_context_as_documents(i)
        all_documents.append((i, docs))
    
    return all_documents
