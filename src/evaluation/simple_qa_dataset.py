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
        "context": "Mark geht oft in verschiedene Restaurants. Als Kind mochte er nur Nudeln. In der Studienzeit aß er hauptsächlich Burger und Fast Food. Seine Frau bevorzugt Sushi und versucht ihn zu überzeugen, gesünder zu essen. Der Sohn mag Burger am meisten. Seit drei Jahren isst Mark am liebsten Pizza, besonders mit Salami und Pilzen. Am Wochenende bestellt die Familie meistens gemeinsam beim Italiener.",
        "answer": "Pizza",
        "evidence": "Seit drei Jahren isst Mark am liebsten Pizza, besonders mit Salami und Pilzen."
    },
    {
        "question": "Ist Emma noch minderjährig?",
        "context": "Emma wurde 1998 geboren und hat dieses Jahr ihren Geburtstag bereits gefeiert. Ihr älterer Bruder Felix kam 1993 zur Welt und arbeitet seit fünf Jahren als Arzt. Die jüngste Schwester Sophie ist erst 18 Jahre alt und macht gerade Abitur. Ihre Mutter wurde 1970 geboren. Die Familie plant eine große Feier, wenn Emma nächstes Jahr ihren runden Geburtstag hat. Ihr Vater scherzt oft, dass Emma jetzt im besten Alter für eine Karriere ist.",
        "answer": "Nein",
        "evidence": "Emma wurde 1998 geboren und hat dieses Jahr ihren Geburtstag bereits gefeiert."
    },
    {
        "question": "Wie heißen David seine Kinder?",
        "context": "David ist 45 Jahre alt und seit 15 Jahren verheiratet. Sein Kollege Stefan ist kinderlos und reist viel. Die Nachbarin hat drei Kinder und beneidet David manchmal. Davids Sohn Max ist 12 Jahre alt und spielt gerne Fußball. Davids Tochter Lisa ist 8 und geht in die zweite Klasse. David arbeitet von zu Hause aus, um mehr Zeit mit seiner Familie zu verbringen. Seine Frau arbeitet Teilzeit als Architektin. Am Wochenende unternimmt die ganze Familie oft Ausflüge in die Natur.",
        "answer": "Max und Lisa",
        "evidence": "Davids Sohn Max ist 12 Jahre alt und spielt gerne Fußball. Seine Tochter Lisa ist 8 und geht in die zweite Klasse."
    },
    {
        "question": "Wohin reist Maria in den Urlaub?",
        "context": "Maria plant ihre Sommerreise sehr sorgfältig. Letztes Jahr war sie in Italien und hat Rom besichtigt. Ihre beste Freundin Laura fährt nach Frankreich an die Côte d'Azur. Maria hatte ursprünglich vor, nach Griechenland zu fliegen, aber die Flüge waren zu teuer. Dann überlegte sie, nach Portugal zu fahren. Letzte Woche hat sie sich aber endgültig entschieden und Flüge nach Barcelona gebucht. Sie freut sich sehr auf die spanische Küche und die Architektur von Gaudí. Ihr Bruder empfiehlt ihr, auch Madrid zu besuchen, aber dafür hat sie nur eine Woche Zeit.",
        "answer": "Barcelona",
        "evidence": "Letzte Woche hat sie sich aber endgültig entschieden und Flüge nach Barcelona gebucht."
    },
    {
        "question": "Welche Sportart betreibt Tim?",
        "context": "Tim ist sehr sportlich und aktiv. Als Kind nahm er Schwimmunterricht, genau wie seine Schwester heute noch gerne schwimmt. In der Schule spielte er zwei Jahre lang Basketball. Sein Vater spielt jeden Sonntag Golf im Country Club. Letztes Jahr probierte Tim Yoga aus, machte aber nach drei Monaten nicht weiter. Seit vier Jahren trainiert Tim dreimal pro Woche Fußball im Verein TSV München. Er spielt dort im Mittelfeld und hat letztes Jahr sogar ein Tor im Finale geschossen. Nebenbei geht er manchmal joggen, aber das ist nur zum Fitbleiben. Sein Trainer sagt, er hätte das Zeug zum Profispieler.",
        "answer": "Fußball",
        "evidence": "Seit vier Jahren trainiert Tim dreimal pro Woche Fußball im Verein TSV München."
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
