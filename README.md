## Installation

### 1. Python Environment Setup

```bash
# Optional: Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Ihr braucht Ollama mit LLama3.2 damit die Embeddings erstellt werden können!

### 2. Gemini & OpenAI API Key

Setze deinen Gemini / OpenAI API Key als Environment Variable unter .env:

```bash
OPENAI_API_KEY='sk-your-api-key-here'
GOOGLE_API_KEY='sk-your-api-key-here'
```

## Verwendung

### Starten des Systems

```bash
python main.py # Ollama is default

# Für OpenAI / Gemini:
python main.py --provider openai
python main.py --provider gemini
```

### Beim ersten Start

Das System wird automatisch:
1. Den HotPotQA Test-Datensatz herunterladen (~50MB)
2. Dokumente aus dem Datensatz extrahieren
3. Embeddings generieren und FAISS-Index erstellen (dauert einige Minuten)
4. Den Index für zukünftige Verwendung speichern

Beim nächsten Start lädt das System einfach den existierenden FAISS-Index!

### Befehle

- **Normal fragen**: Tippe deine Frage und drücke Enter
- **`clear`**: Löscht die Conversation History
- **`exit` oder `quit`**: Beendet das Programm

## HotPotQA Dataset

### Was ist HotPotQA?

HotPotQA ist ein Question-Answering-Dataset, das Multi-Hop-Reasoning benötigt:
- **Multi-Hop**: Fragen erfordern das Kombinieren von Informationen aus mehreren Dokumenten
- **Explainable**: Enthält "supporting facts" die zeigen, welche Sätze zur Antwort beitragen
- **Wikipedia-basiert**: Alle Kontexte stammen aus Wikipedia-Artikeln

### Dataset-Format

```json
{
  "_id": "5a8b57f25542995d1e6f1371",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "answer": "yes",
  "supporting_facts": [
    ["Scott Derrickson", 0],
    ["Ed Wood", 0]
  ],
  "context": [
    ["Scott Derrickson", ["Scott Derrickson is an American filmmaker..."]],
    ["Ed Wood", ["Edward Davis Wood Jr. was an American filmmaker..."]]
  ]
}
```

### Wie das RAG-System HotPotQA nutzt

1. **Document Creation**: Jeder Wikipedia-Artikel im Context wird zu einem LangChain Document
2. **Metadata**: Jedes Dokument speichert:
   - Title des Artikels
   - Zugehörige Frage und Antwort
   - Ob es ein "supporting fact" ist
3. **Text Splitting**: Lange Artikel werden in Chunks für besseres Retrieval aufgeteilt
4. **Embedding & Indexing**: FAISS-Index für schnelle Similarity-Search
5. **Retrieval**: Bei Fragen werden die 4 relevantesten Chunks abgerufen

## Conversational RAG für Multi-Hop Questions

Das System ist besonders stark bei Multi-Hop-Fragen:

**Beispiel 1: Iteratives Reasoning**
```
Q: Who directed the movie that won Best Picture in 2020?
A: [System sucht nach Best Picture 2020 → "Parasite"]

Q: Where was he born?
A: [System versteht "he" = Bong Joon-ho, sucht nach seiner Biografie]
```

**Beispiel 2: Follow-up Details**
```
Q: What university did the founder of Microsoft attend?
A: [Findet Bill Gates → Harvard]

Q: When did he drop out?
A: [Versteht Kontext, sucht nach spezifischen Daten]
```
