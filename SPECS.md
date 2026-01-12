## Evaluation Metrics

- SEED Verwenden
- Einheitlichen Datensatz
    - StatpearlLoader
- Einheitliches Fragenset
    - MedQA
        - Davon 10-100 Fragen die in der Statpearl existieren und festhalten

-> Eine Config erstellen und verwenden

### Metriken (Quantitativ)

- Faithfulness (NLI Score): We employ Natural Language Inference to determine if the generated answer is logically entailed by the retrieved snippets ensuring the explanation is not a "persuasive hallucination."

- Interpretability (Jaccard Coefficient): We use the Jaccard similarity to measure the overlap between system-generated highlights and annotated evidence.

- Efficiency: Measured via response latency and token consumption to assess the overhead of generating explanations.

### Metriken (Qualitativ)

- perceived trust
- usefulness
- satisfaction