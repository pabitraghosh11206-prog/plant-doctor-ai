# Plant Doctor AI

A student project that uses a CNN to classify tomato leaf images and prepares tomato care guides for a retrieval-augmented generation (RAG) system.

## CNN disease classes

- Tomato Early Blight
- Tomato Late Blight
- Healthy Tomato

## Current project work

- Trained CNN model and image prediction script
- Tomato care guides stored as PDFs
- PDF text extracted to `rag/data/extracted_pages.jsonl`
- Source-linked text chunks stored in `rag/data/chunks.jsonl`
- RAG retrieval and application integration are in progress

## Run a prediction

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

From the project root, run:

```bash
python cnn/predict.py path/to/leaf_image.jpg cnn/best_model.h5
```

Replace `path/to/leaf_image.jpg` with the path to your tomato leaf image.

## Project folders

- `cnn/` — model, class names, and prediction code
- `rag/documents/tomato/` — tomato care guide PDFs
- `rag/metadata/` — guide descriptions and source information
- `rag/knowledge/loader.py` — extracts text from the PDFs
- `rag/data/` — extracted pages and text chunks

## Disclaimer

Predictions and care information are for educational purposes. They may be inaccurate and should not replace advice from a qualified agricultural expert.
