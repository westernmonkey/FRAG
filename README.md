# FRAG — Financial Reconciliation AI Agent

One-file Streamlit app that reconciles internal ledgers against SEC 10-K filings.

## Pipeline (inside FRAG.py)

| Stage | What it does | Tool |
|-------|-------------|------|
| 1 | Extract text from 10-K PDF, find financial pages, chunk for RAG | `pdfplumber` |
| 2 | Embed chunks, retrieve relevant context per account | `sentence-transformers` + cosine similarity |
| 3 | Extract dollar values from context | InstructLab fine-tuned model → regex fallback |
| 4 | Compare ledger vs 10-K, compute variance, flag discrepancies | `pandas` |
| 5 | Display results + export | `streamlit` + `plotly` |

## Run

```bash
pip install -r requirements.txt
streamlit run FRAG.py
```

## InstructLab Fine-tuning

The `taxonomy/qna.yaml` contains seed examples for InstructLab training:

```bash
pip install instructlab
ilab config init
# copy taxonomy/ into ~/.local/share/instructlab/taxonomy/knowledge/finance/
ilab data generate
ilab model train
ilab model serve --model-path models/financial-extraction-ft
```

The app auto-detects the model on `localhost:8000`. Without it, regex extraction handles standard 10-K formats.
