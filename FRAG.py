import streamlit as st
import pandas as pd
import pdfplumber
import re
import json
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="FRAG", layout="wide")
st.title("📊 FRAG — Financial Reconciliation Agent")

col1, col2 = st.columns(2)
with col1:
    ten_k_file = st.file_uploader("Upload 10-K (PDF)", type="pdf")
with col2:
    ledger_file = st.file_uploader("Upload Ledger (CSV)", type="csv")


# ── STEP 1: Extract financial text from PDF ───────────────────────

def extract_financial_text(pdf_file):
    """Use pdfplumber to pull text from pages that contain financial data."""
    keywords = ["total net sales", "cash and cash equivalents", "total assets",
                "net income", "accounts receivable", "inventories", "gross margin"]
    pages = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if sum(1 for kw in keywords if kw in text.lower()) >= 2:
                pages.append(text)
    return "\n\n".join(pages)


# ── STEP 2: Extract values (fine-tuned model or regex) ────────────

def try_finetuned_model(text, accounts):
    """Call InstructLab fine-tuned model if it's running on localhost."""
    prompt = (
        f"Extract the most recent fiscal year values for: {', '.join(accounts)}.\n\n"
        f"Context:\n{text[:3000]}\n\n"
        f'Return ONLY JSON: {{"accounts": [{{"name": "...", "value": 12345}}]}}'
    )
    try:
        resp = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "instructlab-granite-7b-lab-trained/instructlab-granite-7b-lab-Q4_K_M.gguf",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 500,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return {a["name"].lower(): a["value"] for a in parsed["accounts"]}, "Fine-tuned Model"
    except Exception:
        pass
    return None, None


def extract_with_regex(text):
    """Pull account names and first dollar value from each line."""
    extracted = {}
    for line in text.split("\n"):
        if len(line.strip()) < 10:
            continue
        amounts = re.findall(r'[\d,]{3,}', line)
        if amounts:
            name_match = re.match(r'^(.*?)\s+\$?\s*[\d,]', line)
            if name_match:
                name = name_match.group(1).strip()
                if len(name) > 3:
                    try:
                        val = int(amounts[0].replace(",", ""))
                        if val > 100:
                            extracted[name.lower()] = val
                    except ValueError:
                        continue
    return extracted


# ── STEP 3: Match and reconcile ───────────────────────────────────

def fuzzy_match(name, candidates):
    """Find closest matching account name."""
    from difflib import SequenceMatcher
    best, best_score = None, 0
    for c in candidates:
        score = SequenceMatcher(None, name.lower(), c.lower()).ratio()
        if score > best_score:
            best, best_score = c, score
    return best if best_score > 0.55 else None


def reconcile(ledger_df, extracted):
    """Compare ledger to 10-K values, compute variance."""
    rows = []
    for _, row in ledger_df.iterrows():
        account = row["Account_Name"]
        ledger_val = abs(int(row["Balance"]))
        match = fuzzy_match(account, list(extracted.keys()))

        if match:
            tenk_val = extracted[match]
            variance = tenk_val - ledger_val
            var_pct = (variance / tenk_val * 100) if tenk_val else 0
            status = "MATCH" if variance == 0 else ("MINOR" if abs(var_pct) < 0.1 else "DISCREPANCY")
        else:
            tenk_val, variance, var_pct, status = None, None, None, "NOT FOUND"

        rows.append({
            "Account": account,
            "Ledger ($M)": ledger_val,
            "10-K ($M)": tenk_val,
            "Variance ($M)": variance,
            "Variance %": round(var_pct, 4) if var_pct is not None else None,
            "Status": status,
        })
    return pd.DataFrame(rows)


# ── RUN ───────────────────────────────────────────────────────────

if st.button("Run Reconciliation", type="primary", use_container_width=True):
    if not ten_k_file or not ledger_file:
        st.error("Upload both files.")
        st.stop()

    ledger_df = pd.read_csv(ledger_file)
    accounts = ledger_df["Account_Name"].tolist()

    with st.status("Running...", expanded=True) as status:
        st.write("Extracting financial pages from 10-K...")
        financial_text = extract_financial_text(ten_k_file)

        st.write("Extracting values...")
        extracted, method = try_finetuned_model(financial_text, accounts)
        if extracted is None:
            extracted = extract_with_regex(financial_text)
            method = "Regex"
        st.write(f"  → Used: {method} | Found {len(extracted)} values")

        st.write("Reconciling...")
        results = reconcile(ledger_df, extracted)
        status.update(label="Done", state="complete")

    # Results
    st.divider()
    st.subheader("Results")

    matches = len(results[results["Status"] == "MATCH"])
    discrep = len(results[results["Status"] == "DISCREPANCY"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Accounts", len(results))
    c2.metric("Matches", matches)
    c3.metric("Discrepancies", discrep)

    st.dataframe(results, use_container_width=True, hide_index=True)

    # Chart
    chart_df = results[results["Variance ($M)"].notna()]
    if not chart_df.empty:
        colors = ["#22c55e" if s == "MATCH" else "#f59e0b" if s == "MINOR" else "#ef4444"
                  for s in chart_df["Status"]]
        fig = go.Figure(go.Bar(
            x=chart_df["Account"], y=chart_df["Variance ($M)"],
            marker_color=colors,
            text=chart_df["Variance ($M)"].apply(lambda v: f"${v:+,.0f}M"),
            textposition="outside",
        ))
        fig.update_layout(yaxis_title="Variance ($M)", height=380,
                          margin=dict(t=20), plot_bgcolor="rgba(0,0,0,0)")
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        st.plotly_chart(fig, use_container_width=True)

    st.download_button("📥 Download Audit Report (CSV)",
                       results.to_csv(index=False),
                       "reconciliation_report.csv", "text/csv",
                       use_container_width=True)