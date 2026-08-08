# Conversational Analytics for SaaS Metrics

A natural-language query interface over a SaaS customer dataset (7,057 customers,
monthly economics, 2023-24). Ask a question in plain English; the app parses it
into a structured query, runs it against the data, and returns a chart with a
written summary. The parse is shown next to each answer, so the translation is
transparent and verifiable.

## Run it

```bash
pip install streamlit plotly pandas
streamlit run saas_app.py
```

Opens at http://localhost:8501. Click an example or type your own question.

## What it answers

- Comparisons: "Which region has the highest churn?", "Average plan price by tier", "LTV/CAC by acquisition channel"
- Trends: "How did ARPU change over time?", "Churn trend by quarter"
- Single values: "Average churn for Europe", "How many Enterprise customers are there?"

Metrics: churn, plan price, ARPU, gross margin, marketing spend, LTV/CAC, customer count.
Dimensions: region, tier, acquisition channel, signup source. Plus year filters and quarterly trends.

## How it works

1. Rule-based parser (parse_question): deterministic. Extracts a metric, aggregation,
   filters, group-by dimension, time window, and intent (single / compare / trend).
   Each decision is recorded and displayed, so every answer can be traced back to how
   the question was read.
2. Query engine (run_query): executes the parsed spec in pandas. No free-form code is
   generated or run.
3. Optional LLM assist (llm_parse + validate_spec): off by default. If wired to a model,
   its proposed query is validated against a whitelist (known metric, known aggregation,
   real columns) before it can run.

## Configuration

Point it at a different dataset by editing DATA_PATH, and adjust the METRICS and
DIMENSIONS dictionaries near the top of saas_app.py to match your columns.

## Files

- saas_app.py: the Streamlit application
- test_saas.py: headless test harness covering every query path
- raw/cac_ltv_model.csv: the dataset
