"""
Conversational Analytics for SaaS Metrics
-----------------------------------------
A natural-language query interface over a SaaS customer dataset. The user asks a
question in plain English; the app parses it into a structured query, runs it
against the data with pandas, and returns a chart plus a written summary. The
parse is displayed alongside each answer so the translation is transparent.

Design: a deterministic rule-based parser handles the common question shapes and
keeps every result explainable. An optional LLM assist (disabled by default) is
gated behind schema validation, so a language model can never run an unchecked
query.

Configuration lives in DATA_PATH, METRICS, and DIMENSIONS below.
"""

import re
import pandas as pd
import streamlit as st
import plotly.express as px

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DATA_PATH = "Raw Project Datasets/CAC-LTV Model Analysis for SaaS Business Insights/cac_ltv_model.csv"

# Metrics the user can ask about: label -> (column, how to format)
METRICS = {
    "churn":       ("churn_rate", "pct"),
    "price":       ("plan_price", "money"),
    "arpu":        ("arpu", "money"),
    "margin":      ("gross_margin", "pct"),
    "marketing":   ("marketing_spend", "money"),
    "ltv_cac":     ("ltv_cac", "ratio"),
    "count":       ("count", "int"),
}
# Categorical dimensions the user can group/filter by
DIMENSIONS = {
    "region":  "region",
    "tier":    "customer_tier",
    "channel": "acquisition_channel",
    "source":  "signup_source",
}

# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["period"] = pd.to_datetime(df["date"], format="%b-%y")
    df["quarter"] = df["period"].dt.to_period("Q").astype(str)
    # derived economics: LTV, CAC, and their ratio
    df["monthly_churn"] = df["churn_rate"].clip(lower=0.005)
    df["ltv"] = df["arpu"] * df["gross_margin"] * (1 / df["monthly_churn"])
    df["cac"] = df["marketing_spend"].clip(lower=1.0)
    df["ltv_cac"] = df["ltv"] / df["cac"]
    return df

# ----------------------------------------------------------------------------
# 1. RULE-BASED PARSER
# ----------------------------------------------------------------------------
METRIC_WORDS = {
    "churn":     ["churn", "attrition", "leave", "cancel", "retention"],
    "price":     ["price", "plan price", "pricing", "cost"],
    "arpu":      ["arpu", "revenue per", "average revenue"],
    "margin":    ["margin", "gross margin", "profitability"],
    "marketing": ["marketing", "cac", "acquisition cost", "spend"],
    "ltv_cac":   ["ltv/cac", "ltv to cac", "ltv cac", "unit economics", "ltv/cac ratio"],
    "count":     ["how many", "number of", "count", "customers", "volume"],
}
AGG_WORDS = {
    "mean":   ["average", "avg", "mean", "typical"],
    "median": ["median", "middle"],
    "max":    ["highest", "most", "max", "top", "worst" ],
    "min":    ["lowest", "least", "min", "best", "cheapest"],
    "sum":    ["total", "sum", "combined"],
}
TREND_WORDS = ["over time", "trend", "by month", "by quarter", "monthly",
               "quarterly", "over the year", "each quarter", "each month"]
COMPARE_WORDS = ["by region", "by tier", "by channel", "by source",
                 "which", "compare", "rank", "across", "highest", "lowest",
                 "best", "worst", "most", "least", "per region", "per tier"]


def _find(t, words):
    return any(w in t for w in words)


def parse_question(q, df):
    t = q.lower().strip()
    spec = {"metric": None, "agg": "mean", "group_by": None, "filters": {},
            "time": None, "intent": None, "sort": "desc", "limit": 12}
    reasons = []

    # --- metric --- (order matters: check compound terms before their substrings)
    metric_order = ["ltv_cac", "arpu", "churn", "price", "margin", "marketing", "count"]
    for m in metric_order:
        if _find(t, METRIC_WORDS[m]):
            spec["metric"] = m
            reasons.append(f"metric={m}")
            break
    if spec["metric"] is None:
        spec["metric"] = "churn"        # sensible default for this dataset
        reasons.append("metric=churn (default)")

    # --- aggregation ---
    # "highest/lowest/most/least" are RANKING words in a comparison (they set sort
    # direction), not aggregation words. Only "total/sum" and "median" change the
    # aggregation itself. Everything else averages within each group.
    RANK_HIGH = ["highest", "most", "top", "worst", "max"]
    RANK_LOW = ["lowest", "least", "min", "best", "cheapest"]
    if _find(t, RANK_HIGH):
        spec["sort"] = "desc"; reasons.append("rank=highest")
    elif _find(t, RANK_LOW):
        spec["sort"] = "asc"; reasons.append("rank=lowest")

    if _find(t, AGG_WORDS["sum"]):
        spec["agg"] = "sum"; reasons.append("agg=sum")
    elif _find(t, AGG_WORDS["median"]):
        spec["agg"] = "median"; reasons.append("agg=median")
    else:
        spec["agg"] = "sum" if spec["metric"] == "count" else "mean"
        reasons.append(f"agg={spec['agg']} (default)")

    # --- filters (dimension = value) ---
    for dim, col in DIMENSIONS.items():
        for val in df[col].unique():
            if str(val).lower() in t:
                spec["filters"][col] = val
                reasons.append(f"filter {dim}={val}")

    # --- time window ---
    yr = re.search(r"\b(20\d{2})\b", t)
    if yr:
        spec["time"] = ("year", int(yr.group(1)))
        reasons.append(f"time year={yr.group(1)}")

    # --- intent ---
    if _find(t, TREND_WORDS):
        spec["intent"] = "trend"
        reasons.append("intent=trend")
    else:
        # detect an explicit group dimension for comparison
        gb = None
        for dim, col in DIMENSIONS.items():
            if dim in t or f"by {dim}" in t or col.lower() in t:
                # don't group by a dimension we already filtered to a single value
                if col not in spec["filters"]:
                    gb = col
                    break
        if gb is None and _find(t, COMPARE_WORDS):
            gb = DIMENSIONS["region"]   # default comparison axis
        if gb:
            spec["intent"] = "compare"
            spec["group_by"] = gb
            reasons.append(f"intent=compare, group_by={gb}")
        else:
            spec["intent"] = "single"
            reasons.append("intent=single value")

    # --- confidence ---
    signal = 0
    if any("metric=" in r and "default" not in r for r in reasons): signal += 1
    if spec["intent"] in ("trend", "compare"): signal += 1
    if spec["filters"]: signal += 1
    if spec["time"]: signal += 1
    confidence = min(1.0, 0.4 + 0.15 * signal)
    return spec, confidence, reasons


# ----------------------------------------------------------------------------
# 2. QUERY ENGINE
# ----------------------------------------------------------------------------
def _apply_filters(df, spec):
    d = df.copy()
    for col, val in spec["filters"].items():
        d = d[d[col] == val]
    if spec["time"] and spec["time"][0] == "year":
        d = d[d["year"] == spec["time"][1]]
    return d


def _agg(d, spec, by):
    m = spec["metric"]
    if m == "count":
        return d.groupby(by).size()
    col = METRICS[m][0]
    return getattr(d.groupby(by)[col], spec["agg"])()


def run_query(df, spec):
    d = _apply_filters(df, spec)
    if len(d) == 0:
        return {"kind": "empty", "n": 0}

    if spec["intent"] == "trend":
        s = _agg(d, spec, "quarter").reset_index()
        s.columns = ["quarter", "value"]
        return {"kind": "trend", "data": s, "n": len(d), "by": "quarter"}

    if spec["intent"] == "compare":
        s = _agg(d, spec, spec["group_by"]).reset_index()
        s.columns = [spec["group_by"], "value"]
        s = s.sort_values("value", ascending=(spec["sort"] == "asc")).head(spec["limit"])
        return {"kind": "compare", "data": s, "n": len(d), "by": spec["group_by"]}

    # single value
    m = spec["metric"]
    if m == "count":
        val = len(d)
    else:
        val = getattr(d[METRICS[m][0]], spec["agg"])()
    return {"kind": "single", "value": val, "n": len(d)}


# ----------------------------------------------------------------------------
# 3. FORMATTING + READOUT
# ----------------------------------------------------------------------------
def _fmt(metric, v):
    kind = METRICS[metric][1]
    if kind == "money":  return f"${v:,.2f}"
    if kind == "pct":    return f"{v*100:.2f}%"
    if kind == "ratio":  return f"{v:.1f}x"
    if kind == "int":    return f"{v:,.0f}"
    return f"{v:,.2f}"


LABELS = {"churn": "monthly churn", "price": "plan price", "arpu": "ARPU",
          "margin": "gross margin", "marketing": "marketing spend",
          "ltv_cac": "LTV/CAC", "count": "customer count"}


def narrate(spec, result):
    if result["kind"] == "empty":
        return "No customers match those filters. Try loosening a filter or the time window."
    m = spec["metric"]
    label = LABELS[m]

    if result["kind"] == "single":
        if m == "count":
            return f"There are **{result['value']:,} customers** matching those filters."
        return (f"Across {result['n']:,} matching customers, the {spec['agg']} "
                f"{label} is **{_fmt(m, result['value'])}**.")

    if result["kind"] == "compare":
        d = result["data"]
        top = d.iloc[0]
        return (f"Comparing {label} by {result['by']} across {result['n']:,} customers. "
                f"**{top[result['by']]}** leads at **{_fmt(m, top['value'])}**. "
                f"Full breakdown below.")

    if result["kind"] == "trend":
        d = result["data"]
        first, last = d.iloc[0]["value"], d.iloc[-1]["value"]
        pct = (last - first) / first * 100 if first else 0
        direction = "up" if pct >= 0 else "down"
        return (f"{label.capitalize()} moved **{direction} {abs(pct):.1f}%** "
                f"from {d.iloc[0]['quarter']} to {d.iloc[-1]['quarter']}, "
                f"across {result['n']:,} customers.")
    return ""


# ----------------------------------------------------------------------------
# OPTIONAL LLM ASSIST (disabled by default; schema-validated)
# ----------------------------------------------------------------------------
def validate_spec(cand, df):
    try:
        if cand.get("metric") not in METRICS: return None
        if cand.get("agg") not in {"mean", "median", "max", "min", "sum"}: return None
        for col in cand.get("filters", {}):
            if col not in df.columns: return None
        return cand
    except Exception:
        return None


def llm_parse(q, df):
    return None  # returns None so the app runs offline; validate_spec() guards any real call


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="SaaS Metrics — Ask the Data", layout="wide")
df = load_data()

st.title("Ask the SaaS data")
st.caption("Ask a question in plain English. It is parsed into a query, run "
           "against 7,057 customer records, and answered with a chart and summary. "
           "The parse is shown so you can verify it.")

examples = [
    "Which region has the highest churn?",
    "How did ARPU change over time?",
    "Average plan price by tier",
    "LTV/CAC by acquisition channel",
    "How many Enterprise customers are there?",
    "Lowest margin by region",
]
cols = st.columns(3)
picked = None
for i, ex in enumerate(examples):
    if cols[i % 3].button(ex, use_container_width=True):
        picked = ex

q = st.text_input("Your question", value=picked or "Which region has the highest churn?")

if q:
    spec, conf, reasons = parse_question(q, df)
    used_llm = False
    if conf < 0.55:
        cand = llm_parse(q, df)
        clean = validate_spec(cand, df) if cand else None
        if clean:
            spec, used_llm = clean, True

    result = run_query(df, spec)
    left, right = st.columns([2, 1])

    with left:
        st.markdown("### Answer")
        st.markdown(narrate(spec, result))
        if result["kind"] == "trend":
            fig = px.line(result["data"], x="quarter", y="value", markers=True)
            fig.update_layout(height=380, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        elif result["kind"] == "compare":
            fig = px.bar(result["data"], x="value", y=result["by"], orientation="h")
            fig.update_layout(height=420, xaxis_title="", yaxis_title="",
                              yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        elif result["kind"] == "single":
            st.metric(f"{spec['agg']} {LABELS[spec['metric']]}",
                      _fmt(spec["metric"], result["value"]),
                      help=f"{result['n']:,} matching customers")

    with right:
        st.markdown("### How this was read")
        st.caption(f"Confidence: {conf:.0%}" +
                   ("  · LLM-assisted (validated)" if used_llm else "  · rule-based"))
        for r in reasons:
            st.write("• " + r)
        with st.expander("Query spec (what actually ran)"):
            st.json({k: v for k, v in spec.items() if v not in (None, {}, [])})
