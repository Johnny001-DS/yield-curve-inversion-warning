
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss
)


def _proc_root() -> Path:
    
    for p in [Path("data/processed"), Path("Data/processed")]:
        if p.exists():
            return p
    st.error("Couldn't find data")
    st.stop()

PROC = _proc_root()

@st.cache_data(show_spinner=False)
def load_preds(which: str) -> pd.DataFrame:
    """
    Load predictions for model key:
      - 'logit'   -> preds_logit_h12_fixed.csv
      - 'histgb'  -> preds_histgb_h12.csv
    Returns DF with columns: date index, inv_true (0/1), inv_prob [0..1]
    """
    fname = {"logit": "preds_logit_h12_fixed.csv", "histgb": "preds_histgb_h12.csv"}[which]
    fpath = PROC / fname
    if not fpath.exists():
        st.error(f"Missing file: {fpath}")
        st.stop()

    df = pd.read_csv(fpath)
    
    date_col = None
    for c in ["date", "Date", "DATE", "Unnamed: 0", "index"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        
        st.error(f"Could not find a date column in {fpath}. Add one named 'date'.")
        st.stop()

    # dates & index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    
    rename = {}
    if "y_te" in df.columns and "inv_true" not in df.columns:
        rename["y_te"] = "inv_true"
    if "y_true" in df.columns:
        rename["y_true"] = "inv_true"
    if "proba" in df.columns and "inv_prob" not in df.columns:
        rename["proba"] = "inv_prob"
    if "prob" in df.columns and "inv_prob" not in df.columns:
        rename["prob"] = "inv_prob"

    df = df.rename(columns=rename)

    missing = {"inv_true", "inv_prob"} - set(df.columns)
    if missing:
        st.error(f"{fpath} must contain columns {missing}. Found: {list(df.columns)}")
        st.stop()

    
    df["inv_prob"] = df["inv_prob"].astype(float).clip(0, 1)
    df["inv_true"] = df["inv_true"].astype(int).clip(0, 1)
    return df

@st.cache_data(show_spinner=False)
def load_backtest() -> pd.DataFrame | None:
    f = PROC / "backtest_inv_h12.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return df

@st.cache_data(show_spinner=False)
def load_threshold_sweep() -> pd.DataFrame | None:
    f = PROC / "threshold_sweep_inv_h12.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return df

# Metrics
def safe_roc_auc(y_true, prob):
    try:
        if len(np.unique(y_true)) == 2:
            return float(roc_auc_score(y_true, prob))
    except Exception:
        pass
    return np.nan

def safe_pr_auc(y_true, prob):
    try:
        if len(np.unique(y_true)) == 2:
            return float(average_precision_score(y_true, prob))
    except Exception:
        pass
    return np.nan

def compute_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    rec  = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1": round(f1, 3),
        "AUC": round(safe_roc_auc(y_true, prob), 3),
        "PR_AUC": round(safe_pr_auc(y_true, prob), 3),
        "Brier": round(float(brier_score_loss(y_true, prob)), 3)
    }

# Plot
def segments_above_threshold(df: pd.DataFrame, thr: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    cond = (df["inv_prob"] >= thr).astype(int)
    
    starts = df.index[(cond.diff() == 1).fillna(cond.iloc[0] == 1)]
    ends   = df.index[(cond.diff() == -1)]
    
    if len(ends) < len(starts):
        ends = ends.append(pd.Index([df.index[-1]]))
    return list(zip(starts, ends))

def hero_chart(df: pd.DataFrame, thr: float, lookback_years: int, title_suffix: str) -> go.Figure:
    if lookback_years is not None and lookback_years > 0:
        cutoff = df.index.max() - pd.DateOffset(years=lookback_years)
        df = df.loc[df.index >= cutoff].copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["inv_prob"],
        mode="lines", name="12w inversion probability",
        line=dict(width=3, color="#F6FF8B")  
    ))

    # threshold
    fig.add_hline(y=thr, line_dash="dash", line_color="#BBBBBB", opacity=0.8)

    # shade model-positives
    for s, e in segments_above_threshold(df, thr):
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(85, 120, 255, 0.15)", line_width=0)

    # actual inversions as ticks near bottom
    y_ticks = np.where(df["inv_true"] == 1, 0.02, np.nan)
    fig.add_trace(go.Scatter(
        x=df.index, y=y_ticks, mode="markers",
        marker=dict(symbol="line-ns-open", size=10, color="#87CEFA"),
        name="actual inversion (tick)"
    ))

    
    now_y = float(df["inv_prob"].iloc[-1])
    fig.add_annotation(
        x=df.index[-1], y=now_y, ax=40, ay=-40,
        text=f"Now: {now_y:.2f}", showarrow=True,
        font=dict(size=12, color="#FFFFFF"), arrowcolor="#FFFFFF"
    )

    fig.update_layout(
        title=f"12-week Inversion Probability — last {lookback_years}y {title_suffix}",
        yaxis=dict(range=[0, 1.02], title="Probability"),
        xaxis=dict(title=""),
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    )
    return fig

def confusion_heatmap(metrics: dict) -> go.Figure:
    z = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
    fig = go.Figure(data=go.Heatmap(
        z=z, colorscale="Blues", showscale=True,
        x=["Pred 0", "Pred 1"], y=["True 0", "True 1"], hoverongaps=False
    ))
    for i in range(2):
        for j in range(2):
            fig.add_annotation(x=["Pred 0","Pred 1"][j], y=["True 0","True 1"][i], text=str(z[i, j]), showarrow=False, font=dict(color="white" if z[i, j] > z.max()/2 else "black"))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), template="plotly_dark", title=f"Confusion (thr={st.session_state['thr']:.2f})")
    return fig

def roc_pr_curves(y_true: np.ndarray, prob: np.ndarray) -> tuple[go.Figure, go.Figure]:
    # ROC
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, prob)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
        roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
        roc_fig.update_layout(template="plotly_dark", height=320, margin=dict(l=20,r=20,t=40,b=20),
                              title=f"ROC (AUC={safe_roc_auc(y_true, prob):.3f})",
                              xaxis_title="FPR", yaxis_title="TPR")
    else:
        roc_fig = go.Figure().update_layout(template="plotly_dark", height=320, title="ROC (single-class in window)")

    # PR
    if len(np.unique(y_true)) == 2:
        prec, rec, _ = precision_recall_curve(y_true, prob)
        pr_fig = go.Figure()
        pr_fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR"))
        pr_fig.update_layout(template="plotly_dark", height=320, margin=dict(l=20,r=20,t=40,b=20), title=f"PR (AP={safe_pr_auc(y_true, prob):.3f})", xaxis_title="Recall", yaxis_title="Precision")
    else:
        pr_fig = go.Figure().update_layout(template="plotly_dark", height=320, title="PR (single-class in window)")
    return roc_fig, pr_fig

def boxplot_metric(df_bt: pd.DataFrame, metric: str) -> go.Figure:
    if df_bt is None or metric not in df_bt.columns:
        return go.Figure().update_layout(template="plotly_dark", height=10)
    fig = go.Figure()
    for name, g in df_bt.groupby("Model"):
        fig.add_trace(go.Box(y=g[metric], name=name, boxmean=True))
    fig.update_layout(template="plotly_dark", height=340, margin=dict(l=20,r=20,t=40,b=20), title=f"{metric} by Model (rolling-origin)")
    return fig


st.set_page_config(page_title="Yield-Curve Inversion Risk (h=12w)", layout="wide")
st.title("Yield-Curve Inversion Risk — 12 weeks ahead")

with st.sidebar:
    st.markdown("### Controls")
    model_key = st.selectbox("Model", ["logit", "histgb"], format_func=lambda k: {"logit":"Logit + state","histgb":"HistGB (cal)"}[k])
    thr = st.slider("Decision threshold", 0.05, 0.95, 0.40, 0.01)
    st.session_state["thr"] = thr
    lookback = st.slider("Lookback (years)", 2, 10, 5, 1)
    st.caption("Tip: Use ~5y for talks; increase for full context.")

# Data
preds = load_preds(model_key)
bt = load_backtest()
sweep = load_threshold_sweep()

# Current window metrics
m = compute_metrics(preds["inv_true"].values, preds["inv_prob"].values, thr)

# KPIs
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("AUC", f"{m['AUC']:.3f}")
kpi2.metric("PR_AUC", f"{m['PR_AUC']:.3f}")
kpi3.metric("Brier", f"{m['Brier']:.3f}")
kpi4.metric("Recall", f"{m['Recall']:.2f}")
kpi5.metric("Precision", f"{m['Precision']:.2f}")


hero, cm = st.columns([3, 1.5])
with hero:
    st.plotly_chart(hero_chart(preds, thr, lookback, f"— {('Logit + state' if model_key=='logit' else 'HistGB (cal)')}"),
                    use_container_width=True, theme="streamlit")
with cm:
    st.plotly_chart(confusion_heatmap(m), use_container_width=True)

# Curves
roc_col, pr_col = st.columns(2)
roc_fig, pr_fig = roc_pr_curves(preds["inv_true"].values, preds["inv_prob"].values)
roc_col.plotly_chart(roc_fig, use_container_width=True)
pr_col.plotly_chart(pr_fig, use_container_width=True)

# Backtest plots 
if bt is not None and {"Model","AUC","PR_AUC","Brier"}.issubset(set(bt.columns)):
    st.markdown("### Rolling-origin backtests")
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(boxplot_metric(bt, "AUC"), use_container_width=True)
    c2.plotly_chart(boxplot_metric(bt, "PR_AUC"), use_container_width=True)
    c3.plotly_chart(boxplot_metric(bt, "Brier"), use_container_width=True)
else:
    st.info("Backtest file not found (data/processed/backtest_inv_h12.csv). Skipping boxplots.")

# Threshold sweep 
with st.expander("Threshold sweep table (precision/recall/F1)", expanded=False):
    if sweep is not None and {"thr","TP","FP","TN","FN","Precision","Recall","F1"}.issubset(set(sweep.columns)):
        st.dataframe(sweep.round(3), use_container_width=True, height=300)
        
        f1_idx = sweep["F1"].idxmax()
        rec_target = 0.80
        idx_rec = (sweep["Recall"]-rec_target).abs().idxmin()
        st.caption(
            f"Best F1 → thr={sweep.loc[f1_idx,'thr']:.2f}, F1={sweep.loc[f1_idx,'F1']:.3f} | "
            f"Recall≈{rec_target:.0%} → thr={sweep.loc[idx_rec,'thr']:.2f}, "
            f"Prec={sweep.loc[idx_rec,'Precision']:.3f}"
        )
    else:
        st.write("Upload data/processed/threshold_sweep_inv_h12.csv to see this table.")