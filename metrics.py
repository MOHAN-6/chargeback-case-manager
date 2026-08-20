"""
metrics.py
----------
Honest performance measurement for the multi-agent case manager, plus an
explicit false-positive cost model.
"""

from __future__ import annotations
import pandas as pd


def confusion_counts(y_true: pd.Series, y_pred_positive: pd.Series) -> dict:
    tp = int(((y_true == True) & (y_pred_positive == True)).sum())
    fp = int(((y_true == False) & (y_pred_positive == True)).sum())
    tn = int(((y_true == False) & (y_pred_positive == False)).sum())
    fn = int(((y_true == True) & (y_pred_positive == False)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def prf1(counts: dict) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(results_df: pd.DataFrame) -> dict:
    """results_df must have columns: is_fraud (bool), decision (str)."""
    out = {}

    # ---- View 1: auto-decided only (exclude MANUAL_REVIEW) ----
    auto = results_df[results_df["decision"] != "MANUAL_REVIEW"].copy()
    auto_pred_positive = auto["decision"] == "FRAUD_CONFIRMED"
    auto_counts = confusion_counts(auto["is_fraud"], auto_pred_positive)
    out["auto_decided"] = {
        "n_cases": int(len(auto)),
        "counts": auto_counts,
        **prf1(auto_counts),
    }

    # ---- View 2: full funnel (MANUAL_REVIEW treated as "flagged") ----
    full_pred_positive = results_df["decision"].isin(["FRAUD_CONFIRMED", "MANUAL_REVIEW"])
    full_counts = confusion_counts(results_df["is_fraud"], full_pred_positive)
    out["full_funnel_flagged"] = {
        "n_cases": int(len(results_df)),
        "counts": full_counts,
        **prf1(full_counts),
    }

    # ---- Manual review load ----
    n_review = int((results_df["decision"] == "MANUAL_REVIEW").sum())
    out["manual_review"] = {
        "n_cases": n_review,
        "pct_of_total": n_review / len(results_df) if len(results_df) else 0.0,
        "fraud_rate_within_review": float(
            results_df.loc[results_df["decision"] == "MANUAL_REVIEW", "is_fraud"].mean()
        ) if n_review else None,
    }

    return out


# --------------------------------------------------------------------------
# False-positive cost model
# --------------------------------------------------------------------------

def false_positive_cost(
    results_df: pd.DataFrame,
    avg_customer_ltv_impact_inr: float = 1200.0,
    review_labor_cost_inr: float = 45.0,
) -> dict:
    """
    Estimates the business cost of the system's errors.
    """
    auto = results_df[results_df["decision"] != "MANUAL_REVIEW"].copy()

    fp_mask = (auto["is_fraud"] == False) & (auto["decision"] == "FRAUD_CONFIRMED")
    fn_mask = (auto["is_fraud"] == True) & (auto["decision"] == "LEGITIMATE_DISPUTE")

    n_fp = int(fp_mask.sum())
    n_fn = int(fn_mask.sum())
    n_review = int((results_df["decision"] == "MANUAL_REVIEW").sum())

    fp_cost = n_fp * avg_customer_ltv_impact_inr
    fn_cost = float(auto.loc[fn_mask, "amount_inr"].sum())
    review_cost = n_review * review_labor_cost_inr

    return {
        "n_false_positive": n_fp,
        "n_false_negative": n_fn,
        "n_manual_review": n_review,
        "cost_false_positive_inr": round(fp_cost, 2),
        "cost_false_negative_inr": round(fn_cost, 2),
        "cost_manual_review_inr": round(review_cost, 2),
        "total_estimated_cost_inr": round(fp_cost + fn_cost + review_cost, 2),
        "assumptions": {
            "avg_customer_ltv_impact_inr_per_fp": avg_customer_ltv_impact_inr,
            "review_labor_cost_inr_per_case": review_labor_cost_inr,
        },
    }


def threshold_sweep(scored_df: pd.DataFrame, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) -> pd.DataFrame:
    """scored_df needs columns: is_fraud (bool), risk_score (float)."""
    rows = []
    for t in thresholds:
        pred_positive = scored_df["risk_score"] >= t
        c = confusion_counts(scored_df["is_fraud"], pred_positive)
        m = prf1(c)
        rows.append({"threshold": t, **c, **m})
    return pd.DataFrame(rows)
