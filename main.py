"""
main.py
-------
End-to-end run of the Multi-Agent Chargeback Case Manager:
  1. Generate a synthetic dispute dataset.
  2. Run every case through InvestigatorAgent -> AnalyzerAgent.
  3. Persist a full, human-readable audit trail.
  4. Compute honest precision/recall/F1 and false-positive cost.
  5. Save a threshold-sensitivity table + chart.
"""

import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_generator import generate_dataset
from agents import investigate_case
from metrics import evaluate, false_positive_cost, threshold_sweep


def run_pipeline(n_cases: int = 1200, seed: int = 42):
    df = generate_dataset(n_cases=n_cases, seed=seed)

    results = []
    audit_lines = []

    for _, row in df.iterrows():
        txn = row.to_dict()
        result, audit = investigate_case(txn)
        results.append(result)
        audit_lines.extend(audit.as_list())

    results_df = pd.DataFrame(results)
    merged = df.merge(results_df, on="case_id")

    # ---- persist audit trail ----
    with open("audit_trail.jsonl", "w") as f:
        for entry in audit_lines:
            f.write(json.dumps(entry) + "\n")

    # ---- metrics ----
    perf = evaluate(merged[["is_fraud", "decision"]])
    cost = false_positive_cost(merged)
    sweep = threshold_sweep(merged[["is_fraud", "risk_score"]])

    merged.to_csv("case_results.csv", index=False)
    sweep.to_csv("threshold_sweep.csv", index=False)

    with open("metrics_summary.json", "w") as f:
        json.dump({"performance": perf, "false_positive_cost": cost}, f, indent=2)

    # ---- threshold chart ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sweep["threshold"], sweep["precision"], marker="o", label="Precision")
    ax.plot(sweep["threshold"], sweep["recall"], marker="o", label="Recall")
    ax.plot(sweep["threshold"], sweep["f1"], marker="o", label="F1")
    ax.axvline(0.70, color="gray", linestyle="--", linewidth=1, label="DECLINE threshold (0.70)")
    ax.axvline(0.40, color="lightgray", linestyle="--", linewidth=1, label="REVIEW threshold (0.40)")
    ax.set_xlabel("Risk score threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs. decision threshold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig("threshold_sensitivity.png", dpi=150)
    plt.close(fig)

    return merged, perf, cost, sweep


def print_summary(perf, cost):
    print("=" * 70)
    print("PERFORMANCE — AUTO-DECIDED CASES (MANUAL_REVIEW excluded)")
    print("=" * 70)
    a = perf["auto_decided"]
    print(f"  n = {a['n_cases']}")
    print(f"  confusion: {a['counts']}")
    print(f"  precision = {a['precision']:.3f}   recall = {a['recall']:.3f}   f1 = {a['f1']:.3f}")

    print("\n" + "=" * 70)
    print("PERFORMANCE — FULL FUNNEL (MANUAL_REVIEW counted as 'flagged')")
    print("=" * 70)
    b = perf["full_funnel_flagged"]
    print(f"  n = {b['n_cases']}")
    print(f"  confusion: {b['counts']}")
    print(f"  precision = {b['precision']:.3f}   recall = {b['recall']:.3f}   f1 = {b['f1']:.3f}")

    r = perf["manual_review"]
    print("\n" + "=" * 70)
    print("MANUAL REVIEW LOAD")
    print("=" * 70)
    print(f"  {r['n_cases']} cases ({r['pct_of_total']:.1%} of total) escalated to a human.")
    print(f"  Fraud rate within escalated cases: {r['fraud_rate_within_review']:.1%}"
          if r['fraud_rate_within_review'] is not None else "  (no cases escalated)")

    print("\n" + "=" * 70)
    print("FALSE-POSITIVE / ERROR COST ESTIMATE")
    print("=" * 70)
    print(f"  False positives (legit dispute wrongly declined): {cost['n_false_positive']} cases"
          f" -> ₹{cost['cost_false_positive_inr']:,.0f}")
    print(f"  False negatives (fraud wrongly approved):          {cost['n_false_negative']} cases"
          f" -> ₹{cost['cost_false_negative_inr']:,.0f}")
    print(f"  Manual review labor:                               {cost['n_manual_review']} cases"
          f" -> ₹{cost['cost_manual_review_inr']:,.0f}")
    print(f"  TOTAL estimated error+ops cost: ₹{cost['total_estimated_cost_inr']:,.0f}")


def example_case_walkthrough(merged: pd.DataFrame):
    """Re-run one FRAUD_CONFIRMED case and one MANUAL_REVIEW case verbosely."""
    from agents import investigate_case

    picks = []
    fraud_case = merged[merged["decision"] == "FRAUD_CONFIRMED"].head(1)
    review_case = merged[merged["decision"] == "MANUAL_REVIEW"].head(1)
    if len(fraud_case):
        picks.append(fraud_case.iloc[0]["case_id"])
    if len(review_case):
        picks.append(review_case.iloc[0]["case_id"])

    df = merged.set_index("case_id")
    out_text = []
    for cid in picks:
        row = df.loc[cid]
        txn = {
            "case_id": cid,
            "merchant_category": row["merchant_category"],
            "amount_inr": row["amount_inr"],
            "customer_account_age_days": row["customer_account_age_days"],
            "merchant_velocity_1h": row["merchant_velocity_1h"],
            "geo_mismatch_km": row["geo_mismatch_km"],
            "new_device": row["new_device"],
            "ip_risk_score": row["ip_risk_score"],
            "prior_chargebacks_180d": row["prior_chargebacks_180d"],
            "session_to_checkout_min": row["session_to_checkout_min"],
            "saved_card_age_days": row["saved_card_age_days"],
        }
        _, audit = investigate_case(txn)
        out_text.append(audit.pretty())
        out_text.append(f"(ground truth is_fraud = {row['is_fraud']})")
        out_text.append("")
    text = "\n".join(out_text)
    with open("example_case_walkthroughs.txt", "w") as f:
        f.write(text)
    print("\n" + "=" * 70)
    print("EXAMPLE CASE WALKTHROUGH (saved to example_case_walkthroughs.txt)")
    print("=" * 70)
    print(text)


if __name__ == "__main__":
    merged, perf, cost, sweep = run_pipeline(n_cases=1200, seed=42)
    print_summary(perf, cost)
    example_case_walkthrough(merged)
    print("\nArtifacts written: audit_trail.jsonl, case_results.csv, metrics_summary.json,"
          " threshold_sweep.csv, threshold_sensitivity.png, example_case_walkthroughs.txt")
