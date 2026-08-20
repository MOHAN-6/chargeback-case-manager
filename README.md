# 🛡️ Aegis — Multi-Agent Chargeback Case Manager

<div align="center">

![Track](https://img.shields.io/badge/Track-02%20AI%20Risk%20Manager-6C5CE7?style=for-the-badge)
![Hackathon](https://img.shields.io/badge/Razorpay-Buildathon-0B3558?style=for-the-badge&logo=razorpay&logoColor=white)
![Status](https://img.shields.io/badge/Status-Defense--Only-2ECC71?style=for-the-badge)

![Agents](https://img.shields.io/badge/Agents-2%20Cooperating-FF6B6B?style=flat-square)
![Cases](https://img.shields.io/badge/Evaluated-1%2C200%20cases-4ECDC4?style=flat-square)
![Precision](https://img.shields.io/badge/Precision-0.826-45B7D1?style=flat-square)
![Recall](https://img.shields.io/badge/Recall-0.881-96CEB4?style=flat-square)
![F1](https://img.shields.io/badge/F1-0.852-FFEAA7?style=flat-square&logoColor=black)
![Manual Review](https://img.shields.io/badge/Manual%20Review-5.3%25-FDCB6E?style=flat-square)

</div>

Stops merchants losing money to chargeback fraud by auto-resolving disputes with two cooperating agents, honest measured performance, and a complete audit trail — **strictly defense-only**.

---

## 👀 Two ways to see this run

| | |
|---|---|
| 🕹️ **`AegisConsole.jsx`** | An interactive console where you pick a real disputed-transaction case and watch a **genuinely autonomous** Investigator agent decide which tools to call (Claude with live tool-calling, not a scripted sequence), hand its evidence dossier to an Analyzer agent, and get a verdict with a reconstructable reasoning chain — same shape as the offline pipeline below, but you watch the LLM agents reason **live** instead of reading a log file. |
| 📊 **`main.py`** | The batch pipeline that produced the 1,200-case offline evaluation (`metrics_summary.json`, `audit_trail.jsonl`) the console's metrics tabs are built from. |

---

## 🏗️ 0. Two-tier architecture — why an LLM agent, and why not *only* an LLM agent

Calling an LLM on every one of a payments company's disputes is slow and expensive, and most disputes aren't actually ambiguous. Aegis runs two tiers:

- 🟢 **Tier 1 — deterministic rules engine.** Runs on 100% of disputes, sub-millisecond, same weighted-logit scorer as before. Clears the ~90%+ of cases where the evidence is unambiguous.
- 🟣 **Tier 2 — autonomous LLM agents (this console).** Only cases Tier 1 finds ambiguous, or above a value threshold, get escalated here. `InvestigatorAgent` and `AnalyzerAgent` are real Claude calls with tool-calling — the Investigator genuinely decides which of its five tools to call and in what order based on what it's found so far, rather than running a fixed checklist.

> [!NOTE]
> This mirrors how production risk systems are actually built (cheap model triages, expensive reasoning model handles the edge cases) and is why the 1,200-case offline metrics below were computed with the Tier-1 engine — calling an LLM 1,200 times for a benchmark isn't what a real system does either. The console's live demo shows what happens the moment a case *does* reach Tier 2.

---

## ❓ 1. The problem

A customer disputes a payment. Today a human analyst manually pulls velocity, device, IP and history data to decide: is this genuine unauthorized-use fraud (merchant should lose the dispute), or is it friendly/illegitimate fraud (merchant should contest and win)? This is slow, inconsistent, and doesn't scale with dispute volume.

---

## 🧩 2. Architecture — two agents, one case manager

```
Dispute filed
      │
      ▼
┌─────────────────────┐        ┌────────────────────┐
│  InvestigatorAgent   │──────▶│   AnalyzerAgent      │──────▶ Decision
│  (evidence gathering)│ dossier│  (risk judgment)     │        + reasoning
└─────────────────────┘        └────────────────────┘
      │                               │
      ▼                               ▼
  5 independent tool calls      weighted, inspectable
  logged individually            scoring + thresholds
```

### 🔍 Agent 1 — `InvestigatorAgent`
Runs five independent, auditable checks against the disputed transaction and **never makes a judgment call itself**:

| 🛠️ Tool | 📡 Signal | 🚩 Flags when |
|---|---|---|
| `check_merchant_velocity` | txns/hour from this merchant | ≥ 6 |
| `check_geo_consistency` | billing vs. IP-geolocation distance | ≥ 150 km |
| `check_device_ip_reputation` | new device **and** IP risk score | new device AND risk ≥ 0.5 |
| `check_dispute_history` | prior chargebacks (180d) | ≥ 2 — a *serial-disputer* signal |
| `check_behavioural_session` | checkout speed + saved-card age | fast, scripted-looking session on a young card |

### ⚖️ Agent 2 — `AnalyzerAgent`
Never sees the raw transaction — only the Investigator's dossier. It converts each flag into a **signed, fixed weight** on a logit scale (no black-box model), so every decision is fully reconstructable:

```
risk_score = sigmoid(bias + Σ weight_i × flag_i)

🔴 risk_score ≥ 0.70            → FRAUD_CONFIRMED
🟡 0.40 ≤ risk_score < 0.70      → MANUAL_REVIEW (escalate to human)
🟢 risk_score < 0.40             → LEGITIMATE_DISPUTE
```

> [!IMPORTANT]
> `check_dispute_history` carries a **negative** weight: a customer who has filed several chargebacks recently is evidence of serial-disputer / friendly-fraud abuse, not genuine unauthorized use — the two agents explicitly model that this signal points the *opposite* direction from the others.

This design is deliberately explainable rather than a black-box classifier, because a merchant-facing risk decision that can't produce a "why" fails the audit requirement even if its accuracy looks good.

---

## 📜 3. Audit trail

Every tool call and every decision is written to `audit_trail.jsonl` (12,000 lines for the 1,200-case test run — 10 entries per case). Each line is a single JSON object: `{ts, case_id, agent, action, finding, rationale, severity}`.

Example — one case, human-readable form (`example_case_walkthroughs.txt`):

```
AUDIT TRAIL — CB-100000
  [CaseManager] open_case: Dispute received for CB-100000 (electronics, ₹26861.10).
⚑ [InvestigatorAgent] check_merchant_velocity: 16 txns in the surrounding hour (threshold=6)
⚑ [InvestigatorAgent] check_geo_consistency: distance = 230 km (threshold=150 km)
⚑ [InvestigatorAgent] check_device_ip_reputation: new_device=True, ip_risk_score=0.79
  [InvestigatorAgent] check_dispute_history: 0 prior chargebacks (threshold=2)
  [InvestigatorAgent] check_behavioural_session: session=1.23min, saved_card_age=4d
  [InvestigatorAgent] compile_dossier: 3/5 checks flagged.
➤ [AnalyzerAgent] score_case: risk_score=0.982 → FRAUD_CONFIRMED.
   Top factors: device_ip_flag (+1.75), geo_flag (+1.55), velocity_flag (+1.35)
➤ [AnalyzerAgent] final_decision: risk≥0.70 → FRAUD_CONFIRMED. Applied: FRAUD_CONFIRMED.
  [CaseManager] close_case: decision=FRAUD_CONFIRMED.
(ground truth is_fraud = True)  ✓ correct
```

A merchant or auditor can trace *exactly* which evidence, at what threshold, produced the call — no step is opaque.

---

## 📈 4. Honest metrics

Evaluated on a **1,200-case synthetic test set** (`data_generator.py`), built with realistic imperfection baked in on purpose: 16% of fraud cases are "sophisticated" (evade the classic signals) and 10% of genuine disputes are "unlucky" (a real customer travelling, on a new phone, at a merchant having a genuine traffic spike) — so the system cannot trivially separate the classes. Full numbers in `metrics_summary.json`.

**🎯 View 1 — auto-decided cases only** (excludes the 5.3% escalated to a human; this is the honest number for "how good is the system when it commits to an answer"):

| n | Precision | Recall | F1 |
|---|---|---|---|
| 1,136 | 🟦 **0.826** | 🟩 **0.881** | 🟨 **0.852** |

Confusion matrix: ✅ TP 398 · ❌ FP 84 · ✅ TN 600 · ❌ FN 54

**🔭 View 2 — full funnel** (manual-review cases counted as "flagged for scrutiny" — the relevant view for "how much genuine fraud gets zero human eyes on it"):

| n | Precision | Recall | F1 |
|---|---|---|---|
| 1,200 | 0.753 | 0.884 | 0.813 |

**🙋 Manual review load:** 64 cases (5.3%) escalated; fraud rate within that escalated bucket is **20.3%** — roughly 4x the base rate, meaning the review threshold is successfully concentrating genuinely ambiguous, higher-risk cases in front of a human rather than dumping the whole queue on them.

> [!TIP]
> Threshold sensitivity (`threshold_sensitivity.png`, generated by `threshold_sweep.csv`) shows precision and recall as a single hard cutoff moves from 0.3 → 0.8, justifying why 0.40/0.70 were chosen as the two operating thresholds — 0.70 sits at the point where precision climbs sharply without recall collapsing.

---

## 💸 5. False-positive cost (the part that's easy to skip and shouldn't be)

A false positive here means: a **genuine** customer's dispute gets auto-declined. That customer doesn't just lose one transaction argument — they've been told by the system that their real purchase was fraud. The real cost is relationship damage, not the transaction itself, so it's priced accordingly rather than at face value:

| Error type | Count | Unit cost basis | Estimated cost |
|---|---|---|---|
| 🟠 False positive (legit → declined) | 84 | ₹1,200 assumed customer LTV/goodwill impact | ₹100,800 |
| 🔴 False negative (fraud → approved) | 54 | actual disputed transaction amount | ₹322,898 |
| 🟡 Manual review | 64 | ₹45 assumed analyst-minutes cost | ₹2,880 |
| **💰 Total estimated error + ops cost** | | | **₹426,578** |

> [!WARNING]
> This makes the trade-off explicit: tightening the DECLINE threshold reduces false positives (customer-goodwill cost) but pushes more cases into MANUAL_REVIEW or LEGITIMATE_DISPUTE, raising false-negative (direct-fraud-loss) cost. `threshold_sweep.csv` is what you'd hand to a risk-ops lead to pick the threshold for their actual cost ratios — the 0.70/0.40 split used here is one defensible choice, not the only one.

---

## 🔒 6. Why this stays defense-only

The system only ever *reads* signals that already exist in a legitimate risk pipeline (velocity, geo, device/IP reputation, dispute history, session timing) and only ever *outputs* a classification + explanation for a human/merchant-facing decision. It has no capability to act on other systems, generate synthetic identities/documents, or probe for detection gaps — it cannot be repurposed to manufacture the fraud patterns it looks for.

---

## ▶️ 7. How to run it

**🕹️ Interactive console:** open `AegisConsole.jsx` as a Claude artifact — pick a case, click "Run agent investigation," watch the Investigator decide which tools to call and the Analyzer rule on the dossier, both live LLM calls with real tool-use.

**⚙️ Offline batch pipeline:**
```bash
python3 main.py
```
Produces: `audit_trail.jsonl`, `case_results.csv`, `metrics_summary.json`, `threshold_sweep.csv`, `threshold_sensitivity.png`, `example_case_walkthroughs.txt`.

---

## 📁 8. Files

| File | Purpose |
|---|---|
| 🕹️ `AegisConsole.jsx` | Interactive console — live Tier-2 LLM agents (Investigator + Analyzer) with real tool-calling, plus the metrics/cost/threshold dashboard |
| 🧪 `data_generator.py` | Synthetic dispute dataset with realistic, imperfect fraud correlations |
| 🤖 `agents.py` | Tier-1 `InvestigatorAgent`, `AnalyzerAgent`, `AuditTrail`, orchestration (Python, deterministic — used for the 1,200-case offline benchmark) |
| 📊 `metrics.py` | Precision/recall/F1 (two views), confusion counts, false-positive cost model, threshold sweep |
| 🚀 `main.py` | End-to-end offline pipeline runner + report printer |

---

## 🔭 9. What's next (if this were productionized)

- 🔁 Replace the fixed-weight scorer with a monitored, versioned model while **keeping** the per-flag audit trail — explainability is the requirement, not the specific model class.
- ✍️ Add a third agent: an **evidence-responder** that drafts the documentation packet for the acquiring bank when `FRAUD_CONFIRMED` / contest is the call (this is the "Chargeback evidence responder" direction from the track brief) — natural extension of the same dossier this Investigator already builds.
- 🔌 Replace synthetic velocity/geo/IP-reputation signals with real service calls; the tool-call structure in `InvestigatorAgent` is already shaped for that swap.
