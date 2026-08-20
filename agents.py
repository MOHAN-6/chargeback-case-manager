"""
agents.py
---------
A two-agent case-manager pipeline for chargeback / dispute investigation.

Agent 1 - InvestigatorAgent
    Autonomously runs a fixed set of "tools" against a disputed transaction
    (velocity check, geo-consistency check, device/IP reputation check,
    dispute-history check, behavioural-session check) and assembles a
    structured evidence dossier. Every tool call is written to the audit
    trail with its raw finding and a one-line human-readable rationale.

Agent 2 - AnalyzerAgent
    Consumes the Investigator's dossier (never touches raw transaction data
    directly - this separation of concerns mirrors how Razorpay describes
    its autonomous, tool-using risk agents). It converts each evidence flag
    into a signed, weighted contribution to a risk score using a fixed,
    inspectable weight table (no black-box model), thresholds that score
    into a decision, and logs its full reasoning chain - including which
    factors pushed the decision up or down - to the audit trail.

Design choice: we use an explainable, hand-specified weighted-logit scorer
rather than a black-box classifier on purpose. RiskOps auditors and
merchants need to see *why* a case was declined/approved; a system that
cannot produce that "why" for every case fails the audit-trail requirement
even if its accuracy is good.
"""

from __future__ import annotations
import json
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


# --------------------------------------------------------------------------
# Audit trail
# --------------------------------------------------------------------------

class AuditTrail:
    """Append-only, human-readable log of every agent action for one case."""

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.entries: list[dict] = []

    def log(self, agent: str, action: str, finding: Any, rationale: str, severity: str = "info"):
        self.entries.append(
            {
                "ts": round(time.time(), 3),
                "case_id": self.case_id,
                "agent": agent,
                "action": action,
                "finding": finding,
                "rationale": rationale,
                "severity": severity,  # info | flag | decision
            }
        )

    def as_list(self):
        return self.entries

    def pretty(self) -> str:
        lines = [f"AUDIT TRAIL — {self.case_id}"]
        for e in self.entries:
            tag = {"info": "  ", "flag": "⚑ ", "decision": "➤ "}.get(e["severity"], "  ")
            lines.append(f"{tag}[{e['agent']}] {e['action']}: {e['rationale']}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Agent 1: Investigator
# --------------------------------------------------------------------------

@dataclass
class EvidenceDossier:
    case_id: str
    velocity_flag: bool
    velocity_detail: str
    geo_flag: bool
    geo_detail: str
    device_ip_flag: bool
    device_ip_detail: str
    history_flag: bool
    history_detail: str
    behavioural_flag: bool
    behavioural_detail: str
    raw: dict = field(default_factory=dict)


class InvestigatorAgent:
    """Agent 1 — gathers evidence about a disputed transaction via a fixed
    tool-belt of checks. Each `_tool_*` method is an independent, auditable
    unit (in a production system these would be real calls out to a
    velocity service, a geo/IP service, a device-fingerprint service, and
    the dispute-history store)."""

    name = "InvestigatorAgent"

    # thresholds are declared up front so they can be audited/tuned
    VELOCITY_THRESHOLD = 6          # txns/hour from the merchant
    GEO_MISMATCH_THRESHOLD_KM = 150
    IP_RISK_THRESHOLD = 0.5
    SESSION_FAST_THRESHOLD_MIN = 1.0
    STALE_CARD_THRESHOLD_DAYS = 30  # "young" saved card

    def investigate(self, txn: dict, audit: AuditTrail) -> EvidenceDossier:
        v_flag, v_detail = self._tool_velocity(txn, audit)
        g_flag, g_detail = self._tool_geo_consistency(txn, audit)
        d_flag, d_detail = self._tool_device_ip(txn, audit)
        h_flag, h_detail = self._tool_dispute_history(txn, audit)
        b_flag, b_detail = self._tool_behavioural(txn, audit)

        dossier = EvidenceDossier(
            case_id=txn["case_id"],
            velocity_flag=v_flag, velocity_detail=v_detail,
            geo_flag=g_flag, geo_detail=g_detail,
            device_ip_flag=d_flag, device_ip_detail=d_detail,
            history_flag=h_flag, history_detail=h_detail,
            behavioural_flag=b_flag, behavioural_detail=b_detail,
            raw=txn,
        )
        audit.log(
            self.name, "compile_dossier",
            finding={"flags_raised": sum([v_flag, g_flag, d_flag, h_flag, b_flag])},
            rationale=f"Evidence gathering complete. {sum([v_flag, g_flag, d_flag, h_flag, b_flag])}/5 checks flagged.",
        )
        return dossier

    def _tool_velocity(self, txn, audit):
        v = txn["merchant_velocity_1h"]
        flag = v >= self.VELOCITY_THRESHOLD
        detail = f"{v} txns from this merchant in the surrounding hour (threshold={self.VELOCITY_THRESHOLD})"
        audit.log(self.name, "check_merchant_velocity", finding=v, rationale=detail,
                   severity="flag" if flag else "info")
        return flag, detail

    def _tool_geo_consistency(self, txn, audit):
        km = txn["geo_mismatch_km"]
        flag = km >= self.GEO_MISMATCH_THRESHOLD_KM
        detail = f"Billing address vs IP-geolocation distance = {km:.0f} km (threshold={self.GEO_MISMATCH_THRESHOLD_KM} km)"
        audit.log(self.name, "check_geo_consistency", finding=km, rationale=detail,
                   severity="flag" if flag else "info")
        return flag, detail

    def _tool_device_ip(self, txn, audit):
        new_device = bool(txn["new_device"])
        ip_risk = txn["ip_risk_score"]
        flag = new_device and ip_risk >= self.IP_RISK_THRESHOLD
        detail = (f"new_device={new_device}, ip_risk_score={ip_risk:.2f} "
                  f"(flag requires new device AND ip_risk>={self.IP_RISK_THRESHOLD})")
        audit.log(self.name, "check_device_ip_reputation", finding={"new_device": new_device, "ip_risk": ip_risk},
                   rationale=detail, severity="flag" if flag else "info")
        return flag, detail

    def _tool_dispute_history(self, txn, audit):
        prior = txn["prior_chargebacks_180d"]
        # NOTE: a high prior-dispute count is a *serial-disputer* signal,
        # i.e. evidence AGAINST genuine fraud (points toward friendly fraud
        # / abuse). We flag it as its own category, sign handled by analyzer.
        flag = prior >= 2
        detail = f"{prior} prior chargebacks filed by this customer in last 180 days (serial-disputer threshold=2)"
        audit.log(self.name, "check_dispute_history", finding=prior, rationale=detail,
                   severity="flag" if flag else "info")
        return flag, detail

    def _tool_behavioural(self, txn, audit):
        fast_session = txn["session_to_checkout_min"] <= self.SESSION_FAST_THRESHOLD_MIN
        stale_saved_card = txn["saved_card_age_days"] <= self.STALE_CARD_THRESHOLD_DAYS
        flag = fast_session and stale_saved_card
        detail = (f"session_to_checkout={txn['session_to_checkout_min']:.2f} min "
                  f"(scripted-flow threshold<={self.SESSION_FAST_THRESHOLD_MIN}), "
                  f"saved_card_age={txn['saved_card_age_days']:.0f} days "
                  f"(new-card threshold<={self.STALE_CARD_THRESHOLD_DAYS})")
        audit.log(self.name, "check_behavioural_session", finding={"fast_session": fast_session, "stale_card": stale_saved_card},
                   rationale=detail, severity="flag" if flag else "info")
        return flag, detail


# --------------------------------------------------------------------------
# Agent 2: Analyzer
# --------------------------------------------------------------------------

# Fixed, inspectable weight table mapping each evidence flag to a signed
# contribution on the logit scale. Positive weight = pushes toward FRAUD.
# Negative weight = pushes toward LEGITIMATE (friendly-fraud/abuse signal).
WEIGHTS = {
    "bias": -0.65,                 # prior: most disputes are NOT genuine fraud
    "velocity_flag": 1.35,
    "geo_flag": 1.55,
    "device_ip_flag": 1.75,
    "history_flag": -1.60,         # serial disputer -> pushes AWAY from fraud
    "behavioural_flag": 1.20,
}

DECLINE_THRESHOLD = 0.70   # risk_score >= this -> confirm fraud / decline dispute in merchant's favor... 
REVIEW_THRESHOLD = 0.40    # risk_score in [REVIEW, DECLINE) -> manual review


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class AnalyzerAgent:
    """Agent 2 — converts the Investigator's evidence dossier into a risk
    score and a final decision, with a fully reconstructable reasoning
    chain (every weight, every flag, the resulting score, and the decision
    threshold applied)."""

    name = "AnalyzerAgent"

    def analyze(self, dossier: EvidenceDossier, audit: AuditTrail) -> dict:
        contributions = {
            "bias": WEIGHTS["bias"],
            "velocity_flag": WEIGHTS["velocity_flag"] if dossier.velocity_flag else 0.0,
            "geo_flag": WEIGHTS["geo_flag"] if dossier.geo_flag else 0.0,
            "device_ip_flag": WEIGHTS["device_ip_flag"] if dossier.device_ip_flag else 0.0,
            "history_flag": WEIGHTS["history_flag"] if dossier.history_flag else 0.0,
            "behavioural_flag": WEIGHTS["behavioural_flag"] if dossier.behavioural_flag else 0.0,
        }
        logit = sum(contributions.values())
        risk_score = _sigmoid(logit)

        if risk_score >= DECLINE_THRESHOLD:
            decision = "FRAUD_CONFIRMED"
        elif risk_score >= REVIEW_THRESHOLD:
            decision = "MANUAL_REVIEW"
        else:
            decision = "LEGITIMATE_DISPUTE"

        # Rank contributing factors by absolute impact for the explanation
        ranked = sorted(
            ((k, v) for k, v in contributions.items() if k != "bias"),
            key=lambda kv: abs(kv[1]), reverse=True,
        )
        top_factors = [f"{k} ({'+' if v >= 0 else ''}{v:.2f})" for k, v in ranked if v != 0]
        if not top_factors:
            top_factors = ["no risk flags raised — decision driven by base prior only"]

        rationale = (
            f"risk_score={risk_score:.3f} (logit={logit:.2f}) -> {decision}. "
            f"Top contributing factors: {', '.join(top_factors[:3])}."
        )

        audit.log(
            self.name, "score_case",
            finding={"risk_score": round(risk_score, 4), "contributions": contributions},
            rationale=rationale,
            severity="decision",
        )
        audit.log(
            self.name, "final_decision",
            finding=decision,
            rationale=(f"Decision thresholds: risk>={DECLINE_THRESHOLD} -> FRAUD_CONFIRMED, "
                       f"{REVIEW_THRESHOLD}<=risk<{DECLINE_THRESHOLD} -> MANUAL_REVIEW, "
                       f"risk<{REVIEW_THRESHOLD} -> LEGITIMATE_DISPUTE. Applied: {decision}."),
            severity="decision",
        )

        return {
            "case_id": dossier.case_id,
            "risk_score": risk_score,
            "decision": decision,
            "contributions": contributions,
            "top_factors": top_factors,
        }


# --------------------------------------------------------------------------
# Orchestrator: runs both agents for one case
# --------------------------------------------------------------------------

def investigate_case(txn: dict) -> tuple[dict, AuditTrail]:
    audit = AuditTrail(txn["case_id"])
    audit.log("CaseManager", "open_case", finding=None,
              rationale=f"Dispute received for {txn['case_id']} ({txn['merchant_category']}, "
                        f"₹{txn['amount_inr']:.2f}). Routing to InvestigatorAgent.")
    investigator = InvestigatorAgent()
    analyzer = AnalyzerAgent()
    dossier = investigator.investigate(txn, audit)
    result = analyzer.analyze(dossier, audit)
    audit.log("CaseManager", "close_case", finding=result["decision"],
              rationale=f"Case {txn['case_id']} closed with decision={result['decision']}.")
    return result, audit
