import React, { useState, useMemo, useRef, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell,
} from "recharts";

/* ---------------------------------------------------------------------- */
/* Design tokens                                                          */
/* ---------------------------------------------------------------------- */
const T = {
  ink: "#0E1116",
  surface: "#161A21",
  surfaceRaised: "#1C212A",
  hair: "#2A3040",
  hairStrong: "#394054",
  textPri: "#E9EBF1",
  textSec: "#9098AC",
  textMute: "#5C6376",
  amber: "#D9861F",
  amberDim: "#8A611E",
  teal: "#3FA48B",
  tealDim: "#2C6E5E",
  mustard: "#C9A227",
  mustardDim: "#8A7020",
  brick: "#C1453C",
  brickDim: "#7D2E29",
  wire: "#5A85B0",
};

const FONT_DISPLAY = "'Space Grotesk', 'Arial Narrow', sans-serif";
const FONT_MONO = "'IBM Plex Mono', 'SF Mono', ui-monospace, monospace";
const FONT_BODY = "'IBM Plex Sans', -apple-system, sans-serif";

/* ---------------------------------------------------------------------- */
/* Case data — pulled from the 1,200-case offline evaluation set          */
/* ---------------------------------------------------------------------- */
const CASES = [
  {
    case_id: "CB-100000", tag: "Classic pattern", merchant_category: "electronics",
    amount_inr: 26861.10, customer_account_age_days: 56, merchant_velocity_1h: 16,
    geo_mismatch_km: 230.4, new_device: true, ip_risk_score: 0.792,
    prior_chargebacks_180d: 0, session_to_checkout_min: 1.23, saved_card_age_days: 4,
    is_fraud: true,
  },
  {
    case_id: "CB-100017", tag: "Serial disputer", merchant_category: "fashion",
    amount_inr: 7270.60, customer_account_age_days: 490, merchant_velocity_1h: 4,
    geo_mismatch_km: 3.6, new_device: false, ip_risk_score: 0.217,
    prior_chargebacks_180d: 3, session_to_checkout_min: 3.1, saved_card_age_days: 86,
    is_fraud: false,
  },
  {
    case_id: "CB-100055", tag: "Evasive pattern", merchant_category: "food_delivery",
    amount_inr: 4618.62, customer_account_age_days: 9, merchant_velocity_1h: 0,
    geo_mismatch_km: 134.5, new_device: false, ip_risk_score: 0.254,
    prior_chargebacks_180d: 0, session_to_checkout_min: 12.68, saved_card_age_days: 3,
    is_fraud: true,
  },
  {
    case_id: "CB-100043", tag: "Traveling customer", merchant_category: "fashion",
    amount_inr: 2925.88, customer_account_age_days: 238, merchant_velocity_1h: 12,
    geo_mismatch_km: 731.3, new_device: false, ip_risk_score: 0.69,
    prior_chargebacks_180d: 1, session_to_checkout_min: 4.83, saved_card_age_days: 2,
    is_fraud: false,
  },
  {
    case_id: "CB-100104", tag: "Borderline", merchant_category: "electronics",
    amount_inr: 1765.49, customer_account_age_days: 44, merchant_velocity_1h: 6,
    geo_mismatch_km: 133.2, new_device: false, ip_risk_score: 0.603,
    prior_chargebacks_180d: 0, session_to_checkout_min: 1.86, saved_card_age_days: 30,
    is_fraud: true,
  },
  {
    case_id: "CB-100036", tag: "Borderline", merchant_category: "travel",
    amount_inr: 2399.27, customer_account_age_days: 389, merchant_velocity_1h: 6,
    geo_mismatch_km: 50.6, new_device: false, ip_risk_score: 0.649,
    prior_chargebacks_180d: 0, session_to_checkout_min: 12.0, saved_card_age_days: 64,
    is_fraud: false,
  },
];

/* ---------------------------------------------------------------------- */
/* Tier-1 deterministic rules engine (mirrors the offline-evaluated model)*/
/* ---------------------------------------------------------------------- */
const TIER1_WEIGHTS = { bias: -0.65, velocity: 1.35, geo: 1.55, deviceIp: 1.75, history: -1.60, behavioural: 1.20 };
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

function tier1Score(c) {
  const velocity_flag = c.merchant_velocity_1h >= 6;
  const geo_flag = c.geo_mismatch_km >= 150;
  const device_ip_flag = c.new_device && c.ip_risk_score >= 0.5;
  const history_flag = c.prior_chargebacks_180d >= 2;
  const behavioural_flag = c.session_to_checkout_min <= 1.0 && c.saved_card_age_days <= 30;
  const logit = TIER1_WEIGHTS.bias
    + (velocity_flag ? TIER1_WEIGHTS.velocity : 0)
    + (geo_flag ? TIER1_WEIGHTS.geo : 0)
    + (device_ip_flag ? TIER1_WEIGHTS.deviceIp : 0)
    + (history_flag ? TIER1_WEIGHTS.history : 0)
    + (behavioural_flag ? TIER1_WEIGHTS.behavioural : 0);
  const score = sigmoid(logit);
  const decision = score >= 0.70 ? "FRAUD_CONFIRMED" : score >= 0.40 ? "MANUAL_REVIEW" : "LEGITIMATE_DISPUTE";
  return { score, decision, flags: { velocity_flag, geo_flag, device_ip_flag, history_flag, behavioural_flag } };
}

/* ---------------------------------------------------------------------- */
/* Tool implementations the Investigator agent can call                  */
/* ---------------------------------------------------------------------- */
function runTool(name, c) {
  switch (name) {
    case "check_merchant_velocity": {
      const flag = c.merchant_velocity_1h >= 6;
      return { flag, value: c.merchant_velocity_1h, threshold: 6,
        detail: `${c.merchant_velocity_1h} transactions from this merchant in the surrounding hour (threshold = 6).` };
    }
    case "check_geo_consistency": {
      const flag = c.geo_mismatch_km >= 150;
      return { flag, value: c.geo_mismatch_km, threshold: 150,
        detail: `Billing address vs. IP-geolocation distance = ${c.geo_mismatch_km.toFixed(0)} km (threshold = 150 km).` };
    }
    case "check_device_ip_reputation": {
      const flag = c.new_device && c.ip_risk_score >= 0.5;
      return { flag, new_device: c.new_device, ip_risk_score: c.ip_risk_score,
        detail: `new_device=${c.new_device}, ip_risk_score=${c.ip_risk_score.toFixed(2)} (flag requires new device AND ip_risk >= 0.5).` };
    }
    case "check_dispute_history": {
      const flag = c.prior_chargebacks_180d >= 2;
      return { flag, value: c.prior_chargebacks_180d, threshold: 2,
        detail: `${c.prior_chargebacks_180d} prior chargebacks filed by this customer in the last 180 days (serial-disputer threshold = 2). A high count here is evidence AGAINST genuine fraud.` };
    }
    case "check_behavioural_session": {
      const fast = c.session_to_checkout_min <= 1.0;
      const stale = c.saved_card_age_days <= 30;
      const flag = fast && stale;
      return { flag, session_to_checkout_min: c.session_to_checkout_min, saved_card_age_days: c.saved_card_age_days,
        detail: `session_to_checkout=${c.session_to_checkout_min.toFixed(2)} min (scripted-flow threshold <= 1.0), saved_card_age=${c.saved_card_age_days.toFixed(0)} days (new-card threshold <= 30).` };
    }
    default:
      return { flag: false, detail: "Unknown tool." };
  }
}

const TOOL_DEFS = [
  { name: "check_merchant_velocity", description: "Checks how many transactions this merchant processed in the hour surrounding the disputed transaction. A spike can indicate a card-testing or fraud-ring event hitting the merchant.", input_schema: { type: "object", properties: {} } },
  { name: "check_geo_consistency", description: "Compares the billing-address geolocation to the IP-derived geolocation at the time of the transaction and returns the distance in km.", input_schema: { type: "object", properties: {} } },
  { name: "check_device_ip_reputation", description: "Checks whether the device fingerprint used was new to this account, and returns an IP reputation risk score (0=clean, 1=known-bad / proxy / datacenter).", input_schema: { type: "object", properties: {} } },
  { name: "check_dispute_history", description: "Returns how many chargebacks this same customer has filed in the last 180 days. Note: a high count is a serial-disputer / friendly-fraud signal, not a genuine-fraud signal.", input_schema: { type: "object", properties: {} } },
  { name: "check_behavioural_session", description: "Returns session timing signals: how quickly the customer went from login to checkout, and how old the saved card token is. Very fast, scripted-looking sessions on newly-saved cards are a fraud signal.", input_schema: { type: "object", properties: {} } },
];

const INVESTIGATOR_SYSTEM = `You are the Investigator agent inside Aegis, a multi-agent chargeback-fraud case manager for an Indian payments company. A dispute has been escalated to you because Aegis's Tier-1 rules engine flagged it as ambiguous or high-value and needs a deeper look before a human or the Analyzer agent rules on it.

You have five independent tools that each pull one risk signal for THIS case (no parameters needed). Call whichever tools you judge necessary to build a solid evidence dossier - use your judgment about how many you need, you do not have to call all five if the picture is already clear early, but for a case worth escalating you should typically call most of them so the Analyzer isn't working blind.

After you've gathered what you need, respond with a short plain-text evidence summary (2-4 sentences, no more tool calls, no decision - the decision is the Analyzer's job, not yours).`;

const ANALYZER_SYSTEM = `You are the Analyzer agent inside Aegis, a multi-agent chargeback-fraud case manager. You never see the raw transaction - only the Investigator agent's evidence dossier below. Weigh the evidence and rule on this dispute.

Remember: a HIGH count of prior chargebacks from this same customer is itself evidence of serial-disputer / friendly-fraud abuse, not genuine unauthorized use - it should REDUCE, not raise, your fraud assessment.

Respond with ONLY a JSON object, no markdown code fences, no other text, in exactly this shape:
{"risk_score": <0-1 float, your calibrated probability this is genuine unauthorized-use fraud>, "decision": "FRAUD_CONFIRMED" | "MANUAL_REVIEW" | "LEGITIMATE_DISPUTE", "reasoning": "<3-5 sentences citing specific evidence>", "top_factors": ["<short factor 1>", "<short factor 2>", "<short factor 3>"]}

Operating guidance (use judgment, not hard rules): risk_score >= 0.70 -> FRAUD_CONFIRMED, 0.40-0.70 -> MANUAL_REVIEW, < 0.40 -> LEGITIMATE_DISPUTE.`;

/* ---------------------------------------------------------------------- */
/* Offline metrics (computed on the 1,200-case synthetic evaluation set) */
/* ---------------------------------------------------------------------- */
const PERF_AUTO = { n: 1136, tp: 398, fp: 84, tn: 600, fn: 54, precision: 0.826, recall: 0.881, f1: 0.852 };
const PERF_FULL = { n: 1200, tp: 411, fp: 135, tn: 600, fn: 54, precision: 0.753, recall: 0.884, f1: 0.813 };
const REVIEW = { n: 64, pct: 0.053, fraudRateWithin: 0.203 };
const COST = { fp: 100800, fn: 322898, review: 2880, total: 426578 };
const SWEEP = [
  { threshold: 0.3, precision: 0.446, recall: 1.0, f1: 0.617 },
  { threshold: 0.4, precision: 0.753, recall: 0.884, f1: 0.813 },
  { threshold: 0.5, precision: 0.753, recall: 0.884, f1: 0.813 },
  { threshold: 0.6, precision: 0.753, recall: 0.884, f1: 0.813 },
  { threshold: 0.7, precision: 0.826, recall: 0.856, f1: 0.841 },
  { threshold: 0.8, precision: 0.850, recall: 0.815, f1: 0.832 },
];

/* ---------------------------------------------------------------------- */
/* Small UI atoms                                                         */
/* ---------------------------------------------------------------------- */
function Badge({ children, color, dim }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 6, fontFamily: FONT_MONO,
      fontSize: 11, letterSpacing: "0.04em", textTransform: "uppercase",
      color, background: `${color}18`, border: `1px solid ${color}55`,
      borderRadius: 3, padding: "3px 8px",
    }}>{children}</span>
  );
}

function decisionColor(decision) {
  if (decision === "FRAUD_CONFIRMED") return T.brick;
  if (decision === "MANUAL_REVIEW") return T.mustard;
  if (decision === "LEGITIMATE_DISPUTE") return T.teal;
  return T.textSec;
}
function decisionLabel(decision) {
  if (decision === "FRAUD_CONFIRMED") return "Fraud confirmed";
  if (decision === "MANUAL_REVIEW") return "Manual review";
  if (decision === "LEGITIMATE_DISPUTE") return "Legitimate dispute";
  return decision;
}

/* ---------------------------------------------------------------------- */
/* Main component                                                         */
/* ---------------------------------------------------------------------- */
export default function AegisConsole() {
  const [selectedId, setSelectedId] = useState(CASES[0].case_id);
  const selected = useMemo(() => CASES.find((c) => c.case_id === selectedId), [selectedId]);
  const tier1 = useMemo(() => tier1Score(selected), [selected]);

  const [status, setStatus] = useState("idle"); // idle | investigating | analyzing | done | error
  const [log, setLog] = useState([]);
  const [verdict, setVerdict] = useState(null);
  const [revealed, setRevealed] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [activeTab, setActiveTab] = useState("performance");
  const logEndRef = useRef(null);

  useEffect(() => {
    if (logEndRef.current) logEndRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [log]);

  function resetForNewCase(id) {
    setSelectedId(id);
    setStatus("idle");
    setLog([]);
    setVerdict(null);
    setRevealed(false);
    setErrorMsg("");
  }

  function pushLog(entry) {
    setLog((prev) => [...prev, { ...entry, id: prev.length, ts: Date.now() }]);
  }

  async function callClaude(messages, system, tools) {
    const body = { model: "claude-sonnet-4-6", max_tokens: 1000, system, messages };
    if (tools) body.tools = tools;
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const t = await res.text().catch(() => "");
      throw new Error(`API error ${res.status}: ${t.slice(0, 200)}`);
    }
    return res.json();
  }

  async function runInvestigation() {
    setStatus("investigating");
    setLog([]);
    setVerdict(null);
    setRevealed(false);
    setErrorMsg("");

    pushLog({ agent: "CaseManager", action: "open_case", severity: "info",
      rationale: `Case ${selected.case_id} escalated by Tier-1 (score ${tier1.score.toFixed(2)}). Routing to InvestigatorAgent.` });

    const caseBrief = `New dispute: ${selected.case_id}. Merchant category: ${selected.merchant_category}. Amount: ₹${selected.amount_inr.toFixed(2)}. Customer account age: ${selected.customer_account_age_days} days. Investigate this case.`;

    let messages = [{ role: "user", content: caseBrief }];
    let dossierFlags = {};
    let narrative = "";

    try {
      for (let turn = 0; turn < 6; turn++) {
        const resp = await callClaude(messages, INVESTIGATOR_SYSTEM, TOOL_DEFS);
        const content = resp.content || [];
        const toolUses = content.filter((b) => b.type === "tool_use");
        const textBlocks = content.filter((b) => b.type === "text");

        if (toolUses.length === 0) {
          narrative = textBlocks.map((b) => b.text).join(" ").trim();
          pushLog({ agent: "InvestigatorAgent", action: "compile_dossier", severity: "info",
            rationale: narrative || "Evidence gathering complete." });
          break;
        }

        messages.push({ role: "assistant", content });
        const toolResults = [];
        for (const tu of toolUses) {
          const result = runTool(tu.name, selected);
          dossierFlags[tu.name] = result;
          pushLog({ agent: "InvestigatorAgent", action: tu.name,
            severity: result.flag ? "flag" : "info", rationale: result.detail });
          toolResults.push({ type: "tool_result", tool_use_id: tu.id, content: JSON.stringify(result) });
        }
        messages.push({ role: "user", content: toolResults });
      }

      const flagsRaised = Object.values(dossierFlags).filter((f) => f.flag).length;
      pushLog({ agent: "InvestigatorAgent", action: "dossier_stats", severity: "info",
        rationale: `${Object.keys(dossierFlags).length}/5 tools called, ${flagsRaised} flagged.` });

      setStatus("analyzing");
      const dossierText = Object.entries(dossierFlags)
        .map(([name, r]) => `- ${name}: ${r.detail}`).join("\n");
      const analyzerInput = `Case ${selected.case_id} (${selected.merchant_category}, ₹${selected.amount_inr.toFixed(2)})\n\nInvestigator's evidence:\n${dossierText}\n\nInvestigator's summary: ${narrative}`;

      const aResp = await callClaude([{ role: "user", content: analyzerInput }], ANALYZER_SYSTEM, null);
      const aText = (aResp.content || []).filter((b) => b.type === "text").map((b) => b.text).join("").trim();
      const cleaned = aText.replace(/^```json/i, "").replace(/^```/, "").replace(/```$/, "").trim();
      const parsed = JSON.parse(cleaned);

      pushLog({ agent: "AnalyzerAgent", action: "score_case", severity: "decision",
        rationale: `risk_score=${Number(parsed.risk_score).toFixed(3)} -> ${decisionLabel(parsed.decision)}. ${parsed.reasoning}` });
      pushLog({ agent: "CaseManager", action: "close_case", severity: "decision",
        rationale: `Case ${selected.case_id} closed with decision=${decisionLabel(parsed.decision)}.` });

      setVerdict(parsed);
      setStatus("done");
    } catch (err) {
      setErrorMsg(err.message || String(err));
      setStatus("error");
      pushLog({ agent: "CaseManager", action: "error", severity: "flag",
        rationale: `Investigation failed: ${err.message || err}` });
    }
  }

  return (
    <div style={{
      fontFamily: FONT_BODY, background: T.ink, color: T.textPri,
      padding: "28px 24px", borderRadius: 14, maxWidth: 1080, margin: "0 auto",
      border: `1px solid ${T.hair}`,
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap');
        @keyframes stampIn { 0% { opacity:0; transform: scale(2.2) rotate(-14deg);} 60% { opacity:1; transform: scale(0.92) rotate(-8deg);} 100% { opacity:1; transform: scale(1) rotate(-8deg);} }
        @keyframes fadeUp { from { opacity:0; transform: translateY(4px);} to {opacity:1; transform: translateY(0);} }
        .aegis-stamp { animation: stampIn 0.45s cubic-bezier(.2,.8,.3,1.2); }
        .aegis-row { animation: fadeUp 0.25s ease-out; }
        .aegis-case-card:hover { border-color: ${T.hairStrong} !important; }
        .aegis-btn:hover { filter: brightness(1.12); }
        .aegis-btn:active { transform: scale(0.98); }
        .aegis-tab { cursor:pointer; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-thumb { background: ${T.hairStrong}; border-radius: 3px; }
      `}</style>

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 22, borderBottom: `1px solid ${T.hair}`, paddingBottom: 18 }}>
        <div>
          <div style={{ fontFamily: FONT_MONO, fontSize: 11, letterSpacing: "0.12em", color: T.amber, marginBottom: 6 }}>AEGIS · CASE MANAGER</div>
          <div style={{ fontFamily: FONT_DISPLAY, fontWeight: 600, fontSize: 26, letterSpacing: "-0.01em" }}>Chargeback investigation console</div>
        </div>
        <div style={{ textAlign: "right", fontFamily: FONT_MONO, fontSize: 11, color: T.textMute }}>
          <div>1,200 cases evaluated offline</div>
          <div>Track 02 — AI Risk Manager</div>
        </div>
      </div>

      {/* Tier funnel strip */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 24, fontFamily: FONT_MONO, fontSize: 12 }}>
        <FunnelStep label="Disputes filed" value="1,200" color={T.textSec} />
        <Arrow />
        <FunnelStep label="Tier-1 rules engine (instant)" value="auto-clears 94.7%" color={T.wire} />
        <Arrow />
        <FunnelStep label="Tier-2 agents (this console)" value="ambiguous / high-value" color={T.amber} highlight />
        <Arrow />
        <FunnelStep label="Human review" value="5.3% of total" color={T.mustard} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr", gap: 20 }}>
        {/* Case selector */}
        <div>
          <div style={{ fontFamily: FONT_MONO, fontSize: 11, letterSpacing: "0.08em", color: T.textMute, marginBottom: 10 }}>SAMPLE CASES</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {CASES.map((c) => {
              const t1 = tier1Score(c);
              const active = c.case_id === selectedId;
              return (
                <div key={c.case_id} className="aegis-case-card" onClick={() => resetForNewCase(c.case_id)}
                  style={{
                    cursor: "pointer", padding: "10px 12px", borderRadius: 8,
                    border: `1px solid ${active ? T.amber : T.hair}`,
                    background: active ? "#241C11" : T.surface, transition: "border-color 0.15s",
                  }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontFamily: FONT_MONO, fontSize: 12, color: T.textPri }}>{c.case_id}</span>
                    <span style={{ fontFamily: FONT_MONO, fontSize: 10, color: decisionColor(t1.decision) }}>{t1.score.toFixed(2)}</span>
                  </div>
                  <div style={{ fontSize: 12, color: T.textSec, marginTop: 3 }}>{c.tag}</div>
                  <div style={{ fontFamily: FONT_MONO, fontSize: 11, color: T.textMute, marginTop: 2 }}>₹{c.amount_inr.toLocaleString("en-IN")} · {c.merchant_category}</div>
                </div>
              );
            })}
          </div>

          {/* Tier-1 readout for selected case */}
          <div style={{ marginTop: 16, padding: 12, borderRadius: 8, background: T.surface, border: `1px solid ${T.hair}` }}>
            <div style={{ fontFamily: FONT_MONO, fontSize: 10, color: T.textMute, marginBottom: 6 }}>TIER-1 SCORE (INSTANT)</div>
            <div style={{ fontFamily: FONT_MONO, fontSize: 20, color: decisionColor(tier1.decision) }}>{tier1.score.toFixed(3)}</div>
            <div style={{ fontSize: 12, color: T.textSec, marginTop: 2 }}>{decisionLabel(tier1.decision)}</div>
          </div>
        </div>

        {/* Investigation panel */}
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <div>
              <div style={{ fontFamily: FONT_DISPLAY, fontWeight: 600, fontSize: 17 }}>{selected.case_id}</div>
              <div style={{ fontSize: 12, color: T.textSec, marginTop: 2 }}>
                {selected.merchant_category} · ₹{selected.amount_inr.toLocaleString("en-IN")} · account age {selected.customer_account_age_days}d
              </div>
            </div>
            <button className="aegis-btn" disabled={status === "investigating" || status === "analyzing"} onClick={runInvestigation}
              style={{
                fontFamily: FONT_MONO, fontSize: 12, letterSpacing: "0.04em", color: "#1A1206",
                background: T.amber, border: "none", borderRadius: 6, padding: "9px 16px",
                cursor: status === "investigating" || status === "analyzing" ? "default" : "pointer",
                opacity: status === "investigating" || status === "analyzing" ? 0.6 : 1,
              }}>
              {status === "investigating" ? "INVESTIGATING…" : status === "analyzing" ? "ANALYZING…" : "RUN AGENT INVESTIGATION"}
            </button>
          </div>

          {/* Live audit feed */}
          <div style={{
            background: T.surface, border: `1px solid ${T.hair}`, borderRadius: 8,
            padding: 14, minHeight: 160, maxHeight: 300, overflowY: "auto",
          }}>
            {log.length === 0 && status === "idle" && (
              <div style={{ fontSize: 13, color: T.textMute, fontStyle: "italic" }}>
                No investigation run yet. Click "Run agent investigation" to send this case to the live Investigator and Analyzer agents.
              </div>
            )}
            {log.map((e) => (
              <div key={e.id} className="aegis-row" style={{ display: "flex", gap: 10, padding: "6px 0", borderBottom: `1px solid ${T.hair}` }}>
                <span style={{
                  fontFamily: FONT_MONO, fontSize: 10, minWidth: 92, textAlign: "right",
                  color: e.agent === "InvestigatorAgent" ? T.wire : e.agent === "AnalyzerAgent" ? T.amber : T.textMute,
                  paddingTop: 1,
                }}>{e.agent.replace("Agent", "")}</span>
                <span style={{ width: 6, height: 6, borderRadius: "50%", marginTop: 5, flexShrink: 0,
                  background: e.severity === "flag" ? T.mustard : e.severity === "decision" ? T.amber : T.hairStrong }} />
                <div style={{ fontSize: 12.5, lineHeight: 1.5 }}>
                  <span style={{ fontFamily: FONT_MONO, fontSize: 11, color: T.textSec }}>{e.action}</span>
                  <span style={{ color: T.textSec }}> — </span>
                  <span style={{ color: T.textPri }}>{e.rationale}</span>
                </div>
              </div>
            ))}
            <div ref={logEndRef} />
          </div>

          {status === "error" && (
            <div style={{ marginTop: 10, fontSize: 12.5, color: T.brick, fontFamily: FONT_MONO }}>{errorMsg}</div>
          )}

          {/* Verdict stamp */}
          {verdict && (
            <div style={{ marginTop: 18, display: "flex", gap: 16, alignItems: "flex-start" }}>
              <div className="aegis-stamp" style={{
                border: `3px solid ${decisionColor(verdict.decision)}`, borderRadius: 8,
                padding: "10px 16px", transform: "rotate(-8deg)", flexShrink: 0,
              }}>
                <div style={{ fontFamily: FONT_DISPLAY, fontWeight: 600, fontSize: 15, letterSpacing: "0.04em",
                  color: decisionColor(verdict.decision), textTransform: "uppercase", whiteSpace: "nowrap" }}>
                  {decisionLabel(verdict.decision)}
                </div>
                <div style={{ fontFamily: FONT_MONO, fontSize: 11, color: T.textSec, marginTop: 2 }}>
                  risk {Number(verdict.risk_score).toFixed(3)}
                </div>
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 13, lineHeight: 1.6, color: T.textPri, marginBottom: 8 }}>{verdict.reasoning}</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
                  {(verdict.top_factors || []).map((f, i) => (
                    <Badge key={i} color={T.amber}>{f}</Badge>
                  ))}
                </div>
                {!revealed ? (
                  <button className="aegis-btn" onClick={() => setRevealed(true)}
                    style={{ fontFamily: FONT_MONO, fontSize: 11, background: "transparent", color: T.textSec,
                      border: `1px solid ${T.hairStrong}`, borderRadius: 6, padding: "6px 12px", cursor: "pointer" }}>
                    REVEAL GROUND TRUTH
                  </button>
                ) : (
                  <div style={{ fontFamily: FONT_MONO, fontSize: 12 }}>
                    Ground truth: <span style={{ color: selected.is_fraud ? T.brick : T.teal }}>{selected.is_fraud ? "genuine fraud" : "legitimate dispute"}</span>
                    {" · "}
                    <span style={{ color: outcomeColor(verdict.decision, selected.is_fraud) }}>
                      agent {verdictCorrect(verdict.decision, selected.is_fraud)}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Metrics dashboard */}
      <div style={{ marginTop: 32, borderTop: `1px solid ${T.hair}`, paddingTop: 22 }}>
        <div style={{ display: "flex", gap: 18, marginBottom: 18 }}>
          {["performance", "cost", "threshold"].map((t) => (
            <div key={t} className="aegis-tab" onClick={() => setActiveTab(t)}
              style={{
                fontFamily: FONT_MONO, fontSize: 12, letterSpacing: "0.06em", textTransform: "uppercase",
                color: activeTab === t ? T.amber : T.textMute, paddingBottom: 8,
                borderBottom: activeTab === t ? `2px solid ${T.amber}` : "2px solid transparent",
              }}>
              {t === "performance" ? "Performance" : t === "cost" ? "Cost analysis" : "Threshold tuning"}
            </div>
          ))}
        </div>

        {activeTab === "performance" && <PerformanceTab />}
        {activeTab === "cost" && <CostTab />}
        {activeTab === "threshold" && <ThresholdTab />}
      </div>
    </div>
  );
}

function verdictCorrect(decision, isFraud) {
  if (decision === "MANUAL_REVIEW") return "escalated (no auto-call made)";
  const predictedFraud = decision === "FRAUD_CONFIRMED";
  return predictedFraud === isFraud ? "correct" : "incorrect";
}

function outcomeColor(decision, isFraud) {
  if (decision === "MANUAL_REVIEW") return T.mustard;
  const predictedFraud = decision === "FRAUD_CONFIRMED";
  return predictedFraud === isFraud ? T.teal : T.brick;
}

function Arrow() {
  return <span style={{ color: T.hairStrong }}>→</span>;
}

function FunnelStep({ label, value, color, highlight }) {
  return (
    <div style={{
      flex: 1, padding: "8px 10px", borderRadius: 6,
      border: `1px solid ${highlight ? color : T.hair}`,
      background: highlight ? `${color}14` : T.surface,
    }}>
      <div style={{ color: T.textMute, fontSize: 10 }}>{label}</div>
      <div style={{ color, fontSize: 13, marginTop: 2 }}>{value}</div>
    </div>
  );
}

/* ---------------------------------------------------------------------- */
/* Metrics tabs                                                           */
/* ---------------------------------------------------------------------- */
function StatCard({ label, value, sub, color }) {
  return (
    <div style={{ background: T.surface, border: `1px solid ${T.hair}`, borderRadius: 8, padding: "14px 16px", flex: 1 }}>
      <div style={{ fontFamily: FONT_MONO, fontSize: 10, color: T.textMute, letterSpacing: "0.05em" }}>{label}</div>
      <div style={{ fontFamily: FONT_DISPLAY, fontWeight: 600, fontSize: 24, color: color || T.textPri, marginTop: 4 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: T.textSec, marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

function ConfusionGrid({ counts, title }) {
  const cellStyle = (bg, fg) => ({
    background: bg, color: fg, borderRadius: 6, padding: "12px 10px", textAlign: "center",
  });
  return (
    <div style={{ flex: 1 }}>
      <div style={{ fontFamily: FONT_MONO, fontSize: 11, color: T.textMute, marginBottom: 8 }}>{title}</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
        <div style={cellStyle(`${T.teal}22`, T.teal)}>
          <div style={{ fontFamily: FONT_MONO, fontSize: 20 }}>{counts.tn}</div>
          <div style={{ fontSize: 10, marginTop: 2 }}>true negative</div>
        </div>
        <div style={cellStyle(`${T.brick}22`, T.brick)}>
          <div style={{ fontFamily: FONT_MONO, fontSize: 20 }}>{counts.fp}</div>
          <div style={{ fontSize: 10, marginTop: 2 }}>false positive</div>
        </div>
        <div style={cellStyle(`${T.brick}22`, T.brick)}>
          <div style={{ fontFamily: FONT_MONO, fontSize: 20 }}>{counts.fn}</div>
          <div style={{ fontSize: 10, marginTop: 2 }}>false negative</div>
        </div>
        <div style={cellStyle(`${T.teal}22`, T.teal)}>
          <div style={{ fontFamily: FONT_MONO, fontSize: 20 }}>{counts.tp}</div>
          <div style={{ fontSize: 10, marginTop: 2 }}>true positive</div>
        </div>
      </div>
    </div>
  );
}

function PerformanceTab() {
  return (
    <div>
      <div style={{ display: "flex", gap: 12, marginBottom: 18 }}>
        <StatCard label="PRECISION (AUTO-DECIDED)" value={PERF_AUTO.precision.toFixed(3)} color={T.amber} />
        <StatCard label="RECALL (AUTO-DECIDED)" value={PERF_AUTO.recall.toFixed(3)} color={T.amber} />
        <StatCard label="F1 (AUTO-DECIDED)" value={PERF_AUTO.f1.toFixed(3)} color={T.amber} />
        <StatCard label="MANUAL REVIEW LOAD" value={`${(REVIEW.pct * 100).toFixed(1)}%`}
          sub={`${(REVIEW.fraudRateWithin * 100).toFixed(1)}% fraud rate within reviewed`} />
      </div>
      <div style={{ display: "flex", gap: 20 }}>
        <ConfusionGrid counts={PERF_AUTO} title={`AUTO-DECIDED ONLY · n=${PERF_AUTO.n}`} />
        <ConfusionGrid counts={PERF_FULL} title={`FULL FUNNEL (REVIEW = FLAGGED) · n=${PERF_FULL.n}`} />
      </div>
      <div style={{ fontSize: 12, color: T.textSec, marginTop: 14, lineHeight: 1.6 }}>
        Auto-decided view excludes the 5.3% of cases escalated to a human, and answers "how good is the system when it commits to a call."
        Full-funnel view treats manual review as a positive flag and answers "how much genuine fraud gets zero human eyes on it."
      </div>
    </div>
  );
}

function CostTab() {
  const data = [
    { name: "False positives", value: COST.fp, color: T.brick, note: "84 legitimate disputes wrongly declined · ₹1,200 assumed goodwill/LTV impact each" },
    { name: "False negatives", value: COST.fn, color: T.brick, note: "54 genuine fraud cases wrongly approved · priced at actual disputed amount" },
    { name: "Manual review", value: COST.review, color: T.mustard, note: "64 cases escalated · ₹45 assumed analyst-minutes cost each" },
  ];
  return (
    <div>
      <div style={{ display: "flex", gap: 12, marginBottom: 18 }}>
        <StatCard label="TOTAL ESTIMATED COST" value={`₹${COST.total.toLocaleString("en-IN")}`} color={T.amber} sub="on 1,200 cases in the evaluation set" />
        <StatCard label="FALSE-POSITIVE COST" value={`₹${COST.fp.toLocaleString("en-IN")}`} color={T.brick} />
        <StatCard label="FALSE-NEGATIVE COST" value={`₹${COST.fn.toLocaleString("en-IN")}`} color={T.brick} />
      </div>
      <div style={{ height: 160 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 90, right: 20, top: 4, bottom: 4 }}>
            <CartesianGrid horizontal={false} stroke={T.hair} />
            <XAxis type="number" tick={{ fill: T.textMute, fontSize: 11 }} stroke={T.hairStrong}
              tickFormatter={(v) => `₹${(v / 1000).toFixed(0)}k`} />
            <YAxis type="category" dataKey="name" tick={{ fill: T.textSec, fontSize: 12 }} stroke={T.hairStrong} width={110} />
            <Tooltip contentStyle={{ background: T.surfaceRaised, border: `1px solid ${T.hair}`, borderRadius: 6, fontSize: 12 }}
              labelStyle={{ color: T.textPri }} formatter={(v) => [`₹${v.toLocaleString("en-IN")}`, "cost"]} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {data.map((d, i) => <Cell key={i} fill={d.color} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 4 }}>
        {data.map((d, i) => (
          <div key={i} style={{ fontSize: 11.5, color: T.textSec }}><span style={{ color: d.color }}>■</span> {d.note}</div>
        ))}
      </div>
    </div>
  );
}

function ThresholdTab() {
  return (
    <div>
      <div style={{ fontSize: 12, color: T.textSec, marginBottom: 12, lineHeight: 1.6 }}>
        Sweeping a single hard threshold across the Tier-1 risk score (instead of the three-way decline / review / approve policy) shows where precision and recall trade off — this justifies why 0.40 and 0.70 were chosen as the two operating thresholds.
      </div>
      <div style={{ height: 260 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={SWEEP} margin={{ left: 0, right: 10, top: 10, bottom: 0 }}>
            <CartesianGrid stroke={T.hair} vertical={false} />
            <XAxis dataKey="threshold" tick={{ fill: T.textMute, fontSize: 11 }} stroke={T.hairStrong} />
            <YAxis domain={[0, 1]} tick={{ fill: T.textMute, fontSize: 11 }} stroke={T.hairStrong} />
            <Tooltip contentStyle={{ background: T.surfaceRaised, border: `1px solid ${T.hair}`, borderRadius: 6, fontSize: 12 }}
              labelStyle={{ color: T.textPri }} />
            <Legend wrapperStyle={{ fontSize: 12, color: T.textSec }} />
            <ReferenceLine x={0.4} stroke={T.mustard} strokeDasharray="4 4" label={{ value: "review", fill: T.mustard, fontSize: 10, position: "top" }} />
            <ReferenceLine x={0.7} stroke={T.brick} strokeDasharray="4 4" label={{ value: "decline", fill: T.brick, fontSize: 10, position: "top" }} />
            <Line type="monotone" dataKey="precision" stroke={T.amber} strokeWidth={2} dot={{ r: 3 }} />
            <Line type="monotone" dataKey="recall" stroke={T.wire} strokeWidth={2} dot={{ r: 3 }} />
            <Line type="monotone" dataKey="f1" stroke={T.teal} strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
