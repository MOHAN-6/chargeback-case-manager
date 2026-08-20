from fastapi import FastAPI
from agents import investigate_case
import pandas as pd

app = FastAPI()

@app.get("/cases")
def get_cases():
    df = pd.read_csv("disputes_dataset.csv")
    return df.head(10).to_dict(orient="records")

@app.post("/investigate/{case_id}")
def investigate(case_id: str):
    df = pd.read_csv("disputes_dataset.csv")
    case = df[df["case_id"] == case_id].iloc[0].to_dict()
    result, audit = investigate_case(case)
    return {
        "decision": result["decision"],
        "risk_score": result["risk_score"],
        "audit": [entry["rationale"] for entry in audit.as_list()]
    }
