import os
import json
import sqlite3
import hashlib
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ---------- CONFIG ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
CHAT_MODEL = "deepseek-chat"
DB_PATH = "synthesise_clone.db"

if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY is not set")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# Embedding model runs on the server, not on your computer.
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ---------- DATABASE ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS charters (
            id TEXT PRIMARY KEY,
            uvz_input TEXT,
            uvz_score REAL,
            charter_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            charter_id TEXT,
            history_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# ---------- PYDANTIC MODELS ----------
class UVZRequest(BaseModel):
    skills: str


class UVZResponse(BaseModel):
    input: str
    ranked_uvz: List[dict]
    top_score: float


class ValidationRequest(BaseModel):
    idea: str


class ValidationResponse(BaseModel):
    idea: str
    go_no_go: str
    confidence: float
    aggregate_scores: dict
    heatmap: List[dict]


class CharterRequest(BaseModel):
    validated_idea: str
    uvz_context: Optional[str] = ""


class CharterResponse(BaseModel):
    charter_id: str
    title: str
    modules: List[str]
    pricing_model: str
    deliverable_tree: List[dict]
    persona: dict
    raw_json: dict


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    charter_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class WorkflowTrigger(BaseModel):
    event: str
    charter_id: str
    payload: Optional[dict] = {}


# ---------- HELPERS ----------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


MONETIZABLE_PROBLEMS = [
    "freelancers struggle to find consistent clients",
    "small businesses need help with SEO and content marketing",
    "creators want to package their knowledge into a course",
    "coaches need a repeatable client onboarding process",
    "e-commerce stores want higher conversion rates",
    "professionals want to learn AI skills quickly",
    "startups need help validating product ideas",
    "teachers want to sell digital lesson plans",
    "consultants need a productized service offering",
    "beginners want a step-by-step guide to a new skill",
]

_problem_embeddings = None


def get_problem_embeddings():
    global _problem_embeddings
    if _problem_embeddings is None:
        _problem_embeddings = embedder.encode(MONETIZABLE_PROBLEMS, convert_to_numpy=True)
    return _problem_embeddings


# ---------- APP LIFECYCLE ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    get_problem_embeddings()
    yield


app = FastAPI(title="Synthesise Clone API", version="0.1.0", lifespan=lifespan)


# ---------- ROUTES ----------
@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/uvz/score", response_model=UVZResponse)
def uvz_score(req: UVZRequest):
    user_emb = embedder.encode(req.skills, convert_to_numpy=True)
    problem_embs = get_problem_embeddings()

    scored = []
    for i, p in enumerate(MONETIZABLE_PROBLEMS):
        sim = cosine_similarity(user_emb, problem_embs[i])
        scored.append({"problem": p, "score": round(sim, 4)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:5]
    return UVZResponse(
        input=req.skills,
        ranked_uvz=top,
        top_score=top[0]["score"] if top else 0.0,
    )


@app.post("/v1/validate", response_model=ValidationResponse)
def validate_concept(req: ValidationRequest):
    system_prompt = (
        "You are a market-validation analyst. Score the given product idea on four dimensions "
        "from 0 to 10: demand, competition (reverse-scored, 10 = low competition), willingness_to_pay, "
        "and feasibility. Return ONLY a JSON object with these four integer keys."
    )
    scores = []
    for _ in range(3):
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.idea},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        try:
            scores.append(json.loads(resp.choices[0].message.content))
        except json.JSONDecodeError:
            continue

    if not scores:
        raise HTTPException(status_code=502, detail="Validation model failed to return JSON")

    keys = ["demand", "competition", "willingness_to_pay", "feasibility"]
    agg = {k: sum(s.get(k, 5) for s in scores) / len(scores) for k in keys}
    overall = sum(agg.values()) / len(agg)
    go_no_go = "GO" if overall >= 6.0 else "NO-GO"

    heatmap = [{"dimension": k, "score": round(agg[k], 2)} for k in keys]

    return ValidationResponse(
        idea=req.idea,
        go_no_go=go_no_go,
        confidence=round(overall / 10, 2),
        aggregate_scores={k: round(v, 2) for k, v in agg.items()},
        heatmap=heatmap,
    )


@app.post("/v1/charter/generate", response_model=CharterResponse)
def generate_charter(req: CharterRequest):
    system_prompt = (
        "You are a product architect. Given a validated idea, produce a complete product charter. "
        "Return ONLY valid JSON with the following keys: title (string), modules (array of strings), "
        "pricing_model (string), deliverable_tree (array of objects with 'name' and 'description'), "
        "persona (object with 'name', 'pain_points', 'goals')."
    )
    user_prompt = f"Idea: {req.validated_idea}\nUVZ context: {req.uvz_context}"

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
    )

    try:
        data = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Charter model failed to return JSON")

    charter_id = hashlib.sha256(
        (req.validated_idea + str(data.get("title", ""))).encode()
    ).hexdigest()[:12]

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO charters (id, uvz_input, uvz_score, charter_json) VALUES (?, ?, ?, ?)",
        (charter_id, req.uvz_context, 0.0, json.dumps(data)),
    )
    conn.commit()
    conn.close()

    return CharterResponse(
        charter_id=charter_id,
        title=data.get("title", "Untitled Product"),
        modules=data.get("modules", []),
        pricing_model=data.get("pricing_model", ""),
        deliverable_tree=data.get("deliverable_tree", []),
        persona=data.get("persona", {}),
        raw_json=data,
    )


@app.post("/v1/chatbot/query", response_model=ChatResponse)
def chatbot_query(req: ChatRequest):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT charter_json FROM charters WHERE id = ?", (req.charter_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Charter not found")

    charter_text = row[0]
    system_prompt = (
        "You are a helpful product assistant. Answer the user's question using ONLY the "
        "product charter below. If the answer is not in the charter, say you don't know.\n\n"
        f"PRODUCT CHARTER:\n{charter_text}"
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message},
        ],
        temperature=0.3,
    )
    reply = resp.choices[0].message.content

    session_id = req.session_id or hashlib.sha256(req.message.encode()).hexdigest()[:10]
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO chat_sessions (session_id, charter_id, history_json) VALUES (?, ?, ?)",
        (session_id, req.charter_id, json.dumps([{"user": req.message, "bot": reply}])),
    )
    conn.commit()
    conn.close()

    return ChatResponse(session_id=session_id, reply=reply)


@app.post("/v1/workflow/trigger")
def trigger_workflow(wf: WorkflowTrigger):
    plan = {
        "event": wf.event,
        "charter_id": wf.charter_id,
        "actions": [],
    }
    if wf.event == "purchase":
        plan["actions"] = [
            {"type": "email", "template": "welcome_sequence"},
            {"type": "webhook", "url": "https://your-app.com/on-purchase"},
        ]
    elif wf.event == "form_submit":
        plan["actions"] = [{"type": "email", "template": "lead_magnet_delivery"}]
    elif wf.event == "abandonment":
        plan["actions"] = [{"type": "email", "template": "abandoned_cart_recovery"}]

    return {"status": "planned", "plan": plan}