"""
SkillBridge AI — FastAPI Backend
Endpoints for skill analysis, curriculum generation, and AI tutor
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import asyncio

app = FastAPI(title="SkillBridge AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request/Response Models ----

class SkillGapRequest(BaseModel):
    resume_text: str
    target_role: str

class SkillGapResponse(BaseModel):
    target_role: str
    skills_matched: List[str]
    skill_gaps: List[str]
    extra_skills: List[str]
    readiness_score: float
    top_3_to_learn: List[str]

class CurriculumRequest(BaseModel):
    target_role: str
    current_skills: List[str]
    skill_gaps: List[str]
    hours_per_week: int = 10
    learning_style: str = "visual + hands-on"
    weeks: int = 8

class TutorMessage(BaseModel):
    session_id: str
    user_message: str
    language: str = "en"  # supports: en, hi, ta, te, bn, mr, kn, gu, ml, pa


# ---- Endpoints ----

@app.get("/health")
def health():
    return {"status": "ok", "model_backend": "AMD MI300X ROCm"}


@app.post("/api/skill-gap", response_model=SkillGapResponse)
async def analyze_skill_gap(request: SkillGapRequest):
    """
    Analyze skill gaps between resume and target role.
    Uses BERT-based NER + skill taxonomy matching.
    """
    from skill_gap_analyzer import SkillGapAnalyzer
    try:
        analyzer = SkillGapAnalyzer()
        report = analyzer.analyze(request.resume_text, request.target_role)
        return SkillGapResponse(**report)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/curriculum")
async def generate_curriculum(request: CurriculumRequest):
    """
    Generate personalized learning curriculum using Llama 3 on AMD MI300X.
    Returns structured week-by-week plan.
    """
    from curriculum_generator import generate_curriculum as gen
    curriculum = gen(
        target_role=request.target_role,
        current_skills=request.current_skills,
        skill_gaps=request.skill_gaps,
        hours_per_week=request.hours_per_week,
        learning_style=request.learning_style,
        weeks=request.weeks,
    )
    return {"curriculum": curriculum}


@app.post("/api/scenario/generate")
async def generate_scenario(role: str, difficulty: str = "medium"):
    """
    Generate a real-world job scenario for practice.
    E.g., for 'Software Engineer': a system design or debugging challenge.
    """
    # In production: calls Llama 3 with RAG over industry knowledge base
    scenarios = {
        "Software Engineer": {
            "title": "Debug a Production Memory Leak",
            "description": (
                "Your Node.js service has been running for 48 hours and memory usage "
                "has grown from 200MB to 1.8GB. Users are reporting slow responses. "
                "You have access to heap snapshots and CPU profiles. "
                "Identify the root cause and propose a fix."
            ),
            "evaluation_criteria": [
                "Correctly identifies heap snapshot analysis approach",
                "Mentions closure leaks or event listener accumulation",
                "Proposes a concrete fix with code",
                "Suggests monitoring to prevent recurrence",
            ],
        }
    }
    scenario = scenarios.get(role, {"error": "No scenario for this role yet"})
    return {"role": role, "difficulty": difficulty, "scenario": scenario}


@app.websocket("/ws/tutor/{session_id}")
async def tutor_websocket(websocket: WebSocket, session_id: str):
    """
    Real-time AI tutor via WebSocket.
    Supports multilingual responses via IndicBERT + Llama 3.
    """
    await websocket.accept()
    conversation_history = []
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            user_message = msg.get("message", "")
            language = msg.get("language", "en")

            # Build prompt with conversation history
            conversation_history.append({"role": "user", "content": user_message})

            # In production: calls AMD MI300X LLM endpoint
            # For demo: echo with mock response
            ai_response = (
                f"[SkillBridge AI Tutor | lang={language}] "
                f"Great question! Let me explain '{user_message[:50]}...' "
                f"in simple terms with an example."
            )
            conversation_history.append({"role": "assistant", "content": ai_response})

            await websocket.send_text(json.dumps({
                "type": "tutor_response",
                "message": ai_response,
                "session_id": session_id,
            }))
    except WebSocketDisconnect:
        print(f"Session {session_id} disconnected")


# ---- Run ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
