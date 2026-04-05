"""
main.py — FastAPI server for the Adaptive AI Project Manager (OpenEnv).

Endpoints
---------
POST /reset              → Reset environment, return success + observation
POST /step               → Submit one action, return success + obs/reward/done/info
GET  /state              → Success + read-only current observation snapshot
GET  /tasks              → Success + structured task list (with urgency scores)
GET  /grader             → Success + deterministic completed-episode grade
GET  /baseline           → Success + heuristic baseline across all 3 formal tasks
GET  /timeline           → Success + full episode timeline JSON
GET  /demo               → One-shot full episode for judges (no stepping needed)
GET  /health             → Liveness probe


Run with:
    uvicorn main:app --host 0.0.0.0 --port 7860 --reload
"""

from __future__ import annotations

import uuid
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api_models import (
    ActionRequest,
    GraderResponse,
    ObservationResponse,
    ResetRequest,
    StateResponse,
    TaskListResponse,
)
from environment import ProjectManagerEnv
from inference import run_scenario
from session_store import SessionStore
from baseline_runner import BaselineRunner
from timeline import EpisodeTimeline
from demo import PriorityAwareAgent, run_episode
from pydantic import BaseModel

from enum import Enum

from fastapi.responses import HTMLResponse

from fastapi import FastAPI, Request
import requests

from fastapi import FastAPI





class ActionTypeEnum(str, Enum):
    assign_task = "assign_task"
    unassign_task = "unassign_task"
    reprioritize = "reprioritize"
    rest_developer = "rest_developer"
    split_task = "split_task"
    pair_program = "pair_program"
    noop = "noop"
    AUTO = "auto"

class ScenarioTypeEnum(str, Enum):
    easy="easy"
    medium="medium"
    hard="hard"

class ResetResponse(BaseModel):
    status:str
    data:dict



app = FastAPI(
    title="Adaptive AI Project Manager",
    description=(
        "An OpenEnv-compliant reinforcement learning environment simulating "
        "a realistic Agile sprint. An AI agent assigns developers to tasks, "
        "manages fatigue, responds to dynamic events, and maximises delivery."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


store = SessionStore()


def _without_status(payload: dict) -> dict:
    """Drop nested status so envelope stays uniform."""
    data = dict(payload)
    data.pop("status", None)
    return data




@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness probe — returns 200 if server is running."""
    return {"status": "success", "data": {"version": "2.1.0"}}



try:
    from api_models import ResetRequestModel, ActionRequestModel
    _PYDANTIC_MODELS = True
except ImportError:
    _PYDANTIC_MODELS = False

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Adaptive AI Project Manager</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<style>
body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f9fafb;
    color: #111827;
}

.container {
    max-width: 800px;
    margin: 120px auto;
    text-align: center;
}

/* Title */
.title {
    font-size: 32px;
    font-weight: 600;
}

.subtitle {
    color: #6b7280;
    margin-top: 10px;
    margin-bottom: 40px;
    font-size: 16px;
}

/* Card */
.card {
    background: white;
    padding: 40px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}

/* Button */
button {
    background: #2563eb;
    color: white;
    border: none;
    padding: 12px 22px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 15px;
    transition: 0.2s;
}

button:hover {
    background: #1d4ed8;
}

/* Secondary */
.docs-btn {
    margin-top: 15px;
    background: white;
    color: #2563eb;
    border: 1px solid #2563eb;
}

.docs-btn:hover {
    background: #eff6ff;
}

/* Footer */
.footer {
    margin-top: 40px;
    color: #9ca3af;
    font-size: 13px;
}
</style>
</head>

<body>

<div class="container">

    <div class="title">Adaptive AI Project Manager</div>
    <div class="subtitle">
        Reinforcement Learning-based Agile Simulation API
    </div>

    <div class="card">
        <p>
            This API simulates real-world Agile project management using intelligent decision-making.
        </p>

        <button onclick="openDocs()">Open API Documentation</button>
    </div>

    <div class="footer">
        Powered by FastAPI · Deployed on Hugging Face
    </div>

</div>

<script>
function openDocs() {
    window.location.href = "/docs";
}
</script>

</body>
</html>
"""


@app.post("/reset", response_model=ResetResponse, tags=["openenv"])
def reset(body: "ResetRequestModel",


) -> dict:
    """
    Initialise (or re-initialise) an episode.

    - **scenario**: one of `easy`, `medium`, `hard`
    - **seed**: integer for full reproducibility
    - **session_id**: reuse an existing session or omit to create a new one
    """
    dc = body.to_dc()
    session_id = dc.session_id if dc.session_id and dc.session_id != "string" else str(uuid.uuid4())
    env = ProjectManagerEnv(
        scenario=dc.scenario,
       seed= dc.seed
    )
    obs = env.reset()
    store.put(session_id, env, EpisodeTimeline())
    return {
        "status": "success",
        "data": {
            "session_id": session_id,
            "observation": _without_status(
                ObservationResponse.from_obs(obs, session_id=session_id).to_dict()
            ),
        "example_step": {
        "session_id": "session_id",
            "action_type": "assign_task"
    }
        },
    }


@app.post("/step", response_model=dict, tags=["openenv"])
def step(body: "ActionRequestModel",
         session_id: str = Query(..., description="Active session UUID"),
         task_id: str =Query(..., description=""),
         action_type: ActionTypeEnum=Query(...,description="")) -> dict:
    """
    Advance the environment by one time step.

    Returns a structured success response with OpenEnv fields:
    `status`, `step`, `reward`, `done`, `info`, task/developer snapshots.

    The `info` field includes:
    - `action_valid` — whether the submitted action was legal
    - `reasoning`    — natural-language explanation of what happened
    - `action_echo`  — the parsed action that was executed

    """
    dc = body.to_dc()

    dc.session_id = session_id
    dc.task_id=task_id
    dc.action_type=action_type
    env = store.get(dc.session_id)


    if env is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    obs= env.state

    tasks = obs.tasks
    developers = obs.developers

    task = next((t for t in tasks if t.id == dc.task_id), None)
    dev = next((d for d in obs.developers if d.id == dc.developer_id), None)

    if not task:
        raise HTTPException(400, "Invalid task_id")



    if dc.action_type == "assign_task":
        missing_skills =[
            s for s in task.required_skills if s not in dev.skills
        ]
        if missing_skills:
            print("pair skills")
        if task:
            for dev in developers:
                dev_skills = dev.skills
                task_skills = task.required_skills

                if all(skill in dev_skills for skill in task_skills):
                    dc.developer_id = dev.id
                    print("AUTO ASSIGNED:", dev.id)
                    break
    try:
        action = dc.to_action()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    timeline = store.get_timeline(dc.session_id)

    try:
        obs = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if timeline is not None:
        timeline.record(
            obs,
            action={

                 "type": str(action.action_type).split(".")[-1].lower(),
                 "assign_exists": action.assign is not None,
                "task_id": getattr(action.assign, "task_id", "MISSING"),
                "developer_id": getattr(action.assign, "developer_id", "MISSING"),

                "primary_developer_id": action.pair.primary_developer_id if action.pair else None,
                "secondary_developer_id": action.pair.secondary_developer_id if action.pair else None,
                "new_priority": action.reprioritize.new_priority.value if action.reprioritize else None,
                "split_ratio": action.split.split_ratio if action.split else None,
            },
        )

    store.put(dc.session_id, env, timeline)
    return {
        "status": "success",
        "data": {
            "session_id": dc.session_id,
            "observation": _without_status(
                ObservationResponse.from_obs(obs, session_id=dc.session_id).to_dict()
            ),
        },
    }




@app.get("/state", response_model=dict, tags=["openenv"])
def state(session_id: str = Query(..., description="Active session UUID")) -> dict:
    """
    Return a structured success snapshot of the current environment state.
    Does **not** advance the simulation clock.
    """
    env = store.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    obs = env.state
    return {
        "status": "success",
        "data": {
            "session_id": session_id,
            "observation": _without_status(
                StateResponse.from_obs(obs, session_id=session_id).to_dict()
            ),
        },
    }




@app.get("/tasks", response_model=dict, tags=["openenv"])
def tasks(session_id: str = Query(..., description="Active session UUID")) -> dict:
    """
    Return a structured success response with urgency-sorted tasks and dependency graph.

    Urgency is calculated as: (priority × business_value) / steps_remaining.
    """
    env = store.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    obs = env.state
    return {
        "status": "success",
        "data": {
            **_without_status(asdict(TaskListResponse.from_obs(obs, session_id=session_id))),
        },
    }




@app.get("/grader", response_model=dict, tags=["openenv"])
def grader(session_id: str = Query(..., description="Active session UUID")) -> dict:
    """
    Deterministically grade the completed episode.

    Requires the episode to be **done** (all tasks terminal or max_steps reached).
    Returns per-dimension scores, weighted total, letter grade, and diagnostic notes.
    """
    env = store.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    try:
        breakdown = env.grade()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "status": "success",
        "data": {
            **_without_status(
                asdict(GraderResponse.from_breakdown(breakdown, session_id=session_id))
            ),
        },
    }




@app.get("/baseline", response_model=dict, tags=["openenv"])
def baseline(seed: int = Query(42, description="RNG seed for reproducibility")) -> dict:
    """
    Run the `PriorityAwareAgent` heuristic baseline across all 3 formal tasks.

    Results are **deterministic** given the same seed.
    Use this to compare any RL agent against the heuristic reference line.
    """
    runner = BaselineRunner(seed=seed)
    result = runner.run_all()
    return {"status": "success", "data": _without_status(asdict(result))}



@app.get("/timeline", tags=["openenv"])
def timeline(session_id: str = Query(..., description="Active session UUID")) -> dict:
    """
    Return the full episode timeline JSON.

    Includes per-step task status counts, developer assignments, reward signal,
    events fired, and delivery %. Grows with every `/step` call.
    """
    env = store.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    tl = store.get_timeline(session_id)
    if tl is None:
        raise HTTPException(status_code=404, detail="No timeline found for session.")
    data = tl.to_json()
    return {
        "status": "success",
        "data": [
            {
                "step":    s.get("step"),
                "action":  (s.get("action") or {}).get("type", "noop"),
                "reward":  s.get("reward"),
                "summary": s.get("state_summary"),
            }
            for s in data.get("steps", [])
        ],
    }


@app.get("/demo", tags=["meta"])
def demo(
    scenario: str = Query("", description="Scenario: easy | medium | hard"),
    seed: int     = Query(42,       description="RNG seed for reproducibility"),
) -> dict:
    """
    Run a complete episode with the `PriorityAwareAgent` and return everything
    a judge needs in one API call:

    - `observation`: final state after the episode
    - `grade`: 8-dimension score breakdown
    - `timeline`: full step-by-step JSON timeline
    - `ascii_timeline`: human-readable ASCII rendering

    """
    env   = ProjectManagerEnv(scenario=scenario, seed=seed)
    agent = PriorityAwareAgent()
    tl    = EpisodeTimeline()
    obs   = env.reset()

    while not obs.done:
        action = agent.act(obs)
        obs    = env.step(action)
        tl.record(obs)

    breakdown = env.grade()

    return {
        "status": "success",
        "data": {
            "scenario":      scenario,
            "seed":          seed,
            "steps_taken":   obs.step,
            "grade":         breakdown.grade,
            "weighted_total": round(breakdown.weighted_total, 6),
            "dimensions": {
                "delivery":     round(breakdown.delivery_score,     4),
                "value":        round(breakdown.value_score,        4),
                "timeliness":   round(breakdown.timeliness_score,   4),
                "priority":     round(breakdown.priority_score,     4),
                "team_health":  round(breakdown.team_health_score,  4),
                "adaptability": round(breakdown.adaptability_score, 4),
                "efficiency":   round(breakdown.efficiency_score,   4),
                "dependency":   round(breakdown.dependency_score,   4),
            },
            "score_report":   breakdown.report(),
            "timeline":       tl.to_json(),
            "ascii_timeline": tl.to_ascii(),
        },
    }

