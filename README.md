---

title: Adaptive AI Project Manager
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: main.py
pinned: false
---

# Adaptive AI Project Manager

OpenEnv RL environment for Agile sprint simulation.

## Version

2.1.0 — Python 3.10–12

An OpenEnv-compliant RL environment simulating a realistic Agile sprint.
An AI agent acts as engineering lead — assigning developers, managing fatigue,
reacting to dynamic disruptions, and maximising delivery within a fixed horizon.

## Quickstart

```bash
pip install -r requirements.txt   # No Rust required — works on Python 3.10–3.14
python quickstart.py              # Zero-config terminal demo
uvicorn main:app --host 0.0.0.0 --port 7860 --reload  # API server
open http://localhost:7860/docs   # Interactive API docs
```

### Docker
```bash
docker build -t pm-env .
docker run -p 7860:7860 pm-env
```

---

## Bug Fixes Applied (v2.0 → v2.1)

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| **1** | `pip install` crashes with Rust error | `pydantic==2.7.1` pinned `pydantic-core==2.18.2` which has no wheel for Python 3.13/3.14 | Loose bounds: `pydantic>=2.7,<3` |
| **2** | Easy scenario always grades F | Deadline slack `randint(2,3)` → deadlines at step 6-8 out of 20; tasks fail before agent reacts | Slack raised to `randint(max_steps//4, max_steps//2)` with hard floor |
| **3** | Agent NOOP deadlock (steps 8–10) | `_best_idle_dev()` required empty `current_tasks`; once all 3 devs were working, idle pool = 0 → NOOP → tasks timeout | Two-tier selection: idle preferred, any-available as fallback |
| **4** | `POST /reset` and `POST /step` silently broken | Routes typed as stdlib `@dataclass`; FastAPI can't parse JSON into them | Changed to Pydantic `ResetRequestModel` / `ActionRequestModel` |
| **5** | `/step` timeline recorder crashes | Accessed `action.task_id` but `Action` dataclass has no such attribute | Fixed to `action.assign.task_id`, `action.pair.primary_developer_id`, etc. |
| **6** | `grader.team_health` always 0 | `overtime_pen = raw_count × 0.005` always hit the 0.40 cap (4 devs × 30 steps = 120 overtime steps possible) | Normalise by `n_devs × max_steps`; penalty = fraction of team-time in overtime |
| **7** | `openenv.yaml` spec ambiguity | No `http_method` specified per endpoint; readers could assume `/reset` is GET | Added `http_method: POST` and `http_path` to all endpoints |
| **8** | Grader didn't receive `max_steps` | `_team_health` had no episode-length context for normalisation | Pass `obs.max_steps` through `grade()` → `_team_health()` |

---

## API Reference

All responses: `{"status": "success", "data": {...}}`

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start episode. Body: `{scenario, seed, session_id}` |
| `POST` | `/step` | Execute action. Body: `{session_id, action_type, ...}` |
| `GET` | `/state` | Read-only state snapshot |
| `GET` | `/tasks` | Urgency-sorted task list + dependency graph |
| `GET` | `/grader` | 8-dimension grade (requires `done=true`) |
| `GET` | `/baseline` | Heuristic baseline across all 3 tiers |
| `GET` | `/timeline` | Full per-step episode timeline |
| `GET` | `/demo` | One-shot complete episode — best for judges |
| `GET` | `/health` | Liveness probe |

### Action Types
```json
{"action_type": "assign_task",    "task_id": "...", "developer_id": "..."}
{"action_type": "unassign_task",  "task_id": "...", "developer_id": "..."}
{"action_type": "reprioritize",   "task_id": "...", "new_priority": "CRITICAL"}
{"action_type": "rest_developer", "developer_id": "..."}
{"action_type": "split_task",     "task_id": "...", "split_ratio": 0.5}
{"action_type": "pair_program",   "task_id": "...", "primary_developer_id": "...", "secondary_developer_id": "..."}
{"action_type": "noop"}
```

---

## Grader — 8 Dimensions

| Dimension | Weight | Measures |
|-----------|--------|---------|
| Delivery | 0.25 | Story points delivered / total |
| Business Value | 0.20 | Value captured / total value |
| Timeliness | 0.15 | On-time completions (late = 0.4× credit) |
| Priority Order | 0.10 | High-priority tasks not sacrificed for low |
| Team Health | 0.10 | Fatigue + normalised overtime + churn |
| Adaptability | 0.10 | Event-injected tasks completed |
| Efficiency | 0.07 | SP delivered vs theoretical capacity |
| Dependencies | 0.03 | No causal-order violations in task graph |

Scores are `[0.0, 1.0]`, fully deterministic for any `(scenario, seed, actions)`.

---

## Baseline Scores (PriorityAwareAgent, seed=42)

```
easy    ~0.54  [D]   ← target B–A with a trained RL agent
medium  ~0.60  [C]
hard    ~0.69  [C]
avg     ~0.61
```

---

## Reproducibility

```python
env1 = ProjectManagerEnv(scenario="medium", seed=42)
env2 = ProjectManagerEnv(scenario="medium", seed=42)
# Identical action sequence → bit-identical scores, guaranteed
```

---

## File Structure

```
main.py              FastAPI server — all endpoints (Bug 4, 5, 7 fixed)
environment.py       Core env: TaskGraph, EventEngine, RewardEngine (Bug 2 fixed)
models.py            Domain models: Task, Developer, Action, Observation
grader.py            8-dimension deterministic grader (Bug 6, 8 fixed)
demo.py              PriorityAwareAgent heuristic baseline (Bug 3 fixed)
baseline_runner.py   Runs baseline across all 3 formal tasks
api_models.py        Pydantic request/response schemas
session_store.py     Thread-safe in-memory session registry (512 sessions, FIFO eviction)
timeline.py          Episode timeline recorder — JSON + ASCII render
quickstart.py        Zero-config terminal demo
tests.py             31-test suite — runs without pytest
openenv.yaml         OpenEnv spec: metadata, action schema, observation schema (Bug 7 fixed)
Dockerfile           Multi-stage build, port 7860, non-root user
requirements.txt     Loose version bounds — no Rust required (Bug 1 fixed)
```

## Observation Space
```
{
  "tasks": [
    {"id": int, "difficulty": int, "deadline": int, "progress": float}
  ],
  "developers": [
    {"id": int, "skill": list, "fatigue": float}
  ],
  "time_step": int
}
```
## Action Space
```
{
  "assignments": [
    {"developer_id": int, "task_id": int}
  ]
}
```
##  Reward Function
```
The environment uses a **dense reward function** normalized to the range:

**Reward ∈ [0.0, 1.0]**

The reward is composed of multiple real-world performance signals:
```
###  Positive Components
```
- **Task Completion (+)**  
  Rewards agents for successfully completing tasks.

- **Deadline Adherence (+)**  
  Higher reward when tasks are completed before or on deadlines.

- **Efficient Allocation (+)**  
  Encourages assigning tasks to developers with matching skills.
```
###  Negative Components
```
- **Overwork Penalty (-)**  
  Penalizes assigning too many tasks to a single developer (fatigue impact).

- **Idle Penalty (-)**  
  Penalizes leaving developers unassigned when tasks are available.

---
```
###  Reward Intuition
```
The reward balances **productivity vs efficiency**:

- Max reward (≈1.0):  
  All tasks completed on time with optimal allocation

- Medium reward (~0.5):  
  Partial completion or inefficient assignments

- Low reward (≈0.0):  
  Missed deadlines, poor allocation, or idle workforce

---

###  Example

```json
{
  "tasks_completed": 10,
  "on_time": 8,
  "efficiency_score": 0.75,
  "fatigue_penalty": 0.1,
  "idle_penalty": 0.05,
  "final_reward": 0.78
}
```
##  OpenEnv Compliance Checklist
```
- [x] Real-world simulation (Agile sprint management)
- [x] step(), reset(), state() implemented
- [x] openenv.yaml present
- [x] 3 difficulty tasks (easy, medium, hard)
- [x] Reward function (0.0–1.0, dense signals)
- [x] Baseline reproducible evaluation
- [x] Dockerfile included
- [x] HuggingFace Spaces deployment ready
```
##  Baseline Inference
```
The baseline agent uses a deterministic heuristic (`PriorityAwareAgent`) to simulate decision-making.

Note: This project is compatible with LLM-based agents using OpenAI API by replacing the agent logic in `baseline_runner.py` and reading API keys from environment variables (`OPENAI_API_KEY`).
```
```
sdk: docker
tags:
  - openenv
```
##  LLM Integration
```
The environment supports LLM-based agents via the OpenAI client. 
For this submission, a deterministic heuristic agent is used to ensure reproducibility and stable evaluation.

The inference script includes OpenAI client initialization and can be extended to use LLM-based decision-making.
```