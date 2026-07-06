# StudyFlow AI — High-Availability Academic Q&A Agent

> **Kaggle Capstone Submission** · Built on [Google ADK](https://google.github.io/adk-docs/) · Python 3.12

[![Tests](https://img.shields.io/badge/tests-20%20passed-brightgreen)](tests/) [![Pass Rate](https://img.shields.io/badge/eval%20pass%20rate-100%25-brightgreen)](scripts/evaluate_qna.py) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml) [![License](https://img.shields.io/badge/license-MIT-blue)](#license)

---

## Overview

**StudyFlow AI** is a production-grade, multi-agent academic assistant that helps students learn more effectively through AI-powered tutoring, personalised study planning, and progress monitoring. It is architected as a fully orchestrated multi-agent system using the **Google Agent Development Kit (ADK)**, with structured JSON telemetry, fault-tolerant resiliency patterns, and a comprehensive automated test suite.

---

## Core Features

### 🤖 Multi-Agent System (Google ADK)
A five-agent hierarchy coordinated through a Root Agent and Coordinator:

| Agent | Responsibility |
| :--- | :--- |
| **CoordinatorAgent** | Receives all requests, classifies intent, delegates to sub-agents |
| **IngestionAgent** | Parses uploaded syllabi and extracts structured `AcademicState` |
| **PlannerAgent** | Generates personalised `StudyTask` schedules from deadlines and availability |
| **TutorAgent** | Answers academic questions grounded in RAG-retrieved course material (Socratic method) |
| **MonitoringAgent** | Reads `ProgressState`, analyses streaks, and identifies at-risk deadlines |

### 📊 Structured Telemetry Logging
Every workflow execution emits structured JSON logs via `studyflow_ai/utils/telemetry.py`:

```json
{"timestamp": "2026-07-06T09:10:10+00:00Z", "level": "INFO", "message": "Starting step: ExecuteWorkflow", "step_name": "ExecuteWorkflow", "status": "START"}
{"timestamp": "2026-07-06T09:10:10+00:00Z", "level": "INFO", "message": "Completed step: AcademicQnARun", "step_name": "AcademicQnARun", "status": "SUCCESS", "duration_ms": 0.01}
```

The `@trace_step(step_name)` async decorator captures start/end, duration in milliseconds, and `SUCCESS` or `ERROR` status for every instrumented coroutine.

### 🛡️ Resiliency Patterns (No External Libraries)
Implemented natively in `studyflow_ai/workflows/academic_qna.py`:

**Retry with Exponential Backoff**
- Retries transient `ConnectionError` / `asyncio.TimeoutError` up to **3 times**.
- Delay grows exponentially: `2^attempt × 0.1s` (0.1s → 0.2s → 0.4s).
- Emits a structured warning log on each retry attempt.

**Circuit Breaker**
- Trips to **`OPEN`** after **5 consecutive hard failures**.
- Holds open for a **30-second cooldown window**.
- Any call while `OPEN` fast-fails immediately with: `"Circuit open: External LLM API is temporarily unavailable."`
- Automatically resets to **`CLOSED`** after the cooldown expires.

---

## Project Structure

```
studyflow_ai/
├── agents/             # CoordinatorAgent, IngestionAgent, PlannerAgent, TutorAgent, MonitoringAgent
├── config/             # Constants and environment configuration
├── memory/             # ConversationMemory, AcademicMemory, UserMemory wrappers
├── models/             # Pydantic request/response/state schemas
├── prompts/            # Plain-text system prompt files per agent
├── repositories/       # Repository interfaces + InMemory implementations
├── services/           # CalendarService, NotificationService, StorageService, VectorService
├── tools/              # ADK Tool Layer (calendar_tool, vector_search_tool, progress_tool, etc.)
├── utils/
│   └── telemetry.py    # JSON structured logging + @trace_step decorator
├── workflows/          # Workflow state machines (5 workflows)
├── runner.py           # StudyFlowRunner — central async execution engine
├── session.py          # SessionManager — multi-turn state lifecycle
├── studyflow_agent.py  # Root Agent — entry point orchestrating all sub-agents
└── cli.py              # CLI entrypoint (ingest / create / update / qna / monitor)

scripts/
└── evaluate_qna.py     # Offline ADK evaluation script

tests/
├── evaluation/
│   └── qna_ground_truth.json     # 3-case ground truth dataset
└── integration/
    ├── test_workflow_e2e.py       # 18 end-to-end workflow integration tests
    └── test_resiliency.py         # 2 resiliency tests (retry + circuit breaker)
```

---

## Setup & Installation

### Prerequisites
- Python 3.10 or higher
- `git`

### 1. Clone the Repository

```bash
git clone https://github.com/<YOUR_USERNAME>/studyflow-ai.git
cd studyflow-ai
```

### 2. Create and Activate a Virtual Environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -e ".[dev]"
```

---

## Running the System

### Run the ADK Evaluation Suite

Executes all 3 ground-truth Q&A cases through the live workflow pipeline and outputs a Markdown pass-rate report:

```bash
python scripts/evaluate_qna.py
```

**Example output:**
```
## StudyFlow AI - Q&A Evaluation Report
============================================================
| Metric           | Value             |
| :---             | :---              |
| Total Cases      | 3                 |
| Passed Cases     | 3                 |
| Average Latency  | 0.0002 seconds    |
| Pass Rate (%)    | 100.0%            |

| Case | Status | Latency  | Match details   |
| :--- | :---   | :---     | :---            |
| 1    | PASS   | 0.0003s  | Matches: 3/3    |
| 2    | PASS   | 0.0002s  | Matches: 3/3    |
| 3    | PASS   | 0.0002s  | Matches: 2/2    |

[EVAL SUCCESS] Evaluation passed! Pass rate 100.0% meets threshold of 66.0%.
```

### Run the Full 20-Case Pytest Suite

```bash
python -m pytest tests/ -v
```

**Expected output:**
```
collected 20 items

tests/integration/test_resiliency.py::test_academic_qna_retry_on_transient_error  PASSED [  5%]
tests/integration/test_resiliency.py::test_academic_qna_circuit_breaker_trips      PASSED [ 10%]
tests/integration/test_workflow_e2e.py::TestIngestSyllabusWorkflow::...            PASSED [ 15%]
...
======================= 20 passed in 1.94s ==============================
```

### Run the CLI

```bash
# Ingest a syllabus
python -m studyflow_ai.cli ingest --user-id u1 --course-id chem101 --file-uri gs://bucket/syllabus.pdf

# Ask a tutoring question
python -m studyflow_ai.cli qna --user-id u1 --course-id chem101 --question "What is covalent bonding?"

# Monitor progress
python -m studyflow_ai.cli monitor --user-id u1
```

---

## Evaluation Results

### ADK Q&A Evaluation — Pass Rate Report

| Metric | Value |
| :--- | :--- |
| **Total Evaluation Cases** | 3 |
| **Passed Cases** | 3 |
| **Average Latency** | 0.0002 seconds |
| **Pass Rate** | **100.0%** |
| **Threshold** | 66.0% |
| **Result** | ✅ PASSED |

| Case | Subject | Status | Keywords Matched |
| :--- | :--- | :--- | :--- |
| 1 | Chemistry — Ionic vs Covalent Bonds | ✅ PASS | 3 / 3 |
| 2 | Computer Science — Python Recursion | ✅ PASS | 3 / 3 |
| 3 | Mathematics — Derivative of x² | ✅ PASS | 2 / 2 |

### Integration Test Suite — 20/20 Passing

| Test Suite | Tests | Passed | Status |
| :--- | :--- | :--- | :--- |
| `test_resiliency.py` | 2 | 2 | ✅ |
| `test_workflow_e2e.py` | 18 | 18 | ✅ |
| **Total** | **20** | **20** | **✅ 100%** |

---

## Architecture

```
User / CLI
     │
     ▼
StudyFlowAgent (Root Agent)
     │
     ▼
CoordinatorAgent ──────────────────────────────────────────┐
     │                                                      │
     ├─── IngestionAgent   (syllabus parsing)               │
     ├─── PlannerAgent     (study scheduling)               │
     ├─── TutorAgent       (RAG-grounded Q&A)               │
     └─── MonitoringAgent  (streak & deadline tracking)     │
                                                            │
StudyFlowRunner ◄──────────────────────────────────────────┘
     │
     ├── IngestSyllabusWorkflow
     ├── CreateStudyPlanWorkflow
     ├── UpdateStudyPlanWorkflow
     ├── AcademicQnAWorkflow  ←── @trace_step + Retry + Circuit Breaker
     └── ProgressMonitorWorkflow
          │
     SessionManager  (multi-turn memory lifecycle)
          │
     ┌────┴──────────┐
     │               │
  Repositories    Services
  (User, Academic, (Calendar, Vector,
   Progress)       Storage, Notification)
```

---

## Technology Stack

| Component | Technology |
| :--- | :--- |
| Agent Framework | Google ADK (`google-adk >= 2.0.0`) |
| Language | Python 3.12 |
| Data Validation | Pydantic v2 |
| Structured Logging | `logging` + custom `StructuredJsonFormatter` |
| Async Runtime | `asyncio` |
| Test Framework | `pytest` + `pytest-asyncio` |
| Resiliency | Native Python (no external circuit-breaker libraries) |

---

## License

This project is released under the MIT License for the purposes of the Kaggle Capstone evaluation.
