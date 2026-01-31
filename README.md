# Autonomous Agent System

**An AI system that actually thinks before it acts.**

Give it a task. Watch it break the problem down, pick the right tools, execute the plan, validate the results, and learn from what happened. No hand-holding required.

Built with FastAPI, PostgreSQL, Next.js, and Claude AI.

---

## The Problem

Most "AI automation" is just fancy if-else statements with an LLM wrapper. They execute blindly, fail silently, and learn nothing.

This system is different. It has a **reasoning layer** that analyzes tasks before execution, a **coordinator** that picks the right strategy, and **specialized agents** that work together like an actual team.

---

## What It Actually Does

Feed it complex, multi-step tasks:
- "Research the top 10 AI papers from 2024, summarize key findings, and create a comparison table"
- "Analyze this CSV, find anomalies, generate visualizations, and write a report"  
- "Scrape these 5 websites, extract product data, merge it, and export to JSON"
- "Review this codebase, identify security issues, suggest fixes with code examples"

The system figures out how to do it. No templates. No rigid workflows. Just intelligent execution.

---

## How It Works

```
USER
  ↓
  "Analyze sales data and create a report"
  ↓
FASTAPI BACKEND
  ↓ Creates task, starts background job
  ↓
ORCHESTRATOR V3
  ↓ Manages lifecycle, handles failures, logs everything
  ↓
REASONER AGENT
  ↓ "This is a data analysis task with medium complexity"
  ↓ "Need: file reading, Python execution, report generation"
  ↓
COORDINATOR AGENT  
  ↓ Assigns: Planner → Executor → Critic → Reflection
  ↓
SPECIALIZED AGENTS
  │
  ├─ PLANNER: "Step 1: Read CSV, Step 2: Analyze, Step 3: Generate report"
  ├─ EXECUTOR: Runs Python analysis, creates visualizations
  ├─ CRITIC: "Data looks good, but add statistical summary"
  └─ REFLECTION: "Learned: CSV analysis tasks need data validation first"
  ↓
RESULT
  "sales_report.pdf generated successfully"
```

**The key insight**: Each agent has a specific job. They communicate, adapt, and improve. Just like a real team.

---

## Core Components

### 1. Reasoner Agent
The strategist. Looks at your task and thinks:
- What type of problem is this?
- How complex is it?
- What's the best approach?
- What could go wrong?

### 2. Coordinator Agent  
The manager. Decides:
- Which agents to use
- In what order
- How to handle dependencies
- When to retry vs. abort

### 3. Specialized Agents
The workers:
- **Planner**: Breaks tasks into atomic steps
- **Executor**: Picks and runs the right tools
- **Critic**: Validates output quality
- **Reflection**: Learns from outcomes

### 4. Tool Layer
The actual capabilities:
- Python code execution (sandboxed)
- File operations (read/write/delete)
- Web requests and scraping
- Shell commands (optional, disabled by default)
- Persistent workspace

---

## Getting Started

**Prerequisites**: Python 3.11+, PostgreSQL, Node.js 18+

### Option 1: Docker (Fastest)

```bash
git clone https://github.com/SunnyRajput9198/Automatic-AI.git
cd Automatic-AI

# Add your Anthropic API key to .env
echo "ANTHROPIC_API_KEY=your_key_here" > agent-system/.env

# Start everything
docker-compose up

# Open http://localhost:3000
```

### Option 2: Local Setup

```bash
# Backend
cd agent-system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create database
createdb autonomous_agents

# Add API key
export ANTHROPIC_API_KEY="your_key_here"

# Run migrations and start server
alembic upgrade head
uvicorn app.main:app --reload

# Frontend (new terminal)
cd Frontend
npm install
npm run dev
```

---

## Try It Out

Send a task:
```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Find the top 5 trending Python libraries this week and explain what they do"
  }'
```

Or use the UI at `http://localhost:3000`

---

## Current Status

**What's Working:**
- ✅ Orchestrator V3 with task lifecycle management
- ✅ Reasoner agent for task analysis
- ✅ Coordinator agent for strategy selection
- ✅ Tool execution layer (Python, files, web)
- ✅ FastAPI backend with PostgreSQL
- ✅ Next.js frontend with real-time updates

**What's Next:**
- 🚧 Planner agent implementation (in progress)
- 🚧 Executor agent refinement
- 🚧 Critic agent for validation
- 🚧 Reflection agent for learning
- 📋 Performance benchmarking
- 📋 Multi-model support (GPT-4, Gemini)
- 📋 Agent marketplace

This is a **work in progress**. It works for basic tasks. Complex workflows are getting there.

---

## Tech Stack

**Backend:**
- FastAPI (async API framework)
- PostgreSQL (task storage)
- SQLAlchemy (ORM)
- Claude Haiku (LLM - fast and cheap)
- Docker (containerization)

**Frontend:**
- Next.js 14 (React framework)
- TypeScript (type safety)
- Tailwind + shadcn/ui (styling)
- React Query (data fetching)

**Why these choices?**
- FastAPI: Async support, auto-docs, fast
- PostgreSQL: Reliable, ACID-compliant
- Claude Haiku: Best price/performance for reasoning tasks
- Next.js: Server components, great DX
- Tailwind: Fast styling without CSS files

---

## Why This Matters

Most "AI agent" projects are demos. This is designed to be **actually useful**.

The architecture is intentionally overbuilt for what it currently does. Why? Because the hard part isn't making an LLM call—it's building the orchestration layer that makes agents reliable, debuggable, and scalable.

This is the foundation for:
- Autonomous research assistants
- Code review bots that actually understand context
- Data processing pipelines that adapt to messy inputs
- Task automation that doesn't break when things change

---


## License

MIT. Do whatever you want with it.

---

## Contact

Issues: [GitHub Issues](https://github.com/SunnyRajput9198/Automatic-AI/issues)

Built by [@SunnyRajput9198](https://github.com/SunnyRajput9198)

---

*If you're reading this and thinking "I could do this better"—you're probably right. Show me.*
