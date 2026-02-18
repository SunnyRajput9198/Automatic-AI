# Autonomous Agent System

**Building an AI system that thinks before it acts.**

Currently: A working orchestration layer with reasoning and coordination. The system can analyze tasks, pick strategies, and execute them using tools.

Still building: The specialized agents that enable full autonomy (planning, smart execution, validation, learning).

Built with FastAPI, PostgreSQL, Next.js, and Claude AI.

---

## The Problem

Most "AI automation" is just fancy if-else statements with an LLM wrapper. They execute blindly, fail silently, and learn nothing.

This system is different. It has a **reasoning layer** that analyzes tasks before execution, a **coordinator** that picks the right strategy, and **specialized agents** that work together like an actual team.

---

## What It Can Do (Right Now)

The Reasoner and Coordinator are working. They can:
- Analyze task complexity and choose execution strategies
- Route tasks to the right tools
- Execute Python code, read/write files, make web requests
- Handle basic multi-step workflows

**What works:**
- Simple data processing (read CSV, run calculations, output results)
- File operations (create, modify, organize files)
- Web requests (fetch data from APIs)
- Basic Python execution (data analysis, simple scripts)

**What doesn't work yet:**
- Complex planning (Planner agent still in progress)
- Smart tool selection (Executor needs refinement)
- Quality validation (Critic agent not done)
- Learning from failures (Reflection agent planned)

Translation: It can execute tasks you give it clear instructions for. It can't yet figure out complex tasks on its own.

---

## How It Works (Current State)

```
USER
  ↓
  "Process this CSV file"
  ↓
FASTAPI BACKEND
  ↓ Creates task, starts background job
  ↓
ORCHESTRATOR V3 ✅ WORKING
  ↓ Manages lifecycle, handles failures, logs everything
  ↓
REASONER AGENT ✅ WORKING
  ↓ "This is a data processing task, low complexity"
  ↓ "Tools needed: file reading, Python execution"
  ↓
COORDINATOR AGENT ✅ WORKING
  ↓ "Route to Python executor, output to file"
  ↓
TOOL LAYER ✅ WORKING
  ↓ Python executor runs analysis
  ↓ File writer saves results
  ↓
RESULT
  "analysis_complete.csv generated"


WHAT'S MISSING (in progress):
  
  🚧 PLANNER AGENT
     Would break complex tasks into steps
  
  🚧 EXECUTOR AGENT  
     Would intelligently select best tools
  
  🚧 CRITIC AGENT
     Would validate outputs
  
  🚧 REFLECTION AGENT
     Would learn from outcomes
```

Right now it's a smart task router with tool execution. The full agent system is being built on top of this foundation.

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

**What actually works today:**

```bash
# Simple file processing
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Read data.csv and calculate the average of column A"
  }'

# Basic web request
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Fetch https://api.github.com/users/SunnyRajput9198 and save to file"
  }'

# Python execution  
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Generate 100 random numbers and calculate statistics"
  }'
```

Or use the UI at `http://localhost:3000`

**Note**: Complex multi-step tasks that require planning and validation aren't reliable yet. Stick to single-step or clearly defined workflows.

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

**Current Reality:**
This is a foundation. The orchestration layer, reasoning, and coordination work. Tool execution works. But the specialized agents that enable true autonomy are still being built.

**The Vision:**
Most "AI agent" projects hard-code workflows or use brittle prompt chains. This is designed to be different—a system where agents actually reason about tasks, coordinate with each other, and improve over time.

**The Hard Part:**
It's not making LLM calls. It's building the orchestration layer that makes agents:
- Reliable (handle failures gracefully)
- Debuggable (trace what happened and why)  
- Scalable (add new agents and tools easily)
- Adaptive (learn from outcomes)

That's what's being built here. The foundation is done. The specialized agents are next.

**End Goal:**
- Autonomous research assistants
- Code review bots that understand context
- Data pipelines that adapt to messy inputs
- Task automation that doesn't break

Not there yet. But the architecture is designed to get there.

---

## Contributing

This is an open project. If you want to:
- Add a new tool
- Build a specialized agent
- Improve the coordinator logic
- Fix bugs

Just fork, make changes, and submit a PR. No formal process.

---

## License

MIT. Do whatever you want with it.

---

## Contact

Issues: [GitHub Issues](https://github.com/SunnyRajput9198/Automatic-AI/issues)

Built by [@SunnyRajput9198](https://github.com/SunnyRajput9198)

---

*If you're reading this and thinking "I could do this better"—you're probably right. Show me.*
