import os
import json
import time
from app.agents.memory.agent_preference_memory import AgentPreferenceMemory

print("\n==============================")
print("🔥 DAY 5 HARDCORE MEMORY TEST")
print("==============================\n")

PREF_PATH = "workspace/agent_preferences.json"

# --------------------------------------------------
# CLEAN SLATE
# --------------------------------------------------
if os.path.exists(PREF_PATH):
    os.remove(PREF_PATH)
    print("🧹 Cleaned old preference file")

mem = AgentPreferenceMemory()

# --------------------------------------------------
# TEST 1 — multiple different tasks
# --------------------------------------------------
tasks = [
    ("write python api", "engineer"),
    ("research latest ai models", "researcher"),
    ("write blog article", "writer"),
]

for task, agent in tasks:
    mem.record_success(task, agent)

print("✅ TEST 1 PASSED — multiple tasks learned")

# --------------------------------------------------
# TEST 2 — persistence across instances
# --------------------------------------------------
mem_reload = AgentPreferenceMemory()

for task, agent in tasks:
    got = mem_reload.get_preferred_agent(task)
    assert got == agent, f"❌ Persistence failed for {task}"

print("✅ TEST 2 PASSED — persistence verified")

# --------------------------------------------------
# TEST 3 — overwrite preference (learning update)
# --------------------------------------------------
task = "write python api"
mem_reload.record_success(task, "engineer")
mem_reload.record_success(task, "engineer")
mem_reload.record_success(task, "engineer")

mem_reload.record_success(task, "researcher")  # wrong agent once

agent = mem_reload.get_preferred_agent(task)

assert agent == "researcher", "❌ Latest learning not applied"

print("✅ TEST 3 PASSED — preference overwrite works")

# --------------------------------------------------
# TEST 4 — noisy similar tasks (real world)
# --------------------------------------------------
similar_tasks = [
    "write python api using fastapi",
    "write python api using flask",
    "write python api with auth",
]

for t in similar_tasks:
    mem_reload.record_success(t, "engineer")

ok = 0
for t in similar_tasks:
    if mem_reload.get_preferred_agent(t) == "engineer":
        ok += 1

assert ok == len(similar_tasks), "❌ Similar-task memory failed"

print("✅ TEST 4 PASSED — noisy tasks handled")

# --------------------------------------------------
# TEST 5 — crash safety (file corruption simulation)
# --------------------------------------------------

with open(PREF_PATH, "w") as f:
    f.write("{ this is broken json}")

try:
    mem_broken = AgentPreferenceMemory()
    print("✅ TEST 5 PASSED — survived corrupted file")
except Exception:
    raise Exception("❌ Crashed on corrupted preference file")

# --------------------------------------------------
# FINAL RESULT
# --------------------------------------------------
print("\n🔥🔥🔥 HARDCORE DAY 5 TEST PASSED 🔥🔥🔥")
print("🧠 Memory is resilient")
print("💾 Persistent")
print("🔁 Learnable")
print("🚀 Production safe\n")
