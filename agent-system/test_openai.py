"""
Quick smoke test — Planner, Critic, Reflection all hitting OpenAI.
Run: python test_openai.py
"""
import asyncio, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from app.agents.planner   import PlannerAgent
from app.agents.critic    import CriticAgent, Verdict
from app.tools.base       import ToolResult

GR, RD, RS, BD = "\033[32m", "\033[31m", "\033[0m", "\033[1m"
ok  = lambda m: print(f"  {GR}✓{RS}  {m}")
bad = lambda m: print(f"  {RD}✗{RS}  {m}")
chk = lambda cond, m: ok(m) if cond else bad(m)
hdr = lambda m: print(f"\n{BD}── {m}{RS}")


async def test_planner():
    hdr("Planner — plan('Research benefits of FastAPI')")
    agent = PlannerAgent()
    steps = await agent.plan("Research benefits of FastAPI")
    print(f"  steps returned: {len(steps)}")
    for s in steps:
        print(f"    [{s['step']}] {s['instruction']}")
    chk(len(steps) > 0,          "at least one step")
    chk(all("instruction" in s for s in steps), "every step has instruction")
    return len(steps) > 0


async def test_critic_pass():
    hdr("Critic — should PASS on relevant result")
    agent = CriticAgent()
    result = ToolResult(
        success=True,
        output="FastAPI is a modern Python web framework known for high performance, "
                "automatic docs, and async support. It is faster than Flask and Django.",
        error=None,
        metadata={"num_results": 5},
    )
    ev = await agent.evaluate(
        step_instruction="Research benefits of FastAPI for Python web development",
        tool_result=result,
        retry_count=0,
    )
    print(f"  verdict: {ev.verdict.value}  |  reason: {ev.reason[:80]}")
    chk(ev.verdict == Verdict.PASS, f"verdict is PASS (got {ev.verdict.value})")
    return ev.verdict == Verdict.PASS


async def test_critic_retry():
    hdr("Critic — should RETRY on off-topic result at attempt 0")
    agent = CriticAgent()
    result = ToolResult(
        success=True,
        output="The Python language was named after Monty Python. It is used in "
                "chip design and embedded systems for the Epiphany coprocessor.",
        error=None,
        metadata={"num_results": 3},
    )
    ev = await agent.evaluate(
        step_instruction="Research benefits of FastAPI for Python web development",
        tool_result=result,
        retry_count=0,
    )
    print(f"  verdict: {ev.verdict.value}  |  suggestions: {ev.suggestions[:80]}")
    chk(ev.verdict == Verdict.RETRY,           f"verdict is RETRY (got {ev.verdict.value})")
    chk(bool(ev.suggestions.strip()),          "suggestions non-empty")
    chk("query" in ev.suggestions.lower(),     "suggestions contain 'query'")
    return ev.verdict == Verdict.RETRY


async def main():
    print(f"\n{BD}{'='*50}")
    print("  OpenAI smoke test — Planner + Critic")
    print(f"{'='*50}{RS}")

    results = {}
    results["Planner"]       = await test_planner()
    results["Critic PASS"]   = await test_critic_pass()
    results["Critic RETRY"]  = await test_critic_retry()

    print(f"\n{BD}── Results {'─'*38}{RS}")
    for name, passed in results.items():
        status = f"{GR}PASS{RS}" if passed else f"{RD}FAIL{RS}"
        print(f"  {name:<20} [{status}]")

    total = sum(results.values())
    print(f"\n  {BD}TOTAL: {total}/{len(results)}{RS}\n")
    return total == len(results)


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) else 1)
