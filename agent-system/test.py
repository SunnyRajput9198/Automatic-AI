import asyncio
from app.agents.specialist.researcher_agent import ResearcherAgent
from app.agents.specialist.enginer_agent import EngineerAgent
from app.agents.specialist.writer_agent import WriterAgent
from app.agents.coordinator.coordinator_agent import CoordinatorAgent


async def main():
    agents = {
        "researcher": ResearcherAgent(),
        "engineer": EngineerAgent(),
        "writer": WriterAgent()
    }

    coordinator = CoordinatorAgent(agents)

    print("\n🔥 PARALLEL TEST")
    result = await coordinator.coordinate(
        "Search Python asyncio best practices and calculate factorial of 20"
    )

    print("Mode:", result.execution_mode)
    print("Successful:", result.successful_agents)
    print("Agents used:", result.total_agents)
    print("\nOutput preview:\n")
    print(result.final_output[:500])


if __name__ == "__main__":
    asyncio.run(main())
