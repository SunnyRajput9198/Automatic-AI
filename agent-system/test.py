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

    print("\n🔥 RUN 1")
    await coordinator.coordinate("Research Python asyncio and write summary")

    print("\n🔥 RUN 2")
    await coordinator.coordinate("Research Python asyncio and write summary")

    print("\n🔥 RUN 3")
    await coordinator.coordinate("Research Python asyncio and write summary")

    print("\n🧠 Check this file:")
    print("workspace/agent_performance.json")


if __name__ == "__main__":
    asyncio.run(main())
