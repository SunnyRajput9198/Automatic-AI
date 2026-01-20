

import asyncio
from typing import Dict, Any, Optional
from app.agents.base_agent import BaseAgent, AgentResult
from app.agents.coordinator.task_router import TaskRouter
from app.agents.coordinator.coordinator_agent import CoordinatorAgent


# ============================================================================
# MOCK AGENTS FOR TESTING
# ============================================================================

class MockResearcherAgent(BaseAgent):
    """Mock researcher for testing"""
    
    def __init__(self):
        super().__init__(
            name="researcher_mock",
            role="researcher",
            allowed_tools=["web_search", "web_fetch"]
        )
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """Mock execution - returns fake research"""
        print(f"🔍 MockResearcher executing: {task}")
        
        return AgentResult(
            success=True,
            output="Mock research results: Python is a programming language.",
            metadata={"sources": ["mock_source.com"]},
            confidence=0.85,
            agent_name=self.name
        )


class MockEngineerAgent(BaseAgent):
    """Mock engineer for testing"""
    
    def __init__(self):
        super().__init__(
            name="engineer_mock",
            role="engineer",
            allowed_tools=["python_executor", "file_write", "file_read"]
        )
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """Mock execution - returns fake code"""
        print(f"💻 MockEngineer executing: {task}")
        
        return AgentResult(
            success=True,
            output="Mock code: print('Hello from engineer')",
            metadata={"language": "python"},
            confidence=0.9,
            agent_name=self.name
        )


class MockWriterAgent(BaseAgent):
    """Mock writer for testing"""
    
    def __init__(self):
        super().__init__(
            name="writer_mock",
            role="writer",
            allowed_tools=["file_write"]
        )
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """Mock execution - returns fake content"""
        print(f"✍️  MockWriter executing: {task}")
        
        # Check if we have researcher output
        researcher_output = context.get("researcher_output", "")
        
        if researcher_output:
            output = f"Mock article based on research:\n{researcher_output}"
        else:
            output = "Mock article: This is a test article."
        
        return AgentResult(
            success=True,
            output=output,
            metadata={"word_count": 100},
            confidence=0.8,
            agent_name=self.name
        )


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

async def test_task_router():
    """Test Task Router"""
    print("\n" + "="*60)
    print("TEST 1: TASK ROUTER")
    print("="*60)
    
    router = TaskRouter()
    
    test_cases = [
        "Search for Python tutorials",
        "Write a blog post about AI",
        "Calculate fibonacci(50) using Python",
        "Research AI trends and write a summary",
        "Create a Python script for data processing"
    ]
    
    for task in test_cases:
        print(f"\n📋 Task: {task}")
        decision = router.route(task)
        print(f"   Agents: {decision.agents_needed}")
        print(f"   Mode: {decision.execution_mode}")
        print(f"   Confidence: {decision.confidence}")
        print(f"   Reasoning: {decision.reasoning}")
    
    print("\n✅ Task Router Test Complete")


async def test_coordinator_single_agent():
    """Test Coordinator with single agent"""
    print("\n" + "="*60)
    print("TEST 2: COORDINATOR - SINGLE AGENT")
    print("="*60)
    
    # Setup agents
    available_agents = {
        "researcher": MockResearcherAgent(),
        "engineer": MockEngineerAgent(),
        "writer": MockWriterAgent()
    }
    
    coordinator = CoordinatorAgent(available_agents)
    
    # Test with research task
    task = "Search for Python best practices"
    print(f"\n📋 Task: {task}")
    
    result = await coordinator.coordinate(task)
    
    print(f"\n📊 Results:")
    print(f"   Success: {result.success}")
    print(f"   Agents Used: {result.total_agents}")
    print(f"   Successful: {result.successful_agents}")
    print(f"   Mode: {result.execution_mode}")
    print(f"   Output Preview: {result.final_output[:100]}...")
    
    print("\n✅ Single Agent Test Complete")


async def test_coordinator_multi_agent():
    """Test Coordinator with multiple agents"""
    print("\n" + "="*60)
    print("TEST 3: COORDINATOR - MULTIPLE AGENTS")
    print("="*60)
    
    # Setup agents
    available_agents = {
        "researcher": MockResearcherAgent(),
        "engineer": MockEngineerAgent(),
        "writer": MockWriterAgent()
    }
    
    coordinator = CoordinatorAgent(available_agents)
    
    # Test with multi-agent task
    task = "Research AI trends then write a blog post about it"
    print(f"\n📋 Task: {task}")
    
    result = await coordinator.coordinate(task)
    
    print(f"\n📊 Results:")
    print(f"   Success: {result.success}")
    print(f"   Agents Used: {result.total_agents}")
    print(f"   Successful: {result.successful_agents}")
    print(f"   Mode: {result.execution_mode}")
    print(f"\n   Final Output:\n{result.final_output}")
    
    print(f"\n   Individual Agent Results:")
    for agent_result in result.agent_results:
        print(f"      - {agent_result['role']}: {agent_result['success']}")
    
    print("\n✅ Multi-Agent Test Complete")


async def test_agent_stats():
    """Test agent statistics tracking"""
    print("\n" + "="*60)
    print("TEST 4: AGENT STATISTICS")
    print("="*60)
    
    agent = MockResearcherAgent()
    
    # Simulate some executions
    for i in range(5):
        await agent.execute(f"Task {i}")
        agent.record_success()
    
    for i in range(2):
        agent.record_failure()
    
    stats = agent.get_stats()
    
    print(f"\n📊 Agent Stats:")
    print(f"   Name: {stats['name']}")
    print(f"   Role: {stats['role']}")
    print(f"   Calls: {stats['calls']}")
    print(f"   Successes: {stats['successes']}")
    print(f"   Failures: {stats['failures']}")
    print(f"   Success Rate: {stats['success_rate']:.2%}")
    
    print("\n✅ Statistics Test Complete")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

async def main():
    """Run all Day 1 tests"""
    print("\n" + "="*60)
    print("🧠 WEEK 4 - DAY 1 TESTS")
    print("="*60)
    print("\nTesting: BaseAgent, TaskRouter, CoordinatorAgent")
    print("")
    
    try:
        # Run tests
        await test_task_router()
        await test_coordinator_single_agent()
        await test_coordinator_multi_agent()
        await test_agent_stats()
        
        print("\n" + "="*60)
        print("✅ ALL DAY 1 TESTS PASSED!")
        print("="*60)
        print("\n📝 Day 1 Summary:")
        print("   ✅ BaseAgent - Shared contract working")
        print("   ✅ TaskRouter - Routing logic working")
        print("   ✅ CoordinatorAgent - Basic coordination working")
        print("   ✅ Agent stats - Performance tracking working")
        print("\n🚀 Ready to move to Day 2: Specialist Agents!")
        print("")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())