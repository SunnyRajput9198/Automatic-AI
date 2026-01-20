import asyncio

from app.agents.specialist.researcher_agent import ResearcherAgent
from app.agents.specialist.enginer_agent import EngineerAgent
from app.agents.specialist.writer_agent import WriterAgent
from app.agents.coordinator.coordinator_agent import CoordinatorAgent
from dotenv import load_dotenv
load_dotenv()

import os
import anthropic

async def test_researcher_agent():
    """Test Researcher Agent"""
    print("\n" + "="*60)
    print("TEST 1: RESEARCHER AGENT")
    print("="*60)
    
    researcher = ResearcherAgent()
    
    task = "Search for Python async best practices"
    print(f"\n📋 Task: {task}")
    
    result = await researcher.execute(task)
    
    print(f"\n📊 Results:")
    print(f"   Success: {result.success}")
    print(f"   Confidence: {result.confidence}")
    print(f"   Duration: {result.duration_sec:.2f}s")
    print(f"   Agent: {result.agent_name}")
    
    if result.success:
        print(f"\n📄 Output Preview:")
        print(result.output[:300] + "..." if len(result.output) > 300 else result.output)
        print(f"\n   Metadata: {result.metadata}")
    else:
        print(f"\n❌ Errors: {result.errors}")
    
    print(f"\n📈 Agent Stats:")
    stats = researcher.get_stats()
    print(f"   Calls: {stats['calls']}")
    print(f"   Success Rate: {stats['success_rate']:.2%}")
    
    print("\n✅ Researcher Agent Test Complete")
    
    return result.success


async def test_engineer_agent():
    """Test Engineer Agent"""
    print("\n" + "="*60)
    print("TEST 2: ENGINEER AGENT")
    print("="*60)
    
    engineer = EngineerAgent()
    
    task = "Calculate the factorial of 15 using Python"
    print(f"\n📋 Task: {task}")
    
    result = await engineer.execute(task)
    
    print(f"\n📊 Results:")
    print(f"   Success: {result.success}")
    print(f"   Confidence: {result.confidence}")
    print(f"   Duration: {result.duration_sec:.2f}s")
    print(f"   Agent: {result.agent_name}")
    
    if result.success:
        print(f"\n📄 Output:")
        print(result.output)
        print(f"\n   Metadata: {result.metadata}")
    else:
        print(f"\n❌ Errors: {result.errors}")
    
    print(f"\n📈 Agent Stats:")
    stats = engineer.get_stats()
    print(f"   Calls: {stats['calls']}")
    print(f"   Success Rate: {stats['success_rate']:.2%}")
    
    print("\n✅ Engineer Agent Test Complete")
    
    return result.success


async def test_writer_agent():
    """Test Writer Agent"""
    print("\n" + "="*60)
    print("TEST 3: WRITER AGENT")
    print("="*60)
    
    writer = WriterAgent()
    
    task = "Write a blog post about Python async programming"
    context = {
        "researcher_output": "Python async allows concurrent execution using asyncio..."
    }
    
    print(f"\n📋 Task: {task}")
    print(f"📦 Context: researcher_output provided")
    
    result = await writer.execute(task, context)
    
    print(f"\n📊 Results:")
    print(f"   Success: {result.success}")
    print(f"   Confidence: {result.confidence}")
    print(f"   Duration: {result.duration_sec:.2f}s")
    print(f"   Agent: {result.agent_name}")
    
    if result.success:
        print(f"\n📄 Output Preview:")
        print(result.output[:400] + "..." if len(result.output) > 400 else result.output)
        print(f"\n   Metadata: {result.metadata}")
    else:
        print(f"\n❌ Errors: {result.errors}")
    
    print(f"\n📈 Agent Stats:")
    stats = writer.get_stats()
    print(f"   Calls: {stats['calls']}")
    print(f"   Success Rate: {stats['success_rate']:.2%}")
    
    print("\n✅ Writer Agent Test Complete")
    
    return result.success


async def test_coordinator_with_real_agents():
    """Test Coordinator with real specialist agents"""
    print("\n" + "="*60)
    print("TEST 4: COORDINATOR WITH REAL AGENTS")
    print("="*60)
    
    # Setup real agents
    available_agents = {
        "researcher": ResearcherAgent(),
        "engineer": EngineerAgent(),
        "writer": WriterAgent()
    }
    
    coordinator = CoordinatorAgent(available_agents)
    
    # Test multi-agent task
    task = "Research Python async patterns and write a summary"
    print(f"\n📋 Task: {task}")
    
    result = await coordinator.coordinate(task)
    
    print(f"\n📊 Coordination Results:")
    print(f"   Success: {result.success}")
    print(f"   Total Agents: {result.total_agents}")
    print(f"   Successful: {result.successful_agents}")
    print(f"   Failed: {result.failed_agents}")
    print(f"   Mode: {result.execution_mode}")
    
    print(f"\n📄 Individual Agent Results:")
    for agent_result in result.agent_results:
        print(f"   - {agent_result['role']}: {agent_result['success']} (confidence: {agent_result.get('confidence', 0):.2f})")
    
    if result.success:
        print(f"\n📝 Final Output Preview:")
        print(result.final_output[:500] + "..." if len(result.final_output) > 500 else result.final_output)
    
    print("\n✅ Coordinator with Real Agents Test Complete")
    
    return result.success


async def main():
    """Run all Day 2 tests"""
    print("\n" + "="*60)
    print("🧠 WEEK 4 - DAY 2 TESTS")
    print("="*60)
    print("\nTesting: Real Specialist Agents (Researcher, Engineer, Writer)")
    print("")
    
    results = []
    
    try:
        # Test each specialist agent
        results.append(await test_researcher_agent())
        results.append(await test_engineer_agent())
        results.append(await test_writer_agent())
        results.append(await test_coordinator_with_real_agents())
        
        # Summary
        print("\n" + "="*60)
        if all(results):
            print("✅ ALL DAY 2 TESTS PASSED!")
        else:
            print("⚠️  SOME TESTS FAILED")
        print("="*60)
        
        print("\n📝 Day 2 Summary:")
        print(f"   {'✅' if results[0] else '❌'} ResearcherAgent - Web research working")
        print(f"   {'✅' if results[1] else '❌'} EngineerAgent - Code generation working")
        print(f"   {'✅' if results[2] else '❌'} WriterAgent - Content creation working")
        print(f"   {'✅' if results[3] else '❌'} Coordinator - Multi-agent coordination working")
        
        if all(results):
            print("\n🚀 Ready to move to Day 3: Parallel Execution!")
        else:
            print("\n⚠️  Fix failing tests before Day 3")
        print("")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())