import asyncio
from app.tools.web_search import WebSearchTool

async def test():
    tool = WebSearchTool()
    result = await tool.run(query='machine learning', max_results=3)
    print('Success:', result.success)
    print('Source:', result.metadata.get('source'))
    print('Num results:', result.metadata.get('num_results'))
    print(result.output[:500])

asyncio.run(test())
