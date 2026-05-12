import asyncio
from ragService import RagService

async def main():
    r = RagService('harusame-md')
    r._ensure_settings_yaml()
    print("Settings yaml generated.")
    try:
        await r._run_graphrag_index()
        print("GraphRAG indexing completed without immediate errors.")
    except Exception as e:
        print("Error during GraphRAG indexing:", e)

asyncio.run(main())