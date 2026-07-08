import asyncio
from app.agents.supervisor import run_agent
from dotenv import load_dotenv
load_dotenv()

async def test():
    try:
        res = await run_agent(message="Hello", session_id="test_session", provider="bedrock")
        print("Success:", res)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
