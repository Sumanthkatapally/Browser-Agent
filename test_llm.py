# test_llm.py
import asyncio
import os
from app.llm import LLM

async def test():
    print(f"Using HF_TOKEN: {os.getenv('HF_TOKEN', 'NOT SET')[:10]}...")
    
    # Test HuggingFace DeepSeek via fireworks-ai provider
    llm = LLM(config_name="default")  # This will use DeepSeek via HF
    print(f"LLM API type: {llm.api_type}")
    print(f"LLM model: {llm.model}")
    
    response = await llm.ask([{"role": "user", "content": "Hello, what is 2+2?"}])
    print("DeepSeek response:", response)

if __name__ == "__main__":
    asyncio.run(test())