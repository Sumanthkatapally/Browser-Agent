import asyncio
import os
from dotenv import load_dotenv
from app.config import config
from app.llm import LLM

# Load environment variables
load_dotenv()

async def test_configuration():
    """Test the TOML configuration with environment variable substitution"""
    
    print("🧪 Testing TOML Configuration with Modal Services")
    print("=" * 55)
    
    # Check environment variables
    print("\n🔍 Environment Variables:")
    qwen_endpoint = os.getenv("MODAL_QWEN_ENDPOINT")
    firellava_endpoint = os.getenv("MODAL_FIRELLAVA_ENDPOINT")
    
    print(f"   MODAL_QWEN_ENDPOINT: {qwen_endpoint}")
    print(f"   MODAL_FIRELLAVA_ENDPOINT: {firellava_endpoint}")
    
    if not qwen_endpoint or not firellava_endpoint:
        print("❌ Environment variables not set! Please check your .env file.")
        return
    
    # Test configuration loading
    print("\n📋 Configuration Status:")
    try:
        default_config = config.llm.get("default")
        qwen_config = config.llm.get("qwen")
        vision_config = config.llm.get("firellava_13b")
        
        print(f"   Default LLM: {default_config.model if default_config else 'Not found'}")
        print(f"   Default API Type: {default_config.api_type if default_config else 'Not found'}")
        print(f"   Default Base URL: {default_config.base_url if default_config else 'Not found'}")
        
        print(f"   Qwen LLM: {qwen_config.model if qwen_config else 'Not found'}")
        print(f"   Qwen Base URL: {qwen_config.base_url if qwen_config else 'Not found'}")
        
        print(f"   Vision LLM: {vision_config.model if vision_config else 'Not found'}")
        print(f"   Vision Base URL: {vision_config.base_url if vision_config else 'Not found'}")
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return
    
    # Test Qwen service
    print("\n🧠 Testing Qwen Modal Service...")
    try:
        qwen_llm = LLM(config_name="default")
        
        messages = [
            {"role": "user", "content": "Say 'TOML Qwen Modal integration working!' if you understand this test."}
        ]
        
        response = await qwen_llm.ask(messages, stream=False)
        print(f"✅ Qwen Response: {response}")
        
    except Exception as e:
        print(f"❌ Qwen test failed: {e}")
    
    # Test FireLLaVA service
    print("\n🔥 Testing FireLLaVA Modal Service...")
    try:
        vision_llm = LLM(config_name="firellava_13b")
        
        messages = [
            {"role": "user", "content": "Say 'TOML FireLLaVA Modal integration working!' if you understand this test."}
        ]
        
        response = await vision_llm.ask(messages, stream=False)
        print(f"✅ FireLLaVA Response: {response}")
        
    except Exception as e:
        print(f"❌ FireLLaVA test failed: {e}")
    
    print("\n🎉 Configuration test complete!")

if __name__ == "__main__":
    asyncio.run(test_configuration())
