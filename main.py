import asyncio
import os
import httpx
from dotenv import load_dotenv
from app.agent.manus import Manus
from app.logger import logger
from app.config import config

# Load environment variables
load_dotenv()

async def test_modal_connectivity():
    """Test connectivity to Modal services before starting agent"""
    
    logger.info("🔍 Testing Modal service connectivity...")
    
    # Get endpoints from TOML config
    try:
        default_llm = config.llm.get("default")
        vision_llm = config.llm.get("firellava_13b") or config.llm.get("vision")
        
        if not default_llm or not vision_llm:
            logger.error("❌ LLM configurations not found in config.toml!")
            return False
        
        qwen_endpoint = default_llm.base_url
        firellava_endpoint = vision_llm.base_url
        
        logger.info(f"🧠 Qwen endpoint: {qwen_endpoint}")
        logger.info(f"🔥 FireLLaVA endpoint: {firellava_endpoint}")
        
        if not qwen_endpoint or not firellava_endpoint:
            logger.error("❌ Modal endpoints not configured!")
            logger.error("Please check:")
            logger.error("   1. .env file contains MODAL_QWEN_ENDPOINT and MODAL_FIRELLAVA_ENDPOINT")
            logger.error("   2. config.toml has correct ${VARIABLE} substitution")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error loading configuration: {e}")
        return False
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test Qwen service
        try:
            health_url = qwen_endpoint.rstrip("/") + "/health"
            response = await client.get(health_url)
            if response.status_code == 200:
                logger.info("✅ Qwen service: Ready")
            elif response.status_code == 503:
                logger.warning("⏳ Qwen service: Still initializing (this is normal)")
                logger.warning("   Services need 2-3 minutes to load models into GPU memory")
            else:
                logger.warning(f"⚠️  Qwen service: Status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Qwen service connection failed: {e}")
            return False
        
        # Test FireLLaVA service
        try:
            health_url = firellava_endpoint.rstrip("/") + "/health"
            response = await client.get(health_url)
            if response.status_code == 200:
                logger.info("✅ FireLLaVA service: Ready")
            elif response.status_code == 503:
                logger.warning("⏳ FireLLaVA service: Still initializing (this is normal)")
            else:
                logger.warning(f"⚠️  FireLLaVA service: Status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ FireLLaVA service connection failed: {e}")
            return False
    
    return True

async def main():
    """Main function with Modal service integration"""
    
    logger.info("🚀 Starting Browser-Use with Modal Services")
    logger.info("📋 Loading configuration from config.toml...")
    
    # Test Modal connectivity first
    if not await test_modal_connectivity():
        logger.error("❌ Modal services not accessible. Please check your configuration.")
        logger.error("💡 Troubleshooting:")
        logger.error("   1. Verify .env file has correct endpoint URLs")
        logger.error("   2. Wait 2-3 minutes if services are still initializing")
        logger.error("   3. Check Modal dashboard for service status")
        return
    
    try:
        prompt = input("\nEnter your prompt: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request with Modal services...")
        
        # Initialize agent (it will use the TOML config automatically)
        agent = Manus()
        
        # Run the agent
        await agent.run(prompt)
        
        logger.info("✅ Request processing completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user.")
    except Exception as e:
        logger.error(f"❌ Agent execution failed: {e}")
        logger.error("💡 Troubleshooting tips:")
        logger.error("   1. Wait 2-3 minutes for services to fully initialize")
        logger.error("   2. Check Modal dashboard for service status")
        logger.error("   3. Verify config/config.toml configuration")

if __name__ == "__main__":
    asyncio.run(main())