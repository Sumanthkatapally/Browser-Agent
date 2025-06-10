#!/usr/bin/env python3
"""
Test script for LLM class integration with Modal services
This tests your actual LLM class configurations
"""

import asyncio
import sys
import os
from pathlib import Path

# Add your app to path
sys.path.append(str(Path(__file__).parent))

try:
    from app.llm import LLM
    from app.schema import Message
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure this script is in your project root directory")
    sys.exit(1)

class LLMIntegrationTester:
    """Test LLM class integration with both Modal services"""
    
    async def test_qwen_text_generation(self):
        """Test Qwen via LLM class for text generation"""
        print("🧠 Testing Qwen Text Generation via LLM Class...")
        
        try:
            # Initialize LLM with qwen config
            llm = LLM(config_name="default")  # Should use Qwen from your config
            print(f"   Model: {llm.model}")
            print(f"   API Type: {llm.api_type}")
            print(f"   Base URL: {llm.base_url}")
            
            # Test simple text generation
            messages = [
                Message.user_message("You are a browser automation expert. Explain how to click a button on a webpage in 2 sentences.")
            ]
            
            print("📤 Sending request...")
            response = await llm.ask(messages, stream=False)
            print(f"✅ Response: {response[:200]}...")
            
            return {"status": "success", "response": response[:200]}
            
        except Exception as e:
            print(f"❌ Qwen text error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def test_qwen_tool_calling(self):
        """Test Qwen tool calling via LLM class"""
        print("\n🔧 Testing Qwen Tool Calling...")
        
        try:
            llm = LLM(config_name="default")
            
            # Define browser automation tools
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "navigate_to_url",
                        "description": "Navigate browser to a specific URL",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "description": "URL to navigate to"}
                            },
                            "required": ["url"]
                        }
                    }
                },
                {
                    "type": "function", 
                    "function": {
                        "name": "click_element",
                        "description": "Click on a web element",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "selector": {"type": "string", "description": "CSS selector for element"},
                                "method": {"type": "string", "enum": ["css", "xpath"], "description": "Selection method"}
                            },
                            "required": ["selector"]
                        }
                    }
                }
            ]
            
            messages = [
                Message.user_message("I need to navigate to https://google.com and click the search button. What tools should I use?")
            ]
            
            print("📤 Sending tool calling request...")
            response = await llm.ask_tool(
                messages=messages,
                tools=tools,
                temperature=0.1
            )
            
            if response:
                print(f"✅ Tool response: {response.content[:200]}...")
                return {"status": "success", "response": response.content[:200]}
            else:
                print("❌ No response received")
                return {"status": "failed", "error": "No response"}
                
        except Exception as e:
            print(f"❌ Tool calling error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def test_firellava_vision(self):
        """Test FireLLaVA vision via LLM class"""
        print("\n👁️ Testing FireLLaVA Vision...")
        
        try:
            # Initialize LLM with vision config
            llm = LLM(config_name="firellava_13b")  # Your vision config
            print(f"   Model: {llm.model}")
            print(f"   API Type: {llm.api_type}")
            print(f"   Base URL: {llm.base_url}")
            
            # Test with base64 image
            sample_image_b64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
            
            messages = [
                {
                    "role": "user",
                    "content": "Describe what you see in this image",
                    "base64_image": sample_image_b64
                }
            ]
            
            print("📤 Sending vision request...")
            response = await llm.ask(messages, stream=False)
            print(f"✅ Vision response: {response[:200]}...")
            
            return {"status": "success", "response": response[:200]}
            
        except Exception as e:
            print(f"❌ Vision error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def test_configuration_validation(self):
        """Test configuration validation"""
        print("\n⚙️ Testing Configuration Validation...")
        
        results = {}
        
        # Test different configs
        configs_to_test = ["default", "qwen", "firellava_13b"]
        
        for config_name in configs_to_test:
            try:
                llm = LLM(config_name=config_name)
                results[config_name] = {
                    "status": "success",
                    "model": llm.model,
                    "api_type": llm.api_type,
                    "base_url": llm.base_url[:50] + "..." if llm.base_url and len(llm.base_url) > 50 else llm.base_url,
                    "has_client": hasattr(llm, 'client') and llm.client is not None
                }
                print(f"✅ {config_name}: {llm.model} ({llm.api_type})")
            except Exception as e:
                results[config_name] = {"status": "error", "error": str(e)}
                print(f"❌ {config_name}: {e}")
        
        return results
    
    async def run_all_tests(self):
        """Run comprehensive LLM integration tests"""
        print("🚀 LLM Integration Test Suite")
        print("=" * 50)
        
        results = {}
        
        # Configuration validation
        results["config"] = await self.test_configuration_validation()
        
        # Qwen tests
        results["qwen_text"] = await self.test_qwen_text_generation()
        results["qwen_tools"] = await self.test_qwen_tool_calling()
        
        # FireLLaVA tests
        results["firellava_vision"] = await self.test_firellava_vision()
        
        return results

async def main():
    """Main test function"""
    print("🔍 LLM Integration Tester")
    print("Testing your LLM class with Modal deployments")
    print()
    
    # Environment check
    print("🔧 Environment Check:")
    required_vars = ["MODAL_QWEN_ENDPOINT", "MODAL_FIRELLAVA_ENDPOINT", "MODAL_TOKEN"]
    for var in required_vars:
        value = os.getenv(var)
        print(f"   {var}: {'✅ Set' if value else '❌ Missing'}")
    print()
    
    tester = LLMIntegrationTester()
    results = await tester.run_all_tests()
    
    print("\n" + "=" * 50)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    for test_name, result in results.items():
        if test_name == "config":
            print("⚙️ Configuration:")
            for config, details in result.items():
                status = "✅" if details["status"] == "success" else "❌"
                print(f"   {status} {config}: {details.get('model', details.get('error', ''))}")
        else:
            status_emoji = {
                "success": "✅",
                "failed": "❌",
                "error": "💥"
            }.get(result.get("status"), "❓")
            print(f"{status_emoji} {test_name}: {result.get('status', 'unknown')}")
            if result.get("error"):
                print(f"   Error: {result['error']}")
    
    print("\n🎯 RECOMMENDATIONS:")
    if results.get("qwen_text", {}).get("status") == "success":
        print("✅ Qwen text generation is working - ready for browser automation!")
    else:
        print("❌ Fix Qwen text generation first")
    
    if results.get("qwen_tools", {}).get("status") == "success":
        print("✅ Qwen tool calling is working - ready for browser tools!")
    else:
        print("❌ Check Qwen tool calling configuration")
    
    if results.get("firellava_vision", {}).get("status") == "success":
        print("✅ FireLLaVA vision is working - ready for screenshot analysis!")
    else:
        print("❌ Fix FireLLaVA configuration for image tasks")

if __name__ == "__main__":
    asyncio.run(main())