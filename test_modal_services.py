# ───────────────────────────────────────────────────────────────────────────────
# FILE 3: test_modal_services.py - Comprehensive Test Script
# ───────────────────────────────────────────────────────────────────────────────

"""
Save this as: test_modal_services.py
Run with: python test_modal_services.py
"""

import asyncio
import os
import httpx
import json
import time

class BulletproofTester:
    """Comprehensive tester for bulletproof services"""
    
    def __init__(self):
        self.qwen_endpoint = os.getenv("MODAL_QWEN_ENDPOINT", "")
        self.firellava_endpoint = os.getenv("MODAL_FIRELLAVA_ENDPOINT", "")
    
    def _create_client(self) -> httpx.AsyncClient:
        """Create HTTP client with all fixes"""
        return httpx.AsyncClient(
            timeout=120.0,  # Longer timeout for health checks
            follow_redirects=True,
            verify=True,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            trust_env=True,  # DNS fix
        )
    
    async def comprehensive_test_qwen(self) -> dict:
        """Comprehensive Qwen testing"""
        if not self.qwen_endpoint:
            return {"status": "skipped", "reason": "No MODAL_QWEN_ENDPOINT"}
        
        print("🧠 Testing Bulletproof A100-40GB Qwen...")
        
        try:
            async with self._create_client() as client:
                # Test 1: Root endpoint
                try:
                    root_resp = await client.get(self.qwen_endpoint)
                    print(f"🏠 Root: {root_resp.status_code}")
                except Exception as e:
                    print(f"⚠️ Root test failed: {e}")
                
                # Test 2: Health check with retries
                health_status = None
                for attempt in range(3):
                    try:
                        health_resp = await client.get(f"{self.qwen_endpoint}/health")
                        health_status = health_resp.status_code
                        print(f"🏥 Health attempt {attempt+1}: {health_status}")
                        
                        if health_status == 200:
                            health_data = health_resp.json()
                            print(f"   Status: {health_data.get('status', 'unknown')}")
                            break
                        elif health_status == 503:
                            print(f"   Service starting up, waiting...")
                            await asyncio.sleep(30)  # Wait for service to initialize
                        
                    except Exception as e:
                        print(f"   Health check failed: {e}")
                        await asyncio.sleep(10)
                
                # Test 3: Chat generation
                if health_status == 200:
                    chat_payload = {
                        "messages": [
                            {"role": "user", "content": "Say 'Bulletproof Qwen A100-40GB working!' if you're functioning correctly"}
                        ],
                        "max_tokens": 100,
                        "temperature": 0.1
                    }
                    
                    chat_resp = await client.post(
                        f"{self.qwen_endpoint}/chat",
                        json=chat_payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if chat_resp.status_code == 200:
                        result = chat_resp.json()
                        return {
                            "status": "success",
                            "gpu": "A100-40GB",
                            "response": result.get('response', '')[:200],
                            "features": ["bulletproof_init", "robust_error_handling"],
                            "health_status": health_status
                        }
                    else:
                        error_details = chat_resp.text
                        return {
                            "status": "chat_failed",
                            "gpu": "A100-40GB", 
                            "error": f"HTTP {chat_resp.status_code}: {error_details}",
                            "health_status": health_status
                        }
                else:
                    return {
                        "status": "unhealthy",
                        "gpu": "A100-40GB",
                        "health_status": health_status,
                        "suggestion": "Service may still be initializing, wait 2-3 minutes"
                    }
                    
        except Exception as e:
            return {"status": "error", "gpu": "A100-40GB", "error": str(e)}
    
    async def comprehensive_test_firellava(self) -> dict:
        """Comprehensive FireLLaVA testing"""
        if not self.firellava_endpoint:
            return {"status": "skipped", "reason": "No MODAL_FIRELLAVA_ENDPOINT"}
        
        print("🔥 Testing Bulletproof A10G FireLLaVA...")
        
        try:
            async with self._create_client() as client:
                # Test 1: Root endpoint
                try:
                    root_resp = await client.get(self.firellava_endpoint)
                    print(f"🏠 Root: {root_resp.status_code}")
                except Exception as e:
                    print(f"⚠️ Root test failed: {e}")
                
                # Test 2: Health check with retries
                health_status = None
                for attempt in range(3):
                    try:
                        health_resp = await client.get(f"{self.firellava_endpoint}/health")
                        health_status = health_resp.status_code
                        print(f"🏥 Health attempt {attempt+1}: {health_status}")
                        
                        if health_status == 200:
                            health_data = health_resp.json()
                            print(f"   Status: {health_data.get('status', 'unknown')}")
                            break
                        elif health_status == 503:
                            print(f"   Service starting up, waiting...")
                            await asyncio.sleep(30)
                        
                    except Exception as e:
                        print(f"   Health check failed: {e}")
                        await asyncio.sleep(10)
                
                # Test 3: Text generation
                if health_status == 200:
                    text_payload = {
                        "messages": [
                            {"role": "user", "content": "Say 'Bulletproof FireLLaVA A10G working!' if you understand"}
                        ],
                        "max_tokens": 100,
                        "temperature": 0.1
                    }
                    
                    resp = await client.post(
                        f"{self.firellava_endpoint}/firellava",
                        json=text_payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        return {
                            "status": "success",
                            "gpu": "A10G",
                            "response": result.get('generated_response', '')[:200],
                            "features": ["bulletproof_init", "image_processing"],
                            "health_status": health_status
                        }
                    else:
                        error_details = resp.text
                        return {
                            "status": "generation_failed",
                            "gpu": "A10G",
                            "error": f"HTTP {resp.status_code}: {error_details}",
                            "health_status": health_status
                        }
                else:
                    return {
                        "status": "unhealthy",
                        "gpu": "A10G",
                        "health_status": health_status,
                        "suggestion": "Service may still be initializing, wait 2-3 minutes"
                    }
                    
        except Exception as e:
            if "getaddrinfo failed" in str(e):
                return {
                    "status": "dns_error",
                    "gpu": "A10G", 
                    "error": f"DNS resolution failed: {str(e)}",
                    "suggestion": "Wait 5-10 minutes for DNS propagation"
                }
            
            return {"status": "error", "gpu": "A10G", "error": str(e)}
    
    async def run_comprehensive_tests(self) -> dict:
        """Run all comprehensive tests"""
        print("🎯 BULLETPROOF MODAL SERVICES COMPREHENSIVE TEST")
        print("=" * 80)
        
        print("🔧 Configuration:")
        print(f"   Qwen endpoint: {'✅ Set' if self.qwen_endpoint else '❌ Missing'}")
        print(f"   FireLLaVA endpoint: {'✅ Set' if self.firellava_endpoint else '❌ Missing'}")
        print()
        
        results = {
            "config": {
                "qwen_endpoint": bool(self.qwen_endpoint),
                "firellava_endpoint": bool(self.firellava_endpoint)
            },
            "timestamp": time.time(),
            "test_version": "bulletproof_v3.0"
        }
        
        # Test both services with comprehensive checks
        try:
            results["qwen_bulletproof"] = await self.comprehensive_test_qwen()
        except Exception as e:
            results["qwen_bulletproof"] = {"status": "test_crashed", "error": str(e)}
        
        try:
            results["firellava_bulletproof"] = await self.comprehensive_test_firellava()
        except Exception as e:
            results["firellava_bulletproof"] = {"status": "test_crashed", "error": str(e)}
        
        return results

async def main():
    """Main comprehensive test function"""
    tester = BulletproofTester()
    results = await tester.run_comprehensive_tests()
    
    print("\n" + "=" * 80)
    print("📊 BULLETPROOF TEST RESULTS SUMMARY")
    print("=" * 80)
    
    # Print detailed results
    for service, result in results.items():
        if service in ["config", "timestamp", "test_version"]:
            continue
            
        if isinstance(result, dict) and "status" in result:
            status_emoji = {
                "success": "✅",
                "partial_success": "⚠️",
                "unhealthy": "🏥",
                "chat_failed": "💬",
                "generation_failed": "🔥",
                "failed": "❌", 
                "error": "💥",
                "dns_error": "🌐",
                "test_crashed": "💥",
                "skipped": "⏭️"
            }.get(result["status"], "❓")
            
            gpu_info = f" ({result.get('gpu', 'Unknown')})" if result.get('gpu') else ""
            print(f"{status_emoji} {service.upper().replace('_', ' ')}{gpu_info}: {result['status']}")
            
            if result.get("error"):
                print(f"   💥 Error: {result['error']}")
            if result.get("suggestion"):
                print(f"   💡 Suggestion: {result['suggestion']}")
            if result.get("features"):
                print(f"   🚀 Features: {', '.join(result['features'])}")
            if result.get("health_status"):
                print(f"   🏥 Health Status: {result['health_status']}")
    
    print("\n🎯 NEXT STEPS:")
    
    qwen_result = results.get("qwen_bulletproof", {})
    firellava_result = results.get("firellava_bulletproof", {})
    
    if qwen_result.get("status") == "success":
        print("✅ Bulletproof A100-40GB Qwen is ready for browser automation!")
    elif qwen_result.get("status") == "unhealthy":
        print("🏥 A100-40GB Qwen is initializing - wait 2-3 minutes and retest")
    else:
        print("❌ Fix A100-40GB Qwen - check logs: modal logs view qwen25-72b-browser-bulletproof")
    
    if firellava_result.get("status") == "success":
        print("✅ Bulletproof A10G FireLLaVA is ready for image analysis!")
    elif firellava_result.get("status") == "unhealthy":
        print("🏥 A10G FireLLaVA is initializing - wait 2-3 minutes and retest")
    elif firellava_result.get("status") == "dns_error":
        print("🌐 A10G FireLLaVA DNS issue - wait 5-10 minutes for propagation")
    else:
        print("❌ Fix A10G FireLLaVA - check logs: modal logs view firellava-13b-bulletproof")
    
    # Save results
    filename = f"bulletproof_test_results_{int(time.time())}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Comprehensive results saved to: {filename}")

if __name__ == "__main__":
    print("🛡️ Bulletproof Modal Services Comprehensive Tester")
    print("Testing A100-40GB Qwen + A10G FireLLaVA with full error handling")
    print()
    
    asyncio.run(main())