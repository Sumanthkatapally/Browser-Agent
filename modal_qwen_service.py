# ───────────────────────────────────────────────────────────────────────────────
# FILE 1: modal_qwen_service.py - Bulletproof A100-40GB Qwen Service
# ───────────────────────────────────────────────────────────────────────────────

"""
Save this as: modal_qwen_service.py
Deploy with: modal deploy modal_qwen_service.py
"""

import os
import modal
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = modal.App("qwen25-72b-browser-bulletproof")

# Robust image with specific working versions
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.1.0",
    "transformers==4.44.0", 
    "accelerate==0.24.0",
    "huggingface-hub>=0.23.2",  # Compatible version range
    "fastapi[standard]==0.104.1",
    "uvicorn[standard]==0.24.0",
    "pydantic==2.5.0"
)

secrets = [modal.Secret.from_name("huggingface-secret")]

@app.cls(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    secrets=secrets,
    scaledown_window=600,
    min_containers=1
)
@modal.concurrent(max_inputs=5)
class QwenBulletproof:
    def __enter__(self):
        """Bulletproof initialization with comprehensive error handling"""
        print("🚀 Loading Qwen2.5-72B with bulletproof initialization...")
        
        # Step 1: Validate environment
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("❌ CRITICAL: HF_TOKEN not found in environment!")
            print("   Please ensure your Modal secret 'huggingface-secret' contains HF_TOKEN")
            print("   Run: modal secret inspect huggingface-secret")
            self.tokenizer = None
            self.model = None
            self.init_error = "HF_TOKEN_MISSING"
            return
        
        if len(hf_token) < 10:
            print(f"❌ CRITICAL: HF_TOKEN appears invalid: '{hf_token}'")
            self.tokenizer = None
            self.model = None
            self.init_error = "HF_TOKEN_INVALID"
            return
        
        print(f"🔑 Valid HF token found: {hf_token[:10]}...")
        
        model_id = "Qwen/Qwen2.5-72B-Instruct"
        
        try:
            # Step 2: Load tokenizer with multiple fallback methods
            print("📚 Loading tokenizer...")
            
            # Try primary method
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    use_auth_token=hf_token,  # Primary auth method
                    use_fast=True,
                    padding_side="left"
                )
                print("✅ Tokenizer loaded with use_auth_token")
            except Exception as e1:
                print(f"⚠️ Primary tokenizer loading failed: {e1}")
                print("🔄 Trying fallback method with token parameter...")
                
                # Fallback method
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        token=hf_token,  # Fallback auth method
                        use_fast=True,
                        padding_side="left"
                    )
                    print("✅ Tokenizer loaded with token parameter")
                except Exception as e2:
                    print(f"❌ Fallback tokenizer loading failed: {e2}")
                    self.tokenizer = None
                    self.model = None
                    self.init_error = f"TOKENIZER_LOAD_FAILED: {str(e2)}"
                    return
            
            # Step 3: Configure tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("🔧 Set pad_token to eos_token")
            
            # Step 4: Test tokenizer
            test_text = "Hello world"
            test_tokens = self.tokenizer.encode(test_text)
            decoded_text = self.tokenizer.decode(test_tokens, skip_special_tokens=True)
            print(f"🧪 Tokenizer test: '{test_text}' -> {len(test_tokens)} tokens -> '{decoded_text}'")
            
            if not test_tokens:
                raise ValueError("Tokenizer test failed - no tokens generated")
            
            print("✅ Tokenizer validated successfully!")
            
            # Step 5: Load model with fallback auth methods
            print("🧠 Loading model...")
            
            # Try primary method
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    use_auth_token=hf_token,  # Primary auth method
                    low_cpu_mem_usage=True,
                    use_cache=True
                )
                print("✅ Model loaded with use_auth_token")
            except Exception as e1:
                print(f"⚠️ Primary model loading failed: {e1}")
                print("🔄 Trying fallback method with token parameter...")
                
                # Fallback method
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True,
                        token=hf_token,  # Fallback auth method
                        low_cpu_mem_usage=True,
                        use_cache=True
                    )
                    print("✅ Model loaded with token parameter")
                except Exception as e2:
                    print(f"❌ Fallback model loading failed: {e2}")
                    self.tokenizer = None
                    self.model = None
                    self.init_error = f"MODEL_LOAD_FAILED: {str(e2)}"
                    return
            
            print("✅ Qwen2.5-72B loaded on A100-40GB successfully!")
            print(f"📊 Model devices: {getattr(self.model, 'hf_device_map', 'auto')}")
            
            # Final validation
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                self.init_error = "TOKENIZER_VALIDATION_FAILED"
                return
            
            if not hasattr(self, 'model') or self.model is None:
                self.init_error = "MODEL_VALIDATION_FAILED"
                return
            
            self.init_error = None
            print("🎉 All components validated - Qwen is ready!")
            
        except Exception as e:
            print(f"❌ Unexpected error during initialization: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            self.tokenizer = None
            self.model = None
            self.init_error = f"UNEXPECTED_ERROR: {str(e)}"

    @modal.method()
    def generate_browser_response(
        self, 
        messages: list, 
        max_tokens: int = 3072,
        temperature: float = 0.1,
        tools: list = None
    ) -> dict:
        """Bulletproof generation with detailed error reporting"""
        
        # Check initialization status
        if hasattr(self, 'init_error') and self.init_error:
            return {
                "response": f"Service initialization failed: {self.init_error}",
                "error": "service_not_initialized",
                "init_error": self.init_error,
                "gpu": "A100-40GB",
                "debug": "Check HF_TOKEN secret and redeploy service"
            }
        
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            return {
                "response": "Error: Tokenizer not available",
                "error": "tokenizer_not_loaded",
                "gpu": "A100-40GB",
                "debug": "Tokenizer failed to initialize"
            }
        
        if not hasattr(self, 'model') or self.model is None:
            return {
                "response": "Error: Model not available",
                "error": "model_not_loaded",
                "gpu": "A100-40GB",
                "debug": "Model failed to initialize"
            }
        
        try:
            # Add browser automation context
            if not messages or messages[0].get("role") != "system":
                browser_system = {
                    "role": "system", 
                    "content": "You are an expert browser automation agent. Provide clear, step-by-step web automation instructions."
                }
                messages = [browser_system] + messages
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=7168,
                padding=False
            ).to(self.model.device)
            
            input_length = inputs['input_ids'].shape[1]
            print(f"🔢 Input tokens: {input_length}")
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 3072),
                    temperature=temperature,
                    do_sample=temperature > 0.0,
                    top_p=0.9 if temperature > 0.0 else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            output_length = len(response_tokens)
            
            print(f"📤 Generated {output_length} tokens")
            
            return {
                "response": response.strip(),
                "input_tokens": input_length,
                "output_tokens": output_length,
                "model": "Qwen/Qwen2.5-72B-Instruct",
                "gpu": "A100-40GB",
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return {
                "response": f"Generation failed: {str(e)}",
                "error": f"generation_failed: {type(e).__name__}",
                "gpu": "A100-40GB"
            }

    @modal.method()
    def health_check(self) -> dict:
        """Comprehensive health check"""
        try:
            if hasattr(self, 'init_error') and self.init_error:
                return {
                    "status": "unhealthy",
                    "error": self.init_error,
                    "gpu": "A100-40GB",
                    "suggestion": "Check HF_TOKEN secret and redeploy"
                }
            
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                return {
                    "status": "unhealthy",
                    "error": "tokenizer_not_available",
                    "gpu": "A100-40GB"
                }
            
            if not hasattr(self, 'model') or self.model is None:
                return {
                    "status": "unhealthy",
                    "error": "model_not_available",
                    "gpu": "A100-40GB"
                }
            
            # Test generation
            test_response = self.generate_browser_response(
                [{"role": "user", "content": "Say 'Bulletproof Qwen is healthy'"}],
                max_tokens=20,
                temperature=0.0
            )
            
            if "error" in test_response:
                return {
                    "status": "unhealthy",
                    "error": test_response["error"],
                    "gpu": "A100-40GB"
                }
            
            return {
                "status": "healthy",
                "gpu": "A100-40GB",
                "model": "Qwen/Qwen2.5-72B-Instruct",
                "test_response": test_response["response"][:100],
                "features": ["bulletproof_init", "fallback_auth", "robust_error_handling"]
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "gpu": "A100-40GB",
                "exception_type": type(e).__name__
            }

# FastAPI for bulletproof Qwen
@app.function(image=image, timeout=300)
@modal.asgi_app()
def qwen_browser_api():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    from typing import List, Optional
    import time
    
    api = FastAPI(
        title="Bulletproof Qwen2.5-72B Browser API",
        description="Bulletproof A100-40GB Qwen deployment with comprehensive error handling",
        version="3.0.0"
    )
    
    class ChatMessage(BaseModel):
        role: str = Field(..., description="Message role")
        content: str = Field(..., description="Message content")
    
    class ChatRequest(BaseModel):
        messages: List[ChatMessage] = Field(..., description="Conversation messages")
        max_tokens: int = Field(3072, ge=1, le=4096, description="Max tokens")
        temperature: float = Field(0.1, ge=0.0, le=2.0, description="Temperature")
        tools: Optional[List[dict]] = Field(None, description="Available tools")
    
    @api.post("/chat")
    async def chat_endpoint(request: ChatRequest):
        """Bulletproof chat endpoint"""
        start_time = time.time()
        
        try:
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            result = QwenBulletproof().generate_browser_response.remote(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                tools=request.tools
            )
            
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            
            if "error" in result:
                return JSONResponse(
                    status_code=500,
                    content={
                        "detail": f"Error: {result['error']}",
                        "debug_info": result,
                        "gpu": "A100-40GB"
                    }
                )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            return JSONResponse(
                status_code=500,
                content={
                    "detail": f"Unexpected error: {str(e)}",
                    "processing_time": processing_time,
                    "gpu": "A100-40GB"
                }
            )
    
    @api.get("/health")
    async def health_endpoint():
        """Bulletproof health check"""
        try:
            result = QwenBulletproof().health_check.remote()
            status_code = 200 if result.get("status") == "healthy" else 503
            return JSONResponse(status_code=status_code, content=result)
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": str(e), "gpu": "A100-40GB"}
            )
    
    @api.get("/")
    async def root():
        """Service information"""
        return {
            "service": "Bulletproof Qwen2.5-72B Browser API",
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "gpu": "A100-40GB",
            "version": "3.0.0",
            "features": [
                "bulletproof_initialization",
                "fallback_auth_methods",
                "comprehensive_error_handling",
                "robust_validation"
            ],
            "endpoints": {
                "chat": "/chat",
                "health": "/health",
                "docs": "/docs"
            }
        }
    
    return api

if __name__ == "__main__":
    print("Deploy bulletproof A100-40GB Qwen with: modal deploy modal_qwen_service.py")
