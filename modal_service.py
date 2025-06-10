# ───────────────────────────────────────────────────────────────────────────────
# FILE 2: modal_service.py - Bulletproof A10G FireLLaVA Service
# ───────────────────────────────────────────────────────────────────────────────

"""
Save this as: modal_service.py
Deploy with: modal deploy modal_service.py
"""

import os
import modal
import base64
from io import BytesIO
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

app = modal.App("firellava-13b-bulletproof")

# Robust image for A10G
image = modal.Image.debian_slim().pip_install(
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "huggingface-hub>=0.20.0",
    "accelerate>=0.20.0",
    "fastapi[standard]",
    "uvicorn[standard]",
    "Pillow>=9.0.0",
    "pydantic>=2.0.0"
)

secrets = [modal.Secret.from_name("huggingface-secret")]

@app.cls(
    image=image, 
    gpu="A10G",
    timeout=1200,
    secrets=secrets,
    scaledown_window=300,
    min_containers=1
)
class FireLLaVA:
    def __enter__(self):
        """Bulletproof FireLLaVA initialization"""
        print("🔥 Loading FireLLaVA-13B with bulletproof initialization...")
        
        # Step 1: Validate environment
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("❌ CRITICAL: HF_TOKEN not found!")
            self.processor = None
            self.model = None
            self.init_error = "HF_TOKEN_MISSING"
            return
        
        print(f"🔑 Valid HF token found: {hf_token[:10]}...")
        
        try:
            # Step 2: Download model with fallback auth
            print("⬇️ Downloading FireLLaVA model...")
            
            try:
                model_dir = snapshot_download(
                    repo_id="fireworks-ai/FireLLaVA-13b",
                    use_auth_token=hf_token
                )
                print("✅ Model downloaded with use_auth_token")
            except Exception as e1:
                print(f"⚠️ Primary download failed: {e1}")
                print("🔄 Trying fallback with token parameter...")
                
                try:
                    model_dir = snapshot_download(
                        repo_id="fireworks-ai/FireLLaVA-13b",
                        token=hf_token
                    )
                    print("✅ Model downloaded with token parameter")
                except Exception as e2:
                    print(f"❌ Fallback download failed: {e2}")
                    self.processor = None
                    self.model = None
                    self.init_error = f"DOWNLOAD_FAILED: {str(e2)}"
                    return
            
            # Step 3: Load processor
            print("📚 Loading processor...")
            self.processor = AutoProcessor.from_pretrained(model_dir)
            print("✅ Processor loaded successfully")
            
            # Step 4: Load model
            print("🧠 Loading FireLLaVA model on A10G...")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            print("✅ FireLLaVA-13B loaded successfully on A10G!")
            print(f"📊 Model device: {self.model.device}")
            
            # Check memory
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**3
                print(f"💾 A10G GPU memory used: {memory_used:.2f} GB / 24 GB")
            
            # Final validation
            if not hasattr(self, 'processor') or self.processor is None:
                self.init_error = "PROCESSOR_VALIDATION_FAILED"
                return
            
            if not hasattr(self, 'model') or self.model is None:
                self.init_error = "MODEL_VALIDATION_FAILED"
                return
            
            self.init_error = None
            print("🎉 FireLLaVA bulletproof initialization complete!")
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            self.processor = None
            self.model = None
            self.init_error = f"UNEXPECTED_ERROR: {str(e)}"

    def _process_base64_image(self, base64_str: str) -> Image.Image:
        """Process base64 image with error handling"""
        try:
            if "base64," in base64_str:
                base64_str = base64_str.split("base64,")[1]
            
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data))
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Optimize for A10G
            max_size = 896
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                print(f"📸 Resized image to {image.size} for A10G")
            
            return image
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {e}")

    @modal.method()
    def generate_response(
        self, 
        messages: list, 
        max_tokens: int = 1536,
        temperature: float = 0.1,
        **kwargs
    ) -> dict:
        """Bulletproof response generation"""
        
        # Check initialization
        if hasattr(self, 'init_error') and self.init_error:
            return {
                "generated_response": f"Service initialization failed: {self.init_error}",
                "error": "service_not_initialized",
                "init_error": self.init_error,
                "gpu": "A10G"
            }
        
        if not hasattr(self, 'processor') or self.processor is None:
            return {
                "generated_response": "Error: Processor not available",
                "error": "processor_not_loaded",
                "gpu": "A10G"
            }
        
        if not hasattr(self, 'model') or self.model is None:
            return {
                "generated_response": "Error: Model not available",
                "error": "model_not_loaded",
                "gpu": "A10G"
            }
        
        try:
            text_prompt = ""
            image = None
            
            # Extract text and image
            for msg in messages:
                if msg["role"] == "user":
                    text_prompt = msg["content"]
                    if "base64_image" in msg:
                        image = self._process_base64_image(msg["base64_image"])
                        break
            
            if not text_prompt:
                return {
                    "generated_response": "Error: No user message found",
                    "error": "no_user_message",
                    "gpu": "A10G"
                }
            
            # Prepare inputs
            if image is not None:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": text_prompt}
                        ]
                    }
                ]
                
                prompt = self.processor.apply_chat_template(
                    conversation, 
                    add_generation_prompt=True
                )
                
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                    max_length=1024
                ).to(self.model.device)
                
                print(f"📸 Processing image + text on A10G")
                
            else:
                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": text_prompt}
                        ]
                    }
                ]
                
                prompt = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True
                )
                
                inputs = self.processor(
                    text=prompt,
                    return_tensors="pt",
                    max_length=1024
                ).to(self.model.device)
                
                print(f"📝 Processing text-only on A10G")
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 1536),
                    temperature=temperature,
                    do_sample=temperature > 0.0,
                    top_p=0.9 if temperature > 0.0 else None,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.processor.decode(response_tokens, skip_special_tokens=True)
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"✅ Generated response on A10G: {len(response)} characters")
            
            return {
                "generated_response": response.strip(),
                "model": "fireworks-ai/FireLLaVA-13b",
                "gpu": "A10G",
                "has_image": image is not None,
                "input_tokens": input_length,
                "output_tokens": len(response_tokens),
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ A10G generation error: {e}")
            return {
                "generated_response": f"Error on A10G: {str(e)}",
                "error": str(e),
                "gpu": "A10G"
            }

    @modal.method()
    def health_check(self) -> dict:
        """Bulletproof health check"""
        try:
            if hasattr(self, 'init_error') and self.init_error:
                return {
                    "status": "unhealthy",
                    "error": self.init_error,
                    "gpu": "A10G",
                    "suggestion": "Check HF_TOKEN secret and redeploy"
                }
            
            if not hasattr(self, 'processor') or self.processor is None:
                return {
                    "status": "unhealthy",
                    "error": "processor_not_available",
                    "gpu": "A10G"
                }
            
            if not hasattr(self, 'model') or self.model is None:
                return {
                    "status": "unhealthy",
                    "error": "model_not_available",
                    "gpu": "A10G"
                }
            
            # Test generation
            test_messages = [
                {"role": "user", "content": "Say 'Bulletproof FireLLaVA is healthy'"}
            ]
            
            result = self.generate_response(
                test_messages,
                max_tokens=50,
                temperature=0.0
            )
            
            if "error" in result:
                return {
                    "status": "unhealthy",
                    "error": result["error"],
                    "gpu": "A10G"
                }
            
            return {
                "status": "healthy",
                "model": "fireworks-ai/FireLLaVA-13b",
                "gpu": "A10G",
                "test_response": result.get("generated_response", "")[:100],
                "features": ["bulletproof_init", "robust_error_handling"]
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "gpu": "A10G",
                "exception_type": type(e).__name__
            }

# FastAPI for bulletproof FireLLaVA
@app.function(image=image, timeout=300)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    from typing import List, Optional
    import time
    
    api = FastAPI(
        title="Bulletproof FireLLaVA-13B A10G API",
        description="Bulletproof FireLLaVA-13B optimized for A10G GPU with comprehensive error handling",
        version="3.0.0"
    )
    
    class Message(BaseModel):
        role: str = Field(..., description="Message role")
        content: str = Field(..., description="Message content")
        base64_image: Optional[str] = Field(None, description="Base64 encoded image")
    
    class FireLLaVARequest(BaseModel):
        messages: List[Message] = Field(..., description="Conversation messages")
        max_tokens: int = Field(1536, ge=1, le=2048, description="Max tokens")
        temperature: float = Field(0.1, ge=0.0, le=2.0, description="Temperature")
    
    @api.post("/firellava")
    async def firellava_endpoint(request: FireLLaVARequest):
        """Bulletproof FireLLaVA endpoint"""
        start_time = time.time()
        
        try:
            messages = [
                {
                    "role": msg.role, 
                    "content": msg.content,
                    **({"base64_image": msg.base64_image} if msg.base64_image else {})
                }
                for msg in request.messages
            ]
            
            result = FireLLaVA().generate_response.remote(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            processing_time = time.time() - start_time
            
            if "error" in result:
                return JSONResponse(
                    status_code=500,
                    content={
                        "detail": f"Error: {result['error']}",
                        "debug_info": result,
                        "gpu": "A10G"
                    }
                )
            
            return {
                "generated_response": result["generated_response"],
                "model": result["model"],
                "gpu": result["gpu"],
                "has_image": result.get("has_image", False),
                "processing_time": processing_time,
                "status": result.get("status", "success")
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return JSONResponse(
                status_code=500,
                content={
                    "detail": f"Unexpected error: {str(e)}",
                    "processing_time": processing_time,
                    "gpu": "A10G"
                }
            )
    
    @api.get("/health")
    async def health_endpoint():
        """Bulletproof health check"""
        try:
            result = FireLLaVA().health_check.remote()
            status_code = 200 if result["status"] == "healthy" else 503
            return JSONResponse(status_code=status_code, content=result)
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": str(e), "gpu": "A10G"}
            )
    
    @api.get("/")
    async def root():
        """Service information"""
        return {
            "service": "Bulletproof FireLLaVA-13B A10G API",
            "model": "fireworks-ai/FireLLaVA-13b",
            "gpu": "A10G (24GB)",
            "version": "3.0.0",
            "features": [
                "bulletproof_initialization",
                "fallback_auth_methods",
                "robust_error_handling",
                "image_processing_optimization"
            ],
            "endpoints": {
                "firellava": "/firellava",
                "health": "/health",
                "docs": "/docs"
            }
        }
    
    return api

if __name__ == "__main__":
    print("Deploy bulletproof A10G FireLLaVA with: modal deploy modal_service.py")
