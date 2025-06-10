import math
import os  # Add this import
import asyncio
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union

import tiktoken
import asyncio
from huggingface_hub import InferenceClient
import httpx
from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat.chat_completion import ChatCompletion  # Add this import
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.bedrock import BedrockClient
from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded  # Make sure this import exists
from app.logger import logger  # Assuming a logger is set up in your app
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)
load_dotenv()

REASONING_MODELS = ["o1", "o3-mini"]
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


class TokenCounter:
    # Token constants
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    # Image processing constants
    MAX_SIZE = 2048
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768
    TILE_SIZE = 512

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        """Calculate tokens for a text string"""
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        """
        Calculate tokens for an image based on detail level and dimensions

        For "low" detail: fixed 85 tokens
        For "high" detail:
        1. Scale to fit in 2048x2048 square
        2. Scale shortest side to 768px
        3. Count 512px tiles (170 tokens each)
        4. Add 85 tokens
        """
        detail = image_item.get("detail", "medium")

        # For low detail, always return fixed token count
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # For medium detail (default in OpenAI), use high detail calculation
        # OpenAI doesn't specify a separate calculation for medium

        # For high detail, calculate based on dimensions if available
        if detail == "high" or detail == "medium":
            # If dimensions are provided in the image_item
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        # Default values when dimensions aren't available or detail level is unknown
        if detail == "high":
            # Default to a 1024x1024 image calculation for high detail
            return self._calculate_high_detail_tokens(1024, 1024)  # 765 tokens
        elif detail == "medium":
            # Default to a medium-sized image for medium detail
            return 1024  # This matches the original default
        else:
            # For unknown detail levels, use medium as default
            return 1024

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """Calculate tokens for high detail images based on dimensions"""
        # Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # Step 3: Count number of 512px tiles
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # Step 4: Calculate final token count
        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """Calculate tokens for message content"""
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """Calculate tokens for tool calls"""
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """Calculate the total number of tokens in a message list"""
        total_tokens = self.FORMAT_TOKENS  # Base format tokens

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # Base tokens per message

            # Add role tokens
            tokens += self.count_text(message.get("role", ""))

            # Add content tokens
            if "content" in message:
                tokens += self.count_content(message["content"])

            # Add tool calls tokens
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # Add name and tool_call_id tokens
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens

# class HFInferenceLLM:
#     """Async wrapper over HF Inference chat endpoint using fireworks-ai provider."""
#     def __init__(self, model: str, api_key: str):
#         # Use fireworks-ai provider as shown in your working example
#         self.client = InferenceClient(
#             provider="fireworks-ai",
#             api_key=api_key  # This should be HF_TOKEN
#         )
#         self.model = model

#     async def chat_completion(self, messages: List[dict], **kwargs) -> str:
#         # Format messages for HF API
#         hf_msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        
#         result = await asyncio.to_thread(
#             lambda: self.client.chat.completions.create(
#                 model=self.model, 
#                 messages=hf_msgs, 
#                 **kwargs
#             )
#         )
#         return result.choices[0].message.content


# class QwenModalLLM:
#     """Qwen2.5-72B via Modal deployment - replaces HFInferenceLLM"""
#     def __init__(self, model: str, base_url: str):
#         if base_url and not base_url.startswith(('http://', 'https://')):
#             base_url = f"https://{base_url}"
#         self.endpoint = base_url.rstrip("/") + "/chat" if base_url else ""
#         self.model = model

#     async def chat_completion(self, messages: List[dict], **kwargs) -> str:
#         if not self.endpoint:
#             raise ValueError("Modal Qwen endpoint URL is not configured")
            
#         headers = {
#             "Content-Type": "application/json"
#         }
        
#         # Add Modal token for auth if available
#         if os.getenv('MODAL_TOKEN'):
#             headers["Authorization"] = f"Bearer {os.getenv('MODAL_TOKEN')}"
        
#         payload = {
#             "messages": messages,
#             "max_tokens": kwargs.get("max_tokens", 4096),
#             "temperature": kwargs.get("temperature", 0.1),
#             "tools": kwargs.get("tools", None)  # Pass tools if provided
#         }
        
#         async with httpx.AsyncClient(timeout=180.0) as client:
#             resp = await client.post(self.endpoint, json=payload, headers=headers)
#             resp.raise_for_status()
#             result = resp.json()
#             return result["response"]

class QwenModalLLM:
    """Qwen2.5-72B via Modal deployment"""
    def __init__(self, model: str, base_url: str):
        if not base_url:
            raise ValueError("Modal Qwen endpoint URL is required")
        
        if base_url and not base_url.startswith(('http://', 'https://')):
            base_url = f"https://{base_url}"
        self.endpoint = base_url.rstrip("/") + "/chat"
        self.model = model
        
        # Use delayed import to avoid circular imports
        try:
            from app.logger import logger
            logger.info(f"🧠 Qwen Modal endpoint configured: {self.endpoint}")
        except ImportError:
            print(f"🧠 Qwen Modal endpoint configured: {self.endpoint}")

    async def chat_completion(self, messages: List[dict], **kwargs) -> str:
        """Chat completion using Modal Qwen service"""
        try:
            from app.logger import logger
        except ImportError:
            import logging
            logger = logging.getLogger(__name__)
        
        if not self.endpoint:
            raise ValueError("Modal Qwen endpoint URL is not configured")
            
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add Modal token for auth if available
        modal_token = os.getenv('MODAL_TOKEN')
        if modal_token:
            headers["Authorization"] = f"Bearer {modal_token}"
        
        payload = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.1),
        }
        
        # Add tools if provided
        if kwargs.get("tools"):
            payload["tools"] = kwargs["tools"]
        
        logger.debug(f"🔄 Sending request to Qwen Modal: {len(messages)} messages")
        
        try:
            async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
                resp = await client.post(self.endpoint, json=payload, headers=headers)
                
                if resp.status_code == 503:
                    logger.warning("⏳ Qwen service is still initializing, retrying...")
                    await asyncio.sleep(5)
                    resp = await client.post(self.endpoint, json=payload, headers=headers)
                
                resp.raise_for_status()
                result = resp.json()
                
                if "response" not in result:
                    logger.error(f"❌ Unexpected response format: {result}")
                    raise ValueError("Invalid response format from Modal Qwen service")
                
                logger.debug(f"✅ Received response from Qwen: {len(result['response'])} characters")
                return result["response"]
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                logger.error("❌ Qwen service unavailable (503). Service may still be initializing.")
                logger.error("   Wait 2-3 minutes for models to load into GPU memory.")
            else:
                logger.error(f"❌ HTTP error from Qwen service: {e.response.status_code}")
                try:
                    logger.error(f"   Response: {e.response.text}")
                except:
                    pass
            raise
        except Exception as e:
            logger.error(f"❌ Error communicating with Qwen Modal service: {e}")
            raise


# class ModalVisionLLM:
#     """Calls your Modal-hosted FireLLaVA service."""
#     def __init__(self, base_url: str):
#         # Ensure base_url has proper protocol
#         if base_url and not base_url.startswith(('http://', 'https://')):
#             base_url = f"https://{base_url}"
#         self.endpoint = base_url.rstrip("/") + "/firellava" if base_url else ""

#     async def chat_completion(self, messages: List[dict], **kwargs) -> str:
#         if not self.endpoint:
#             raise ValueError("Modal endpoint URL is not configured")
            
#         headers = {"Authorization": f"Bearer {os.getenv('MODAL_TOKEN','')}"}
#         payload = {"messages": messages, **kwargs}
#         async with httpx.AsyncClient() as client:
#             resp = await client.post(self.endpoint, json=payload, headers=headers)
#             resp.raise_for_status()
#             return resp.json()["generated_response"]


class ModalVisionLLM:
    """Calls your Modal-hosted FireLLaVA service."""
    def __init__(self, base_url: str):
        if not base_url:
            raise ValueError("Modal FireLLaVA endpoint URL is required")
        
        # Ensure base_url has proper protocol
        if base_url and not base_url.startswith(('http://', 'https://')):
            base_url = f"https://{base_url}"
        self.endpoint = base_url.rstrip("/") + "/firellava"
        
        # Use delayed import to avoid circular imports
        try:
            from app.logger import logger
            logger.info(f"🔥 FireLLaVA Modal endpoint configured: {self.endpoint}")
        except ImportError:
            print(f"🔥 FireLLaVA Modal endpoint configured: {self.endpoint}")

    async def chat_completion(self, messages: List[dict], **kwargs) -> str:
        """Vision completion using Modal FireLLaVA service"""
        try:
            from app.logger import logger
        except ImportError:
            import logging
            logger = logging.getLogger(__name__)
        
        if not self.endpoint:
            raise ValueError("Modal FireLLaVA endpoint URL is not configured")
            
        headers = {"Content-Type": "application/json"}
        
        # Add Modal token for auth if available
        modal_token = os.getenv('MODAL_TOKEN')
        if modal_token:
            headers["Authorization"] = f"Bearer {modal_token}"
        
        payload = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.1)
        }
        
        logger.debug(f"🔄 Sending request to FireLLaVA Modal: {len(messages)} messages")
        
        try:
            async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
                resp = await client.post(self.endpoint, json=payload, headers=headers)
                
                if resp.status_code == 503:
                    logger.warning("⏳ FireLLaVA service is still initializing, retrying...")
                    await asyncio.sleep(5)
                    resp = await client.post(self.endpoint, json=payload, headers=headers)
                
                resp.raise_for_status()
                result = resp.json()
                
                if "generated_response" not in result:
                    logger.error(f"❌ Unexpected response format: {result}")
                    raise ValueError("Invalid response format from Modal FireLLaVA service")
                
                logger.debug(f"✅ Received response from FireLLaVA: {len(result['generated_response'])} characters")
                return result["generated_response"]
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                logger.error("❌ FireLLaVA service unavailable (503). Service may still be initializing.")
                logger.error("   Wait 2-3 minutes for models to load into GPU memory.")
            else:
                logger.error(f"❌ HTTP error from FireLLaVA service: {e.response.status_code}")
                try:
                    logger.error(f"   Response: {e.response.text}")
                except:
                    pass
            raise
        except Exception as e:
            logger.error(f"❌ Error communicating with FireLLaVA Modal service: {e}")
            raise


class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls,
        config_name: str = "default",
        llm_config: Optional[LLMSettings] = None,
        model_name: Optional[str] = None,  # Accept model_name here
        **kwargs  # Accept other kwargs
    ):
        # Create a unique key for the instance based on config_name AND model_name if provided
        instance_key = f"{config_name}_{model_name}" if model_name else config_name

        if instance_key not in cls._instances:
            instance = super().__new__(cls)
            # Pass all arguments including model_name to __init__
            instance.__init__(config_name=config_name, llm_config=llm_config, model_name=model_name, **kwargs)
            cls._instances[instance_key] = instance
        return cls._instances[instance_key]

    def __init__(
        self,
        config_name: str = "default",
        llm_config: Optional[LLMSettings] = None,
        model_name: Optional[str] = None,  # Add model_name override
        **kwargs  # Allow extra kwargs
    ):
        # Prevent re-initialization if already done by __new__
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Determine which config to use
        base_llm_config = llm_config or config.llm

        # Handle config name (e.g., "default" or "vision")
        if config_name in base_llm_config:
            final_llm_config = base_llm_config[config_name]
        else:
            final_llm_config = base_llm_config.get("default", LLMSettings())
            logger.warning(f"LLM config '{config_name}' not found, using default.")

        # Apply explicit model_name override if provided
        self.model = model_name or final_llm_config.model

        # Apply other settings from the resolved config
        self.max_tokens = final_llm_config.max_tokens
        self.temperature = final_llm_config.temperature
        self.api_type = final_llm_config.api_type
        self.api_key = final_llm_config.api_key
        self.api_version = final_llm_config.api_version
        
        # Handle base_url with proper protocol
        base_url = final_llm_config.base_url
        if base_url and not base_url.startswith(('http://', 'https://')):
            # Only add https:// if it's not empty and doesn't already have a protocol
            if base_url.strip():
                base_url = f"https://{base_url}"
        self.base_url = base_url

        # Add token counting related attributes
        self.total_input_tokens = 0
        self.total_completion_tokens = 0
        self.max_input_tokens = getattr(final_llm_config, 'max_input_tokens', None)

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Initialize the appropriate client based on api_type
        if self.api_type == "azure":
            self.client = AsyncAzureOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        elif self.api_type == "aws":
            self.client = BedrockClient()
#        elif self.api_type == "huggingface":
#            # Text via HF Inference
#            self.client = HFInferenceLLM(
#                model=self.model,
#                api_key=os.getenv("HF_TOKEN")
#            )
        elif self.api_type == "qwen-modal":
            # Qwen2.5-72B via Modal deployment
            self.client = QwenModalLLM(
                model=self.model,
                base_url=self.base_url or os.getenv("MODAL_QWEN_ENDPOINT")
            )
        elif self.api_type == "modal":
            # Vision via Modal service
            self.client = ModalVisionLLM(base_url=self.base_url or os.getenv("MODAL_FIRELLAVA_ENDPOINT")) 
        else:  # Default to OpenAI/Fireworks style
            if self.base_url and ("modal" in self.base_url.lower() or self.api_type == "qwen-modal"):
                self.client = QwenModalLLM(model=self.model, base_url=self.base_url)
            else:            
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        self.token_counter = TokenCounter(self.tokenizer)
        self._initialized = True  # Set the flag after successful initialization

    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """Update token counts"""
        # Only track tokens if max_input_tokens is set
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """Check if token limits are exceeded"""
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        # If max_input_tokens is not set, always return True
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        """Generate error message for token limit exceeded"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"

        return "Token limit exceeded"

    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects
            supports_images: Flag indicating if the target model supports image inputs

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            # Convert Message objects to dictionaries
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # If message is a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # Process base64 images if present and model supports images
                if supports_images and message.get("base64_image"):
                    # Initialize or convert content to appropriate format
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # Convert string items to proper text objects
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # Add the image to content
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                            },
                        }
                    )

                    # Remove the base64_image field
                    del message["base64_image"]
                # If model doesn't support images but message has base64_image, handle gracefully
                elif not supports_images and message.get("base64_image"):
                    # Just remove the base64_image field and keep the text content
                    del message["base64_image"]

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # else: do not include the message
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
        stream_callback: Optional[callable] = None,
    ) -> str:
        """
        Send a chat completion request.

        Args:
            messages: List of messages
            system_msgs: Optional system messages
            stream: Whether to stream the response
            temperature: Optional temperature override
            stream_callback: Optional callback for processing streaming chunks

        Returns:
            The chat completion text
        """
        try:
            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS

            # Format system and user messages with image support check
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            params = {
                "model": self.model,
                "messages": messages,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            if not stream:
                # Non-streaming request
                #if self.api_type in ("huggingface", "modal"):
                if self.api_type in ("qwen-modal", "modal"):
                    # both wrappers expose chat_completion()
                    return await self.client.chat_completion(
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )                
                response = await self.client.chat.completions.create(
                    **params, stream=False
                )

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                # Update token counts
                self.update_token_count(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )

                return response.choices[0].message.content

            # Streaming request, For streaming, update estimated token count before making the request
            self.update_token_count(input_tokens)
            #if self.api_type in ("huggingface", "modal"):
            if self.api_type in ("qwen-modal", "modal"):
                # both wrappers expose chat_completion()
                return await self.client.chat_completion(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
            response = await self.client.chat.completions.create(**params, stream=True)

            collected_messages = []
            completion_text = ""
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                completion_text += chunk_message

                # Use the callback if provided, otherwise print to console
                if stream_callback and callable(stream_callback):
                    await stream_callback(chunk_message)
                else:
                    print(chunk_message, end="", flush=True)

            if not stream_callback:
                print()  # Newline after streaming only if not using callback
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            # estimate completion tokens for streaming response
            completion_tokens = self.count_tokens(completion_text)
            logger.info(
                f"Estimated completion tokens for streaming response: {completion_tokens}"
            )
            self.total_completion_tokens += completion_tokens

            return full_response

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError:
            logger.exception(f"Validation error")
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API error")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception:
            logger.exception(f"Unexpected error in ask")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt with images to the LLM and get the response.

        Args:
            messages: List of conversation messages
            images: List of image URLs or image data dictionaries
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # For ask_with_images, we always set supports_images to True because
            # this method should only be called with models that support images
            if self.model not in MULTIMODAL_MODELS:
                raise ValueError(
                    f"Model {self.model} does not support images. Use a model from {MULTIMODAL_MODELS}"
                )

            # Format messages with image support
            formatted_messages = self.format_messages(messages, supports_images=True)

            # Ensure the last message is from the user to attach images
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )

            # Process the last user message to include images
            last_message = formatted_messages[-1]

            # Convert content to multimodal format if needed
            content = last_message["content"]
            multimodal_content = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content
                if isinstance(content, list)
                else []
            )

            # Add images to content
            for image in images:
                if isinstance(image, str):
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                elif isinstance(image, dict) and "url" in image:
                    multimodal_content.append({"type": "image_url", "image_url": image})
                elif isinstance(image, dict) and "image_url" in image:
                    multimodal_content.append(image)
                else:
                    raise ValueError(f"Unsupported image format: {image}")

            # Update the message with multimodal content
            last_message["content"] = multimodal_content

            # Add system messages if provided
            if system_msgs:
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages

            # Calculate tokens and check limits
            input_tokens = self.count_message_tokens(all_messages)
            if not self.check_token_limit(input_tokens):
                raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

            # Set up API parameters
            params = {
                "model": self.model,
                "messages": all_messages,
                "stream": stream,
            }

            # Add model-specific parameters
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # Handle non-streaming request
            if not stream:
                #if self.api_type in ("huggingface", "modal"):
                if self.api_type in ("qwen-modal", "modal"):
                    # delegate to our wrapper for a simple text return
                    return await self.client.chat_completion(
                        messages=all_messages,
                        images=images,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )                
                response = await self.client.chat.completions.create(**params)

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                self.update_token_count(response.usage.prompt_tokens)
                return response.choices[0].message.content

            # Handle streaming request
            self.update_token_count(input_tokens)
            #if self.api_type in ("huggingface", "modal"):
            if self.api_type in ("qwen-modal", "modal"):
                # HF/Modal wrappers don't stream; just get the full text
                return await self.client.chat_completion(
                    messages=all_messages,
                    images=images,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            response = await self.client.chat.completions.create(**params)

            collected_messages = []
            completion_text = ""
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                completion_text += chunk_message

                # Use the callback if provided, otherwise print to console
                if stream_callback and callable(stream_callback):
                    await stream_callback(chunk_message)
                else:
                    print(chunk_message, end="", flush=True)

            if not stream_callback:
                print()  # Newline after streaming only if not using callback
            full_response = "".join(collected_messages).strip()

            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            return full_response

        except TokenLimitExceeded:
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_with_images: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_with_images: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ChatCompletionMessage | None:
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # If there are tools, calculate token count for tool descriptions
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))

            input_tokens += tools_tokens

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            # Set up the completion request
            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )
            # ─── HF/Modal branch ────────────────────────────────────────────────────
            #if self.api_type in ("huggingface", "modal"):
            if self.api_type in ("qwen-modal", "modal"):
                text = await self.client.chat_completion(
                    messages=messages,
                    tools=tools,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return ChatCompletionMessage(role="assistant", content=text)
            
            response: ChatCompletion = await self.client.chat.completions.create(
                **params, stream=False
            )

            # Check if response is valid
            if not response.choices or not response.choices[0].message:
                print(response)
                # raise ValueError("Invalid or empty response from LLM")
                return None

            # Update token counts
            self.update_token_count(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )

            return response.choices[0].message

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise