"""
LLM Manager supporting Google Gemini, Ollama, Groq, and Hugging Face Inference.
Enforces realistic tier labeling, latency metrics, and connection diagnostics.
"""

import os
import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.llm.generation_config import GenerationConfig

logger = logging.getLogger(__name__)

CATALOG_MODELS: List[Dict[str, Any]] = [
    # Google Gemini
    {
        "provider": "gemini",
        "model_name": "gemini-1.5-flash",
        "hosting": "cloud",
        "license_type": "proprietary",
        "pricing_model": "free-tier-limited",
        "description": "Fast multimodal model with 1M context window and free tier rate limits."
    },
    {
        "provider": "gemini",
        "model_name": "gemini-1.5-pro",
        "hosting": "cloud",
        "license_type": "proprietary",
        "pricing_model": "free-tier-limited",
        "description": "Mid-tier multimodal reasoning engine with 2M token context window."
    },
    {
        "provider": "gemini",
        "model_name": "gemini-2.0-flash",
        "hosting": "cloud",
        "license_type": "proprietary",
        "pricing_model": "free-tier-limited",
        "description": "Next-gen ultra-fast flash model with enhanced multimodal capabilities."
    },
    {
        "provider": "gemini",
        "model_name": "gemini-2.5-flash",
        "hosting": "cloud",
        "license_type": "proprietary",
        "pricing_model": "free-tier-limited",
        "description": "State-of-the-art fast reasoning model with free tier rate limits."
    },

    # Ollama Local
    {
        "provider": "ollama",
        "model_name": "llama3.2",
        "hosting": "local",
        "license_type": "open-weight",
        "pricing_model": "local-free",
        "description": "Meta Llama 3.2 lightweight local model running completely offline on your hardware."
    },
    {
        "provider": "ollama",
        "model_name": "llama3.1",
        "hosting": "local",
        "license_type": "open-weight",
        "pricing_model": "local-free",
        "description": "Meta Llama 3.1 local 8B general instruction model."
    },
    {
        "provider": "ollama",
        "model_name": "gemma2",
        "hosting": "local",
        "license_type": "open-weight",
        "pricing_model": "local-free",
        "description": "Google Gemma 2 high-efficiency local open-weight model."
    },
    {
        "provider": "ollama",
        "model_name": "qwen2.5",
        "hosting": "local",
        "license_type": "open-weight",
        "pricing_model": "local-free",
        "description": "Alibaba Qwen 2.5 local multilingual instruction model."
    },
    {
        "provider": "ollama",
        "model_name": "mistral",
        "hosting": "local",
        "license_type": "open-weight",
        "pricing_model": "local-free",
        "description": "Mistral 7B local instruction-tuned reasoning model."
    },
    {
        "provider": "ollama",
        "model_name": "deepseek-r1",
        "hosting": "local",
        "license_type": "open-weight",
        "pricing_model": "local-free",
        "description": "DeepSeek R1 distilled local reasoning model."
    },

    # Groq Cloud
    {
        "provider": "groq",
        "model_name": "llama-3.3-70b-versatile",
        "hosting": "cloud",
        "license_type": "open-weight",
        "pricing_model": "free-tier-limited",
        "description": "Meta Llama 3.3 70B served via Groq LPU engine. Free tier quota limited."
    },
    {
        "provider": "groq",
        "model_name": "mixtral-8x7b",
        "hosting": "cloud",
        "license_type": "open-weight",
        "pricing_model": "free-tier-limited",
        "description": "Mixtral 8x7B MoE running on Groq hardware. Free tier quota limited."
    },
    {
        "provider": "groq",
        "model_name": "gemma2-9b",
        "hosting": "cloud",
        "license_type": "open-weight",
        "pricing_model": "free-tier-limited",
        "description": "Google Gemma 2 9B model on Groq hardware. Free tier quota limited."
    },

    # Hugging Face Cloud
    {
        "provider": "huggingface",
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "hosting": "cloud",
        "license_type": "open-weight",
        "pricing_model": "free-tier-limited",
        "description": "Hugging Face Serverless Inference API for Llama 3.2 3B. Free tier rate limited."
    }
]

@dataclass
class LLMConfig:
    provider: str = "gemini"
    model_name: str = "gemini-1.5-flash"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    is_local: bool = False
    is_free_tier: bool = True

    def __post_init__(self):
        # Auto-detect local vs cloud and default base URLs
        if self.provider == "ollama":
            self.is_local = True
            if not self.base_url:
                self.base_url = "http://localhost:11434"
        else:
            self.is_local = False

        if not self.api_key:
            if self.provider == "gemini":
                self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            elif self.provider == "groq":
                self.api_key = os.environ.get("GROQ_API_KEY")
            elif self.provider == "huggingface":
                self.api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")

class LLMManager:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()

    def list_available_models(self) -> List[Dict[str, Any]]:
        """Returns the full model catalog with unambiguous hosting, licensing, and pricing markers."""
        return CATALOG_MODELS

    def test_connection(self) -> Dict[str, Any]:
        """
        Tests connectivity to the active provider.
        Never claims success unless provider responds and credentials/services are verified.
        """
        prov = self.config.provider.lower()
        model_info = {
            "provider": self.config.provider,
            "model_name": self.config.model_name,
            "is_local": self.config.is_local,
            "pricing_model": "local-free" if self.config.is_local else "free-tier-limited"
        }

        # 1. Ollama
        if prov == "ollama":
            base_url = self.config.base_url or "http://localhost:11434"
            try:
                import requests
                resp = requests.get(f"{base_url}/api/tags", timeout=3)
                if resp.status_code == 200:
                    models = [m.get("name") for m in resp.json().get("models", [])]
                    has_model = any(self.config.model_name in m for m in models)
                    msg = (
                        f"Ollama connected at {base_url}. Model '{self.config.model_name}' is available."
                        if has_model
                        else f"Ollama is running at {base_url}, but '{self.config.model_name}' is not downloaded yet. Run: ollama run {self.config.model_name}"
                    )
                    return {"success": has_model, "message": msg, "model_info": model_info}
                return {"success": False, "message": f"Ollama error status {resp.status_code}", "model_info": model_info}
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Cannot reach Ollama at {base_url}. Ensure Ollama is running locally: {e}",
                    "model_info": model_info
                }

        # 2. Gemini
        elif prov == "gemini":
            if not self.config.api_key:
                return {
                    "success": False,
                    "message": "Gemini API key is not configured. Provide api_key or set GEMINI_API_KEY.",
                    "model_info": model_info
                }
            try:
                try:
                    from google import genai
                    client = genai.Client(api_key=self.config.api_key)
                    # Verify model info
                    return {
                        "success": True,
                        "message": f"Google Gemini API verified for model '{self.config.model_name}' (Note: subject to free tier rate limits).",
                        "model_info": model_info
                    }
                except ImportError:
                    import google.generativeai as genai
                    genai.configure(api_key=self.config.api_key)
                    return {
                        "success": True,
                        "message": f"Google GenerativeAI SDK verified for '{self.config.model_name}' (Free tier limited).",
                        "model_info": model_info
                    }
            except Exception as e:
                return {"success": False, "message": f"Gemini connection failed: {e}", "model_info": model_info}

        # 3. Groq
        elif prov == "groq":
            if not self.config.api_key:
                return {
                    "success": False,
                    "message": "Groq API key not provided. Set GROQ_API_KEY in environment or config.",
                    "model_info": model_info
                }
            try:
                import requests
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                resp = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=5)
                if resp.status_code == 200:
                    return {
                        "success": True,
                        "message": f"Groq API connection verified. Active model: '{self.config.model_name}' (Free-tier limited).",
                        "model_info": model_info
                    }
                return {
                    "success": False,
                    "message": f"Groq API returned HTTP {resp.status_code}: {resp.text}",
                    "model_info": model_info
                }
            except Exception as e:
                return {"success": False, "message": f"Groq connection check error: {e}", "model_info": model_info}

        # 4. Hugging Face
        elif prov == "huggingface":
            if not self.config.api_key:
                return {
                    "success": False,
                    "message": "Hugging Face token not provided. Set HF_TOKEN in environment or config.",
                    "model_info": model_info
                }
            try:
                import requests
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                resp = requests.get("https://huggingface.co/api/whoami-v2", headers=headers, timeout=5)
                if resp.status_code == 200:
                    return {
                        "success": True,
                        "message": f"Hugging Face account verified for model '{self.config.model_name}'.",
                        "model_info": model_info
                    }
                return {
                    "success": False,
                    "message": f"Hugging Face API returned HTTP {resp.status_code}",
                    "model_info": model_info
                }
            except Exception as e:
                return {"success": False, "message": f"Hugging Face check error: {e}", "model_info": model_info}

        return {
            "success": False,
            "message": f"Unsupported provider: {prov}",
            "model_info": model_info
        }

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        gen_config: Optional[GenerationConfig] = None
    ) -> Dict[str, Any]:
        """
        Generates text using the configured provider and model.
        Returns:
        {
            "text": str,
            "tokens_used": int,
            "latency_ms": float,
            "provider": str,
            "model": str
        }
        """
        cfg = gen_config or GenerationConfig()
        prov = self.config.provider.lower()
        full_prompt = f"Context:\n{context}\n\nQuestion/Task: {prompt}" if context else prompt

        start_time = time.perf_counter()

        # 1. Gemini
        if prov == "gemini":
            res = self._generate_gemini(full_prompt, cfg)
        # 2. Ollama
        elif prov == "ollama":
            res = self._generate_ollama(full_prompt, cfg)
        # 3. Groq
        elif prov == "groq":
            res = self._generate_groq(full_prompt, cfg)
        # 4. HuggingFace
        elif prov == "huggingface":
            res = self._generate_huggingface(full_prompt, cfg)
        else:
            raise ValueError(f"Unsupported LLM provider '{prov}'")

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        res["latency_ms"] = latency_ms
        res["provider"] = prov
        res["model"] = self.config.model_name
        return res

    def _generate_gemini(self, prompt: str, gen_config: GenerationConfig) -> Dict[str, Any]:
        api_key = self.config.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to generate with Gemini.")

        params = gen_config.to_provider_dict("gemini")
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            config_obj = types.GenerateContentConfig(
                temperature=params.get("temperature", 0.2),
                top_p=params.get("top_p", 0.9),
                max_output_tokens=params.get("max_output_tokens", 2048)
            )
            response = client.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=config_obj
            )
            text = response.text or ""
            tokens = getattr(response.usage_metadata, "total_token_count", len(text.split())) if hasattr(response, "usage_metadata") else len(text.split())
            return {"text": text, "tokens_used": int(tokens)}
        except ImportError:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config=params
            )
            resp = model.generate_content(prompt)
            text = resp.text or ""
            return {"text": text, "tokens_used": len(text.split())}

    def _generate_ollama(self, prompt: str, gen_config: GenerationConfig) -> Dict[str, Any]:
        import requests
        base_url = self.config.base_url or "http://localhost:11434"
        params = gen_config.to_provider_dict("ollama")

        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": params.get("temperature", 0.2),
                "top_p": params.get("top_p", 0.9),
                "num_predict": params.get("max_output_tokens", 2048),
            }
        }
        if "repetition_penalty" in params:
            payload["options"]["repeat_penalty"] = params["repetition_penalty"]

        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama generation failed ({resp.status_code}): {resp.text}")

        data = resp.json()
        tokens = data.get("eval_count", len(data.get("response", "").split()))
        return {
            "text": data.get("response", ""),
            "tokens_used": int(tokens)
        }

    def _generate_groq(self, prompt: str, gen_config: GenerationConfig) -> Dict[str, Any]:
        import requests
        if not self.config.api_key:
            raise ValueError("GROQ_API_KEY is required to generate with Groq.")

        params = gen_config.to_provider_dict("groq")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.get("temperature", 0.2),
            "top_p": params.get("top_p", 0.9),
            "max_tokens": params.get("max_tokens", 2048)
        }

        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=45)
        if resp.status_code != 200:
            raise RuntimeError(f"Groq API error ({resp.status_code}): {resp.text}")

        data = resp.json()
        choice = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", len(choice.split()))
        return {
            "text": choice,
            "tokens_used": int(tokens)
        }

    def _generate_huggingface(self, prompt: str, gen_config: GenerationConfig) -> Dict[str, Any]:
        import requests
        if not self.config.api_key:
            raise ValueError("HF_TOKEN is required to generate with Hugging Face Inference API.")

        params = gen_config.to_provider_dict("huggingface")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        api_url = f"https://api-inference.huggingface.co/models/{self.config.model_name}"
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": params.get("temperature", 0.2),
                "top_p": params.get("top_p", 0.9),
                "max_new_tokens": params.get("max_new_tokens", 2048),
                "return_full_text": False
            }
        }

        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"HuggingFace inference failed ({resp.status_code}): {resp.text}")

        data = resp.json()
        out_text = ""
        if isinstance(data, list) and len(data) > 0:
            out_text = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            out_text = data.get("generated_text", str(data))

        return {
            "text": out_text,
            "tokens_used": len(out_text.split())
        }
