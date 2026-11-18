"""
Generation Configuration and Parameter Translation Matrix for Multi-Modal RAG LLMs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

PROVIDER_SUPPORT_MATRIX: Dict[str, List[str]] = {
    "gemini": [
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
        "stop_sequences",
        "seed",
        "presence_penalty",
        "frequency_penalty"
    ],
    "ollama": [
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
        "repetition_penalty",
        "stop_sequences",
        "seed",
        "min_p",
        "typical_p",
        "presence_penalty",
        "frequency_penalty"
    ],
    "groq": [
        "temperature",
        "top_p",
        "max_output_tokens",
        "stop_sequences",
        "seed",
        "presence_penalty",
        "frequency_penalty"
    ],
    "huggingface": [
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
        "repetition_penalty",
        "stop_sequences",
        "seed"
    ]
}

@dataclass
class GenerationConfig:
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: Optional[int] = None
    max_output_tokens: int = 2048
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop_sequences: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    min_p: Optional[float] = None
    typical_p: Optional[float] = None

    @classmethod
    def get_supported_params(cls, provider: str) -> List[str]:
        """Returns the list of parameter names supported by the specified provider."""
        return PROVIDER_SUPPORT_MATRIX.get(provider.lower(), list(PROVIDER_SUPPORT_MATRIX["gemini"]))

    def validate(self, provider: str) -> List[str]:
        """
        Validates values within their accepted ranges.
        Returns a list of validation error strings, or an empty list if valid.
        """
        errors = []
        if not (0.0 <= self.temperature <= 2.0):
            errors.append(f"temperature {self.temperature} out of bounds [0.0, 2.0]")
        if not (0.0 <= self.top_p <= 1.0):
            errors.append(f"top_p {self.top_p} out of bounds [0.0, 1.0]")
        if self.top_k is not None and not (1 <= self.top_k <= 100):
            errors.append(f"top_k {self.top_k} out of bounds [1, 100]")
        if not (1 <= self.max_output_tokens <= 32768):
            errors.append(f"max_output_tokens {self.max_output_tokens} out of bounds [1, 32768]")
        if not (-2.0 <= self.frequency_penalty <= 2.0):
            errors.append(f"frequency_penalty {self.frequency_penalty} out of bounds [-2.0, 2.0]")
        if not (-2.0 <= self.presence_penalty <= 2.0):
            errors.append(f"presence_penalty {self.presence_penalty} out of bounds [-2.0, 2.0]")
        if not (0.0 <= self.repetition_penalty <= 2.0):
            errors.append(f"repetition_penalty {self.repetition_penalty} out of bounds [0.0, 2.0]")
        if self.min_p is not None and not (0.0 <= self.min_p <= 1.0):
            errors.append(f"min_p {self.min_p} out of bounds [0.0, 1.0]")
        if self.typical_p is not None and not (0.0 <= self.typical_p <= 1.0):
            errors.append(f"typical_p {self.typical_p} out of bounds [0.0, 1.0]")
        return errors

    def to_provider_dict(self, provider: str) -> Dict[str, Any]:
        """
        Filters and translates configuration fields strictly to the parameters supported by the target provider.
        """
        supported = set(self.get_supported_params(provider))
        prov = provider.lower()
        res: Dict[str, Any] = {}

        if "temperature" in supported:
            res["temperature"] = float(self.temperature)
        if "top_p" in supported:
            res["top_p"] = float(self.top_p)
        if "top_k" in supported and self.top_k is not None:
            res["top_k"] = int(self.top_k)
        if "max_output_tokens" in supported:
            if prov == "groq":
                res["max_tokens"] = int(self.max_output_tokens)
            elif prov == "huggingface":
                res["max_new_tokens"] = int(self.max_output_tokens)
            else:
                res["max_output_tokens"] = int(self.max_output_tokens)
        if "frequency_penalty" in supported and self.frequency_penalty != 0.0:
            res["frequency_penalty"] = float(self.frequency_penalty)
        if "presence_penalty" in supported and self.presence_penalty != 0.0:
            res["presence_penalty"] = float(self.presence_penalty)
        if "repetition_penalty" in supported and self.repetition_penalty != 1.0:
            res["repetition_penalty"] = float(self.repetition_penalty)
        if "stop_sequences" in supported and self.stop_sequences:
            if prov == "groq":
                res["stop"] = self.stop_sequences
            else:
                res["stop_sequences"] = self.stop_sequences
        if "seed" in supported and self.seed is not None:
            res["seed"] = int(self.seed)
        if "min_p" in supported and self.min_p is not None:
            res["min_p"] = float(self.min_p)
        if "typical_p" in supported and self.typical_p is not None:
            res["typical_p"] = float(self.typical_p)

        return res
