"""
Phishing detector for inference
"""
import torch
from typing import Dict, List, Optional, Any
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PhishingDetector:
    """Production-ready phishing detector with reasoning"""

    def __init__(self, model, tokenizer, preprocessor, config):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.config = config

        # Put model in eval mode
        self.model.eval()

        # Resolve actual device from "auto"
        self.device = self._resolve_device(config.device)

    def analyze_email(self,
                     email_text: str,
                     return_reasoning: bool = True) -> Dict[str, Any]:
        """Analyze email for phishing with detailed output"""

        start_time = time.time()

        # Use structured JSON prompt like QwenPhishingModel
        messages = [
            {
                "role": "user",
                "content": f"""Analyze this email for phishing. Respond ONLY with this exact JSON format:

{{
  "classification": "PHISHING" or "LEGITIMATE",
  "confidence": 0.0-1.0,
  "reasoning": "Brief 2-3 sentence analysis",
  "risk_indicators": ["list", "of", "indicators", "found"],
  "recommended_action": "BLOCK" or "QUARANTINE" or "REVIEW" or "ALLOW"
}}

------Email Start------

{email_text}

------Email End------"""
            }
        ]

        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        # Generate response
        response = self._generate(formatted_prompt)

        # Parse JSON response (same as QwenPhishingModel)
        result = self._parse_json_response(response)

        # Add timing
        result['processing_time'] = time.time() - start_time

        if not return_reasoning:
            # Return minimal response
            return {
                'classification': result['classification'],
                'confidence': result['confidence'],
                'processing_time': result['processing_time']
            }

        return result

    def _generate(self, prompt: str) -> str:
        """Generate model response"""
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        response = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

        return response.strip()

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from model"""
        import json
        import re

        try:
            # Try to extract JSON from the response
            # First try to find JSON after </think> if thinking mode was used
            think_match = re.search(r'</think>\s*(\{.*\})', response, re.DOTALL)
            if think_match:
                json_str = think_match.group(1).strip()
            else:
                # Look for any JSON block in the response (fallback)
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group().strip()
                else:
                    raise ValueError("No JSON found in response")

            result = json.loads(json_str)

            # Calculate risk score based on classification and confidence
            if result["classification"] == "PHISHING":
                result["risk_score"] = 0.7 + (result["confidence"] * 0.3)
            elif result["classification"] == "LEGITIMATE":
                result["risk_score"] = 0.1 + (result["confidence"] * 0.2)
            else:
                result["risk_score"] = 0.5

            # Add indicator weight
            result["risk_score"] += len(result.get("risk_indicators", [])) * 0.05
            result["risk_score"] = min(max(result["risk_score"], 0.0), 1.0)

            return result
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")

        # Fallback to original parsing if JSON fails
        return {
            "classification": "UNKNOWN",
            "confidence": 0.5,
            "reasoning": "Failed to parse model response",
            "risk_indicators": [],
            "risk_score": 0.5,
            "recommended_action": "REVIEW"
        }


    def batch_analyze(self, emails: List[str], batch_size: int = 8) -> List[Dict]:
        """Analyze multiple emails in batches"""
        results = []

        for i in range(0, len(emails), batch_size):
            batch = emails[i:i+batch_size]

            for email in batch:
                try:
                    result = self.analyze_email(email)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to analyze email: {e}")
                    results.append({
                        'classification': 'ERROR',
                        'error': str(e)
                    })

        return results

    def _resolve_device(self, device_config: str) -> str:
        """Resolve device configuration to actual device"""
        if device_config == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device_config