"""
Qwen model setup and configuration
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from transformers import TextStreamer
from typing import Optional, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

# Try to import llama-cpp-python for GGUF support
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

class QwenPhishingModel:
    """Qwen model for phishing detection with reasoning"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_gguf = False
        self.llama_model = None

    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load Qwen model with optional checkpoint"""
        import os
        import json

        # Check if it's a GGUF file
        if checkpoint_path and checkpoint_path.endswith('.gguf') and os.path.exists(checkpoint_path):
            if not LLAMA_CPP_AVAILABLE:
                raise RuntimeError("llama-cpp-python is required for GGUF files. Install with: pip install llama-cpp-python")

            logger.info(f"Loading GGUF model: {checkpoint_path}")
            self.is_gguf = True

            # Load with llama-cpp-python
            n_ctx = self.config.max_seq_length
            n_threads = os.cpu_count() if self.config.device == "cpu" else None
            n_gpu_layers = -1 if self.config.device != "cpu" else 0

            self.llama_model = Llama(
                model_path=checkpoint_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            logger.info("GGUF model loaded successfully")
            return

        elif checkpoint_path and os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
            # This is a LoRA checkpoint, load base model first then adapter
            logger.info(f"Loading LoRA checkpoint: {checkpoint_path}")

            # Read adapter config to get base model
            with open(os.path.join(checkpoint_path, "adapter_config.json"), 'r') as f:
                adapter_config = json.load(f)

            base_model_name = adapter_config.get("base_model_name_or_path", self.config.student_model)
            logger.info(f"Loading base model: {base_model_name}")

            # Load base model
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_name,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                dtype=None,
                device_map=self.config.device,
            )

            # Load LoRA adapter
            from peft import PeftModel
            logger.info(f"Loading LoRA adapter from: {checkpoint_path}")
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)

        elif checkpoint_path:
            # Regular checkpoint
            logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=checkpoint_path,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                dtype=None,
                device_map=self.config.device,
            )
        else:
            # Base model only
            logger.info(f"Loading base model: {self.config.student_model}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.student_model,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                dtype=None,
                device_map=self.config.device,
            )

        # Setup chat template for Qwen3-thinking
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="qwen3-thinking",
        )

        logger.info("Model loaded successfully")

    def setup_lora(self):
        """Setup LoRA adapters for efficient fine-tuning"""
        logger.info("Setting up LoRA adapters")

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=self.config.lora_target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.config.seed,
            use_rslora=False,
            loftq_config=None,
        )

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        logger.info(f"Total parameters: {total_params:,}")

    def generate_response(self,
                         prompt: str,
                         max_new_tokens: int = 2000,
                         temperature: float = 0.3,
                         top_p: float = 0.9,
                         top_k: int = 20,
                         streaming: bool = False) -> str:
        """Generate response for a given prompt"""

        # Tokenize input
        # FIX: Remove .to(self.config.device) because device_map="auto" already places the model and inputs on the correct device.
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.model.device) # Use model's device directly

        # Setup streamer if needed
        streamer = TextStreamer(self.tokenizer, skip_prompt=True) if streaming else None

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                streamer=streamer,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response and get only the newly generated text
        response = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

        return response.strip()

    def analyze_email(self, email_text: str, return_reasoning: bool = True) -> Dict[str, Any]:
        """Analyze email for phishing with reasoning"""
        import time

        start_time = time.time()

        if self.is_gguf:
            # Handle GGUF model - request structured output
            prompt = f"""<|im_start|>user
Analyze this email for phishing. Respond with this JSON format on a single line without any extra text:

{{
  "classification": "PHISHING" or "LEGITIMATE",
  "confidence": 0.0-1.0,
  "reasoning": "Brief 2-3 sentence analysis",
  "risk_indicators": ["list", "of", "indicators", "found"],
  "recommended_action": "BLOCK" or "QUARANTINE" or "REVIEW" or "ALLOW"
}}

------Email Start------

{email_text}

------Email End------<|im_end|>
<|im_start|>assistant
"""

            print(f"DEBUG: GGUF Prompt sent to model:\n{'-'*50}")
            print(prompt)
            print(f"{'-'*50}")

            response = self.llama_model(
                prompt,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop=["<|im_end|>"],
                echo=False
            )

            response_text = response['choices'][0]['text']
            print(f"DEBUG: Raw GGUF response:\n{'-'*50}")
            print(response_text)
            print(f"{'-'*50}")

            result = self._parse_response(response_text)
            result['processing_time'] = time.time() - start_time
        else:
            # Handle HuggingFace model - request structured output
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

            # Apply chat template - keep thinking enabled for better reasoning
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True  # Keep thinking enabled for better performance
            )

            print(f"DEBUG: HuggingFace Prompt sent to model:\n{'-'*50}")
            print(prompt)
            print(f"{'-'*50}")

            # Generate response
            response = self.generate_response(prompt)

            print(f"DEBUG: Raw HuggingFace response:\n{'-'*50}")
            print(response)
            print(f"{'-'*50}")

            # Parse response
            result = self._parse_response(response)
            result['processing_time'] = time.time() - start_time

        if not return_reasoning:
            return {"classification": result["classification"]}

        return result

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from model"""
        import json
        import re

        try:
            # Try to extract JSON from the response
            # First try to find JSON after </think> if thinking mode was used
            think_match = re.search(r'</think>\s*(\{.*\})', response, re.DOTALL)
            if think_match:
                json_str = think_match.group(1).strip()
                print(f"DEBUG: Found JSON after </think>: {json_str[:200]}...")
            else:
                # Look for any JSON block in the response (fallback)
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group().strip()
                    print(f"DEBUG: Found JSON block: {json_str[:200]}...")
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
            print(f"DEBUG: Failed to parse JSON response: {e}")
            print(f"DEBUG: Raw response: {response}")

        # Fallback to original parsing if JSON fails
        return {
            "classification": "UNKNOWN",
            "confidence": 0.5,
            "reasoning": "Failed to parse model response",
            "risk_indicators": [],
            "risk_score": 0.5,
            "recommended_action": "REVIEW"
        }

    def save_model(self, save_path: str):
        """Save the fine-tuned model"""
        logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info("Model saved successfully")

    def merge_and_unload(self):
        """Merge LoRA weights with base model and unload adapters"""
        logger.info("Merging LoRA weights with base model")
        if hasattr(self.model, 'merge_and_unload'):
            self.model = self.model.merge_and_unload()
            logger.info("LoRA weights merged successfully")
        else:
            logger.warning("Model doesn't have LoRA adapters to merge")

    def save_gguf(self, save_path: str, quantization_method: str = "q4_k_m"):
        """Save model in GGUF format for llama.cpp"""
        logger.info(f"Saving GGUF model to {save_path}")
        self.model.save_pretrained_gguf(save_path, self.tokenizer, quantization_method=quantization_method)
        logger.info("GGUF model saved successfully")

    def export_merged_model(self, save_path: str, save_method: str = "merged_16bit", quantization_method: str = "q4_k_m"):
        """Export merged model (LoRA + base) using Unsloth's save methods"""
        logger.info(f"Exporting merged model with method: {save_method}")

        # Check if we have a PeftModel with adapters
        if hasattr(self.model, 'merge_and_unload'):
            logger.info("Found LoRA adapters, merging...")
            # First merge the adapters
            merged_model = self.model.merge_and_unload()

            # Now save the merged model
            if save_method == "gguf":
                logger.info(f"Saving merged model to GGUF: {save_path} with quantization: {quantization_method}")
                merged_model.save_pretrained_gguf(save_path, self.tokenizer, quantization_method=quantization_method)
            else:
                logger.info(f"Saving merged model to: {save_path}")
                merged_model.save_pretrained_merged(save_path, self.tokenizer, save_method=save_method)
        else:
            logger.warning("No LoRA adapters found, saving base model only")
            # Just save the base model
            if save_method == "gguf":
                logger.info(f"Saving base model to GGUF: {save_path} with quantization: {quantization_method}")
                self.model.save_pretrained_gguf(save_path, self.tokenizer, quantization_method=quantization_method)
            else:
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)

        logger.info("Merged model export completed")
        return save_path

    def export_merged_gguf(self, save_path: str, quantization_method: str = "q4_k_m"):
        """Export merged model (LoRA + base) in GGUF format"""
        return self.export_merged_model(save_path, save_method="gguf", quantization_method=quantization_method)