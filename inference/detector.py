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
                     return_reasoning: bool = True,
                     extract_features: bool = True) -> Dict[str, Any]:
        """Analyze email for phishing with detailed output"""

        start_time = time.time()

        # Extract features if requested
        features = None
        if extract_features:
            features = self.preprocessor.extract_features(email_text)

        # Create prompt
        if features:
            prompt = self.preprocessor.create_enhanced_prompt(email_text, features)
        else:
            prompt = f"Analyze this email for phishing indicators and provide detailed security reasoning:\n\n{email_text}"

        # Create conversation format
        messages = [
            {"role": "user", "content": prompt}
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

        # Parse response
        result = self._parse_response(response, features)

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

    def _parse_response(self, response: str, features: Optional[Dict] = None) -> Dict[str, Any]:
        """Parse model response into structured output"""

        result = {
            'classification': 'UNKNOWN',
            'confidence': 0.5,
            'reasoning': response,
            'risk_indicators': [],
            'risk_score': 0.0,
            'features': features,
            'recommended_action': 'REVIEW'
        }

        response_lower = response.lower()

        # Extract classification
        if "classification:" in response_lower:
            class_line = response_lower.split("classification:")[1].split('\n')[0]
            if "phishing" in class_line:
                result['classification'] = 'PHISHING'
            elif "legitimate" in class_line:
                result['classification'] = 'LEGITIMATE'
        else:
            # Fallback classification
            phishing_count = response_lower.count("phishing")
            legitimate_count = response_lower.count("legitimate")

            if phishing_count > legitimate_count * 1.5:
                result['classification'] = 'PHISHING'
            elif legitimate_count > phishing_count * 1.5:
                result['classification'] = 'LEGITIMATE'

        # Extract confidence
        result['confidence'] = self._extract_confidence(response)

        # Extract risk indicators
        result['risk_indicators'] = self._extract_risk_indicators(response)

        # Calculate risk score
        result['risk_score'] = self._calculate_risk_score(result, features)

        # Determine recommended action
        result['recommended_action'] = self._determine_action(result)

        return result

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response"""
        response_lower = response.lower()

        # High confidence indicators
        high_confidence = ['definitely', 'clearly', 'certainly', 'obvious', 'conclusive', 'undoubtedly']
        medium_confidence = ['likely', 'probably', 'appears', 'seems', 'suggests']
        low_confidence = ['possibly', 'might', 'could', 'uncertain', 'unclear']

        confidence = 0.7  # Base confidence

        # Adjust based on language
        for word in high_confidence:
            if word in response_lower:
                confidence = max(confidence, 0.9)
                break

        for word in medium_confidence:
            if word in response_lower and confidence < 0.9:
                confidence = max(confidence, 0.75)

        for word in low_confidence:
            if word in response_lower:
                confidence = min(confidence, 0.6)

        return confidence

    def _extract_risk_indicators(self, response: str) -> List[str]:
        """Extract risk indicators from response"""
        indicators = []
        response_lower = response.lower()

        risk_patterns = {
            'suspicious_url': ['suspicious url', 'malicious link', 'phishing url'],
            'urgency': ['urgent', 'immediate action', 'time sensitive'],
            'credential_request': ['password request', 'credential', 'login information'],
            'spoofed_sender': ['spoofed', 'impersonat', 'fake sender'],
            'grammar_issues': ['grammar error', 'spelling mistake', 'poor english'],
            'generic_greeting': ['generic greeting', 'dear customer', 'valued user'],
            'attachment_risk': ['suspicious attachment', 'malicious file', 'dangerous attachment'],
            'social_engineering': ['social engineering', 'psychological', 'manipulation']
        }

        for indicator, patterns in risk_patterns.items():
            if any(pattern in response_lower for pattern in patterns):
                indicators.append(indicator)

        return indicators

    def _calculate_risk_score(self, result: Dict, features: Optional[Dict]) -> float:
        """Calculate overall risk score"""
        risk_score = 0.0

        # Base score from classification
        if result['classification'] == 'PHISHING':
            risk_score = 0.7
        elif result['classification'] == 'LEGITIMATE':
            risk_score = 0.2
        else:
            risk_score = 0.5

        # Adjust based on confidence
        risk_score = risk_score * result['confidence']

        # Adjust based on risk indicators
        indicator_weight = 0.05
        risk_score += len(result['risk_indicators']) * indicator_weight

        # Incorporate features if available
        if features:
            if features.get('urgency_score', 0) > 0.5:
                risk_score += 0.1
            if features.get('suspicious_score', 0) > 0.5:
                risk_score += 0.1
            if features.get('has_shortened_url'):
                risk_score += 0.05
            if features.get('suspicious_attachments'):
                risk_score += 0.15

        # Normalize to 0-1
        return min(max(risk_score, 0.0), 1.0)

    def _determine_action(self, result: Dict) -> str:
        """Determine recommended action based on analysis"""
        risk_score = result['risk_score']
        confidence = result['confidence']

        if risk_score > 0.8:
            return 'BLOCK'
        elif risk_score > 0.6 and confidence > 0.7:
            return 'QUARANTINE'
        elif risk_score > 0.4:
            return 'REVIEW'
        else:
            return 'ALLOW'

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