"""
SIEM integration for phishing detection alerts
"""
import json
import hashlib
import socket
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from kafka import KafkaProducer
from flask import Flask, request, jsonify
import logging
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import requests
import uuid

logger = logging.getLogger(__name__)

class SIEMIntegration:
    """SIEM integration for security alerts"""

    def __init__(self, detector, config):
        self.detector = detector
        self.config = config
        self.producer = None
        self._init_kafka()

        # Alert queue for async processing
        self.alert_queue = queue.Queue(maxsize=1000)
        self.worker_thread = None
        self._start_worker()

    def _init_kafka(self):
        """Initialize Kafka producer"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=[self.config.kafka_broker],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type='gzip'
            )
            logger.info(f"Connected to Kafka broker: {self.config.kafka_broker}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")

    def _start_worker(self):
        """Start background worker for alert processing"""
        self.worker_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self.worker_thread.start()

    def _process_alerts(self):
        """Process alerts from queue"""
        while True:
            try:
                alert = self.alert_queue.get(timeout=1)
                self._send_to_kafka(alert)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")

    def create_alert(self, email_text: str, analysis_result: Dict, email_metadata: Optional[Dict] = None) -> Dict:
        """Create comprehensive SIEM alert from analysis result"""

        # Generate unique ID for the email
        email_hash = hashlib.sha256(email_text.encode()).hexdigest()
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Extract email metadata if available
        if email_metadata is None:
            email_metadata = self._extract_email_metadata(email_text)

        alert = {
            # Core alert information
            'alert_id': f"phish-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{random.randint(1,999):03d}",
            'timestamp': timestamp,
            'source': 'phishing-detector-ai',
            'sourcetype': 'phishing_detection',
            'host': socket.gethostname(),
            'index': 'security',

            'event': {
                
                'event_type': 'email_analysis',
                'category': 'phishing_attempt' if analysis_result.get('classification') == 'PHISHING' else 'email_analysis',
                'severity': self._determine_severity(analysis_result),

                # Email information
                'email': {
                    'sender': email_metadata.get('sender', 'unknown'),
                    'subject': email_metadata.get('subject', 'unknown'),
                    'recipient': email_metadata.get('recipient', 'unknown'),
                    'message_id': email_metadata.get('message_id', f"hash-{email_hash[:16]}"),
                    'size_bytes': len(email_text),
                    'hash': email_hash,
                    'body': email_text
                },

                # Detection results
                'detection': {
                    'classification': analysis_result.get('classification', 'UNKNOWN'),
                    'confidence': analysis_result.get('confidence', 0.0),
                    'risk_score': analysis_result.get('risk_score', 0.0),
                    'recommended_action': analysis_result.get('recommended_action', 'REVIEW'),
                    'processing_time_ms': int(analysis_result.get('processing_time', 0) * 1000),
                    'model_version': 'qwen-phishing-detector-v1.0',
                    'inference_type': getattr(analysis_result, 'inference_type', 'unknown')
                },

                # Detailed analysis
                'analysis': {
                    'reasoning': analysis_result.get('reasoning', ''),
                    'risk_indicators': analysis_result.get('risk_indicators', []),
                    'threat_indicators': self._extract_threat_indicators(analysis_result, email_text)
                },

                # Actions taken
                'actions_taken': {
                    'email_quarantined': analysis_result.get('risk_score', 0) > 0.7,
                    'sender_blocked': analysis_result.get('recommended_action') == 'BLOCK',
                    'soc_alerted': analysis_result.get('risk_score', 0) > 0.8,
                    'threat_intel_updated': analysis_result.get('classification') == 'PHISHING'
                }
            }
        }

        # Preserve alert identifier inside the event payload for downstream systems like Splunk.
        alert['event']['alert_id'] = alert['alert_id']

        # Add MITRE ATT&CK mappings if phishing detected
        if analysis_result.get('classification') == 'PHISHING':
            alert['event']['mitre_attack'] = {
                'tactics': ['T1566', 'T1204'],
                'techniques': [
                    'T1566.002',  # Phishing: Spearphishing Link
                    'T1204.002'   # User Execution: Malicious File
                ],
                'technique_names': ['Phishing: Spearphishing Link', 'User Execution: Malicious File']
            }

        return alert

    def _extract_email_metadata(self, email_text: str) -> Dict:
        """Extract basic email metadata from text"""
        import re

        metadata = {
            'sender': 'unknown',
            'subject': 'unknown',
            'recipient': 'unknown',
            'message_id': 'unknown'
        }

        # Try to extract sender
        from_match = re.search(r'^\s*From\s*:\s*([^\r\n]+)', email_text, re.IGNORECASE | re.MULTILINE)
        if from_match:
            metadata['sender'] = from_match.group(1).strip()

        # Try to extract subject
        subject_match = re.search(r'^\s*Subject\s*:\s*([^\r\n]+)', email_text, re.IGNORECASE | re.MULTILINE)
        if subject_match:
            metadata['subject'] = subject_match.group(1).strip()

        # Try to extract recipient
        to_match = re.search(r'^\s*To\s*:\s*([^\r\n]+)', email_text, re.IGNORECASE | re.MULTILINE)
        if to_match:
            metadata['recipient'] = to_match.group(1).strip()

        return metadata

    def _extract_threat_indicators(self, analysis_result: Dict, email_text: str) -> Dict:
        """Extract threat intelligence indicators"""
        import re

        indicators = {
            'domains': [],
            'ips': [],
            'urls': [],
            'email_addresses': [],
            'social_engineering_tactics': [],
            'impersonation_targets': [],
            'financial_requests': False,
            'pii_requests': [],
            'urgency_language': False
        }

        email_lower = email_text.lower()

        # Extract domains and URLs 
        url_pattern = r'https?://([^\s/]+)'
        urls = re.findall(url_pattern, email_text, re.IGNORECASE)
        indicators['domains'] = list(set(urls))
        indicators['urls'] = re.findall(r'https?://[^\s]+', email_text, re.IGNORECASE)

        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        indicators['email_addresses'] = re.findall(email_pattern, email_text)

        # Detect social engineering tactics
        if any(word in email_lower for word in ['urgent', 'immediate', 'expires', 'limited time']):
            indicators['social_engineering_tactics'].append('urgency')
            indicators['urgency_language'] = True

        if any(word in email_lower for word in ['verify', 'confirm', 'update', 'suspend']):
            indicators['social_engineering_tactics'].append('verification_request')

        if any(word in email_lower for word in ['bank', 'paypal', 'amazon', 'microsoft', 'apple']):
            indicators['social_engineering_tactics'].append('brand_impersonation')

        # Detect financial requests
        if any(word in email_lower for word in ['payment', 'fee', 'money', '$', 'transfer', 'refund']):
            indicators['financial_requests'] = True

        # Detect PII requests
        pii_patterns = ['password', 'ssn', 'social security', 'credit card', 'id', 'license']
        for pattern in pii_patterns:
            if pattern in email_lower:
                indicators['pii_requests'].append(pattern)

        return indicators

    def _determine_severity(self, analysis_result: Dict) -> str:
        """Determine alert severity"""
        risk_score = analysis_result.get('risk_score', 0.0)

        if risk_score > 0.9:
            return 'CRITICAL'
        elif risk_score > 0.7:
            return 'HIGH'
        elif risk_score > 0.5:
            return 'MEDIUM'
        elif risk_score > 0.3:
            return 'LOW'
        else:
            return 'INFO'

    def send_alert(self, alert: Dict):
        """Send alert to SIEM"""
        try:
            # Add to queue for async processing
            self.alert_queue.put_nowait(alert)
            logger.info(f"Alert queued: {alert['alert_id']}")
        except queue.Full:
            logger.error("Alert queue is full, alert dropped")

    def _send_to_kafka(self, alert: Dict):
        """Send alert to Kafka"""
        if self.producer:
            try:
                future = self.producer.send(
                    self.config.kafka_topic,
                    value=alert,
                    key=alert['alert_id'].encode('utf-8')
                )

                # Wait for confirmation 
                result = future.get(timeout=10)
                logger.info(f"Alert sent to Kafka: {alert['alert_id']}")

            except Exception as e:
                logger.error(f"Failed to send alert to Kafka: {e}")

        # Also send to Splunk if configured
        self._send_to_splunk(alert)

    def _send_to_splunk(self, alert: Dict):
        """Send alert to Splunk HTTP Event Collector"""
        if hasattr(self.config, 'splunk_hec_url') and hasattr(self.config, 'splunk_token'):
            try:
                headers = {
                    'Authorization': f'Splunk {self.config.splunk_token}',
                    'Content-Type': 'application/json'
                }
                # Format for Splunk HEC
                # Convert ISO8601 timestamp to epoch seconds; fall back to current time.
                ts = alert.get("timestamp")
                try:
                    if isinstance(ts, (int, float)):
                        epoch_time = float(ts)
                    elif isinstance(ts, str):
                        epoch_time = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                    else:
                        epoch_time = datetime.now(tz=timezone.utc).timestamp()
                except Exception:
                    epoch_time = datetime.now(tz=timezone.utc).timestamp()
                # Format for Splunk HEC
                splunk_event = {
                    "time": epoch_time,
                    'host': alert.get('host'),
                    'source': alert.get('source'),
                    'sourcetype': alert.get('sourcetype'),
                    'index': alert.get('index'),
                    'fields': {
                        'alert_id': alert.get('alert_id')
                    },
                    'event': alert['event']
                }

                response = requests.post(
                    self.config.splunk_hec_url,
                    json=splunk_event,
                    headers=headers,
                    timeout=10, 
                    verify=False
                )

                if response.status_code == 200:
                    logger.info(f"Alert sent to Splunk: {alert['alert_id']}")
                else:
                    logger.error(f"Splunk HEC error: {response.status_code} - {response.text}")

            except Exception as e:
                logger.error(f"Failed to send alert to Splunk: {e}")

    def process_email(self, email_text: str) -> Dict:
        """Process email and generate alert if needed"""

        # Analyze email
        analysis_result = self.detector.analyze_email(email_text)

        # Create alert
        alert = self.create_alert(email_text, analysis_result)

        # Send alert if meets threshold
        if analysis_result.get('risk_score', 0) >= self.config.alert_threshold:
            self.send_alert(alert)

        return {
            'analysis': analysis_result,
            'alert': alert
        }

    def get_stats(self) -> Dict:
        """Get current statistics"""
        return {
            'queue_size': self.alert_queue.qsize(),
            'kafka_connected': self.producer is not None,
            'worker_alive': self.worker_thread.is_alive() if self.worker_thread else False
        }


class PhishingAPI:
    """REST API for phishing detection"""

    def __init__(self, siem_integration):
        self.siem = siem_integration
        self.app = Flask(__name__)
        self._setup_routes()

        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=1)

    def _setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'stats': self.siem.get_stats()  
            })

        @self.app.route('/analyze', methods=['POST'])
        def analyze():
            try:
                data = request.get_json()

                if not data or 'email' not in data:
                    return jsonify({'error': 'Missing email content'}), 400

                email_text = data['email']

                # Process email
                result = self.siem.process_email(email_text)

                return jsonify(result)

            except Exception as e:
                logger.error(f"API error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/batch', methods=['POST'])
        def batch_analyze():
            try:
                data = request.get_json()

                if not data or 'emails' not in data:
                    return jsonify({'error': 'Missing emails'}), 400

                emails = data['emails']

                # Process in parallel
                futures = []
                for email in emails:
                    future = self.executor.submit(self.siem.process_email, email)
                    futures.append(future)

                # Collect results
                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        results.append({'error': str(e)})

                return jsonify({'results': results})

            except Exception as e:
                logger.error(f"Batch API error: {e}")
                return jsonify({'error': str(e)}), 500

    def run(self):
        """Run the API server"""
        self.app.run(
            host=self.siem.config.api_host,
            port=self.siem.config.api_port,
            debug=False
            
        )
