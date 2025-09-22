"""
SIEM integration for phishing detection alerts
"""
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from kafka import KafkaProducer
from flask import Flask, request, jsonify
import logging
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

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

    def create_alert(self, email_text: str, analysis_result: Dict) -> Dict:
        """Create SIEM alert from analysis result"""

        # Generate unique ID for the email
        email_hash = hashlib.sha256(email_text.encode()).hexdigest()

        alert = {
            'alert_id': f"PHISH_{datetime.now().strftime('%Y%m%d%H%M%S')}_{email_hash[:8]}",
            'timestamp': datetime.now().isoformat(),
            'event_type': 'EMAIL_SECURITY',
            'severity': self._determine_severity(analysis_result),
            'classification': analysis_result.get('classification', 'UNKNOWN'),
            'confidence_score': analysis_result.get('confidence', 0.0),
            'risk_score': analysis_result.get('risk_score', 0.0),
            'risk_indicators': analysis_result.get('risk_indicators', []),
            'recommended_action': analysis_result.get('recommended_action', 'REVIEW'),
            'email_metadata': {
                'hash': email_hash,
                'length': len(email_text),
                'processing_time': analysis_result.get('processing_time', 0)
            },
            'reasoning': analysis_result.get('reasoning', ''),
            'features': analysis_result.get('features', {})
        }

        # Add MITRE ATT&CK mappings if phishing detected
        if analysis_result.get('classification') == 'PHISHING':
            alert['mitre_attack'] = {
                'technique': 'T1566',
                'name': 'Phishing',
                'tactic': 'Initial Access'
            }

        return alert

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
        self.executor = ThreadPoolExecutor(max_workers=10)

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