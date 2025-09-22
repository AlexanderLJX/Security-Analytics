def import_cicids_to_wazuh(self, cicids_path="data/CIC-IDS-2017/"):
    """Import CIC-IDS2017 as network alerts"""
    
    df = pd.read_csv(f'{cicids_path}/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
    
    alerts = []
    for _, row in df.iterrows():
        if row['Label'] != 'BENIGN':
            alert = {
                "@timestamp": datetime.now().isoformat(),
                "agent": {
                    "id": "002",
                    "name": "network-sensor",
                    "ip": row.get('Source IP', '0.0.0.0')
                },
                "rule": {
                    "description": f"Network attack detected: {row['Label']}",
                    "level": 12,
                    "id": "5503",
                    "groups": ["network", "ids", row['Label'].lower()]
                },
                "data": {
                    "srcip": row.get('Source IP'),
                    "dstip": row.get('Destination IP'),
                    "srcport": row.get('Source Port'),
                    "dstport": row.get('Destination Port'),
                    "protocol": row.get('Protocol'),
                    "attack_type": row['Label']
                },
                "network": {
                    "protocol": row.get('Protocol', 'tcp'),
                    "bytes": int(row.get('Total Length of Fwd Packets', 0))
                }
            }
            alerts.append(alert)
    
    # Insert into Elasticsearch
    self._bulk_insert(alerts)