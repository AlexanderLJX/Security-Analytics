"""
Evaluation metrics for phishing detection
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class PhishingMetrics:
    """Calculate and visualize metrics for phishing detection"""

    def __init__(self):
        self.results = []

    def calculate_metrics(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_prob: Optional[np.ndarray] = None) -> Dict:
        """Calculate comprehensive metrics"""

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }

        # Add ROC-AUC if probabilities available
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        metrics['true_negatives'] = cm[0, 0]
        metrics['false_positives'] = cm[0, 1]
        metrics['false_negatives'] = cm[1, 0]
        metrics['true_positives'] = cm[1, 1]

        # False positive rate (important for phishing)
        if cm[0, 0] + cm[0, 1] > 0:
            metrics['false_positive_rate'] = cm[0, 1] / (cm[0, 0] + cm[0, 1])
        else:
            metrics['false_positive_rate'] = 0

        # Store results
        self.results.append(metrics)

        return metrics

    def print_metrics(self, metrics: Dict, title: str = "Evaluation Metrics"):
        """Print metrics in a formatted way"""
        print(f"\n{'='*50}")
        print(f"{title:^50}")
        print(f"{'='*50}")

        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")

        if 'roc_auc' in metrics:
            print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

        print(f"FPR:       {metrics['false_positive_rate']:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              Legit  Phish")
        print(f"Actual Legit  {metrics['true_negatives']:5d}  {metrics['false_positives']:5d}")
        print(f"       Phish  {metrics['false_negatives']:5d}  {metrics['true_positives']:5d}")
        print(f"{'='*50}\n")

    def plot_confusion_matrix(self, metrics: Dict, save_path: Optional[str] = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))

        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'])

        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                      save_path: Optional[str] = None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved ROC curve to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                                   save_path: Optional[str] = None):
        """Plot precision-recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved precision-recall curve to {save_path}")
        else:
            plt.show()

        plt.close()

    def evaluate_reasoning_quality(self, predictions: List[Dict]) -> Dict:
        """Evaluate the quality of reasoning generated"""

        quality_metrics = {
            'avg_confidence': np.mean([p.get('confidence', 0.5) for p in predictions]),
            'avg_num_indicators': np.mean([len(p.get('risk_indicators', [])) for p in predictions]),
            'reasoning_provided': sum(1 for p in predictions if p.get('reasoning')),
            'total_predictions': len(predictions)
        }

        # Analyze confidence calibration
        confident_correct = 0
        confident_incorrect = 0

        for pred in predictions:
            if pred.get('confidence', 0.5) > 0.8:
                if pred.get('correct', False):
                    confident_correct += 1
                else:
                    confident_incorrect += 1

        if confident_correct + confident_incorrect > 0:
            quality_metrics['high_confidence_accuracy'] = (
                confident_correct / (confident_correct + confident_incorrect)
            )
        else:
            quality_metrics['high_confidence_accuracy'] = 0

        return quality_metrics

    def cross_dataset_evaluation(self,
                                datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
        """Evaluate model across multiple datasets"""
        results = []

        for dataset_name, (y_true, y_pred) in datasets.items():
            metrics = self.calculate_metrics(y_true, y_pred)
            metrics['dataset'] = dataset_name
            results.append(metrics)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Select key columns
        columns = ['dataset', 'accuracy', 'precision', 'recall', 'f1_score', 'false_positive_rate']
        df = df[columns]

        # Add average row
        avg_row = df.select_dtypes(include=[np.number]).mean()
        avg_row['dataset'] = 'Average'
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        return df

    def export_results(self, output_path: str):
        """Export evaluation results"""
        if not self.results:
            logger.warning("No results to export")
            return

        # Convert to DataFrame
        df = pd.DataFrame(self.results)

        # Save as CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Exported results to {output_path}")