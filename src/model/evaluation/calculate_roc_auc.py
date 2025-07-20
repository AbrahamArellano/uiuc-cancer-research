"""
Calculate ROC-AUC metrics for TabNet prostate cancer variant classifier.
This script loads the trained model and computes multi-class ROC curves and AUC scores.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from datetime import datetime


def load_model_and_data(model_path, data_path):
    """Load trained TabNet model and test dataset."""
    # Load the trained model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    feature_names = model_data['feature_names']
    
    # Load test data (assuming it's saved from training)
    test_data = pd.read_csv(data_path)
    
    return model, preprocessor, feature_names, test_data


def prepare_test_data(test_data, feature_names, preprocessor):
    """Prepare test data for prediction."""
    # Extract features
    X_test = test_data[feature_names]
    
    # Apply preprocessing
    X_test_processed = preprocessor.transform(X_test)
    
    # Get true labels
    y_true = test_data['target'].values
    
    # Map string labels to integers if needed
    label_map = {'Benign': 0, 'Pathogenic': 1, 'VUS': 2}
    if y_true.dtype == 'object':
        y_true = np.array([label_map[label] for label in y_true])
    
    return X_test_processed, y_true


def calculate_multiclass_roc_auc(y_true, y_pred_proba, class_names):
    """Calculate ROC curves and AUC for multi-class classification."""
    n_classes = len(class_names)
    
    # Binarize the output for multi-class ROC
    y_true_binarized = label_binarize(y_true, classes=list(range(n_classes)))
    
    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculate micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_binarized.ravel(), 
        y_pred_proba.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Calculate macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    return fpr, tpr, roc_auc


def plot_roc_curves(fpr, tpr, roc_auc, class_names, output_path):
    """Generate ROC curve plots."""
    plt.figure(figsize=(10, 8))
    
    colors = ['#d62728', '#2ca02c', '#ff7f0e']  # Red, Green, Orange
    
    # Plot ROC curves for each class
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        plt.plot(
            fpr[i], tpr[i], 
            color=color, lw=2,
            label=f'{class_name} (AUC = {roc_auc[i]:.3f})'
        )
    
    # Plot macro-average ROC curve
    plt.plot(
        fpr["macro"], tpr["macro"],
        color='navy', lw=2, linestyle='--',
        label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})'
    )
    
    # Plot chance line
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
    
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - TabNet Prostate Cancer Variant Classification', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curves saved to: {output_path}")


def save_results(fpr, tpr, roc_auc, class_names, output_path):
    """Save ROC-AUC metrics to JSON file."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'class_names': class_names,
        'auc_scores': {
            'individual_classes': {
                class_name: float(roc_auc[i]) 
                for i, class_name in enumerate(class_names)
            },
            'micro_average': float(roc_auc['micro']),
            'macro_average': float(roc_auc['macro'])
        },
        'summary': {
            'overall_auc': float(roc_auc['macro']),
            'pathogenic_auc': float(roc_auc[1]),  # Assuming Pathogenic is index 1
            'benign_auc': float(roc_auc[0]),      # Assuming Benign is index 0
            'vus_auc': float(roc_auc[2])          # Assuming VUS is index 2
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    return results


def main():
    """Main execution function."""
    # Configuration
    model_path = 'tabnet_model_20250708_161747.pkl'
    test_data_path = 'data/processed/test_data.csv'  # Adjust path as needed
    output_dir = 'results/roc_analysis/'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Class names in order
    class_names = ['Benign', 'Pathogenic', 'VUS']
    
    print("Loading model and data...")
    model, preprocessor, feature_names, test_data = load_model_and_data(
        model_path, test_data_path
    )
    
    print("Preparing test data...")
    X_test, y_true = prepare_test_data(test_data, feature_names, preprocessor)
    
    print("Getting predictions...")
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_test)
    
    # Also get class predictions for verification
    y_pred = model.predict(X_test)
    
    # Verify accuracy matches reported value
    accuracy = np.mean(y_pred == y_true)
    print(f"Test accuracy: {accuracy:.3f} (should be ~0.899)")
    
    print("Calculating ROC curves and AUC...")
    fpr, tpr, roc_auc = calculate_multiclass_roc_auc(y_true, y_pred_proba, class_names)
    
    # Print results
    print("\nROC-AUC Results:")
    print(f"Macro-average AUC: {roc_auc['macro']:.3f}")
    print(f"Micro-average AUC: {roc_auc['micro']:.3f}")
    print("\nClass-specific AUC:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {roc_auc[i]:.3f}")
    
    # Save results
    results_path = os.path.join(output_dir, 'roc_auc_metrics.json')
    results = save_results(fpr, tpr, roc_auc, class_names, results_path)
    
    # Generate plots
    plot_path = os.path.join(output_dir, 'roc_curves.png')
    plot_roc_curves(fpr, tpr, roc_auc, class_names, plot_path)
    
    # Print summary for paper
    print("\n" + "="*50)
    print("SUMMARY FOR PAPER:")
    print("="*50)
    print(f"Overall AUC (macro-average): {results['summary']['overall_auc']:.3f}")
    print(f"Pathogenic AUC: {results['summary']['pathogenic_auc']:.3f}")
    print(f"Benign AUC: {results['summary']['benign_auc']:.3f}")
    print(f"VUS AUC: {results['summary']['vus_auc']:.3f}")


if __name__ == "__main__":
    main()