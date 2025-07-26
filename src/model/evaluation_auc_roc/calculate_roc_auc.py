"""
Calculate ROC-AUC metrics for TabNet prostate cancer variant classifier.
Complete version with all fixes for GPU-saved models and PyTorch 2.6 compatibility.

Author: Abraham Arellano (aa107@illinois.edu)
Location: /u/aa107/uiuc-cancer-research/src/model/evaluation_auc_roc/calculate_roc_auc.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from datetime import datetime
import torch
import warnings
warnings.filterwarnings('ignore')


def load_model_and_data(model_path):
    """Load trained TabNet model with full GPU/CPU compatibility."""
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model with GPU to CPU compatibility
    with open(model_path, 'rb') as f:
        try:
            # Try torch.load first with proper settings for PyTorch 2.6+
            model_data = torch.load(f, map_location='cpu', weights_only=False)
            print("✅ Loaded using torch.load with CPU mapping")
        except Exception as e:
            print(f"⚠️  Torch load failed: {e}")
            # Fallback to pure pickle
            f.seek(0)  # Reset file pointer
            model_data = pickle.load(f)
            print("✅ Loaded using pickle.load")
    
    # Extract components based on TabNet save structure
    required_keys = ['tabnet_model', 'scaler', 'label_encoder', 'feature_names']
    for key in required_keys:
        if key not in model_data:
            raise KeyError(f"Missing required component: {key}")
    
    model = model_data['tabnet_model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
    feature_names = model_data['feature_names']
    
    # Move model to CPU if needed (for GPU-saved models)
    if hasattr(model, 'device'):
        model.device = 'cpu'
    
    # Load the original dataset
    data_path = '/u/aa107/uiuc-cancer-research/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv'
    print(f"Loading dataset from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    full_data = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(full_data):,} variants, {len(full_data.columns)} columns")
    
    return model, scaler, label_encoder, feature_names, full_data


def recreate_test_split(full_data, feature_names, label_encoder):
    """Recreate the exact test split used during training."""
    print("\nRecreating test split...")
    
    # Verify all features are present
    missing_features = [f for f in feature_names if f not in full_data.columns]
    if missing_features:
        print(f"⚠️  Warning: Missing features: {missing_features[:5]}...")
        available_features = [f for f in feature_names if f in full_data.columns]
        if not available_features:
            raise ValueError("No features found in dataset!")
        feature_names = available_features
    
    # Prepare features
    X = full_data[feature_names]
    print(f"Features shape: {X.shape}")
    
    # Create target variable (matching training logic)
    target_column = None
    possible_targets = ['CLIN_SIG', 'clinical_significance', 'ClinicalSignificance', 
                       'CLNSIG', 'clinicalsignificance']
    
    for col in possible_targets:
        if col in full_data.columns:
            target_column = col
            print(f"Using target column: {col}")
            break
    
    if target_column is None:
        raise ValueError(f"No clinical significance column found. Available columns: {full_data.columns.tolist()[:10]}...")
    
    # Create target variable
    y = full_data[target_column].fillna('VUS')
    
    # Map values to three classes (same as training)
    value_mapping = {
        'Pathogenic/Likely_pathogenic': 'Pathogenic',
        'Pathogenic': 'Pathogenic',
        'Likely_pathogenic': 'Pathogenic',
        'Benign/Likely_benign': 'Benign',
        'Benign': 'Benign',
        'Likely_benign': 'Benign',
        'Uncertain_significance': 'VUS',
        'Conflicting_interpretations_of_pathogenicity': 'VUS',
        'not_provided': 'VUS',
        'drug_response': 'VUS',
        'other': 'VUS'
    }
    
    y = y.replace(value_mapping)
    
    # Handle any unmapped values
    unique_values = y.unique()
    for val in unique_values:
        if val not in ['Pathogenic', 'Benign', 'VUS']:
            y = y.replace(val, 'VUS')
            print(f"  Mapped '{val}' to VUS")
    
    # Print class distribution
    print("\nTarget distribution:")
    for class_name, count in y.value_counts().items():
        print(f"  {class_name}: {count:,} ({count/len(y)*100:.1f}%)")
    
    # Use same random_state as training to get identical split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTest set size: {len(X_test):,} variants")
    
    # Encode labels
    y_test_encoded = label_encoder.transform(y_test)
    
    return X_test, y_test_encoded, y_test


def calculate_multiclass_roc_auc(y_true, y_pred_proba, class_names):
    """Calculate ROC curves and AUC for multi-class classification."""
    print("\nCalculating ROC curves...")
    n_classes = len(class_names)
    
    # Ensure proper shape
    if y_pred_proba.shape[1] != n_classes:
        raise ValueError(f"Prediction shape mismatch. Expected {n_classes} classes, got {y_pred_proba.shape[1]}")
    
    # Binarize the output
    y_true_binarized = label_binarize(y_true, classes=list(range(n_classes)))
    
    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(f"  {class_names[i]} AUC: {roc_auc[i]:.3f}")
    
    # Micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_binarized.ravel(), 
        y_pred_proba.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    print(f"  Macro-average AUC: {roc_auc['macro']:.3f}")
    print(f"  Micro-average AUC: {roc_auc['micro']:.3f}")
    
    return fpr, tpr, roc_auc


def save_results(fpr, tpr, roc_auc, class_names, output_dir):
    """Save ROC-AUC metrics and curve data."""
    print("\nSaving results...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results dictionary
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'type': 'TabNet',
            'task': 'Prostate Cancer Variant Classification',
            'classes': 3
        },
        'class_names': class_names,
        'auc_scores': {
            'individual_classes': {
                class_names[i]: float(roc_auc[i]) 
                for i in range(len(class_names))
            },
            'micro_average': float(roc_auc['micro']),
            'macro_average': float(roc_auc['macro'])
        },
        'summary': {
            'overall_auc': float(roc_auc['macro']),
            'pathogenic_auc': float(roc_auc[class_names.index('Pathogenic')]) if 'Pathogenic' in class_names else None,
            'benign_auc': float(roc_auc[class_names.index('Benign')]) if 'Benign' in class_names else None,
            'vus_auc': float(roc_auc[class_names.index('VUS')]) if 'VUS' in class_names else None
        }
    }
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'roc_auc_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✅ Metrics saved: {metrics_path}")
    
    # Save FPR/TPR data for plotting
    curve_data = {
        'fpr': {str(k): v.tolist() if hasattr(v, 'tolist') else v for k, v in fpr.items()},
        'tpr': {str(k): v.tolist() if hasattr(v, 'tolist') else v for k, v in tpr.items()},
        'roc_auc': {str(k): float(v) for k, v in roc_auc.items()},
        'class_names': class_names
    }
    
    curve_path = os.path.join(output_dir, 'roc_curve_data.json')
    with open(curve_path, 'w') as f:
        json.dump(curve_data, f, indent=2)
    print(f"  ✅ Curve data saved: {curve_path}")
    
    return results


def plot_roc_curves(fpr, tpr, roc_auc, class_names, output_dir):
    """Generate ROC curve plot."""
    print("\nGenerating ROC curve plot...")
    
    plt.figure(figsize=(8, 6))
    
    # Color scheme
    colors = ['#2ca02c', '#d62728', '#ff7f0e']  # Green, Red, Orange
    
    # Plot individual class curves
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        if i < len(colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
    
    # Plot macro-average curve
    plt.plot(fpr["macro"], tpr["macro"], 'navy', lw=2, linestyle='--',
            label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')
    
    # Plot chance line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    # Formatting
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - TabNet Prostate Cancer Classification', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Plot saved: {plot_path}")


def main():
    """Main execution function."""
    # Configuration
    model_path = '/u/aa107/scratch/tabnet_model_20250719_055806.pkl'
    output_dir = '/u/aa107/uiuc-cancer-research/results/roc_analysis/'
    
    print("="*60)
    print("ROC-AUC CALCULATION FOR TABNET MODEL")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    
    try:
        # Load model and data
        print("\n1. LOADING MODEL AND DATA")
        print("-" * 30)
        model, scaler, label_encoder, feature_names, full_data = load_model_and_data(model_path)
        
        # Get class names from label encoder
        class_names = list(label_encoder.classes_)
        print(f"Classes: {class_names}")
        print(f"Features: {len(feature_names)}")
        
        # Recreate test split
        print("\n2. RECREATING TEST SPLIT")
        print("-" * 30)
        X_test, y_test_encoded, y_test_labels = recreate_test_split(
            full_data, feature_names, label_encoder
        )
        
        # Scale features
        print("\n3. PREPROCESSING")
        print("-" * 30)
        print("Scaling features...")
        X_test_scaled = scaler.transform(X_test)
        
        # Get predictions
        print("\n4. GENERATING PREDICTIONS")
        print("-" * 30)
        print("Getting probability predictions...")
        y_pred_proba = model.predict_proba(X_test_scaled)
        print(f"Prediction shape: {y_pred_proba.shape}")
        
        # Get class predictions to verify accuracy
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        print(f"Test accuracy: {accuracy:.3f} (expected ~0.879)")
        
        # Calculate ROC curves and AUC
        print("\n5. ROC-AUC CALCULATION")
        print("-" * 30)
        fpr, tpr, roc_auc = calculate_multiclass_roc_auc(
            y_test_encoded, y_pred_proba, class_names
        )
        
        # Save all results
        print("\n6. SAVING RESULTS")
        print("-" * 30)
        results = save_results(fpr, tpr, roc_auc, class_names, output_dir)
        
        # Generate plot
        plot_roc_curves(fpr, tpr, roc_auc, class_names, output_dir)
        
        # Print summary
        print("\n" + "="*50)
        print("SUMMARY FOR PAPER:")
        print("="*50)
        print(f"Overall AUC (macro-average): {results['summary']['overall_auc']:.3f}")
        if results['summary']['pathogenic_auc']:
            print(f"Pathogenic detection AUC: {results['summary']['pathogenic_auc']:.3f}")
        if results['summary']['benign_auc']:
            print(f"Benign detection AUC: {results['summary']['benign_auc']:.3f}")
        if results['summary']['vus_auc']:
            print(f"VUS detection AUC: {results['summary']['vus_auc']:.3f}")
        
        print("\n✅ SUCCESS: ROC-AUC analysis completed!")
        print(f"📊 Results saved to: {output_dir}")
        print("\n🎯 Next step: Run compare_baseline_tools.py for publication figures")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()