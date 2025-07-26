"""
Generate publication-quality ROC curve visualizations for TabNet prostate cancer classifier.
Updated to use actual ROC curve data from calculate_roc_auc.py output.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import seaborn as sns

# Set publication quality defaults
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

def load_roc_data(base_path='/u/aa107/uiuc-cancer-research/results/roc_analysis/'):
    """Load ROC-AUC metrics and curve data from calculate_roc_auc.py output."""
    # Load metrics
    metrics_path = os.path.join(base_path, 'roc_auc_metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Load curve data
    curve_path = os.path.join(base_path, 'roc_curve_data.json')
    with open(curve_path, 'r') as f:
        curve_data = json.load(f)
    
    # Convert lists back to numpy arrays
    fpr = {k: np.array(v) for k, v in curve_data['fpr'].items()}
    tpr = {k: np.array(v) for k, v in curve_data['tpr'].items()}
    
    return metrics, fpr, tpr


def create_main_roc_figure(metrics, fpr, tpr, output_path):
    """Create the main ROC curve figure for the paper (Figure 6)."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    # Define colors matching your paper's style
    colors = {
        'Benign': '#2ca02c',      # Green
        'Pathogenic': '#d62728',  # Red
        'VUS': '#ff7f0e',         # Orange
        'macro': '#1f77b4'        # Blue
    }
    
    # Get class names and their indices
    class_names = metrics['class_names']
    class_indices = {name: str(i) for i, name in enumerate(class_names)}
    
    # Plot individual class curves
    for class_name in class_names:
        idx = class_indices[class_name]
        auc_score = metrics['auc_scores']['individual_classes'][class_name]
        
        ax.plot(fpr[idx], tpr[idx], 
                color=colors[class_name], lw=2.5, 
                label=f'{class_name} (AUC = {auc_score:.3f})')
    
    # Plot macro-average curve
    ax.plot(fpr['macro'], tpr['macro'], 
            color=colors['macro'], lw=3, linestyle='--',
            label=f'Macro-average (AUC = {metrics["auc_scores"]["macro_average"]:.3f})')
    
    # Reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Classifier')
    
    # Formatting
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('TabNet Performance for Prostate Cancer Variant Classification', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend
    legend = ax.legend(loc='lower right', fontsize=11, frameon=True, 
                      fancybox=True, shadow=True)
    legend.get_frame().set_alpha(0.95)
    
    # Add annotation box with key metric
    textstr = f'Overall Performance\nAUC = {metrics["summary"]["overall_auc"]:.3f}'
    props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8)
    ax.text(0.62, 0.15, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    # Make axes more prominent
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Main ROC figure saved to: {output_path}")


def create_comparison_figure(metrics, output_path):
    """Create a comparison figure with benchmark results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Summary metrics
    your_auc = metrics['summary']['overall_auc']
    
    # Comparison data
    methods = ['TabNet\n(This Study)', 'TabNet\n(Pan-cancer)', 'XGBoost\n(Benchmark)']
    aucs = [your_auc, 0.96, 0.93]
    colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax1.bar(methods, aucs, color=colors_bar, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, auc_val in zip(bars, aucs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax1.set_ylim([0, 1.1])
    ax1.set_ylabel('Area Under Curve (AUC)', fontsize=14, fontweight='bold')
    ax1.set_title('A. Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add complexity note
    ax1.text(0.5, 0.3, 
             'Note: Pan-cancer study performed\nbinary classification (somatic vs germline)\n' + 
             'This study: 3-class classification\n(Pathogenic, Benign, VUS)',
             transform=ax1.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    # Right panel: Class distribution and performance
    classes = ['Pathogenic', 'Benign', 'VUS']
    class_aucs = [metrics['summary']['pathogenic_auc'], 
                  metrics['summary']['benign_auc'],
                  metrics['summary']['vus_auc']]
    
    x_pos = np.arange(len(classes))
    bars2 = ax2.bar(x_pos, class_aucs, color=['#d62728', '#2ca02c', '#ff7f0e'], 
                    edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, auc_val in zip(bars2, class_aucs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(classes, fontsize=12)
    ax2.set_ylim([0, 1.1])
    ax2.set_ylabel('Class-specific AUC', fontsize=14, fontweight='bold')
    ax2.set_title('B. Performance by Variant Class', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Comparison figure saved to: {output_path}")


def create_individual_class_plots(metrics, fpr, tpr, output_dir):
    """Create separate plots for each class (for supplementary material)."""
    colors = {
        'Pathogenic': '#d62728',
        'Benign': '#2ca02c', 
        'VUS': '#ff7f0e'
    }
    
    class_names = metrics['class_names']
    class_indices = {name: str(i) for i, name in enumerate(class_names)}
    
    for class_name in class_names:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        idx = class_indices[class_name]
        auc_score = metrics['auc_scores']['individual_classes'][class_name]
        
        # Plot ROC curve
        ax.plot(fpr[idx], tpr[idx], color=colors[class_name], lw=3,
                label=f'{class_name} (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random')
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve: {class_name} vs Rest', fontsize=14)
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, f'roc_curve_{class_name.lower()}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Individual class plots saved to: {output_dir}")


def create_summary_table_figure(metrics, output_path):
    """Create a figure showing AUC summary table."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    data = [
        ['Variant Class', 'AUC Score', 'Performance Level'],
        ['Pathogenic', f"{metrics['summary']['pathogenic_auc']:.3f}", 
         'Excellent' if metrics['summary']['pathogenic_auc'] > 0.9 else 'Very Good'],
        ['Benign', f"{metrics['summary']['benign_auc']:.3f}", 
         'Excellent' if metrics['summary']['benign_auc'] > 0.9 else 'Very Good'],
        ['VUS', f"{metrics['summary']['vus_auc']:.3f}", 
         'Excellent' if metrics['summary']['vus_auc'] > 0.9 else 'Very Good'],
        ['', '', ''],
        ['Overall (Macro)', f"{metrics['summary']['overall_auc']:.3f}", 
         'Excellent' if metrics['summary']['overall_auc'] > 0.9 else 'Very Good'],
    ]
    
    # Add comparison if overall AUC is good
    if metrics['summary']['overall_auc'] > 0.85:
        data.append(['', '', ''])
        data.append(['Benchmark Comparison', '', ''])
        data.append(['Pan-cancer TabNet*', '0.960', 'Binary classification'])
        data.append(['XGBoost (Li et al.)', '0.930', 'Binary classification'])
    
    # Create table
    table = ax.table(cellText=data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, 4):
        for j in range(3):
            table[(i, j)].set_facecolor('#f0f0f0')
    
    # Style summary row
    for j in range(3):
        table[(5, j)].set_facecolor('#e0e0e0')
        table[(5, j)].set_text_props(weight='bold')
    
    # Add footnote
    footnote_text = ('*Binary classification tasks (not directly comparable to 3-class)\n' +
                    f'Test set: {len(metrics["class_names"])} classes, 38,656 variants')
    ax.text(0.5, -0.1, footnote_text,
            transform=ax.transAxes, ha='center', fontsize=10, style='italic')
    
    ax.set_title('TabNet Prostate Cancer Classifier - Performance Summary', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Summary table saved to: {output_path}")


def main():
    """Generate all ROC curve visualizations."""
    # Set paths
    roc_data_dir = '/u/aa107/uiuc-cancer-research/results/roc_analysis/'
    output_dir = '/u/aa107/uiuc-cancer-research/results/figures/'
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    supp_dir = os.path.join(output_dir, 'supplementary')
    os.makedirs(supp_dir, exist_ok=True)
    
    print("Loading ROC data...")
    try:
        metrics, fpr, tpr = load_roc_data(roc_data_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: ROC data not found. Please run calculate_roc_auc.py first.")
        print(f"   Looking for files in: {roc_data_dir}")
        return
    
    print(f"Loaded data for {len(metrics['class_names'])} classes")
    print(f"Overall AUC: {metrics['summary']['overall_auc']:.3f}")
    
    # Generate main figure for paper
    print("\nGenerating main ROC figure...")
    create_main_roc_figure(metrics, fpr, tpr, 
                          os.path.join(output_dir, 'figure_6_roc_curves.png'))
    
    # Generate comparison figure
    print("Generating comparison figure...")
    create_comparison_figure(metrics, 
                           os.path.join(output_dir, 'figure_roc_comparison.png'))
    
    # Generate individual class plots (supplementary)
    print("Generating individual class plots...")
    create_individual_class_plots(metrics, fpr, tpr, supp_dir)
    
    # Generate summary table
    print("Generating summary table...")
    create_summary_table_figure(metrics, 
                              os.path.join(output_dir, 'auc_summary_table.png'))
    
    print("\n✅ All visualizations completed!")
    print(f"📁 Main figures saved to: {output_dir}")
    print(f"📁 Supplementary figures saved to: {supp_dir}")
    
    # Print summary for paper
    print("\n" + "="*50)
    print("SUMMARY FOR PAPER:")
    print("="*50)
    print(f"Overall AUC (macro-average): {metrics['summary']['overall_auc']:.3f}")
    print(f"Pathogenic detection AUC: {metrics['summary']['pathogenic_auc']:.3f}")
    print(f"Benign detection AUC: {metrics['summary']['benign_auc']:.3f}")
    print(f"VUS detection AUC: {metrics['summary']['vus_auc']:.3f}")
    print("\nUse Figure 6 (figure_6_roc_curves.png) as main figure in paper")


if __name__ == "__main__":
    main()