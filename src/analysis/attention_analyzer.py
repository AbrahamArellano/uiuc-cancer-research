#!/usr/bin/env python3
"""
TabNet Attention Pattern Analyzer - UPDATED VERSION
Analyzes attention patterns from TabNet model for interpretability
Updated to match 8-tier feature structure and handle low importance values

Location: /u/aa107/uiuc-cancer-research/src/analysis/attention_analyzer.py
Author: PhD Research Student, University of Illinois
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AttentionAnalyzer:
    """Analyzes TabNet attention patterns for clinical interpretability"""
    
    def __init__(self, analysis_dir=None):
        """Initialize attention analyzer"""
        if analysis_dir is None:
            self.analysis_dir = "/u/aa107/uiuc-cancer-research/results/attention_analysis"
        else:
            self.analysis_dir = analysis_dir
        
        self.attention_dir = os.path.join(self.analysis_dir, "attention_weights")
        self.patterns_dir = os.path.join(self.analysis_dir, "pattern_analysis")
        
        # Create output directory
        os.makedirs(self.patterns_dir, exist_ok=True)
        
        self.attention_data = {}
        self.summary_df = None
        
        # Define 8-tier feature groups matching training
        self.feature_groups = {
            'tier1_vep_corrected': ['Consequence', 'DOMAINS', 'PUBMED', 'VAR_SYNONYMS'],
            'tier2_core_vep': ['SYMBOL', 'Gene', 'Feature_type', 'Feature', 'BIOTYPE', 
                               'HGVSc', 'HGVSp', 'Protein_position', 'Amino_acids', 'Existing_variation'],
            'tier3_alphamissense': ['alphamissense_pathogenicity', 'alphamissense_class'],
            'tier4_population': ['AF', 'AFR_AF', 'AMR_AF', 'EAS_AF', 'EUR_AF', 'SAS_AF',
                                'gnomADe_AF', 'gnomADe_AFR_AF', 'gnomADe_AMR_AF', 'gnomADe_ASJ_AF',
                                'gnomADe_EAS_AF', 'gnomADe_FIN_AF', 'gnomADe_MID_AF', 'gnomADe_NFE_AF',
                                'gnomADe_REMAINING_AF', 'gnomADe_SAS_AF', 'af_1kg'],
            'tier5_functional': ['IMPACT', 'sift_score', 'polyphen_score', 'SIFT', 'PolyPhen', 'impact_score'],
            'tier6_clinical': ['SOMATIC', 'PHENO', 'EXON', 'INTRON', 'CCDS'],
            'tier7_variant_props': ['ref_length', 'alt_length', 'variant_size', 'is_indel', 
                                    'is_snv', 'is_lof', 'is_missense', 'is_synonymous'],
            'tier8_prostate_biology': ['is_important_gene', 'dna_repair_pathway', 
                                      'mismatch_repair_pathway', 'hormone_pathway']
        }
        
        # Create reverse mapping for quick lookup
        self.feature_to_group = {}
        for group, features in self.feature_groups.items():
            for feature in features:
                self.feature_to_group[feature] = group
        
        print("📊 TabNet Attention Pattern Analyzer Initialized")
        print(f"📁 Input: {self.attention_dir}")
        print(f"📁 Output: {self.patterns_dir}")

    def load_attention_data(self):
        """Load all attention weight files"""
        print("\n📋 LOADING ATTENTION DATA")
        print("-" * 30)
        
        # Load summary first
        summary_file = os.path.join(self.attention_dir, "attention_summary.csv")
        if os.path.exists(summary_file):
            self.summary_df = pd.read_csv(summary_file)
            print(f"✅ Loaded summary: {len(self.summary_df)} variants")
        
        # Load individual attention files
        attention_files = [f for f in os.listdir(self.attention_dir) 
                          if f.endswith('_attention.csv') and f != 'attention_summary.csv']
        
        print(f"🔍 Found {len(attention_files)} attention files")
        
        for filename in sorted(attention_files):
            variant_id = filename.replace('_attention.csv', '')
            filepath = os.path.join(self.attention_dir, filename)
            
            try:
                df = pd.read_csv(filepath)
                self.attention_data[variant_id] = df
                print(f"   ✅ {variant_id}: {len(df)} features")
            except Exception as e:
                print(f"   ❌ {variant_id}: Failed to load - {e}")
        
        return len(self.attention_data) > 0

    def identify_feature_groups(self):
        """Identify feature groups from loaded data"""
        print("\n🏷️  IDENTIFYING FEATURE GROUPS")
        print("-" * 30)
        
        if not self.attention_data:
            print("❌ No attention data loaded")
            return {}
        
        # Get all unique features from attention data
        all_features = set()
        for variant_id, attention_df in self.attention_data.items():
            all_features.update(attention_df['feature'].tolist())
        
        # Categorize features
        found_groups = {}
        uncategorized = []
        
        for feature in all_features:
            group = self.feature_to_group.get(feature, None)
            if group:
                if group not in found_groups:
                    found_groups[group] = []
                found_groups[group].append(feature)
            else:
                uncategorized.append(feature)
        
        print("📊 Feature group classification:")
        for group in self.feature_groups.keys():
            if group in found_groups:
                print(f"   {group}: {len(found_groups[group])} features")
            else:
                print(f"   {group}: 0 features (not found in data)")
        
        if uncategorized:
            print(f"\n⚠️  Uncategorized features: {len(uncategorized)}")
            for feat in uncategorized[:5]:
                print(f"     - {feat}")
        
        return found_groups

    def analyze_pathogenic_vs_benign_patterns(self):
        """Analyze attention differences between pathogenic and benign variants"""
        print("\n🔍 ANALYZING PATHOGENIC VS BENIGN DIFFERENCES")
        print("-" * 50)
        
        if not self.summary_df is not None:
            print("❌ No summary data available")
            return {}
        
        # Get available classifications
        available_classifications = self.summary_df['classification'].unique()
        print(f"📋 Available classifications: {available_classifications}")
        
        # Separate pathogenic and benign variants
        pathogenic_df = self.summary_df[self.summary_df['classification'] == 'pathogenic']
        benign_df = self.summary_df[self.summary_df['classification'] == 'benign']
        
        print(f"📊 Comparing {len(pathogenic_df)} pathogenic vs {len(benign_df)} benign variants")
        
        if len(pathogenic_df) == 0 or len(benign_df) == 0:
            print("❌ Insufficient data for comparison")
            return {'insufficient_data': True}
        
        # Collect attention scores by category
        pathogenic_attention = {}
        benign_attention = {}
        
        # Process pathogenic variants
        for _, row in pathogenic_df.iterrows():
            variant_id = row['variant_id']
            if variant_id in self.attention_data:
                attention_df = self.attention_data[variant_id]
                for _, att_row in attention_df.iterrows():
                    feature = att_row['feature']
                    importance = att_row['global_importance']
                    if feature not in pathogenic_attention:
                        pathogenic_attention[feature] = []
                    pathogenic_attention[feature].append(importance)
        
        # Process benign variants
        for _, row in benign_df.iterrows():
            variant_id = row['variant_id']
            if variant_id in self.attention_data:
                attention_df = self.attention_data[variant_id]
                for _, att_row in attention_df.iterrows():
                    feature = att_row['feature']
                    importance = att_row['global_importance']
                    if feature not in benign_attention:
                        benign_attention[feature] = []
                    benign_attention[feature].append(importance)
        
        # Calculate average attention for each feature
        pathogenic_avg = {feat: np.mean(scores) for feat, scores in pathogenic_attention.items()}
        benign_avg = {feat: np.mean(scores) for feat, scores in benign_attention.items()}
        
        # Get top features for each category
        pathogenic_top = sorted(pathogenic_avg.items(), key=lambda x: x[1], reverse=True)[:10]
        benign_top = sorted(benign_avg.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\n🔴 Top 10 features for PATHOGENIC variants:")
        for i, (feature, importance) in enumerate(pathogenic_top, 1):
            # Use scientific notation for very small values
            if importance < 0.0001:
                print(f"    {i}. {feature}: {importance:.4e}")
            else:
                print(f"    {i}. {feature}: {importance:.4f}")
        
        print(f"\n🟢 Top 10 features for BENIGN variants:")
        for i, (feature, importance) in enumerate(benign_top, 1):
            if importance < 0.0001:
                print(f"    {i}. {feature}: {importance:.4e}")
            else:
                print(f"    {i}. {feature}: {importance:.4f}")
        
        # Find distinctive features
        all_features = set(pathogenic_avg.keys()) | set(benign_avg.keys())
        distinctive_features = []
        
        for feature in all_features:
            path_score = pathogenic_avg.get(feature, 0)
            benign_score = benign_avg.get(feature, 0)
            
            # Avoid division by zero
            if benign_score > 0:
                ratio = path_score / benign_score
            elif path_score > 0:
                ratio = float('inf')
            else:
                ratio = 1.0
            
            # Consider features distinctive if ratio > 1.5 or < 0.67
            if ratio > 1.5 or ratio < 0.67:
                distinctive_features.append({
                    'feature': feature,
                    'pathogenic_avg': path_score,
                    'benign_avg': benign_score,
                    'ratio': ratio,
                    'category': 'pathogenic' if ratio > 1 else 'benign'
                })
        
        # Sort by absolute difference from 1
        distinctive_features.sort(key=lambda x: abs(x['ratio'] - 1), reverse=True)
        
        print(f"\n📋 Features distinctive to each category:")
        for i, feat_info in enumerate(distinctive_features[:10]):
            feature = feat_info['feature']
            ratio = feat_info['ratio']
            category = feat_info['category']
            print(f"   🏷️  {category.upper()}: {feature} (ratio: {ratio:.2f})")
        
        return {
            'pathogenic_top': pathogenic_top,
            'benign_top': benign_top,
            'distinctive_features': distinctive_features
        }

    def analyze_feature_group_attention(self):
        """Analyze attention patterns by feature groups"""
        print("\n🏷️  FEATURE GROUP ATTENTION ANALYSIS")
        print("-" * 40)
        
        if not self.summary_df is not None:
            print("❌ No summary data available")
            return {}
        
        # Separate by classification
        pathogenic_df = self.summary_df[self.summary_df['classification'] == 'pathogenic']
        benign_df = self.summary_df[self.summary_df['classification'] == 'benign']
        
        # Collect attention by feature group
        group_stats = {}
        
        for group_name in self.feature_groups.keys():
            pathogenic_imp = []
            benign_imp = []
            
            # Collect pathogenic importance
            for _, row in pathogenic_df.iterrows():
                variant_id = row['variant_id']
                if variant_id in self.attention_data:
                    attention_df = self.attention_data[variant_id]
                    group_features = attention_df[attention_df['feature_group'] == group_name]
                    if not group_features.empty:
                        pathogenic_imp.extend(group_features['global_importance'].tolist())
            
            # Collect benign importance
            for _, row in benign_df.iterrows():
                variant_id = row['variant_id']
                if variant_id in self.attention_data:
                    attention_df = self.attention_data[variant_id]
                    group_features = attention_df[attention_df['feature_group'] == group_name]
                    if not group_features.empty:
                        benign_imp.extend(group_features['global_importance'].tolist())
            
            if pathogenic_imp and benign_imp:
                group_stats[group_name] = {
                    'pathogenic_mean': np.mean(pathogenic_imp),
                    'pathogenic_std': np.std(pathogenic_imp),
                    'benign_mean': np.mean(benign_imp),
                    'benign_std': np.std(benign_imp),
                    'difference': np.mean(pathogenic_imp) - np.mean(benign_imp)
                }
        
        print("📊 Average attention by feature group:")
        for group, stats in sorted(group_stats.items(), key=lambda x: abs(x[1]['difference']), reverse=True):
            path_mean = stats['pathogenic_mean']
            benign_mean = stats['benign_mean']
            diff = stats['difference']
            direction = "↑ Pathogenic" if diff > 0 else "↓ Benign"
            
            print(f"   {group}:")
            if path_mean < 0.0001:
                print(f"      Pathogenic: {path_mean:.4e} ± {stats['pathogenic_std']:.4e}")
            else:
                print(f"      Pathogenic: {path_mean:.4f} ± {stats['pathogenic_std']:.4f}")
            
            if benign_mean < 0.0001:
                print(f"      Benign:     {benign_mean:.4e} ± {stats['benign_std']:.4e}")
            else:
                print(f"      Benign:     {benign_mean:.4f} ± {stats['benign_std']:.4f}")
            
            print(f"      Difference: {diff:+.4f} ({direction})")
            print()
        
        return group_stats

    def analyze_decision_step_patterns(self):
        """Analyze how attention changes across TabNet's decision steps"""
        print("\n🔄 DECISION STEP ATTENTION ANALYSIS")
        print("-" * 40)
        
        if not self.attention_data:
            print("❌ No attention data loaded")
            return {}
        
        # Identify how many decision steps we have
        sample_data = next(iter(self.attention_data.values()))
        step_columns = [col for col in sample_data.columns if col.startswith('step_') and col.endswith('_attention')]
        n_steps = len(step_columns)
        
        print(f"📊 Analyzing {n_steps} decision steps")
        
        step_patterns = {}
        
        for variant_id, attention_df in self.attention_data.items():
            variant_steps = {}
            
            for step_idx, step_col in enumerate(step_columns):
                # Get top features for this step
                step_attention = attention_df[['feature', step_col]].copy()
                step_attention = step_attention.sort_values(step_col, ascending=False)
                
                top_features = step_attention.head(3)['feature'].tolist()
                variant_steps[step_idx + 1] = {
                    'top_features': top_features,
                    'total_attention': step_attention[step_col].sum()
                }
            
            step_patterns[variant_id] = variant_steps
        
        # Analyze consistency across steps
        print(f"\n📋 Step-wise attention consistency:")
        
        consistency_scores = []
        for variant_id, steps in list(step_patterns.items())[:5]:  # Show first 5 variants
            print(f"\n   {variant_id}:")
            
            # Check which features appear consistently across steps
            all_top_features = []
            for step_num, step_data in steps.items():
                all_top_features.extend(step_data['top_features'])
            
            # Count feature occurrences
            feature_counts = {}
            for feat in all_top_features:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
            
            # Features appearing in 3+ steps are consistent
            consistent_features = [feat for feat, count in feature_counts.items() if count >= 3]
            
            print(f"      Consistent features (appear in ≥3 steps): {len(consistent_features)}")
            if consistent_features:
                print(f"         - {', '.join(consistent_features[:3])}")
            
            consistency_scores.append(len(consistent_features))
        
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
        print(f"\n📊 Average consistency score: {avg_consistency:.1f} features")
        
        return step_patterns

    def create_visualizations(self):
        """Create attention pattern visualizations"""
        print("\n📊 CREATING ATTENTION VISUALIZATIONS")
        print("-" * 40)
        
        plot_files = []
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Top features comparison plot
        if self.summary_df is not None:
            pathogenic_df = self.summary_df[self.summary_df['classification'] == 'pathogenic']
            benign_df = self.summary_df[self.summary_df['classification'] == 'benign']
            
            if len(pathogenic_df) > 0 and len(benign_df) > 0:
                # Collect average attention for top features
                all_features = {}
                
                for classification, df in [('Pathogenic', pathogenic_df), ('Benign', benign_df)]:
                    for _, row in df.iterrows():
                        variant_id = row['variant_id']
                        if variant_id in self.attention_data:
                            att_df = self.attention_data[variant_id]
                            for _, att_row in att_df.head(10).iterrows():
                                feature = att_row['feature']
                                importance = att_row['global_importance']
                                
                                if feature not in all_features:
                                    all_features[feature] = {'Pathogenic': [], 'Benign': []}
                                all_features[feature][classification].append(importance)
                
                # Calculate averages
                feature_avgs = []
                for feature, values in all_features.items():
                    if values['Pathogenic'] and values['Benign']:
                        feature_avgs.append({
                            'Feature': feature,
                            'Pathogenic': np.mean(values['Pathogenic']),
                            'Benign': np.mean(values['Benign'])
                        })
                
                # Sort by total importance
                feature_avgs.sort(key=lambda x: x['Pathogenic'] + x['Benign'], reverse=True)
                
                # Create comparison plot
                if feature_avgs:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    top_features = feature_avgs[:15]
                    features = [f['Feature'] for f in top_features]
                    pathogenic_values = [f['Pathogenic'] for f in top_features]
                    benign_values = [f['Benign'] for f in top_features]
                    
                    x = np.arange(len(features))
                    width = 0.35
                    
                    bars1 = ax.bar(x - width/2, pathogenic_values, width, label='Pathogenic', alpha=0.8)
                    bars2 = ax.bar(x + width/2, benign_values, width, label='Benign', alpha=0.8)
                    
                    ax.set_xlabel('Features', fontsize=12)
                    ax.set_ylabel('Average Attention Weight', fontsize=12)
                    ax.set_title('Top 15 Features: Pathogenic vs Benign Attention Comparison', fontsize=14)
                    ax.set_xticks(x)
                    ax.set_xticklabels(features, rotation=45, ha='right')
                    ax.legend()
                    
                    plt.tight_layout()
                    plot_file = os.path.join(self.patterns_dir, "top_features_comparison.png")
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"   ✅ Saved: top_features_comparison.png")
                    plot_files.append(plot_file)
        
        # 2. Feature group comparison
        group_stats = self.analyze_feature_group_attention()
        if group_stats:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            groups = list(group_stats.keys())
            pathogenic_means = [group_stats[g]['pathogenic_mean'] for g in groups]
            benign_means = [group_stats[g]['benign_mean'] for g in groups]
            
            x = np.arange(len(groups))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, pathogenic_means, width, label='Pathogenic', alpha=0.8)
            bars2 = ax.bar(x + width/2, benign_means, width, label='Benign', alpha=0.8)
            
            ax.set_xlabel('Feature Groups', fontsize=12)
            ax.set_ylabel('Average Attention Weight', fontsize=12)
            ax.set_title('Feature Group Attention: Pathogenic vs Benign', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels([g.replace('tier', 'Tier ').replace('_', ' ').title() for g in groups], 
                              rotation=45, ha='right')
            ax.legend()
            
            # Use log scale if values are very small
            if min(pathogenic_means + benign_means) < 0.001:
                ax.set_yscale('log')
                ax.set_ylabel('Average Attention Weight (log scale)', fontsize=12)
            
            plt.tight_layout()
            plot_file = os.path.join(self.patterns_dir, "feature_group_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: feature_group_comparison.png")
            plot_files.append(plot_file)
        
        return plot_files

    def save_pattern_analysis(self, category_analysis, group_stats, step_patterns):
        """Save analysis results to text files"""
        print("\n💾 SAVING PATTERN ANALYSIS")
        print("-" * 25)
        
        analysis_files = []
        
        # Save category comparison
        if category_analysis and 'insufficient_data' not in category_analysis:
            category_file = os.path.join(self.patterns_dir, "pathogenic_vs_benign_patterns.txt")
            with open(category_file, 'w') as f:
                f.write("PATHOGENIC VS BENIGN ATTENTION PATTERNS\n")
                f.write("=" * 40 + "\n\n")
                
                f.write("TOP FEATURES FOR PATHOGENIC VARIANTS:\n")
                f.write("-" * 40 + "\n")
                for i, (feature, importance) in enumerate(category_analysis.get('pathogenic_top', []), 1):
                    if importance < 0.0001:
                        f.write(f"{i:2d}. {feature}: {importance:.4e}\n")
                    else:
                        f.write(f"{i:2d}. {feature}: {importance:.4f}\n")
                
                f.write("\nTOP FEATURES FOR BENIGN VARIANTS:\n")
                f.write("-" * 40 + "\n")
                for i, (feature, importance) in enumerate(category_analysis.get('benign_top', []), 1):
                    if importance < 0.0001:
                        f.write(f"{i:2d}. {feature}: {importance:.4e}\n")
                    else:
                        f.write(f"{i:2d}. {feature}: {importance:.4f}\n")
                
                f.write("\nDISTINCTIVE FEATURES:\n")
                f.write("-" * 20 + "\n")
                for item in category_analysis.get('distinctive_features', [])[:10]:
                    cat = item['category'].upper()
                    feature = item['feature']
                    ratio = item['ratio']
                    f.write(f"{cat}: {feature} (ratio: {ratio:.2f})\n")
            
            print(f"   ✅ Category patterns: pathogenic_vs_benign_patterns.txt")
            analysis_files.append(category_file)
        
        # Save feature group analysis
        if group_stats:
            group_file = os.path.join(self.patterns_dir, "feature_group_analysis.txt")
            with open(group_file, 'w') as f:
                f.write("FEATURE GROUP ATTENTION ANALYSIS\n")
                f.write("=" * 40 + "\n\n")
                
                f.write("Average attention weights by feature group:\n\n")
                
                for group, stats in sorted(group_stats.items(), key=lambda x: abs(x[1]['difference']), reverse=True):
                    f.write(f"{group.upper()}:\n")
                    
                    # Handle very small values
                    if stats['pathogenic_mean'] < 0.0001:
                        f.write(f"   Pathogenic: {stats['pathogenic_mean']:.4e} ± {stats['pathogenic_std']:.4e}\n")
                    else:
                        f.write(f"   Pathogenic: {stats['pathogenic_mean']:.4f} ± {stats['pathogenic_std']:.4f}\n")
                    
                    if stats['benign_mean'] < 0.0001:
                        f.write(f"   Benign:     {stats['benign_mean']:.4e} ± {stats['benign_std']:.4e}\n")
                    else:
                        f.write(f"   Benign:     {stats['benign_mean']:.4f} ± {stats['benign_std']:.4f}\n")
                    
                    f.write(f"   Difference: {stats['difference']:+.4f}\n\n")
            
            print(f"   ✅ Group analysis: feature_group_analysis.txt")
            analysis_files.append(group_file)
        
        # Save step patterns summary
        if step_patterns:
            step_file = os.path.join(self.patterns_dir, "decision_step_patterns.txt")
            with open(step_file, 'w') as f:
                f.write("TABNET DECISION STEP ATTENTION PATTERNS\n")
                f.write("=" * 45 + "\n\n")
                
                f.write("Summary of attention evolution across TabNet decision steps\n\n")
                
                for variant_id in list(step_patterns.keys())[:5]:  # First 5 variants as examples
                    f.write(f"{variant_id.upper()}:\n")
                    steps = step_patterns[variant_id]
                    
                    for step_num in sorted(steps.keys()):
                        step_data = steps[step_num]
                        f.write(f"   Step {step_num}: {', '.join(step_data['top_features'][:3])}\n")
                    f.write("\n")
            
            print(f"   ✅ Step patterns: decision_step_patterns.txt")
            analysis_files.append(step_file)
        
        return analysis_files

def main():
    """Main attention analysis pipeline"""
    print("📊 TABNET ATTENTION PATTERN ANALYSIS")
    print("=" * 50)
    print("Purpose: Analyze attention patterns without medical interpretation")
    print("Input: Extracted attention weights from attention_extractor.py")
    print("Output: Documented patterns for clinical expert review")
    print()
    
    try:
        # Initialize analyzer
        analyzer = AttentionAnalyzer()
        
        # Load attention data
        if not analyzer.load_attention_data():
            print("❌ Failed to load attention data")
            return False
        
        # Identify feature groups
        found_groups = analyzer.identify_feature_groups()
        
        # Analyze pathogenic vs benign patterns
        category_analysis = analyzer.analyze_pathogenic_vs_benign_patterns()
        
        # Analyze feature group attention
        group_stats = analyzer.analyze_feature_group_attention()
        
        # Analyze decision step patterns
        step_patterns = analyzer.analyze_decision_step_patterns()
        
        # Create visualizations
        plot_files = analyzer.create_visualizations()
        
        # Save analysis results
        analysis_files = analyzer.save_pattern_analysis(category_analysis, group_stats, step_patterns)
        
        print(f"\n🎉 ATTENTION PATTERN ANALYSIS COMPLETED!")
        print("=" * 50)
        print(f"✅ Analyzed {len(analyzer.attention_data)} variants")
        print(f"📁 Results saved to: {analyzer.patterns_dir}")
        print(f"📊 Visualizations: {len(plot_files)} plots")
        print(f"📋 Analysis files: {len(analysis_files)} reports")
        
        print(f"\n🎯 Key Findings Preview:")
        print(f"   - Feature groups analyzed: {len(found_groups)}")
        if category_analysis and 'distinctive_features' in category_analysis:
            print(f"   - Distinctive features found: {len(category_analysis['distinctive_features'])}")
        print(f"   - Decision steps examined: {len(analyzer.attention_data)}")
        
        print(f"\n🎯 Ready for next step:")
        print(f"   python src/analysis/results_generator.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Attention analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)