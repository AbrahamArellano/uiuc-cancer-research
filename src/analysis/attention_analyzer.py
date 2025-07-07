#!/usr/bin/env python3
"""
TabNet Attention Pattern Analyzer
Analyzes attention patterns from extracted TabNet weights

Location: /u/aa107/uiuc-cancer-research/src/analysis/attention_analyzer.py
Author: PhD Research Student, University of Illinois

Purpose: Document objective patterns in TabNet attention without medical interpretation.
Focus on quantitative differences between pathogenic vs benign attention patterns.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class AttentionAnalyzer:
    """Analyzes TabNet attention patterns for interpretability"""
    
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
        self.feature_groups = None
        
        print("📊 TabNet Attention Pattern Analyzer Initialized")
        print(f"📁 Input: {self.attention_dir}")
        print(f"📁 Output: {self.patterns_dir}")

    def load_attention_data(self):
        """Load all attention weight files"""
        print("\n📋 LOADING ATTENTION DATA")
        print("-" * 30)
        
        if not os.path.exists(self.attention_dir):
            raise FileNotFoundError(f"Attention directory not found: {self.attention_dir}")
        
        # Load summary first
        summary_file = os.path.join(self.attention_dir, "attention_summary.csv")
        if not os.path.exists(summary_file):
            raise FileNotFoundError(f"Attention summary not found: {summary_file}")
        
        summary_df = pd.read_csv(summary_file)
        print(f"✅ Loaded summary: {len(summary_df)} variants")
        
        # Load individual attention files
        attention_files = [f for f in os.listdir(self.attention_dir) if f.endswith('_attention.csv')]
        print(f"🔍 Found {len(attention_files)} attention files")
        
        for file in attention_files:
            variant_id = file.replace('_attention.csv', '')
            file_path = os.path.join(self.attention_dir, file)
            
            try:
                attention_df = pd.read_csv(file_path)
                self.attention_data[variant_id] = attention_df
                print(f"   ✅ {variant_id}: {len(attention_df)} features")
            except Exception as e:
                print(f"   ❌ Error loading {variant_id}: {e}")
        
        return summary_df

    def identify_feature_groups(self):
        """Identify feature groups from the data"""
        print("\n🏷️  IDENTIFYING FEATURE GROUPS")
        print("-" * 30)
        
        # Get a sample attention file to analyze features
        sample_data = next(iter(self.attention_data.values()))
        all_features = sample_data['feature'].tolist()
        
        # Define feature groups based on naming patterns and known tiers
        self.feature_groups = {
            'tier1_vep_corrected': [],
            'tier2_core_vep': [],
            'tier3_alphamissense': [],
            'tier4_population': [],
            'tier5_functional': [],
            'tier6_clinical': [],
            'tier7_variant_props': [],
            'tier8_prostate_biology': []
        }
        
        # Classify features based on patterns
        for feature in all_features:
            feature_lower = feature.lower()
            
            # Tier 1: VEP-corrected (high priority)
            if any(term in feature_lower for term in ['consequence', 'domains', 'pubmed', 'var_synonyms']):
                self.feature_groups['tier1_vep_corrected'].append(feature)
            
            # Tier 2: Core VEP
            elif any(term in feature_lower for term in ['symbol', 'biotype', 'canonical', 'pick', 'hgvs', 'protein_position', 'amino_acids', 'existing_variation', 'variant_class']):
                self.feature_groups['tier2_core_vep'].append(feature)
            
            # Tier 3: AlphaMissense
            elif 'alphamissense' in feature_lower:
                self.feature_groups['tier3_alphamissense'].append(feature)
            
            # Tier 4: Population frequency
            elif any(term in feature_lower for term in ['af', 'gnomad', '1kg']):
                self.feature_groups['tier4_population'].append(feature)
            
            # Tier 5: Functional predictions
            elif any(term in feature_lower for term in ['impact', 'sift', 'polyphen']):
                self.feature_groups['tier5_functional'].append(feature)
            
            # Tier 6: Clinical context
            elif any(term in feature_lower for term in ['somatic', 'pheno', 'exon', 'intron', 'ccds']):
                self.feature_groups['tier6_clinical'].append(feature)
            
            # Tier 7: Variant properties
            elif any(term in feature_lower for term in ['ref_length', 'alt_length', 'variant_size', 'is_indel', 'is_snv', 'is_lof', 'is_missense', 'is_synonymous']):
                self.feature_groups['tier7_variant_props'].append(feature)
            
            # Tier 8: Prostate biology
            elif any(term in feature_lower for term in ['important_gene', 'dna_repair', 'mismatch_repair', 'hormone']):
                self.feature_groups['tier8_prostate_biology'].append(feature)
            
            # Unclassified
            else:
                # Add to most likely group based on patterns
                if '_af' in feature_lower or 'frequency' in feature_lower:
                    self.feature_groups['tier4_population'].append(feature)
                else:
                    self.feature_groups['tier2_core_vep'].append(feature)  # Default to core VEP
        
        print("📊 Feature group classification:")
        for group, features in self.feature_groups.items():
            if features:
                print(f"   {group}: {len(features)} features")
        
        return self.feature_groups

    def analyze_category_differences(self, summary_df):
        """Analyze attention differences between pathogenic and benign variants"""
        print("\n🔍 ANALYZING PATHOGENIC VS BENIGN DIFFERENCES")
        print("-" * 50)
        
        # Separate variants by category
        pathogenic_variants = summary_df[summary_df['category'] == 'pathogenic']['variant_id'].tolist()
        benign_variants = summary_df[summary_df['category'] == 'benign']['variant_id'].tolist()
        
        print(f"📊 Comparing {len(pathogenic_variants)} pathogenic vs {len(benign_variants)} benign variants")
        
        # Analyze top features for each category
        pathogenic_features = defaultdict(list)
        benign_features = defaultdict(list)
        
        # Collect top 5 features for each variant
        for variant_id in pathogenic_variants:
            if variant_id in self.attention_data:
                attention_df = self.attention_data[variant_id]
                top_5 = attention_df.head(5)
                for _, row in top_5.iterrows():
                    pathogenic_features[row['feature']].append(row['global_importance'])
        
        for variant_id in benign_variants:
            if variant_id in self.attention_data:
                attention_df = self.attention_data[variant_id]
                top_5 = attention_df.head(5)
                for _, row in top_5.iterrows():
                    benign_features[row['feature']].append(row['global_importance'])
        
        # Calculate average importance by category
        pathogenic_avg = {feature: np.mean(importances) for feature, importances in pathogenic_features.items()}
        benign_avg = {feature: np.mean(importances) for feature, importances in benign_features.items()}
        
        # Find top features for each category
        pathogenic_top = sorted(pathogenic_avg.items(), key=lambda x: x[1], reverse=True)[:10]
        benign_top = sorted(benign_avg.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("\n🔴 Top 10 features for PATHOGENIC variants:")
        for i, (feature, importance) in enumerate(pathogenic_top, 1):
            print(f"   {i:2d}. {feature}: {importance:.4f}")
        
        print("\n🟢 Top 10 features for BENIGN variants:")
        for i, (feature, importance) in enumerate(benign_top, 1):
            print(f"   {i:2d}. {feature}: {importance:.4f}")
        
        # Find distinctive features
        distinctive_features = self._find_distinctive_features(pathogenic_avg, benign_avg)
        
        return {
            'pathogenic_top': pathogenic_top,
            'benign_top': benign_top,
            'distinctive_features': distinctive_features,
            'pathogenic_features': pathogenic_features,
            'benign_features': benign_features
        }

    def _find_distinctive_features(self, pathogenic_avg, benign_avg):
        """Find features that are distinctively important for one category"""
        print("\n🎯 DISTINCTIVE FEATURES ANALYSIS")
        print("-" * 35)
        
        distinctive = []
        all_features = set(pathogenic_avg.keys()) | set(benign_avg.keys())
        
        for feature in all_features:
            path_imp = pathogenic_avg.get(feature, 0)
            benign_imp = benign_avg.get(feature, 0)
            
            # Calculate difference ratio
            if max(path_imp, benign_imp) > 0:
                if path_imp > benign_imp:
                    ratio = path_imp / (benign_imp + 1e-8)
                    if ratio > 1.5:  # At least 50% more important
                        distinctive.append({
                            'feature': feature,
                            'category': 'pathogenic',
                            'path_importance': path_imp,
                            'benign_importance': benign_imp,
                            'ratio': ratio
                        })
                else:
                    ratio = benign_imp / (path_imp + 1e-8)
                    if ratio > 1.5:
                        distinctive.append({
                            'feature': feature,
                            'category': 'benign',
                            'path_importance': path_imp,
                            'benign_importance': benign_imp,
                            'ratio': ratio
                        })
        
        # Sort by ratio
        distinctive.sort(key=lambda x: x['ratio'], reverse=True)
        
        print("📋 Features distinctive to each category:")
        for item in distinctive[:10]:  # Top 10
            cat = item['category'].upper()
            feature = item['feature']
            ratio = item['ratio']
            print(f"   {cat}: {feature} (ratio: {ratio:.2f})")
        
        return distinctive

    def analyze_feature_group_patterns(self, summary_df):
        """Analyze attention patterns by feature groups"""
        print("\n🏷️  FEATURE GROUP ATTENTION ANALYSIS")
        print("-" * 40)
        
        group_importance = defaultdict(lambda: defaultdict(list))
        
        # Collect importance by group for each variant
        for variant_id, attention_df in self.attention_data.items():
            # Get variant category
            variant_info = summary_df[summary_df['variant_id'] == variant_id]
            if len(variant_info) == 0:
                continue
            category = variant_info.iloc[0]['category']
            
            # Group features by tier
            for group_name, features in self.feature_groups.items():
                if features:
                    group_data = attention_df[attention_df['feature'].isin(features)]
                    if len(group_data) > 0:
                        avg_importance = group_data['global_importance'].mean()
                        group_importance[group_name][category].append(avg_importance)
        
        # Calculate group statistics
        group_stats = {}
        for group_name, category_data in group_importance.items():
            pathogenic_imp = category_data.get('pathogenic', [])
            benign_imp = category_data.get('benign', [])
            
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
            print(f"      Pathogenic: {path_mean:.4f} ± {stats['pathogenic_std']:.4f}")
            print(f"      Benign:     {benign_mean:.4f} ± {stats['benign_std']:.4f}")
            print(f"      Difference: {diff:+.4f} ({direction})")
            print()
        
        return group_stats

    def analyze_decision_step_patterns(self):
        """Analyze how attention changes across TabNet's decision steps"""
        print("\n🔄 DECISION STEP ATTENTION ANALYSIS")
        print("-" * 40)
        
        # Identify how many decision steps we have
        sample_data = next(iter(self.attention_data.values()))
        step_columns = [col for col in sample_data.columns if col.startswith('step_') and col.endswith('_attention')]
        n_steps = len(step_columns)
        
        print(f"📊 Analyzing {n_steps} decision steps")
        
        step_patterns = {}
        
        for variant_id, attention_df in self.attention_data.items():
            variant_steps = {}
            
            for step_col in step_columns:
                step_num = int(step_col.split('_')[1])
                
                # Get top 5 features for this step
                step_data = attention_df.nlargest(5, step_col)
                top_features = step_data['feature'].tolist()
                top_attention = step_data[step_col].tolist()
                
                variant_steps[step_num] = {
                    'top_features': top_features,
                    'top_attention': top_attention
                }
            
            step_patterns[variant_id] = variant_steps
        
        # Analyze step consistency
        print("\n📋 Step-wise attention consistency:")
        
        for variant_id, steps in step_patterns.items():
            print(f"\n   {variant_id}:")
            
            # Check if same features appear across steps
            all_step_features = []
            for step_num, step_data in steps.items():
                all_step_features.extend(step_data['top_features'])
            
            from collections import Counter
            feature_counts = Counter(all_step_features)
            consistent_features = [f for f, count in feature_counts.items() if count >= n_steps//2]
            
            print(f"      Consistent features (appear in ≥{n_steps//2} steps): {len(consistent_features)}")
            for feature in consistent_features[:3]:  # Top 3
                print(f"         - {feature}")
        
        return step_patterns

    def create_attention_visualizations(self, category_analysis, group_stats):
        """Create visualization plots for attention patterns"""
        print("\n📊 CREATING ATTENTION VISUALIZATIONS")
        print("-" * 40)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Top features comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pathogenic top features
        path_features = [item[0] for item in category_analysis['pathogenic_top'][:8]]
        path_importance = [item[1] for item in category_analysis['pathogenic_top'][:8]]
        
        ax1.barh(range(len(path_features)), path_importance, color='red', alpha=0.7)
        ax1.set_yticks(range(len(path_features)))
        ax1.set_yticklabels([f.replace('_', ' ') for f in path_features], fontsize=10)
        ax1.set_xlabel('Average Attention Weight')
        ax1.set_title('Top Features: Pathogenic Variants')
        ax1.invert_yaxis()
        
        # Benign top features
        benign_features = [item[0] for item in category_analysis['benign_top'][:8]]
        benign_importance = [item[1] for item in category_analysis['benign_top'][:8]]
        
        ax2.barh(range(len(benign_features)), benign_importance, color='green', alpha=0.7)
        ax2.set_yticks(range(len(benign_features)))
        ax2.set_yticklabels([f.replace('_', ' ') for f in benign_features], fontsize=10)
        ax2.set_xlabel('Average Attention Weight')
        ax2.set_title('Top Features: Benign Variants')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plot1_path = os.path.join(self.patterns_dir, "top_features_comparison.png")
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: top_features_comparison.png")
        
        # 2. Feature group comparison plot
        if group_stats:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            groups = list(group_stats.keys())
            path_means = [group_stats[g]['pathogenic_mean'] for g in groups]
            benign_means = [group_stats[g]['benign_mean'] for g in groups]
            
            x = np.arange(len(groups))
            width = 0.35
            
            ax.bar(x - width/2, path_means, width, label='Pathogenic', color='red', alpha=0.7)
            ax.bar(x + width/2, benign_means, width, label='Benign', color='green', alpha=0.7)
            
            ax.set_xlabel('Feature Groups')
            ax.set_ylabel('Average Attention Weight')
            ax.set_title('Attention by Feature Group: Pathogenic vs Benign')
            ax.set_xticks(x)
            ax.set_xticklabels([g.replace('tier', 'T').replace('_', ' ') for g in groups], rotation=45)
            ax.legend()
            
            plt.tight_layout()
            plot2_path = os.path.join(self.patterns_dir, "feature_group_comparison.png")
            plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: feature_group_comparison.png")
        
        return [plot1_path, plot2_path] if group_stats else [plot1_path]

    def save_pattern_analysis(self, category_analysis, group_stats, step_patterns):
        """Save detailed pattern analysis results"""
        print("\n💾 SAVING PATTERN ANALYSIS")
        print("-" * 25)
        
        # Save category differences
        category_file = os.path.join(self.patterns_dir, "pathogenic_vs_benign_patterns.txt")
        with open(category_file, 'w') as f:
            f.write("TABNET ATTENTION PATTERNS: PATHOGENIC VS BENIGN\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("TOP FEATURES FOR PATHOGENIC VARIANTS:\n")
            f.write("-" * 40 + "\n")
            for i, (feature, importance) in enumerate(category_analysis['pathogenic_top'], 1):
                f.write(f"{i:2d}. {feature}: {importance:.4f}\n")
            
            f.write("\nTOP FEATURES FOR BENIGN VARIANTS:\n")
            f.write("-" * 40 + "\n")
            for i, (feature, importance) in enumerate(category_analysis['benign_top'], 1):
                f.write(f"{i:2d}. {feature}: {importance:.4f}\n")
            
            f.write("\nDISTINCTIVE FEATURES:\n")
            f.write("-" * 20 + "\n")
            for item in category_analysis['distinctive_features'][:10]:
                cat = item['category'].upper()
                feature = item['feature']
                ratio = item['ratio']
                f.write(f"{cat}: {feature} (ratio: {ratio:.2f})\n")
        
        print(f"   ✅ Category patterns: pathogenic_vs_benign_patterns.txt")
        
        # Save feature group analysis
        if group_stats:
            group_file = os.path.join(self.patterns_dir, "feature_group_analysis.txt")
            with open(group_file, 'w') as f:
                f.write("FEATURE GROUP ATTENTION ANALYSIS\n")
                f.write("=" * 40 + "\n\n")
                
                f.write("Average attention weights by feature group:\n\n")
                
                for group, stats in sorted(group_stats.items(), key=lambda x: abs(x[1]['difference']), reverse=True):
                    f.write(f"{group.upper()}:\n")
                    f.write(f"   Pathogenic: {stats['pathogenic_mean']:.4f} ± {stats['pathogenic_std']:.4f}\n")
                    f.write(f"   Benign:     {stats['benign_mean']:.4f} ± {stats['benign_std']:.4f}\n")
                    f.write(f"   Difference: {stats['difference']:+.4f}\n\n")
            
            print(f"   ✅ Group analysis: feature_group_analysis.txt")
        
        # Save step patterns summary
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
        
        return category_file, group_file, step_file

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
        summary_df = analyzer.load_attention_data()
        
        if not analyzer.attention_data:
            print("❌ No attention data found")
            return False
        
        # Identify feature groups
        analyzer.identify_feature_groups()
        
        # Analyze pathogenic vs benign differences
        category_analysis = analyzer.analyze_category_differences(summary_df)
        
        # Analyze feature group patterns
        group_stats = analyzer.analyze_feature_group_patterns(summary_df)
        
        # Analyze decision step patterns
        step_patterns = analyzer.analyze_decision_step_patterns()
        
        # Create visualizations
        plot_files = analyzer.create_attention_visualizations(category_analysis, group_stats)
        
        # Save detailed analysis
        analysis_files = analyzer.save_pattern_analysis(category_analysis, group_stats, step_patterns)
        
        print(f"\n🎉 ATTENTION PATTERN ANALYSIS COMPLETED!")
        print("=" * 50)
        print(f"✅ Analyzed {len(analyzer.attention_data)} variants")
        print(f"📁 Results saved to: {analyzer.patterns_dir}")
        print(f"📊 Visualizations: {len(plot_files)} plots")
        print(f"📋 Analysis files: {len(analysis_files)} reports")
        
        print(f"\n🎯 Key Findings Preview:")
        print(f"   - Feature groups analyzed: {len(analyzer.feature_groups)}")
        print(f"   - Distinctive features found: {len(category_analysis['distinctive_features'])}")
        print(f"   - Decision steps examined: {len(step_patterns)}")
        
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