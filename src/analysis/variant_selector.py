#!/usr/bin/env python3
"""
Variant Selector for TabNet Attention Analysis
Selects representative pathogenic and benign variants for interpretability study

Location: /u/aa107/uiuc-cancer-research/src/analysis/variant_selector.py
Author: PhD Research Student, University of Illinois

Purpose: Select 5-10 pathogenic and 5-10 benign variants from ClinVar-labeled cases
for detailed TabNet attention analysis. No medical expertise required - uses existing labels.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime

class VariantSelector:
    """Selects representative variants for attention analysis"""
    
    def __init__(self, dataset_path=None, output_dir=None):
        """Initialize variant selector"""
        if dataset_path is None:
            self.dataset_path = "/u/aa107/uiuc-cancer-research/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv"
        else:
            self.dataset_path = dataset_path
            
        if output_dir is None:
            self.output_dir = "/u/aa107/uiuc-cancer-research/results/attention_analysis"
        else:
            self.output_dir = output_dir
            
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("🔍 Variant Selector Initialized")
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"📁 Output: {self.output_dir}")

    def load_dataset(self):
        """Load the clean dataset"""
        print("\n📁 LOADING DATASET")
        print("-" * 30)
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        try:
            df = pd.read_csv(self.dataset_path, low_memory=False)
            print(f"✅ Loaded {len(df):,} variants with {len(df.columns)} columns")
            return df
        except Exception as e:
            raise Exception(f"Error loading dataset: {e}")

    def analyze_clin_sig_distribution(self, df):
        """Analyze CLIN_SIG distribution to understand available labels"""
        print("\n📊 ANALYZING CLIN_SIG DISTRIBUTION")
        print("-" * 40)
        
        if 'CLIN_SIG' not in df.columns:
            print("❌ CLIN_SIG column not found in dataset")
            return None
        
        # Count CLIN_SIG values
        clin_sig_counts = df['CLIN_SIG'].value_counts()
        print("📋 CLIN_SIG Distribution:")
        for value, count in clin_sig_counts.head(10).items():
            print(f"   {value}: {count:,}")
        
        # Identify pathogenic and benign cases
        pathogenic_terms = ['pathogenic', 'likely_pathogenic', 'Pathogenic', 'Likely_pathogenic']
        benign_terms = ['benign', 'likely_benign', 'Benign', 'Likely_benign']
        
        pathogenic_mask = df['CLIN_SIG'].astype(str).str.lower().isin([t.lower() for t in pathogenic_terms])
        benign_mask = df['CLIN_SIG'].astype(str).str.lower().isin([t.lower() for t in benign_terms])
        
        pathogenic_count = pathogenic_mask.sum()
        benign_count = benign_mask.sum()
        
        print(f"\n🎯 Available for Selection:")
        print(f"   Pathogenic variants: {pathogenic_count:,}")
        print(f"   Benign variants: {benign_count:,}")
        
        return {
            'pathogenic_mask': pathogenic_mask,
            'benign_mask': benign_mask,
            'pathogenic_count': pathogenic_count,
            'benign_count': benign_count
        }

    def select_representative_variants(self, df, analysis_data, n_pathogenic=8, n_benign=8):
        """Select representative variants for analysis"""
        print(f"\n🎯 SELECTING REPRESENTATIVE VARIANTS")
        print("-" * 40)
        print(f"Target: {n_pathogenic} pathogenic + {n_benign} benign variants")
        
        selected_variants = []
        
        # Select pathogenic variants
        print(f"\n🔴 Selecting {n_pathogenic} Pathogenic Variants:")
        pathogenic_variants = df[analysis_data['pathogenic_mask']]
        
        if len(pathogenic_variants) < n_pathogenic:
            print(f"⚠️  Only {len(pathogenic_variants)} pathogenic variants available")
            n_pathogenic = len(pathogenic_variants)
        
        # Stratify by gene diversity if possible
        if 'SYMBOL' in pathogenic_variants.columns:
            # Try to get variants from different genes
            gene_counts = pathogenic_variants['SYMBOL'].value_counts()
            selected_pathogenic = []
            
            for gene in gene_counts.index:
                if len(selected_pathogenic) >= n_pathogenic:
                    break
                gene_variants = pathogenic_variants[pathogenic_variants['SYMBOL'] == gene]
                # Randomly select one variant from this gene
                selected_variant = gene_variants.sample(n=1, random_state=42)
                selected_pathogenic.append(selected_variant)
                print(f"   ✅ Selected pathogenic variant from {gene}")
            
            # If we need more variants, randomly sample the rest
            if len(selected_pathogenic) < n_pathogenic:
                remaining = n_pathogenic - len(selected_pathogenic)
                already_selected_idx = pd.concat(selected_pathogenic).index
                remaining_variants = pathogenic_variants.drop(already_selected_idx)
                additional = remaining_variants.sample(n=min(remaining, len(remaining_variants)), random_state=42)
                selected_pathogenic.append(additional)
                print(f"   ✅ Added {len(additional)} additional pathogenic variants")
            
            pathogenic_selection = pd.concat(selected_pathogenic)
        else:
            # Random selection if no gene information
            pathogenic_selection = pathogenic_variants.sample(n=n_pathogenic, random_state=42)
        
        # Select benign variants
        print(f"\n🟢 Selecting {n_benign} Benign Variants:")
        benign_variants = df[analysis_data['benign_mask']]
        
        if len(benign_variants) < n_benign:
            print(f"⚠️  Only {len(benign_variants)} benign variants available")
            n_benign = len(benign_variants)
        
        # Similar stratification for benign variants
        if 'SYMBOL' in benign_variants.columns:
            gene_counts = benign_variants['SYMBOL'].value_counts()
            selected_benign = []
            
            for gene in gene_counts.index:
                if len(selected_benign) >= n_benign:
                    break
                gene_variants = benign_variants[benign_variants['SYMBOL'] == gene]
                selected_variant = gene_variants.sample(n=1, random_state=42)
                selected_benign.append(selected_variant)
                print(f"   ✅ Selected benign variant from {gene}")
            
            if len(selected_benign) < n_benign:
                remaining = n_benign - len(selected_benign)
                already_selected_idx = pd.concat(selected_benign).index
                remaining_variants = benign_variants.drop(already_selected_idx)
                additional = remaining_variants.sample(n=min(remaining, len(remaining_variants)), random_state=42)
                selected_benign.append(additional)
                print(f"   ✅ Added {len(additional)} additional benign variants")
            
            benign_selection = pd.concat(selected_benign)
        else:
            benign_selection = benign_variants.sample(n=n_benign, random_state=42)
        
        # Combine selections
        all_selected = pd.concat([pathogenic_selection, benign_selection])
        
        # Add selection metadata
        all_selected = all_selected.copy()
        all_selected['selection_category'] = ['pathogenic'] * len(pathogenic_selection) + ['benign'] * len(benign_selection)
        all_selected['selection_timestamp'] = datetime.now().isoformat()
        
        print(f"\n✅ SELECTION COMPLETE:")
        print(f"   Total selected: {len(all_selected)} variants")
        print(f"   Pathogenic: {len(pathogenic_selection)}")
        print(f"   Benign: {len(benign_selection)}")
        
        return all_selected

    def save_selected_variants(self, selected_variants):
        """Save selected variants for attention analysis"""
        print(f"\n💾 SAVING SELECTED VARIANTS")
        print("-" * 30)
        
        # Save main selection file
        output_file = os.path.join(self.output_dir, "selected_variants.csv")
        selected_variants.to_csv(output_file, index=False)
        print(f"✅ Saved selected variants: {output_file}")
        
        # Create summary report
        summary_file = os.path.join(self.output_dir, "variant_selection_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("TABNET ATTENTION ANALYSIS - VARIANT SELECTION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total variants selected: {len(selected_variants)}\n\n")
            
            # Selection breakdown
            category_counts = selected_variants['selection_category'].value_counts()
            f.write("SELECTION BREAKDOWN:\n")
            for category, count in category_counts.items():
                f.write(f"   {category.title()}: {count} variants\n")
            
            # Gene diversity
            if 'SYMBOL' in selected_variants.columns:
                unique_genes = selected_variants['SYMBOL'].nunique()
                f.write(f"\nGENE DIVERSITY:\n")
                f.write(f"   Unique genes: {unique_genes}\n")
                
                gene_counts = selected_variants['SYMBOL'].value_counts()
                f.write(f"   Genes represented:\n")
                for gene, count in gene_counts.items():
                    f.write(f"     {gene}: {count} variant(s)\n")
            
            # Key features for analysis
            key_features = ['chromosome', 'position', 'SYMBOL', 'Consequence', 'CLIN_SIG']
            available_features = [f for f in key_features if f in selected_variants.columns]
            
            f.write(f"\nSELECTED VARIANTS PREVIEW:\n")
            for i, (_, variant) in enumerate(selected_variants.iterrows(), 1):
                category = variant['selection_category']
                f.write(f"\n{i}. {category.upper()} VARIANT:\n")
                for feature in available_features:
                    if pd.notna(variant[feature]):
                        f.write(f"   {feature}: {variant[feature]}\n")
            
            f.write(f"\nNEXT STEPS:\n")
            f.write(f"1. Run attention extraction: python src/analysis/attention_extractor.py\n")
            f.write(f"2. Analyze attention patterns: python src/analysis/attention_analyzer.py\n")
            f.write(f"3. Generate final results: python src/analysis/results_generator.py\n")
        
        print(f"✅ Saved summary report: {summary_file}")
        
        # Create individual variant files for detailed analysis
        variants_dir = os.path.join(self.output_dir, "individual_variants")
        os.makedirs(variants_dir, exist_ok=True)
        
        for i, (_, variant) in enumerate(selected_variants.iterrows(), 1):
            category = variant['selection_category']
            gene = variant.get('SYMBOL', 'unknown')
            variant_file = os.path.join(variants_dir, f"variant_{i:02d}_{category}_{gene}.csv")
            
            # Save individual variant with all features
            variant_df = pd.DataFrame([variant])
            variant_df.to_csv(variant_file, index=False)
        
        print(f"✅ Saved individual variant files: {variants_dir}")
        
        return output_file, summary_file

    def generate_analysis_preview(self, selected_variants):
        """Generate preview of what attention analysis will examine"""
        print(f"\n🔍 ATTENTION ANALYSIS PREVIEW")
        print("-" * 40)
        
        print("📋 Selected variants will be analyzed for:")
        print("   1. Which features get highest attention weights")
        print("   2. How attention differs between pathogenic vs benign cases")
        print("   3. Attention patterns across TabNet's 6 decision steps")
        print("   4. Whether VEP-corrected features get high attention")
        print("   5. AlphaMissense attention correlation")
        
        if 'SYMBOL' in selected_variants.columns:
            unique_genes = selected_variants['SYMBOL'].unique()
            print(f"\n🧬 Genes in analysis: {', '.join(unique_genes)}")
        
        print(f"\n📊 Expected outputs:")
        print(f"   - Attention heatmaps for each variant")
        print(f"   - Feature ranking tables")
        print(f"   - Pathogenic vs benign comparison")
        print(f"   - Publication-ready summary")

def main():
    """Main variant selection pipeline"""
    print("🎯 TABNET ATTENTION ANALYSIS - VARIANT SELECTION")
    print("=" * 60)
    print("Purpose: Select representative variants for interpretability study")
    print("Method: Use ClinVar labels (no medical expertise required)")
    print()
    
    try:
        # Initialize selector
        selector = VariantSelector()
        
        # Load dataset
        df = selector.load_dataset()
        
        # Analyze available labels
        analysis_data = selector.analyze_clin_sig_distribution(df)
        
        if analysis_data is None:
            print("❌ Cannot proceed without CLIN_SIG labels")
            return False
        
        # Check if we have enough variants
        min_pathogenic = 5
        min_benign = 5
        
        if analysis_data['pathogenic_count'] < min_pathogenic:
            print(f"❌ Insufficient pathogenic variants: {analysis_data['pathogenic_count']} < {min_pathogenic}")
            return False
        
        if analysis_data['benign_count'] < min_benign:
            print(f"❌ Insufficient benign variants: {analysis_data['benign_count']} < {min_benign}")
            return False
        
        # Select variants
        n_pathogenic = min(8, analysis_data['pathogenic_count'])
        n_benign = min(8, analysis_data['benign_count'])
        
        selected_variants = selector.select_representative_variants(
            df, analysis_data, n_pathogenic, n_benign
        )
        
        # Save results
        output_file, summary_file = selector.save_selected_variants(selected_variants)
        
        # Generate preview
        selector.generate_analysis_preview(selected_variants)
        
        print(f"\n🎉 VARIANT SELECTION COMPLETED!")
        print("=" * 40)
        print(f"✅ Selected {len(selected_variants)} variants for attention analysis")
        print(f"📁 Results saved to: {selector.output_dir}")
        print(f"📋 Main file: {output_file}")
        print(f"📋 Summary: {summary_file}")
        
        print(f"\n🎯 Ready for next step:")
        print(f"   python src/analysis/attention_extractor.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Variant selection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)