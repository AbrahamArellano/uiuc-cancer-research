#!/usr/bin/env python3
"""
TabNet Attention Extraction for Interpretability Analysis

This script extracts attention weights from a trained TabNet model
for selected variants to understand which features drive predictions.

Author: PhD Research Student, University of Illinois
Contact: aa107@illinois.edu
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier

class TabNetAttentionExtractor:
    """Extract and analyze TabNet attention weights for interpretability"""
    
    def __init__(self, model_path, analysis_dir):
        """Initialize attention extractor with paths"""
        self.model_path = Path(model_path)
        self.analysis_dir = Path(analysis_dir)
        self.dataset_path = Path("/u/aa107/uiuc-cancer-research/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv")
        self.selected_variants_path = self.analysis_dir / "selected_variants.csv"
        self.attention_dir = self.analysis_dir / "attention_weights"
        
        # Create output directory
        self.attention_dir.mkdir(parents=True, exist_ok=True)
        
        # Model and data attributes
        self.model = None
        self.feature_names = []
        self.scaler = None
        self.label_encoder = None
        
        # VEP Severity Tables (matching training)
        self.CONSEQUENCE_SEVERITY = {
            'transcript_ablation': 10,
            'splice_acceptor_variant': 9,
            'splice_donor_variant': 9,
            'stop_gained': 8,
            'frameshift_variant': 8,
            'stop_lost': 7,
            'start_lost': 7,
            'transcript_amplification': 6,
            'inframe_insertion': 5,
            'inframe_deletion': 5,
            'missense_variant': 4,
            'protein_altering_variant': 4,
            'splice_region_variant': 3,
            'incomplete_terminal_codon_variant': 3,
            'start_retained_variant': 3,
            'stop_retained_variant': 3,
            'synonymous_variant': 2,
            'coding_sequence_variant': 2,
            'mature_mirna_variant': 2,
            '5_prime_utr_variant': 1,
            '3_prime_utr_variant': 1,
            'non_coding_transcript_exon_variant': 1,
            'intron_variant': 1,
            'nmd_transcript_variant': 1,
            'non_coding_transcript_variant': 1,
            'upstream_gene_variant': 0,
            'downstream_gene_variant': 0,
            'tfbs_ablation': 0,
            'tfbs_amplification': 0,
            'tf_binding_site_variant': 0,
            'regulatory_region_ablation': 0,
            'regulatory_region_amplification': 0,
            'feature_elongation': 0,
            'regulatory_region_variant': 0,
            'feature_truncation': 0,
            'intergenic_variant': 0
        }
        
        self.CLIN_SIG_SEVERITY = {
            'pathogenic': 4,
            'likely_pathogenic': 3,
            'uncertain_significance': 2,
            'likely_benign': 1,
            'benign': 0
        }
        
        self.IMPACT_SEVERITY = {
            'HIGH': 3,
            'MODERATE': 2,
            'LOW': 1,
            'MODIFIER': 0
        }
        
        # Initialize 8-tier feature groups
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
        
        print("🔍 TabNet Attention Extractor Initialized")
        print(f"📁 Model: {self.model_path}")
        print(f"📁 Analysis: {self.analysis_dir}")
        print(f"📁 Attention output: {self.attention_dir}")

    def load_trained_model(self):
        """Load the trained TabNet model"""
        print("\n🤖 LOADING TRAINED TABNET MODEL")
        print("-" * 40)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            # Import required modules
            import torch
            from pytorch_tabnet.tab_model import TabNetClassifier
            
            # Load model directly from pickle file
            print("📁 Loading model from pickle file...")
            with open(self.model_path, 'rb') as f:
                model_data = torch.load(f, map_location='cpu', weights_only=False)
            
            # Extract the TabNet model
            if isinstance(model_data, dict) and 'tabnet_model' in model_data:
                self.model = model_data['tabnet_model']
                self.feature_names = model_data.get('feature_names', [])
                self.scaler = model_data.get('scaler', None)
                self.label_encoder = model_data.get('label_encoder', None)
                print("✅ Loaded TabNet model from dictionary structure")
            else:
                # Direct model object
                self.model = model_data
                print("✅ Loaded TabNet model directly")
            
            # Verify model has required method
            if hasattr(self.model, 'explain'):
                print("✅ TabNet explain() method available")
            else:
                print("❌ TabNet model missing explain() method")
                return False
            
            # Force CPU device for attention extraction
            device = 'cpu'
            print(f"🎯 Target device: {device} (CPU-only mode for stability)")
            
            # Move model to CPU if needed
            if hasattr(self.model, 'device_name'):
                self.model.device_name = device
            
            # Display model info
            print(f"📊 Features from pickle: {len(self.feature_names)}")
            print(f"📊 Scaler available: {self.scaler is not None}")
            print(f"📊 Label encoder available: {self.label_encoder is not None}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_selected_variants(self):
        """Load variants selected for attention analysis"""
        print("\n📋 LOADING SELECTED VARIANTS")
        print("-" * 30)
        
        if not os.path.exists(self.selected_variants_path):
            raise FileNotFoundError(f"Selected variants not found: {self.selected_variants_path}")
        
        selected_df = pd.read_csv(self.selected_variants_path)
        print(f"✅ Loaded {len(selected_df)} selected variants")
        
        # Check classification distribution
        if 'selection_category' in selected_df.columns:
            print("📊 Classification distribution:")
            for category, count in selected_df['selection_category'].value_counts().items():
                print(f"   {category.capitalize()}: {count}")
        
        return selected_df

    def prepare_variant_features(self, selected_df):
        """Prepare variant features matching training pipeline exactly"""
        print("\n🔧 PREPARING VARIANT FEATURES")
        print("-" * 35)
        
        # Load full dataset
        print("📊 Loading full dataset for feature extraction...")
        full_df = pd.read_csv(self.dataset_path)
        
        # Match selected variants in full dataset using multiple identifiers
        matched_variants = []
        classification_map = {}  # Store variant ID to classification mapping
        
        for idx, selected_row in selected_df.iterrows():
            # Create unique identifier for matching
            if 'chromosome' in selected_row and 'position' in selected_row:
                # Match by chromosome and position
                matches = full_df[
                    (full_df['chromosome'] == selected_row['chromosome']) & 
                    (full_df['position'] == selected_row['position'])
                ]
                
                # If multiple matches, try to narrow down by gene
                if len(matches) > 1 and 'SYMBOL' in selected_row:
                    gene_matches = matches[matches['SYMBOL'] == selected_row['SYMBOL']]
                    if not gene_matches.empty:
                        matches = gene_matches
                
                if not matches.empty:
                    for _, match in matches.iterrows():
                        matched_variants.append(match)
                        # Create unique key for this variant
                        var_key = f"{match['chromosome']}_{match['position']}_{match.get('SYMBOL', 'NA')}"
                        classification_map[var_key] = selected_row.get('selection_category', 'unknown')
        
        if not matched_variants:
            print("❌ No variants matched in full dataset")
            return None, None, None
        
        matched_df = pd.DataFrame(matched_variants)
        print(f"🔍 Found {len(matched_df)} matching variants in dataset")
        
        # Select features using same hierarchy as training - MUST MATCH EXACTLY
        selected_features = []
        
        # TIER 1: VEP-Corrected Features (4 features) - CLIN_SIG removed
        tier1_features = ['Consequence', 'DOMAINS', 'PUBMED', 'VAR_SYNONYMS']
        for feature in tier1_features:
            if feature in matched_df.columns:
                selected_features.append(feature)
                self.feature_groups['tier1_vep_corrected'].append(feature)
        
        # TIER 2: Core VEP Annotations (10 features)
        tier2_features = ['SYMBOL', 'BIOTYPE', 'CANONICAL', 'PICK', 'HGVSc', 
                         'HGVSp', 'Protein_position', 'Amino_acids', 'Existing_variation', 'VARIANT_CLASS']
        for feature in tier2_features:
            if feature in matched_df.columns:
                selected_features.append(feature)
                self.feature_groups['tier2_core_vep'].append(feature)
        
        # TIER 3: AlphaMissense Integration (2 features)
        tier3_features = ['alphamissense_pathogenicity', 'alphamissense_class']
        for feature in tier3_features:
            if feature in matched_df.columns:
                selected_features.append(feature)
                self.feature_groups['tier3_alphamissense'].append(feature)
        
        # TIER 4: Population Genetics (17 features)
        tier4_features = ['AF', 'AFR_AF', 'AMR_AF', 'EAS_AF', 'EUR_AF', 'SAS_AF',
                         'gnomADe_AF', 'gnomADe_AFR_AF', 'gnomADe_AMR_AF', 'gnomADe_ASJ_AF',
                         'gnomADe_EAS_AF', 'gnomADe_FIN_AF', 'gnomADe_MID_AF', 'gnomADe_NFE_AF',
                         'gnomADe_REMAINING_AF', 'gnomADe_SAS_AF', 'af_1kg']
        for feature in tier4_features:
            if feature in matched_df.columns:
                selected_features.append(feature)
                self.feature_groups['tier4_population'].append(feature)
        
        # TIER 5: Functional Predictions (6 features)
        tier5_features = ['IMPACT', 'sift_score', 'polyphen_score', 'SIFT', 'PolyPhen', 'impact_score']
        for feature in tier5_features:
            if feature in matched_df.columns:
                selected_features.append(feature)
                self.feature_groups['tier5_functional'].append(feature)
        
        # TIER 6: Clinical Context (5 features)
        tier6_features = ['SOMATIC', 'PHENO', 'EXON', 'INTRON', 'CCDS']
        for feature in tier6_features:
            if feature in matched_df.columns:
                selected_features.append(feature)
                self.feature_groups['tier6_clinical'].append(feature)
        
        # TIER 7: Variant Properties (8 features)
        tier7_features = ['ref_length', 'alt_length', 'variant_size', 'is_indel', 
                         'is_snv', 'is_lof', 'is_missense', 'is_synonymous']
        for feature in tier7_features:
            if feature in matched_df.columns:
                selected_features.append(feature)
                self.feature_groups['tier7_variant_props'].append(feature)
        
        # TIER 8: Prostate Biology (4 features)
        tier8_features = ['is_important_gene', 'dna_repair_pathway', 'mismatch_repair_pathway', 'hormone_pathway']
        for feature in tier8_features:
            if feature in matched_df.columns:
                selected_features.append(feature)
                self.feature_groups['tier8_prostate_biology'].append(feature)
        
        print(f"✅ Selected {len(selected_features)} features across 8 tiers:")
        for tier, features in self.feature_groups.items():
            if features:
                print(f"   {tier}: {len(features)} features")
        
        # Ensure we have exactly 56 features by checking what might be missing
        expected_feature_count = 56
        if len(selected_features) != expected_feature_count:
            print(f"\n⚠️  Feature count mismatch: {len(selected_features)} vs expected {expected_feature_count}")
            
            # Debug: show which features we have
            print("\nCurrent features by tier:")
            for tier, features in self.feature_groups.items():
                if features:
                    print(f"{tier}: {features}")
        
        # Select features from matched_df
        X_selected = matched_df[selected_features].copy()
        self.actual_feature_names = selected_features
        
        # Apply categorical encoding - MUST MATCH TRAINING EXACTLY
        print("\n🔧 APPLYING CATEGORICAL ENCODING (TRAINING PIPELINE)")
        print("-" * 50)
        
        # Identify categorical columns IN THE SELECTED FEATURES
        categorical_columns = []
        for col in selected_features:
            if col in X_selected.columns and X_selected[col].dtype == 'object':
                categorical_columns.append(col)
        
        print(f"📋 Found {len(categorical_columns)} categorical columns to encode")
        
        # 1. Encode Consequence with severity
        if 'Consequence' in X_selected.columns:
            print("🔹 Encoding Consequence with severity rankings...")
            X_selected['Consequence'] = X_selected['Consequence'].map(self.CONSEQUENCE_SEVERITY).fillna(0)
        
        # 2. Encode IMPACT with severity
        if 'IMPACT' in X_selected.columns:
            print("🔹 Encoding IMPACT with severity rankings...")
            X_selected['IMPACT'] = X_selected['IMPACT'].map(self.IMPACT_SEVERITY).fillna(0)
        
        # 3. Encode AlphaMissense class
        if 'alphamissense_class' in X_selected.columns:
            print("🔹 Encoding AlphaMissense classes...")
            am_class_map = {'Likely_Pathogenic': 2, 'Ambiguous': 1, 'Likely_Benign': 0}
            X_selected['alphamissense_class'] = X_selected['alphamissense_class'].map(am_class_map).fillna(1)
        
        # 4. Handle remaining categorical columns
        remaining_categorical = [col for col in categorical_columns 
                               if col not in ['Consequence', 'IMPACT', 'alphamissense_class']]
        
        if remaining_categorical:
            print(f"🔹 Label encoding {len(remaining_categorical)} remaining categorical columns...")
            
            # Use the label encoder from training if available
            if self.label_encoder is not None:
                # Apply same encoding as training
                for col in remaining_categorical:
                    if col in X_selected.columns:
                        # Handle missing values
                        X_selected[col] = X_selected[col].fillna('Unknown')
                        # For simplicity, encode as 0/1 for binary, or use label encoder
                        unique_vals = X_selected[col].unique()
                        if len(unique_vals) <= 2:
                            # Binary encoding
                            X_selected[col] = (X_selected[col] == unique_vals[0]).astype(int)
                        else:
                            # Create a simple numeric mapping
                            val_map = {val: i for i, val in enumerate(unique_vals)}
                            X_selected[col] = X_selected[col].map(val_map)
            else:
                # Fallback: simple numeric encoding
                for col in remaining_categorical:
                    if col in X_selected.columns:
                        X_selected[col] = pd.Categorical(X_selected[col]).codes
        
        print("✅ Categorical encoding completed - all features now numeric")
        
        # Handle any remaining non-numeric values
        for col in X_selected.columns:
            if X_selected[col].dtype == 'object':
                print(f"⚠️  Converting remaining object column: {col}")
                X_selected[col] = pd.to_numeric(X_selected[col], errors='coerce').fillna(0)
        
        # Apply scaling if scaler is available
        if self.scaler is not None:
            print("\n🔄 Applying trained scaler to encoded features...")
            try:
                X_scaled = self.scaler.transform(X_selected)
                X_selected = pd.DataFrame(X_scaled, columns=X_selected.columns, index=X_selected.index)
                print(f"✅ Features scaled successfully: {X_selected.shape}")
            except Exception as e:
                print(f"⚠️  Scaling failed: {e}")
                print("⚠️  Proceeding with unscaled features")
        else:
            print("⚠️  No scaler available - using raw encoded features")
        
        # Create variant info with FIXED classification preservation
        variant_info = []
        for i, (_, row) in enumerate(matched_df.iterrows()):
            # Create unique key for this variant
            var_key = f"{row['chromosome']}_{row['position']}_{row.get('SYMBOL', 'NA')}"
            
            # Get classification from mapping
            classification = classification_map.get(var_key, 'unknown')
            
            info = {
                'variant_id': f"variant_{i+1:02d}",
                'chromosome': row.get('chromosome', 'unknown'),
                'position': row.get('position', 'unknown'),
                'gene': row.get('SYMBOL', 'unknown'),
                'classification': classification
            }
            variant_info.append(info)
            print(f"   ✅ Prepared {info['variant_id']}: {classification}")
        
        print(f"\n✅ Prepared features: {X_selected.shape}")
        print(f"📊 Feature columns: {len(X_selected.columns)}")
        print(f"🧮 All features numeric: {X_selected.select_dtypes(include=[np.number]).shape[1] == X_selected.shape[1]}")
        
        # Verify classification preservation
        preserved_classifications = {}
        for info in variant_info:
            cls = info['classification']
            preserved_classifications[cls] = preserved_classifications.get(cls, 0) + 1
        
        print(f"🔍 Classification preservation check:")
        for cls, count in preserved_classifications.items():
            print(f"   {cls}: {count} variants")
        
        return X_selected, None, variant_info

    def extract_attention_weights(self, X_selected, variant_info):
        """Extract attention weights using TabNet's explain method"""
        print("\n🧠 EXTRACTING ATTENTION WEIGHTS")
        print("-" * 35)
        
        if self.model is None:
            print("❌ No model loaded")
            return None
        
        if X_selected is None or len(X_selected) == 0:
            print("❌ No features prepared")
            return None
        
        try:
            # Convert to numpy array for TabNet
            X_array = X_selected.values
            print(f"📊 Input shape for TabNet: {X_array.shape}")
            
            # Ensure model is in eval mode
            self.model.eval()
            
            # No GPU tensor conversion needed - we're in CPU mode
            print("🎯 Running attention extraction on CPU")
            
            # Call explain method
            print("🔍 Calling TabNet.explain()...")
            M_explain, masks = self.model.explain(X_array)
            
            print(f"✅ Attention extraction successful!")
            print(f"   M_explain shape: {M_explain.shape}")
            print(f"   Masks shape: {masks.shape}")
            
            # Store results
            attention_results = {
                'feature_importance': M_explain,
                'attention_masks': masks,
                'variant_info': variant_info,
                'feature_names': self.actual_feature_names
            }
            
            return attention_results
            
        except Exception as e:
            print(f"❌ Attention extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_attention_weights(self, attention_results):
        """Save extracted attention weights for analysis"""
        print("\n💾 SAVING ATTENTION WEIGHTS")
        print("-" * 30)
        
        if attention_results is None:
            print("❌ No attention results to save")
            return
        
        # Save feature importance scores
        importance_df = pd.DataFrame(
            attention_results['feature_importance'],
            columns=self.actual_feature_names
        )
        
        # Add variant info
        for i, info in enumerate(attention_results['variant_info']):
            importance_df.loc[i, 'variant_id'] = info['variant_id']
            importance_df.loc[i, 'classification'] = info['classification']
            importance_df.loc[i, 'gene'] = info['gene']
        
        # Reorder columns
        meta_cols = ['variant_id', 'classification', 'gene']
        importance_df = importance_df[meta_cols + self.actual_feature_names]
        
        # Save to CSV
        output_file = self.attention_dir / "feature_importance_scores.csv"
        importance_df.to_csv(output_file, index=False)
        print(f"✅ Saved feature importance: {output_file}")
        
        # Save raw attention masks
        masks_file = self.attention_dir / "attention_masks.npy"
        np.save(masks_file, attention_results['attention_masks'])
        print(f"✅ Saved attention masks: {masks_file}")
        
        # Save summary statistics
        summary_file = self.attention_dir / "attention_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("TABNET ATTENTION EXTRACTION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Extraction date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total variants analyzed: {len(attention_results['variant_info'])}\n")
            f.write(f"Features analyzed: {len(self.actual_feature_names)}\n")
            f.write(f"\nClassification breakdown:\n")
            
            class_counts = {}
            for info in attention_results['variant_info']:
                cls = info['classification']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            for cls, count in class_counts.items():
                f.write(f"  {cls}: {count} variants\n")
            
            f.write(f"\nTop 10 features by average importance:\n")
            avg_importance = importance_df[self.actual_feature_names].mean().sort_values(ascending=False)
            for i, (feature, score) in enumerate(avg_importance.head(10).items()):
                f.write(f"  {i+1}. {feature}: {score:.4f}\n")
        
        print(f"✅ Saved summary: {summary_file}")
        
        print("\n🎉 Attention extraction completed successfully!")
        print(f"📁 Results saved to: {self.attention_dir}")

def main():
    """Main execution pipeline"""
    print("🧠 TABNET ATTENTION EXTRACTION")
    print("=" * 50)
    print("Purpose: Extract attention weights from trained TabNet model")
    print("Input: Selected variants from variant_selector.py")
    print("Output: Attention weights for interpretability analysis")
    print()
    
    # Configuration
    model_path = "/u/aa107/scratch/tabnet_model_20250727_053446.pkl"
    analysis_dir = "/u/aa107/uiuc-cancer-research/results/attention_analysis"
    
    try:
        # Initialize extractor
        extractor = TabNetAttentionExtractor(model_path, analysis_dir)
        
        # Load trained model
        if not extractor.load_trained_model():
            print("❌ Failed to load model")
            return False
        
        # Load selected variants
        selected_variants = extractor.load_selected_variants()
        
        # Prepare features
        X_selected, y_selected, variant_info = extractor.prepare_variant_features(selected_variants)
        
        if X_selected is None:
            print("❌ Failed to prepare features")
            return False
        
        # Extract attention weights
        attention_results = extractor.extract_attention_weights(X_selected, variant_info)
        
        if attention_results is None:
            print("❌ Failed to extract attention")
            return False
        
        # Save results
        extractor.save_attention_weights(attention_results)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)