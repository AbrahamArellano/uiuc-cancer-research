#!/usr/bin/env python3
"""
TabNet Attention Weight Extractor - CORRECTED VERSION
Extracts attention weights from trained TabNet model for selected variants
Fixed to match exact 56-feature training configuration

Location: /u/aa107/uiuc-cancer-research/src/analysis/attention_extractor.py
Author: PhD Research Student, University of Illinois
"""

import pandas as pd
import numpy as np
import os
import pickle
import sys
import re
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
sys.path.append('/u/aa107/uiuc-cancer-research/src')

class AttentionExtractor:
    """Extracts TabNet attention weights for interpretability analysis"""
    
    def __init__(self, model_path=None, analysis_dir=None):
        """Initialize attention extractor"""
        if model_path is None:
            # Use the latest model from successful training
            self.model_path = "/u/aa107/scratch/tabnet_model_20250727_053446.pkl"
        else:
            self.model_path = model_path
            
        if analysis_dir is None:
            self.analysis_dir = "/u/aa107/uiuc-cancer-research/results/attention_analysis"
        else:
            self.analysis_dir = analysis_dir
        
        self.dataset_path = "/u/aa107/uiuc-cancer-research/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv"
        self.selected_variants_path = os.path.join(self.analysis_dir, "selected_variants.csv")
        
        # Create output directories
        self.attention_dir = os.path.join(self.analysis_dir, "attention_weights")
        os.makedirs(self.attention_dir, exist_ok=True)
        
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.label_encoder = None
        self.actual_feature_names = None
        
        # VEP Severity Tables (from training script)
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
                model_data = pickle.load(f)
            
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
            
            # Check device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"🎯 Target device: {device}")
            
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
        
        # TIER 1: VEP-Corrected Features (4 features) - CLIN_SIG excluded to prevent leakage
        tier1_features = ['Consequence', 'DOMAINS', 'PUBMED', 'VAR_SYNONYMS']
        for feature in tier1_features:
            if feature in matched_df.columns:
                selected_features.append(feature)
                self.feature_groups['tier1_vep_corrected'].append(feature)
        
        # TIER 2: Core VEP Annotations (10 features) - MUST MATCH TRAINING EXACTLY
        tier2_features = ['SYMBOL', 'BIOTYPE', 'CANONICAL', 'PICK', 'HGVSc', 
                         'HGVSp', 'Protein_position', 'Amino_acids', 'Existing_variation', 'VARIANT_CLASS']
        for feature in tier2_features:
            if feature in matched_df.columns:
                selected_features.append(feature)
                self.feature_groups['tier2_core_vep'].append(feature)
        
        # TIER 3: AlphaMissense (2 features)
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
        
        # TIER 5: Functional Predictions (6 features) - INCLUDE BOTH RAW AND PARSED
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
        
        # 3. Parse SIFT scores if sift_score not already in features
        if 'SIFT' in X_selected.columns and 'sift_score' not in X_selected.columns:
            print("🔹 Parsing SIFT scores...")
            def parse_sift(sift_value):
                if pd.isna(sift_value):
                    return np.nan
                try:
                    match = re.search(r'\(([\d.]+)\)', str(sift_value))
                    if match:
                        return float(match.group(1))
                except:
                    pass
                return np.nan
            X_selected['sift_score'] = X_selected['SIFT'].apply(parse_sift)
            # IMPORTANT: Do NOT drop SIFT column - keep both!
        
        # 4. Parse PolyPhen scores if polyphen_score not already in features
        if 'PolyPhen' in X_selected.columns and 'polyphen_score' not in X_selected.columns:
            print("🔹 Parsing PolyPhen scores...")
            def parse_polyphen(pp_value):
                if pd.isna(pp_value):
                    return np.nan
                try:
                    match = re.search(r'\(([\d.]+)\)', str(pp_value))
                    if match:
                        return float(match.group(1))
                except:
                    pass
                return np.nan
            X_selected['polyphen_score'] = X_selected['PolyPhen'].apply(parse_polyphen)
            # IMPORTANT: Do NOT drop PolyPhen column - keep both!
        
        # 5. Encode AlphaMissense class
        if 'alphamissense_class' in X_selected.columns:
            print("🔹 Encoding AlphaMissense classes...")
            am_class_map = {'likely_benign': 0, 'ambiguous': 1, 'likely_pathogenic': 2}
            X_selected['alphamissense_class'] = X_selected['alphamissense_class'].map(am_class_map).fillna(0)
        
        # 6. Label encode remaining categorical features
        remaining_categorical = X_selected.select_dtypes(include=['object']).columns.tolist()
        if remaining_categorical:
            print(f"🔹 Label encoding {len(remaining_categorical)} remaining categorical columns...")
            for col in remaining_categorical:
                le = LabelEncoder()
                # Handle missing values
                mask = X_selected[col].notna()
                if mask.sum() > 0:  # Only encode if there are non-null values
                    X_selected.loc[mask, col] = le.fit_transform(X_selected.loc[mask, col])
                X_selected[col] = X_selected[col].fillna(0).astype(int)
        
        print("✅ Categorical encoding completed - all features now numeric")
        
        # Verify all features are numeric
        numeric_check = X_selected.select_dtypes(exclude=[np.number]).columns.tolist()
        if numeric_check:
            print(f"❌ WARNING: {len(numeric_check)} features still non-numeric:")
            for col in numeric_check[:5]:
                print(f"     - {col}: {X_selected[col].dtype}")
            return None, None, None
        
        # Apply scaling using the trained scaler if available
        if self.scaler is not None:
            print("\n🔄 Applying trained scaler to encoded features...")
            try:
                # The scaler expects exactly the same features it was trained on
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
            
            # Call explain method
            print("🔍 Calling TabNet.explain()...")
            M_explain, masks = self.model.explain(X_array)
            
            print(f"✅ Attention extraction successful!")
            print(f"📊 Explanation shape: {M_explain.shape}")
            print(f"📊 Masks shape: {len(masks)} decision steps")
            
            # Process attention data for each variant
            attention_data = []
            feature_names = list(X_selected.columns)  # Use actual column names
            
            for i, variant in enumerate(variant_info):
                variant_attention = {
                    'variant_info': variant,
                    'attention_by_step': []
                }
                
                # Extract attention for each decision step
                for step in range(len(masks)):
                    step_attention = {}
                    
                    # Get attention weights for this variant and step
                    if i < masks[step].shape[0]:
                        attention_weights = masks[step][i]
                        
                        # Map to feature names
                        for j, feature_name in enumerate(feature_names):
                            if j < len(attention_weights):
                                step_attention[feature_name] = float(attention_weights[j])
                    
                    variant_attention['attention_by_step'].append(step_attention)
                
                attention_data.append(variant_attention)
                print(f"   ✅ Processed {variant['variant_id']}: {variant['classification']}")
            
            print(f"✅ Extracted attention for {len(attention_data)} variants")
            return attention_data
            
        except Exception as e:
            print(f"❌ Attention extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_attention_data(self, attention_data):
        """Save attention weights to CSV files"""
        print("\n💾 SAVING ATTENTION DATA")
        print("-" * 25)
        
        if not attention_data:
            print("❌ No attention data to save")
            return None, None, None
        
        saved_files = []
        summary_data = []
        
        # Get feature names from first variant
        if attention_data and attention_data[0]['attention_by_step']:
            feature_names = list(attention_data[0]['attention_by_step'][0].keys())
        else:
            feature_names = self.actual_feature_names if self.actual_feature_names else self.feature_names
        
        # Save individual variant attention files
        for variant_data in attention_data:
            variant_info = variant_data['variant_info']
            variant_id = variant_info['variant_id']
            
            # Create DataFrame for this variant
            attention_df = pd.DataFrame()
            
            # Add variant info
            attention_df['feature'] = feature_names
            
            # Add attention from each step
            for step_idx, step_attention in enumerate(variant_data['attention_by_step']):
                step_col = f"step_{step_idx+1}_attention"
                attention_df[step_col] = [step_attention.get(feat, 0) for feat in feature_names]
            
            # Calculate global importance (average across steps)
            step_cols = [col for col in attention_df.columns if col.endswith('_attention')]
            attention_df['global_importance'] = attention_df[step_cols].mean(axis=1)
            
            # Add feature group
            attention_df['feature_group'] = attention_df['feature'].apply(self._get_feature_group)
            
            # Sort by global importance
            attention_df = attention_df.sort_values('global_importance', ascending=False)
            
            # Save to CSV
            filename = f"{variant_id}_attention.csv"
            filepath = os.path.join(self.attention_dir, filename)
            attention_df.to_csv(filepath, index=False)
            saved_files.append(filepath)
            print(f"   ✅ {filename}")
            
            # Collect summary data
            top_features = attention_df.nlargest(3, 'global_importance')
            
            summary_data.append({
                'variant_id': variant_info['variant_id'],
                'gene': variant_info['gene'],
                'classification': variant_info['classification'],
                'top_feature_1': top_features.iloc[0]['feature'] if len(top_features) > 0 else '',
                'top_attention_1': top_features.iloc[0]['global_importance'] if len(top_features) > 0 else 0,
                'top_feature_2': top_features.iloc[1]['feature'] if len(top_features) > 1 else '',
                'top_attention_2': top_features.iloc[1]['global_importance'] if len(top_features) > 1 else 0,
                'top_feature_3': top_features.iloc[2]['feature'] if len(top_features) > 2 else '',
                'top_attention_3': top_features.iloc[2]['global_importance'] if len(top_features) > 2 else 0
            })
        
        # Save summary file
        summary_file = os.path.join(self.attention_dir, "attention_summary.csv")
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        print(f"   ✅ attention_summary.csv")
        
        # Verify classification preservation in summary
        print(f"\n🔍 Final summary classification verification:")
        summary_classifications = summary_df['classification'].value_counts()
        for classification, count in summary_classifications.items():
            print(f"   {classification}: {count} variants")
        
        # Save metadata
        metadata = {
            'extraction_date': datetime.now().isoformat(),
            'model_path': self.model_path,
            'variants_processed': len(attention_data),
            'features_analyzed': len(feature_names),
            'decision_steps': len(attention_data[0]['attention_by_step']) if attention_data else 0,
            'feature_groups': {k: len(v) for k, v in self.feature_groups.items() if v}
        }
        
        metadata_file = os.path.join(self.attention_dir, "extraction_metadata.json")
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ extraction_metadata.json")
        
        return saved_files, summary_file, metadata_file

    def _get_feature_group(self, feature):
        """Get the tier group for a feature"""
        for group_name, features in self.feature_groups.items():
            if feature in features:
                return group_name
        return 'unknown'

    def generate_extraction_summary(self, attention_data):
        """Generate a summary of the extraction process"""
        print(f"\n📋 EXTRACTION SUMMARY")
        print("-" * 25)
        
        if not attention_data:
            print("❌ No attention data to summarize")
            return
        
        print(f"✅ Successfully extracted attention for {len(attention_data)} variants")
        
        # Analyze by classification
        classifications = {}
        for variant_data in attention_data:
            classification = variant_data['variant_info']['classification']
            if classification not in classifications:
                classifications[classification] = 0
            classifications[classification] += 1
        
        print(f"\n📊 Variants by classification:")
        for classification, count in classifications.items():
            print(f"   {classification}: {count}")
        
        # Check for proper classification preservation
        if 'unknown' in classifications and len(classifications) == 1:
            print(f"\n⚠️  CLASSIFICATION PRESERVATION WARNING!")
            print(f"   All variants marked as 'unknown' - check variant matching logic")
        else:
            print(f"\n✅ Classification preservation successful!")
        
        # Analyze feature attention patterns
        if attention_data:
            sample_variant = attention_data[0]
            step_count = len(sample_variant['attention_by_step'])
            feature_count = len(sample_variant['attention_by_step'][0]) if sample_variant['attention_by_step'] else 0
            
            print(f"\n🧠 Attention analysis details:")
            print(f"   Decision steps: {step_count}")
            print(f"   Features per step: {feature_count}")
            print(f"   Total attention weights: {len(attention_data) * step_count * feature_count}")
            
            # Show feature group distribution
            print(f"\n📊 Feature groups analyzed:")
            for group_name, features in self.feature_groups.items():
                if features:
                    print(f"   {group_name}: {len(features)} features")

def main():
    """Main attention extraction pipeline"""
    print("🧠 TABNET ATTENTION EXTRACTION")
    print("=" * 50)
    print("Purpose: Extract attention weights from trained TabNet model")
    print("Input: Selected variants from variant_selector.py")
    print("Output: Attention weights for interpretability analysis")
    print()
    
    try:
        # Initialize extractor
        extractor = AttentionExtractor()
        
        # Load trained model
        if not extractor.load_trained_model():
            print("❌ Failed to load model")
            return False
        
        # Load selected variants
        selected_df = extractor.load_selected_variants()
        
        # Prepare features
        X_selected, y_selected, variant_info = extractor.prepare_variant_features(selected_df)
        
        if X_selected is None:
            print("❌ Failed to prepare features")
            return False
        
        # Extract attention weights
        attention_data = extractor.extract_attention_weights(X_selected, variant_info)
        
        if attention_data is None:
            print("❌ Failed to extract attention")
            return False
        
        # Save results
        saved_files, summary_file, metadata_file = extractor.save_attention_data(attention_data)
        
        if saved_files is None:
            print("❌ Failed to save attention data")
            return False
        
        # Generate summary
        extractor.generate_extraction_summary(attention_data)
        
        print(f"\n🎉 ATTENTION EXTRACTION COMPLETED!")
        print("=" * 40)
        print(f"✅ Processed {len(attention_data)} variants")
        print(f"📁 Results saved to: {extractor.attention_dir}")
        print(f"📋 Files created: {len(saved_files)} individual + summary + metadata")
        
        print(f"\n🎯 Ready for next step:")
        print(f"   python src/analysis/attention_analyzer.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Attention extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)