#!/usr/bin/env python3
"""
TabNet Attention Weight Extractor - CORRECTED VERSION
Extracts attention weights from trained TabNet model for selected variants

Location: /u/aa107/uiuc-cancer-research/src/analysis/attention_extractor.py
Author: PhD Research Student, University of Illinois

CRITICAL FIX: Implements complete categorical encoding pipeline from training
to resolve preprocessing mismatch between training and inference phases.
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
            self.model_path = "/u/aa107/scratch/tabnet_model_20250706_151358.pkl"
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
                print(f"📊 Features from pickle: {len(self.feature_names)}")
                print(f"📊 Scaler available: {self.scaler is not None}")
                print(f"📊 Label encoder available: {self.label_encoder is not None}")
            else:
                # If the pickle file contains the model directly
                self.model = model_data
                print("✅ Loaded model directly from pickle")
                
                # Load feature names from metadata file
                metadata_path = self.model_path.replace('.pkl', '_metadata.txt')
                if os.path.exists(metadata_path):
                    print(f"📋 Loading feature names from metadata: {metadata_path}")
                    with open(metadata_path, 'r') as f:
                        content = f.read()
                        # Extract feature names from metadata if available
                        if 'Feature names:' in content:
                            features_line = [line for line in content.split('\n') if 'Feature names:' in line][0]
                            self.feature_names = eval(features_line.split('Feature names:')[1].strip())
                            print(f"✅ Loaded {len(self.feature_names)} feature names from metadata")
            
            # Handle device placement after loading
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"🎯 Target device: {device}")
            
            # Move model to appropriate device if needed
            if hasattr(self.model, 'to'):
                print(f"🔄 Moving model to {device}...")
                self.model = self.model.to(device)
            elif hasattr(self.model, 'cpu') and device.type == 'cpu':
                print("🔄 Moving model to CPU...")
                self.model = self.model.cpu()
            elif hasattr(self.model, 'cuda') and device.type == 'cuda':
                print("🔄 Moving model to CUDA...")
                self.model = self.model.cuda()
            
            print(f"✅ Model loaded successfully")
            print(f"📊 Features: {len(self.feature_names) if self.feature_names else 'Unknown'}")
            
            # Only load from dataset if no feature names found anywhere
            if not self.feature_names:
                print("⚠️  No feature names found - loading from dataset as fallback...")
                dataset_df = pd.read_csv(self.dataset_path)
                # Exclude target columns
                exclude_cols = ['variant_classification', 'CLIN_SIG', 'chromosome', 'position']
                self.feature_names = [col for col in dataset_df.columns if col not in exclude_cols]
                print(f"📊 Features loaded from dataset: {len(self.feature_names)}")
            
            # Verify model type
            if hasattr(self.model, 'explain'):
                print("✅ TabNet explain() method available")
            else:
                print("⚠️  Model may not support attention extraction")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_selected_variants(self):
        """Load the selected variants from variant_selector.py"""
        print("\n📋 LOADING SELECTED VARIANTS")
        print("-" * 30)
        
        if not os.path.exists(self.selected_variants_path):
            raise FileNotFoundError(f"Selected variants not found: {self.selected_variants_path}")
        
        selected_df = pd.read_csv(self.selected_variants_path)
        print(f"✅ Loaded {len(selected_df)} selected variants")
        
        # Show selection breakdown
        if 'selection_category' in selected_df.columns:
            category_counts = selected_df['selection_category'].value_counts()
            for category, count in category_counts.items():
                print(f"   {category.title()}: {count}")
        
        return selected_df

    def _encode_categorical_features(self, df):
        """Encode categorical features using the EXACT same logic as training"""
        print("🔧 APPLYING CATEGORICAL ENCODING (TRAINING PIPELINE)")
        print("-" * 50)
        df_encoded = df.copy()
        
        # Encode Consequence using severity ranking
        if 'Consequence' in df.columns:
            print("🔹 Encoding Consequence with severity rankings...")
            df_encoded['Consequence_encoded'] = df['Consequence'].map(
                lambda x: self.CONSEQUENCE_SEVERITY.get(str(x).lower(), 0) if pd.notna(x) else 0
            )
            df_encoded = df_encoded.drop('Consequence', axis=1)
            df_encoded = df_encoded.rename(columns={'Consequence_encoded': 'Consequence'})
        
        # Encode CLIN_SIG using severity ranking (for target creation, not features)
        if 'CLIN_SIG' in df.columns:
            print("🔹 Encoding CLIN_SIG with severity rankings...")
            df_encoded['CLIN_SIG_encoded'] = df['CLIN_SIG'].map(
                lambda x: self.CLIN_SIG_SEVERITY.get(str(x).lower(), 0) if pd.notna(x) else 0
            )
            df_encoded = df_encoded.drop('CLIN_SIG', axis=1)
            df_encoded = df_encoded.rename(columns={'CLIN_SIG_encoded': 'CLIN_SIG'})
        
        # Encode IMPACT using severity ranking
        if 'IMPACT' in df.columns:
            print("🔹 Encoding IMPACT with severity rankings...")
            df_encoded['IMPACT_encoded'] = df['IMPACT'].map(
                lambda x: self.IMPACT_SEVERITY.get(str(x).upper(), 1) if pd.notna(x) else 1
            )
            df_encoded = df_encoded.drop('IMPACT', axis=1)
            df_encoded = df_encoded.rename(columns={'IMPACT_encoded': 'IMPACT'})
        
        # Parse SIFT scores from format "deleterious(0.01)" -> 0.01
        if 'SIFT' in df.columns:
            print("🔹 Parsing SIFT scores...")
            def parse_sift(x):
                if pd.isna(x) or x == '' or str(x).lower() == 'unknown':
                    return np.nan
                match = re.search(r'\(([\d.]+)\)', str(x))
                return float(match.group(1)) if match else np.nan
            
            df_encoded['SIFT_parsed'] = df['SIFT'].apply(parse_sift)
            df_encoded = df_encoded.drop('SIFT', axis=1)
            df_encoded = df_encoded.rename(columns={'SIFT_parsed': 'SIFT'})
        
        # Parse PolyPhen scores from format "probably_damaging(0.95)" -> 0.95
        if 'PolyPhen' in df.columns:
            print("🔹 Parsing PolyPhen scores...")
            def parse_polyphen(x):
                if pd.isna(x) or x == '' or str(x).lower() == 'unknown':
                    return np.nan
                match = re.search(r'\(([\d.]+)\)', str(x))
                return float(match.group(1)) if match else np.nan
            
            df_encoded['PolyPhen_parsed'] = df['PolyPhen'].apply(parse_polyphen)
            df_encoded = df_encoded.drop('PolyPhen', axis=1)
            df_encoded = df_encoded.rename(columns={'PolyPhen_parsed': 'PolyPhen'})
        
        # Encode AlphaMissense class
        if 'alphamissense_class' in df.columns:
            print("🔹 Encoding AlphaMissense classes...")
            am_encoding = {'likely_pathogenic': 2, 'ambiguous': 1, 'likely_benign': 0}
            df_encoded['alphamissense_class_encoded'] = df['alphamissense_class'].map(
                lambda x: am_encoding.get(str(x).lower(), 1) if pd.notna(x) else 1
            )
            df_encoded = df_encoded.drop('alphamissense_class', axis=1)
            df_encoded = df_encoded.rename(columns={'alphamissense_class_encoded': 'alphamissense_class'})
        
        # Convert remaining categorical to numeric using label encoding
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"🔹 Label encoding {len(categorical_cols)} remaining categorical columns...")
            for col in categorical_cols:
                if col in df_encoded.columns:
                    # Handle "Unknown" values
                    df_encoded[col] = df_encoded[col].replace(['Unknown', 'unknown', ''], np.nan)
                    # Fill missing with most frequent value
                    if not df_encoded[col].dropna().empty:
                        most_frequent = df_encoded[col].mode().iloc[0] if not df_encoded[col].mode().empty else 'missing'
                        df_encoded[col] = df_encoded[col].fillna(most_frequent)
                        # Label encode
                        le = LabelEncoder()
                        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    else:
                        df_encoded[col] = 0
        
        print(f"✅ Categorical encoding completed - all features now numeric")
        return df_encoded

    def prepare_variant_features(self, selected_df):
        """Prepare features for the selected variants with COMPLETE preprocessing pipeline"""
        print("\n🔧 PREPARING VARIANT FEATURES")
        print("-" * 35)
        
        # Load the full dataset to get features
        print("📊 Loading full dataset for feature extraction...")
        full_df = pd.read_csv(self.dataset_path)
        
        # Check if dataset has categorical features (this is EXPECTED for raw data)
        categorical_cols = full_df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"📋 Dataset contains {len(categorical_cols)} categorical columns (expected)")
            print("🔧 Applying complete categorical encoding pipeline...")
        else:
            print("📋 Dataset appears to be already preprocessed")
        
        # Create lookup keys for selected variants
        if 'chromosome' in selected_df.columns and 'position' in selected_df.columns:
            selected_keys = set(zip(selected_df['chromosome'], selected_df['position']))
            
            # Find matching rows in full dataset
            matching_mask = full_df.apply(
                lambda row: (row['chromosome'], row['position']) in selected_keys, axis=1
            )
            
            matched_df = full_df[matching_mask].copy()
            print(f"✅ Found {len(matched_df)} matching variants in dataset")
        else:
            print("⚠️  No chromosome/position columns - using first N rows")
            matched_df = full_df.head(len(selected_df)).copy()
        
        if len(matched_df) == 0:
            print("❌ No matching variants found")
            return None, None, None
        
        # CRITICAL FIX: Apply categorical encoding BEFORE feature selection
        if len(categorical_cols) > 0:
            matched_df_encoded = self._encode_categorical_features(matched_df)
        else:
            matched_df_encoded = matched_df.copy()
        
        # Prepare features for TabNet - use the exact 56 feature names from training
        feature_cols = self.feature_names
        
        # Check which features exist in the dataset after encoding
        available_features = [col for col in feature_cols if col in matched_df_encoded.columns]
        missing_features = [col for col in feature_cols if col not in matched_df_encoded.columns]
        
        if missing_features:
            print(f"⚠️  Missing {len(missing_features)} features in dataset:")
            for feat in missing_features[:5]:  # Show first 5 missing
                print(f"     - {feat}")
            if len(missing_features) > 5:
                print(f"     ... and {len(missing_features) - 5} more")
            print(f"📊 Using {len(available_features)} available features")
            feature_cols = available_features
        
        # Extract features (now all numeric after encoding)
        X_selected = matched_df_encoded[feature_cols].copy()
        y_selected = matched_df_encoded.get('variant_classification', ['Unknown'] * len(matched_df_encoded))
        
        # Handle missing values (same as training)
        X_selected = X_selected.fillna(0)
        
        # Verify all features are numeric before scaling
        numeric_check = X_selected.select_dtypes(include=['object']).columns
        if len(numeric_check) > 0:
            print(f"❌ ERROR: {len(numeric_check)} features still categorical after encoding!")
            for col in numeric_check[:5]:
                print(f"     - {col}: {X_selected[col].dtype}")
            return None, None, None
        
        # Apply scaling using the EXACT same scaler from training
        if self.scaler is not None:
            print("🔄 Applying trained scaler to encoded features...")
            try:
                # The scaler expects exactly the same features it was trained on
                X_scaled = self.scaler.transform(X_selected)
                X_selected = pd.DataFrame(X_scaled, columns=X_selected.columns, index=X_selected.index)
                print(f"✅ Features scaled successfully: {X_selected.shape}")
            except Exception as e:
                print(f"❌ Scaling failed: {e}")
                print("💡 Feature dimensions or types don't match training")
                return None, None, None
        else:
            print("⚠️  No scaler available - using raw encoded features")
        
        # Create variant info for tracking
        variant_info = []
        for i, (_, row) in enumerate(matched_df.iterrows()):
            info = {
                'variant_id': f"variant_{i+1:02d}",
                'chromosome': row.get('chromosome', 'unknown'),
                'position': row.get('position', 'unknown'),
                'gene': row.get('SYMBOL', 'unknown'),
                'classification': row.get('variant_classification', 'unknown')
            }
            variant_info.append(info)
        
        print(f"✅ Prepared features: {X_selected.shape}")
        print(f"📊 Feature columns: {len(X_selected.columns)}")
        print(f"🧮 All features numeric: {X_selected.select_dtypes(include=[np.number]).shape[1] == X_selected.shape[1]}")
        
        # Update feature names to match the actual features used
        self.actual_feature_names = list(X_selected.columns)
        
        return X_selected, y_selected, variant_info

    def extract_attention_weights(self, X_selected, variant_info):
        """Extract attention weights using TabNet's explain() method"""
        print("\n🧠 EXTRACTING ATTENTION WEIGHTS")
        print("-" * 35)
        
        if self.model is None:
            print("❌ No model loaded")
            return None
        
        if not hasattr(self.model, 'explain'):
            print("❌ Model does not support attention extraction")
            return None
        
        try:
            # Convert to numpy array for TabNet
            X_numpy = X_selected.values
            print(f"📊 Input shape for TabNet: {X_numpy.shape}")
            
            # Extract attention weights using TabNet's explain method
            print("🔍 Calling TabNet.explain()...")
            M_explain, masks = self.model.explain(X_numpy)
            
            print(f"✅ Attention extraction successful!")
            print(f"📊 Explanation shape: {M_explain.shape}")
            print(f"📊 Masks shape: {len(masks)} decision steps")
            
            # Process attention data for each variant
            attention_data = []
            feature_names = self.actual_feature_names if self.actual_feature_names else self.feature_names
            
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
                print(f"   ✅ Processed {variant['variant_id']}")
            
            print(f"✅ Extracted attention for {len(attention_data)} variants")
            return attention_data
            
        except Exception as e:
            print(f"❌ Attention extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_attention_data(self, attention_data):
        """Save attention weights to files"""
        print("\n💾 SAVING ATTENTION DATA")
        print("-" * 25)
        
        if not attention_data:
            print("❌ No attention data to save")
            return None, None, None
        
        saved_files = []
        
        # Save individual attention files for each variant
        for variant_data in attention_data:
            variant_id = variant_data['variant_info']['variant_id']
            
            # Create attention dataframe
            attention_rows = []
            for step_idx, step_data in enumerate(variant_data['attention_by_step']):
                for feature, attention in step_data.items():
                    attention_rows.append({
                        'decision_step': step_idx + 1,
                        'feature': feature,
                        'attention_weight': attention
                    })
            
            # Save to CSV
            attention_df = pd.DataFrame(attention_rows)
            file_path = os.path.join(self.attention_dir, f"{variant_id}_attention.csv")
            attention_df.to_csv(file_path, index=False)
            saved_files.append(file_path)
            print(f"   ✅ {variant_id}_attention.csv")
        
        # Create summary with top features per variant
        summary_data = []
        for variant_data in attention_data:
            variant_info = variant_data['variant_info']
            
            # Calculate average attention across all steps
            feature_attention = {}
            for step_data in variant_data['attention_by_step']:
                for feature, attention in step_data.items():
                    if feature not in feature_attention:
                        feature_attention[feature] = []
                    feature_attention[feature].append(attention)
            
            # Average and sort
            avg_attention = {feature: np.mean(weights) for feature, weights in feature_attention.items()}
            top_features = sorted(avg_attention.items(), key=lambda x: x[1], reverse=True)[:3]
            
            summary_data.append({
                'variant_id': variant_info['variant_id'],
                'gene': variant_info['gene'],
                'classification': variant_info['classification'],
                'top_feature_1': top_features[0][0] if len(top_features) > 0 else '',
                'top_attention_1': top_features[0][1] if len(top_features) > 0 else 0,
                'top_feature_2': top_features[1][0] if len(top_features) > 1 else '',
                'top_attention_2': top_features[1][1] if len(top_features) > 1 else 0,
                'top_feature_3': top_features[2][0] if len(top_features) > 2 else '',
                'top_attention_3': top_features[2][1] if len(top_features) > 2 else 0
            })
        
        summary_file = os.path.join(self.attention_dir, "attention_summary.csv")
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        print(f"   ✅ attention_summary.csv")
        
        # Save metadata
        metadata = {
            'extraction_date': datetime.now().isoformat(),
            'model_path': self.model_path,
            'variants_processed': len(attention_data),
            'features_analyzed': len(self.actual_feature_names) if self.actual_feature_names else len(self.feature_names),
            'decision_steps': len(attention_data[0]['attention_by_step']) if attention_data else 0
        }
        
        metadata_file = os.path.join(self.attention_dir, "extraction_metadata.json")
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ extraction_metadata.json")
        
        return saved_files, summary_file, metadata_file

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
        
        # Analyze feature attention patterns
        if attention_data:
            sample_variant = attention_data[0]
            step_count = len(sample_variant['attention_by_step'])
            feature_count = len(self.actual_feature_names) if self.actual_feature_names else len(self.feature_names)
            
            print(f"\n🧠 Attention analysis details:")
            print(f"   Decision steps: {step_count}")
            print(f"   Features per step: {feature_count}")
            print(f"   Total attention weights: {len(attention_data) * step_count * feature_count}")

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