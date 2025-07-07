#!/usr/bin/env python3
"""
TabNet Attention Weight Extractor
Extracts attention weights from trained TabNet model for selected variants

Location: /u/aa107/uiuc-cancer-research/src/analysis/attention_extractor.py
Author: PhD Research Student, University of Illinois

Purpose: Use TabNet's built-in explain() function to extract attention weights
across all 6 decision steps for the selected representative variants.
"""

import pandas as pd
import numpy as np
import os
import pickle
import sys
from pathlib import Path
from datetime import datetime
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
            if hasattr(self.model, 'device_'):
                current_device = getattr(self.model, 'device_', 'unknown')
                print(f"📍 Model current device: {current_device}")
            
            # For TabNet models, ensure compatibility with current environment
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

    def prepare_variant_features(self, selected_df):
        """Prepare features for the selected variants"""
        print("\n🔧 PREPARING VARIANT FEATURES")
        print("-" * 35)
        
        # Load the full dataset to get features
        print("📊 Loading full dataset for feature extraction...")
        full_df = pd.read_csv(self.dataset_path)
        
        # Check if dataset has categorical features (indicates need for encoding)
        categorical_cols = full_df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"⚠️  Dataset contains {len(categorical_cols)} categorical columns")
            print("💡 This dataset needs categorical encoding before scaling")
            print("📋 Expected: Fully preprocessed numeric dataset from training")
            print("❌ The training pipeline used a different preprocessed dataset")
            return None, None, None
        
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
        
        # Prepare features for TabNet - use the exact 56 feature names from training
        feature_cols = self.feature_names
        
        # Check which features exist in the dataset
        available_features = [col for col in feature_cols if col in matched_df.columns]
        missing_features = [col for col in feature_cols if col not in matched_df.columns]
        
        if missing_features:
            print(f"⚠️  Missing {len(missing_features)} features in dataset:")
            for feat in missing_features[:5]:  # Show first 5 missing
                print(f"     - {feat}")
            if len(missing_features) > 5:
                print(f"     ... and {len(missing_features) - 5} more")
            print(f"📊 Using {len(available_features)} available features")
            feature_cols = available_features
        
        # Extract ALL features (including categorical) - same as training
        X_selected = matched_df[feature_cols].copy()
        y_selected = matched_df.get('variant_classification', ['Unknown'] * len(matched_df))
        
        # Handle missing values (same as training)
        X_selected = X_selected.fillna(0)
        
        # Apply the EXACT same preprocessing pipeline used during training
        if self.scaler is not None:
            print("🔄 Applying trained scaler to preprocessed features...")
            try:
                # The scaler expects exactly the same features it was trained on
                X_scaled = self.scaler.transform(X_selected)
                X_selected = pd.DataFrame(X_scaled, columns=X_selected.columns, index=X_selected.index)
                print(f"✅ Features scaled successfully: {X_selected.shape}")
            except Exception as e:
                print(f"❌ Scaling failed: {e}")
                print("💡 This indicates the dataset needs categorical encoding first")
                return None, None, None
        else:
            print("⚠️  No scaler available - using raw features")
        
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
        
        # Update feature names to match the actual features used
        self.actual_feature_names = list(X_selected.columns)
        
        return X_selected, y_selected, variant_info

    def extract_attention_weights(self, X_selected, variant_info):
        """Extract attention weights using TabNet's explain function"""
        print("\n🧠 EXTRACTING ATTENTION WEIGHTS")
        print("-" * 35)
        
        try:
            # Use TabNet's explain function to get attention weights
            print("🔍 Getting attention weights from TabNet...")
            
            # Convert to numpy if pandas DataFrame
            if isinstance(X_selected, pd.DataFrame):
                X_array = X_selected.values
            else:
                X_array = X_selected
            
            # Get attention weights - this returns masks for each decision step
            masks, _ = self.model.explain(X_array)
            
            print(f"✅ Extracted attention for {len(variant_info)} variants")
            print(f"📊 Decision steps: {len(masks)}")
            print(f"📊 Mask shape per step: {masks[0].shape}")
            
            # Process attention data
            attention_data = []
            
            for variant_idx, variant in enumerate(variant_info):
                variant_attention = {
                    'variant_info': variant,
                    'attention_by_step': {}
                }
                
                # Extract attention for each decision step
                for step_idx, mask in enumerate(masks):
                    step_attention = mask[variant_idx]  # Attention for this variant at this step
                    
                    # Create feature-attention pairs
                    feature_attention = {}
                    for feat_idx, attention_weight in enumerate(step_attention):
                        if feat_idx < len(self.actual_feature_names):
                            feature_name = self.actual_feature_names[feat_idx]
                            feature_attention[feature_name] = float(attention_weight)
                    
                    variant_attention['attention_by_step'][f'step_{step_idx+1}'] = feature_attention
                
                attention_data.append(variant_attention)
            
            print(f"✅ Processed attention data for all variants")
            return attention_data
            
        except Exception as e:
            print(f"❌ Error extracting attention: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_attention_data(self, attention_data):
        """Save attention weights to files"""
        print("\n💾 SAVING ATTENTION DATA")
        print("-" * 25)
        
        saved_files = []
        
        # Save individual variant attention files
        for variant_data in attention_data:
            variant_info = variant_data['variant_info']
            variant_id = variant_info['variant_id']
            gene = variant_info['gene']
            classification = variant_info['classification']
            
            # Create filename
            filename = f"{variant_id}_{classification}_{gene}_attention.csv"
            filepath = os.path.join(self.attention_dir, filename)
            
            # Prepare data for CSV
            csv_data = []
            for step_name, feature_attention in variant_data['attention_by_step'].items():
                for feature, attention in feature_attention.items():
                    csv_data.append({
                        'variant_id': variant_id,
                        'decision_step': step_name,
                        'feature': feature,
                        'attention_weight': attention
                    })
            
            # Save to CSV
            if csv_data:
                csv_df = pd.DataFrame(csv_data)
                csv_df.to_csv(filepath, index=False)
                saved_files.append(filepath)
                print(f"   ✅ {filename}")
        
        # Create summary file
        summary_data = []
        for variant_data in attention_data:
            variant_info = variant_data['variant_info']
            
            # Calculate average attention across all steps for each feature
            feature_totals = {}
            step_count = len(variant_data['attention_by_step'])
            
            for step_attention in variant_data['attention_by_step'].values():
                for feature, attention in step_attention.items():
                    if feature not in feature_totals:
                        feature_totals[feature] = 0
                    feature_totals[feature] += attention
            
            # Average and get top features
            avg_attention = {feat: total/step_count for feat, total in feature_totals.items()}
            top_features = sorted(avg_attention.items(), key=lambda x: x[1], reverse=True)[:10]
            
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