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
            # Import the model class
            from model.tabnet_prostate_variant_classifier import ProstateVariantTabNet
            
            # Create model instance and load
            self.model = ProstateVariantTabNet()
            self.model.load_model(self.model_path)
            
            print(f"✅ Model loaded successfully")
            print(f"📊 Features: {len(self.model.feature_names)}")
            print(f"🎯 Classes: {list(self.model.label_encoder.classes_)}")
            print(f"🏗️  Architecture: n_d={self.model.n_d}, n_a={self.model.n_a}, n_steps={self.model.n_steps}")
            
            self.feature_names = self.model.feature_names
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_selected_variants(self):
        """Load the variants selected for analysis"""
        print("\n📋 LOADING SELECTED VARIANTS")
        print("-" * 30)
        
        if not os.path.exists(self.selected_variants_path):
            raise FileNotFoundError(f"Selected variants not found: {self.selected_variants_path}")
        
        try:
            selected_df = pd.read_csv(self.selected_variants_path)
            print(f"✅ Loaded {len(selected_df)} selected variants")
            
            # Display selection summary
            if 'selection_category' in selected_df.columns:
                category_counts = selected_df['selection_category'].value_counts()
                print("📊 Variant categories:")
                for category, count in category_counts.items():
                    print(f"   {category.title()}: {count}")
            
            return selected_df
            
        except Exception as e:
            raise Exception(f"Error loading selected variants: {e}")

    def prepare_variant_features(self, selected_df):
        """Prepare feature matrix for the selected variants matching model training"""
        print("\n🔧 PREPARING VARIANT FEATURES")
        print("-" * 30)
        
        # Need to process selected variants the same way as training
        # This means applying the same encoding and feature selection
        
        try:
            # Load full dataset to get proper feature processing
            full_df = pd.read_csv(self.dataset_path, low_memory=False)
            print(f"✅ Loaded full dataset: {len(full_df):,} variants")
            
            # Get indices of selected variants by matching key features
            # Using multiple columns to ensure unique matching
            key_cols = ['chromosome', 'position']
            if 'reference_allele' in selected_df.columns and 'reference_allele' in full_df.columns:
                key_cols.append('reference_allele')
            if 'alternate_allele' in selected_df.columns and 'alternate_allele' in full_df.columns:
                key_cols.append('alternate_allele')
            
            # Find matching rows in full dataset
            selected_indices = []
            for _, variant in selected_df.iterrows():
                # Create match condition
                match_condition = True
                for col in key_cols:
                    if col in variant and col in full_df.columns:
                        match_condition = match_condition & (full_df[col] == variant[col])
                
                matched_indices = full_df[match_condition].index.tolist()
                if matched_indices:
                    selected_indices.append(matched_indices[0])  # Take first match
                else:
                    print(f"⚠️  Warning: Could not find match for variant at {variant.get('chromosome', 'unknown')}:{variant.get('position', 'unknown')}")
            
            if not selected_indices:
                raise Exception("No matching variants found in full dataset")
            
            print(f"✅ Matched {len(selected_indices)} variants in full dataset")
            
            # Extract matched variants from full dataset
            matched_variants = full_df.iloc[selected_indices].copy()
            
            # Process features using the same pipeline as training
            X, y = self.model.load_data()  # This processes the full dataset
            
            # Extract the same rows from processed data
            X_selected = X.iloc[selected_indices].copy()
            y_selected = y[selected_indices] if y is not None else None
            
            print(f"✅ Prepared feature matrix: {X_selected.shape[0]} variants × {X_selected.shape[1]} features")
            
            # Verify feature names match
            if list(X_selected.columns) != self.feature_names:
                print("⚠️  Warning: Feature names don't match exactly")
                # Reorder to match training
                X_selected = X_selected[self.feature_names]
            
            return X_selected, y_selected, matched_variants
            
        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def extract_attention_weights(self, X_selected, variant_info):
        """Extract attention weights using TabNet's explain function"""
        print(f"\n🧠 EXTRACTING ATTENTION WEIGHTS")
        print("-" * 35)
        print(f"Processing {len(X_selected)} selected variants across {self.model.n_steps} decision steps")
        
        try:
            # Get predictions and explanations
            results = self.model.predict_with_explanation(X_selected, self.feature_names)
            
            predictions = results['predictions']
            explanations = results['explanations']  # Global feature importance
            attention_masks = results['attention_masks']  # Step-wise attention
            
            print(f"✅ Extracted attention data:")
            print(f"   Predictions: {len(predictions)} variants")
            print(f"   Global explanations shape: {explanations.shape}")
            print(f"   Attention masks shape: {attention_masks.shape}")
            print(f"   Decision steps: {attention_masks.shape[0]}")
            
            # Process each variant
            attention_data = []
            
            for i, (prediction, variant_row) in enumerate(zip(predictions, variant_info.iterrows())):
                variant_idx, variant = variant_row
                
                # Get variant identifier
                variant_id = f"variant_{i+1:02d}"
                if 'SYMBOL' in variant:
                    variant_id += f"_{variant['SYMBOL']}"
                if 'selection_category' in variant:
                    variant_id += f"_{variant['selection_category']}"
                
                print(f"   Processing {variant_id}: {prediction}")
                
                # Extract attention for this variant
                variant_attention = {
                    'variant_id': variant_id,
                    'variant_index': i,
                    'prediction': prediction,
                    'category': variant.get('selection_category', 'unknown'),
                    'gene': variant.get('SYMBOL', 'unknown'),
                    'chromosome': variant.get('chromosome', 'unknown'),
                    'position': variant.get('position', 'unknown'),
                    'global_importance': explanations[i],  # Global feature importance for this variant
                    'step_attention': attention_masks[:, i, :],  # Attention across all steps for this variant
                    'feature_names': self.feature_names
                }
                
                attention_data.append(variant_attention)
            
            return attention_data
            
        except Exception as e:
            print(f"❌ Error extracting attention: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_attention_data(self, attention_data):
        """Save attention weights for each variant"""
        print(f"\n💾 SAVING ATTENTION DATA")
        print("-" * 25)
        
        saved_files = []
        
        try:
            for variant_data in attention_data:
                variant_id = variant_data['variant_id']
                
                # Save detailed attention data
                attention_file = os.path.join(self.attention_dir, f"{variant_id}_attention.csv")
                
                # Create comprehensive attention DataFrame
                attention_df = pd.DataFrame({
                    'feature': variant_data['feature_names'],
                    'global_importance': variant_data['global_importance']
                })
                
                # Add step-wise attention
                step_attention = variant_data['step_attention']
                for step in range(step_attention.shape[0]):
                    attention_df[f'step_{step+1}_attention'] = step_attention[step, :]
                
                # Add variant metadata
                attention_df['variant_id'] = variant_id
                attention_df['prediction'] = variant_data['prediction']
                attention_df['category'] = variant_data['category']
                attention_df['gene'] = variant_data['gene']
                
                # Sort by global importance
                attention_df = attention_df.sort_values('global_importance', ascending=False)
                
                # Save
                attention_df.to_csv(attention_file, index=False)
                saved_files.append(attention_file)
                print(f"   ✅ {variant_id}")
            
            # Save summary data
            summary_file = os.path.join(self.attention_dir, "attention_summary.csv")
            
            summary_data = []
            for variant_data in attention_data:
                # Get top 5 features for each variant
                top_features_idx = np.argsort(variant_data['global_importance'])[-5:][::-1]
                top_features = [variant_data['feature_names'][i] for i in top_features_idx]
                top_importances = variant_data['global_importance'][top_features_idx]
                
                summary_data.append({
                    'variant_id': variant_data['variant_id'],
                    'prediction': variant_data['prediction'],
                    'category': variant_data['category'],
                    'gene': variant_data['gene'],
                    'top_feature_1': top_features[0],
                    'top_importance_1': top_importances[0],
                    'top_feature_2': top_features[1],
                    'top_importance_2': top_importances[1],
                    'top_feature_3': top_features[2],
                    'top_importance_3': top_importances[2],
                    'top_feature_4': top_features[3],
                    'top_importance_4': top_importances[3],
                    'top_feature_5': top_features[4],
                    'top_importance_5': top_importances[4]
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_file, index=False)
            print(f"   ✅ Summary: attention_summary.csv")
            
            # Save metadata
            metadata_file = os.path.join(self.attention_dir, "extraction_metadata.txt")
            with open(metadata_file, 'w') as f:
                f.write("TABNET ATTENTION EXTRACTION METADATA\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Extraction date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model used: {self.model_path}\n")
                f.write(f"Variants processed: {len(attention_data)}\n")
                f.write(f"Features analyzed: {len(self.feature_names)}\n")
                f.write(f"Decision steps: {self.model.n_steps}\n\n")
                
                f.write("VARIANT SUMMARY:\n")
                for variant_data in attention_data:
                    f.write(f"   {variant_data['variant_id']}: {variant_data['prediction']} ({variant_data['category']})\n")
                
                f.write(f"\nFILES CREATED:\n")
                for file_path in saved_files:
                    f.write(f"   {os.path.basename(file_path)}\n")
                f.write(f"   {os.path.basename(summary_file)}\n")
                
                f.write(f"\nNEXT STEPS:\n")
                f.write(f"1. Analyze attention patterns: python src/analysis/attention_analyzer.py\n")
                f.write(f"2. Generate final results: python src/analysis/results_generator.py\n")
            
            print(f"   ✅ Metadata: extraction_metadata.txt")
            
            return saved_files, summary_file, metadata_file
            
        except Exception as e:
            print(f"❌ Error saving attention data: {e}")
            return None, None, None

    def generate_extraction_summary(self, attention_data):
        """Generate summary of extraction results"""
        print(f"\n📊 EXTRACTION SUMMARY")
        print("-" * 20)
        
        total_variants = len(attention_data)
        categories = [v['category'] for v in attention_data]
        predictions = [v['prediction'] for v in attention_data]
        
        print(f"✅ Successfully extracted attention for {total_variants} variants")
        
        # Category breakdown
        from collections import Counter
        category_counts = Counter(categories)
        print(f"📋 Categories:")
        for category, count in category_counts.items():
            print(f"   {category.title()}: {count}")
        
        # Prediction breakdown
        prediction_counts = Counter(predictions)
        print(f"🎯 Predictions:")
        for prediction, count in prediction_counts.items():
            print(f"   {prediction}: {count}")
        
        # Feature analysis preview
        all_importances = np.concatenate([v['global_importance'] for v in attention_data])
        avg_importance = np.mean(all_importances.reshape(total_variants, -1), axis=0)
        top_features_idx = np.argsort(avg_importance)[-5:][::-1]
        
        print(f"🔝 Top 5 average features across all variants:")
        for i, idx in enumerate(top_features_idx, 1):
            feature_name = self.feature_names[idx]
            importance = avg_importance[idx]
            print(f"   {i}. {feature_name}: {importance:.4f}")

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