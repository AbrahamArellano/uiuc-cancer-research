#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import pickle
import torch

class TabNetAttentionExtractor:
    def __init__(self, model_path, analysis_dir):
        """Initialize TabNet attention extractor for exclusive GPU access"""
        self.model_path = model_path
        self.analysis_dir = analysis_dir
        self.attention_dir = os.path.join(analysis_dir, 'attention_weights')
        self.selected_variants_path = os.path.join(analysis_dir, 'selected_variants.csv')
        
        # Create attention weights directory
        os.makedirs(self.attention_dir, exist_ok=True)
        
        # Initialize model components
        self.model = None
        self.feature_names = []
        self.actual_feature_names = []
        self.scaler = None
        self.label_encoder = None
        
        print("🔍 TabNet Attention Extractor Initialized")
        print(f"📁 Model: {self.model_path}")
        print(f"📁 Analysis: {self.analysis_dir}")
        print(f"📁 Attention output: {self.attention_dir}")

    def load_trained_model(self):
        """Load the trained TabNet model with exclusive GPU access"""
        print("\n🤖 LOADING TRAINED TABNET MODEL")
        print("-" * 40)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            # Import required modules
            import torch
            from pytorch_tabnet.tab_model import TabNetClassifier
            
            # Check GPU availability
            if not torch.cuda.is_available():
                print("❌ CUDA not available")
                return False
            
            # Set device to GPU (we have exclusive access)
            device = 'cuda'
            print(f"🎯 Using device: {device} (exclusive access)")
            
            # Load model from pickle file
            print("📁 Loading model from pickle file...")
            with open(self.model_path, 'rb') as f:
                model_data = torch.load(f, map_location=device, weights_only=False)
            
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
            
            # Set model to GPU
            if hasattr(self.model, 'device_name'):
                self.model.device_name = device
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Display model info
            print(f"📊 Features from pickle: {len(self.feature_names)}")
            print(f"📊 Scaler available: {self.scaler is not None}")
            print(f"📊 Label encoder available: {self.label_encoder is not None}")
            print(f"📊 Model device: {device}")
            
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

    def prepare_variant_features(self, selected_variants):
        """Prepare feature matrix for selected variants"""
        print("\n🧮 PREPARING VARIANT FEATURES")
        print("-" * 30)
        
        try:
            # Get feature columns (exclude metadata)
            exclude_cols = ['variant_id', 'selection_category', 'gene_symbol', 
                          'clin_sig', 'consequence', 'chromosome', 'position']
            
            feature_cols = [col for col in selected_variants.columns if col not in exclude_cols]
            self.actual_feature_names = feature_cols
            
            # Extract features
            X_selected = selected_variants[feature_cols].copy()
            
            # Handle missing values
            X_selected = X_selected.fillna(0)
            
            # Ensure all features are numeric
            for col in X_selected.columns:
                X_selected[col] = pd.to_numeric(X_selected[col], errors='coerce').fillna(0)
            
            # Apply scaler if available
            if self.scaler is not None:
                print("🔧 Applying saved scaler...")
                X_selected = pd.DataFrame(
                    self.scaler.transform(X_selected),
                    columns=X_selected.columns,
                    index=X_selected.index
                )
            
            # Prepare variant info for tracking
            variant_info = []
            for idx, row in selected_variants.iterrows():
                info = {
                    'variant_id': row.get('variant_id', f'variant_{idx}'),
                    'gene_symbol': row.get('gene_symbol', 'Unknown'),
                    'classification': row.get('selection_category', 'Unknown'),
                    'clin_sig': row.get('clin_sig', 'Unknown')
                }
                variant_info.append(info)
            
            print(f"✅ Prepared features: {X_selected.shape}")
            print(f"📊 Feature columns: {len(X_selected.columns)}")
            print(f"🧮 All features numeric: {X_selected.select_dtypes(include=[np.number]).shape[1] == X_selected.shape[1]}")
            
            return X_selected, None, variant_info
            
        except Exception as e:
            print(f"❌ Feature preparation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def extract_attention_weights(self, X_selected, variant_info):
        """Extract attention weights using TabNet's explain method with GPU acceleration"""
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
            
            print("🎯 Running attention extraction on GPU (exclusive access)")
            
            # Call explain method - TabNet will use GPU automatically
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
        
        try:
            # Save feature importance scores
            importance_df = pd.DataFrame(
                attention_results['feature_importance'],
                columns=attention_results['feature_names']
            )
            
            # Add variant information
            for i, info in enumerate(attention_results['variant_info']):
                importance_df.loc[i, 'variant_id'] = info['variant_id']
                importance_df.loc[i, 'gene_symbol'] = info['gene_symbol']
                importance_df.loc[i, 'classification'] = info['classification']
            
            # Save importance scores
            importance_file = os.path.join(self.attention_dir, 'feature_importance_scores.csv')
            importance_df.to_csv(importance_file, index=False)
            print(f"✅ Saved importance scores: {importance_file}")
            
            # Save attention masks
            masks_file = os.path.join(self.attention_dir, 'attention_masks.npy')
            np.save(masks_file, attention_results['attention_masks'])
            print(f"✅ Saved attention masks: {masks_file}")
            
            # Save feature names
            features_file = os.path.join(self.attention_dir, 'feature_names.txt')
            with open(features_file, 'w') as f:
                for feature in attention_results['feature_names']:
                    f.write(f"{feature}\n")
            print(f"✅ Saved feature names: {features_file}")
            
            # Generate summary
            summary_file = os.path.join(self.attention_dir, 'extraction_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("TabNet Attention Extraction Summary\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Total variants analyzed: {len(attention_results['variant_info'])}\n")
                f.write(f"Features analyzed: {len(attention_results['feature_names'])}\n")
                f.write(f"Attention masks shape: {attention_results['attention_masks'].shape}\n")
                f.write(f"Feature importance shape: {attention_results['feature_importance'].shape}\n\n")
                
                # Top features by average importance
                avg_importance = np.mean(attention_results['feature_importance'], axis=0)
                top_indices = np.argsort(avg_importance)[-10:][::-1]
                
                f.write("Top 10 Most Important Features:\n")
                f.write("-" * 30 + "\n")
                for i, idx in enumerate(top_indices, 1):
                    feature = attention_results['feature_names'][idx]
                    score = avg_importance[idx]
                    f.write(f"{i:2d}. {feature}: {score:.4f}\n")
            
            print(f"✅ Saved summary: {summary_file}")
            
            print("\n🎉 Attention extraction completed successfully!")
            print(f"📁 Results saved to: {self.attention_dir}")
            
        except Exception as e:
            print(f"❌ Failed to save attention weights: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution pipeline"""
    print("🧠 TABNET ATTENTION EXTRACTION")
    print("=" * 50)
    print("Purpose: Extract attention weights from trained TabNet model")
    print("Input: Selected variants from variant_selector.py")
    print("Output: Attention weights for interpretability analysis")
    print()
    
    # Configuration
    model_path = "/u/aa107/uiuc-cancer-research/results/training/tabnet_training_20250728_075348/tabnet_model_20250728_072312.pkl"
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