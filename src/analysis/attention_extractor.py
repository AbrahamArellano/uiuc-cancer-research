#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import pickle
import torch
import json

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
        """Load the trained TabNet model using the same approach as training validation"""
        print("\n🤖 LOADING TRAINED TABNET MODEL")
        print("-" * 40)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            
            print("📁 Loading TabNet native format (.zip)...")
            
            # Use exact same approach as training validation
            self.model = TabNetClassifier()
            self.model.load_model(self.model_path)
            
            print("✅ Loaded TabNet model from native format")
                        
            # Use exact same dynamic feature selection as training
            self.feature_names = self._select_training_features()
            
            # Create basic metadata
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            self.scaler = StandardScaler()
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(['Benign', 'Pathogenic', 'VUS'])
            
            print(f"✅ Model ready: {len(self.feature_names)} features")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _select_training_features(self):
        """Replicate exact feature selection logic from training"""
        # Load dataset to check available features
        data_path = '/u/aa107/uiuc-cancer-research/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv'
        df = pd.read_csv(data_path, nrows=1)  # Just header check
        
        selected_features = []
        
        # TIER 1: VEP-Corrected Features (4 features) - CLIN_SIG removed
        tier1_features = ['Consequence', 'DOMAINS', 'PUBMED', 'VAR_SYNONYMS']
        for feature in tier1_features:
            if feature in df.columns:
                selected_features.append(feature)
        
        # TIER 2: Core VEP Annotations (10 features)
        tier2_features = ['SYMBOL', 'BIOTYPE', 'CANONICAL', 'PICK', 'HGVSc', 
                         'HGVSp', 'Protein_position', 'Amino_acids', 'Existing_variation', 'VARIANT_CLASS']
        for feature in tier2_features:
            if feature in df.columns:
                selected_features.append(feature)
        
        # TIER 3: AlphaMissense Integration (2 features)
        tier3_features = ['alphamissense_pathogenicity', 'alphamissense_class']
        for feature in tier3_features:
            if feature in df.columns:
                selected_features.append(feature)
        
        # TIER 4: Population Genetics (17 features)
        tier4_features = ['AF', 'AFR_AF', 'AMR_AF', 'EAS_AF', 'EUR_AF', 'SAS_AF',
                         'gnomADe_AF', 'gnomADe_AFR_AF', 'gnomADe_AMR_AF', 'gnomADe_ASJ_AF',
                         'gnomADe_EAS_AF', 'gnomADe_FIN_AF', 'gnomADe_MID_AF', 'gnomADe_NFE_AF',
                         'gnomADe_REMAINING_AF', 'gnomADe_SAS_AF', 'af_1kg']
        for feature in tier4_features:
            if feature in df.columns:
                selected_features.append(feature)
        
        # TIER 5: Functional Predictions (6 features)
        tier5_features = ['IMPACT', 'sift_score', 'polyphen_score', 'SIFT', 'PolyPhen', 'impact_score']
        for feature in tier5_features:
            if feature in df.columns:
                selected_features.append(feature)
        
        # TIER 6: Clinical Context (5 features)
        tier6_features = ['SOMATIC', 'PHENO', 'EXON', 'INTRON', 'CCDS']
        for feature in tier6_features:
            if feature in df.columns:
                selected_features.append(feature)
        
        # TIER 7: Variant Properties (8 features)
        tier7_features = ['ref_length', 'alt_length', 'variant_size', 'is_indel', 
                         'is_snv', 'is_lof', 'is_missense', 'is_synonymous']
        for feature in tier7_features:
            if feature in df.columns:
                selected_features.append(feature)
        
        # TIER 8: Prostate Biology (4 features)
        tier8_features = ['is_important_gene', 'dna_repair_pathway', 'mismatch_repair_pathway', 'hormone_pathway']
        for feature in tier8_features:
            if feature in df.columns:
                selected_features.append(feature)
        
        return selected_features

    def load_selected_variants(self):
        """Load variants selected for attention analysis"""
        print("\n📋 LOADING SELECTED VARIANTS")
        print("-" * 30)
        
        if not os.path.exists(self.selected_variants_path):
            raise FileNotFoundError(f"Selected variants file not found: {self.selected_variants_path}")
        
        try:
            # Load selected variants
            selected_df = pd.read_csv(self.selected_variants_path)
            print(f"✅ Loaded {len(selected_df)} selected variants")
            
            # Display variant distribution
            if 'classification' in selected_df.columns:
                print(f"📊 Variant distribution:")
                for cls, count in selected_df['classification'].value_counts().items():
                    print(f"   {cls}: {count} variants")
            
            return selected_df
            
        except Exception as e:
            print(f"❌ Failed to load selected variants: {e}")
            return None

    def prepare_variant_features(self, selected_variants):
        """Prepare feature matrix for selected variants"""
        print("\n🔧 PREPARING VARIANT FEATURES")
        print("-" * 35)
        
        try:
            # Load full dataset
            data_path = '/u/aa107/uiuc-cancer-research/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv'
            print(f"📁 Loading full dataset from: {data_path}")
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset not found: {data_path}")
            
            full_df = pd.read_csv(data_path, low_memory=False)
            print(f"✅ Loaded full dataset: {len(full_df):,} variants")
            
            # Create variant identifiers for matching
            if 'variant_id' not in selected_variants.columns:
                # Try to create variant IDs from available columns
                if all(col in selected_variants.columns for col in ['CHROM', 'POS', 'REF', 'ALT']):
                    selected_variants['variant_id'] = (
                        selected_variants['CHROM'].astype(str) + '_' +
                        selected_variants['POS'].astype(str) + '_' +
                        selected_variants['REF'].astype(str) + '_' +
                        selected_variants['ALT'].astype(str)
                    )
                else:
                    print("❌ Cannot create variant IDs - missing required columns")
                    return None, None, None
            
            # Match selected variants with full dataset
            if 'variant_id' not in full_df.columns:
                if all(col in full_df.columns for col in ['CHROM', 'POS', 'REF', 'ALT']):
                    full_df['variant_id'] = (
                        full_df['CHROM'].astype(str) + '_' +
                        full_df['POS'].astype(str) + '_' +
                        full_df['REF'].astype(str) + '_' +
                        full_df['ALT'].astype(str)
                    )
                else:
                    print("❌ Cannot create variant IDs in full dataset")
                    return None, None, None
            
            # Find matching variants
            matched_df = full_df[full_df['variant_id'].isin(selected_variants['variant_id'])].copy()
            print(f"✅ Matched {len(matched_df)} variants from selection")
            
            if len(matched_df) == 0:
                print("❌ No matching variants found")
                return None, None, None
            
            # Prepare features using the same approach as training
            if len(self.feature_names) == 0:
                print("❌ No feature names available from model")
                return None, None, None
            
            # Use only available features (skip missing engineered features)
            available_features = [f for f in self.feature_names if f in matched_df.columns]
            missing_features = set(self.feature_names) - set(available_features)
            if missing_features:
                print(f"⚠️  Missing features ({len(missing_features)}): continuing with available features")
            
            print(f"✅ Using {len(available_features)} of {len(self.feature_names)} features")
            
            X = matched_df[available_features].copy()

            # Handle missing values (same as training)
            for feature in available_features:
                X[feature] = pd.to_numeric(X[feature], errors='coerce')
                X[feature] = X[feature].fillna(X[feature].median())
                X[feature] = X[feature].fillna(0)
            
            # Use only available features and skip scaling
            available_features = [f for f in self.feature_names if f in matched_df.columns]
            
            # Handle missing values and convert to numeric
            for feature in available_features:
                X[feature] = pd.to_numeric(X[feature], errors='coerce')
                X[feature] = X[feature].fillna(X[feature].median())
                X[feature] = X[feature].fillna(0)
            
            # Skip scaling since we don't have fitted training scaler
            X_scaled = X.values
            print(f"⚠️  Using raw features (no scaling) - {len(available_features)} of {len(self.feature_names)} features available")
            
            # Create targets if available
            y = None
            if 'classification' in matched_df.columns and self.label_encoder is not None:
                try:
                    y = self.label_encoder.transform(matched_df['classification'])
                    print("✅ Target labels encoded")
                except Exception as e:
                    print(f"⚠️  Could not encode targets: {e}")
            
            # Create variant info for tracking
            variant_info = matched_df[['variant_id'] + (['classification'] if 'classification' in matched_df.columns else [])].copy()
            
            print(f"✅ Prepared {X_scaled.shape[0]} variants × {X_scaled.shape[1]} features")
            
            return X_scaled, y, variant_info
            
        except Exception as e:
            print(f"❌ Failed to prepare features: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def extract_attention_weights(self, X_scaled, variant_info):
        """Extract TabNet attention weights for variants"""
        print("\n🧠 EXTRACTING ATTENTION WEIGHTS")
        print("-" * 40)
        
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            print(f"📊 Extracting attention for {X_scaled.shape[0]} variants...")
            
            # Get predictions and explanations
            predictions = self.model.predict(X_scaled)
            explanations, masks = self.model.explain(X_scaled)
            
            print(f"✅ Extracted attention weights")
            print(f"📊 Explanations shape: {explanations.shape}")
            print(f"📊 Masks shape: {len(masks)} decision steps")
            
            # Convert predictions back to labels if possible
            predicted_labels = None
            if self.label_encoder is not None:
                try:
                    predicted_labels = self.label_encoder.inverse_transform(predictions)
                    print("✅ Predictions converted to labels")
                except Exception as e:
                    print(f"⚠️  Could not convert predictions: {e}")
            
            # Package results
            results = {
                'explanations': explanations,
                'masks': masks,
                'predictions': predictions,
                'predicted_labels': predicted_labels,
                'variant_info': variant_info,
                'feature_names': self.feature_names
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Failed to extract attention: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_attention_weights(self, attention_results):
        """Save attention weights and analysis results"""
        print("\n💾 SAVING ATTENTION WEIGHTS")
        print("-" * 35)
        
        try:
            explanations = attention_results['explanations']
            masks = attention_results['masks']
            variant_info = attention_results['variant_info']
            feature_names = attention_results['feature_names']
            
            # Save individual variant attention weights
            for i, (_, variant) in enumerate(variant_info.iterrows()):
                variant_id = variant['variant_id']
                
                # Create attention dataframe for this variant
                variant_attention = pd.DataFrame({
                    'feature': feature_names,
                    'attention_weight': explanations[i, :len(feature_names)]
                }).sort_values('attention_weight', ascending=False)
                
                # Save individual file
                variant_file = os.path.join(self.attention_dir, f"{variant_id}_attention.csv")
                variant_attention.to_csv(variant_file, index=False)
            
            print(f"✅ Saved individual attention files for {len(variant_info)} variants")
            
            # Create summary with top features per variant
            summary_data = []
            for i, (_, variant) in enumerate(variant_info.iterrows()):
                variant_id = variant['variant_id']
                classification = variant.get('classification', 'Unknown')
                
                # Get top 10 features for this variant
                variant_attention = explanations[i, :len(feature_names)]
                top_indices = np.argsort(variant_attention)[-10:][::-1]
                
                summary_data.append({
                    'variant_id': variant_id,
                    'classification': classification,
                    'top_feature': feature_names[top_indices[0]],
                    'top_attention': variant_attention[top_indices[0]],
                    'prediction': attention_results['predictions'][i] if attention_results['predictions'] is not None else None,
                    'predicted_label': attention_results['predicted_labels'][i] if attention_results['predicted_labels'] is not None else None
                })
            
            # Save summary
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(self.attention_dir, "attention_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"✅ Saved summary: {summary_file}")
            
            # Save metadata
            metadata = {
                'extraction_timestamp': pd.Timestamp.now().isoformat(),
                'num_variants': len(variant_info),
                'num_features': len(feature_names),
                'num_decision_steps': len(masks),
                'feature_names': feature_names
            }
            
            metadata_file = os.path.join(self.attention_dir, "extraction_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✅ Saved metadata: {metadata_file}")
            
            # Feature importance summary
            overall_importance = np.mean(explanations[:, :len(feature_names)], axis=0)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'mean_attention': overall_importance
            }).sort_values('mean_attention', ascending=False)
            
            importance_file = os.path.join(self.attention_dir, "feature_importance_summary.csv")
            importance_df.to_csv(importance_file, index=False)
            
            print(f"✅ Top 5 most important features overall:")
            for _, row in importance_df.head(5).iterrows():
                print(f"   {row['feature']}: {row['mean_attention']:.4f}")
            
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
    
    # Configuration - Updated to use latest successful model
    model_path = "/u/aa107/scratch/tabnet_model_20250731_032824_tabnet.zip"
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
        
        if selected_variants is None:
            print("❌ Failed to load selected variants")
            return False
        
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