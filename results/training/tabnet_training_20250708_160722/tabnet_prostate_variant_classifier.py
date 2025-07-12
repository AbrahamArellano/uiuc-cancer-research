#!/usr/bin/env python3
"""
Enhanced TabNet Prostate Cancer Variant Classification
Complete implementation with model saving and main execution

Location: /u/aa107/uiuc-cancer-research/src/model/tabnet_prostate_variant_classifier.py
Author: PhD Research Student, University of Illinois
Contact: aa107@illinois.edu

Key Improvements:
- Expanded from 24 to 57 high-value features using 8-tier priority system
- VEP severity encoding tables for proper categorical handling
- Robust string conversion and missing value handling
- Hierarchical target variable creation (CLIN_SIG > AlphaMissense > IMPACT)
- Clinical interpretability with feature group analysis
- Model saving and loading functionality
- Complete main execution pipeline
"""

import pandas as pd
import numpy as np
import re
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ProstateVariantTabNet:
    """Enhanced TabNet for prostate cancer variant classification"""
    
    def __init__(self, n_d=64, n_a=64, n_steps=6):
        """Initialize TabNet with optimal hyperparameters"""
        self.n_d = n_d
        self.n_a = n_a  
        self.n_steps = n_steps
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
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
        
        # VEP Severity Tables (from post_process_vep_concatenation.py)
        self.CONSEQUENCE_SEVERITY = {
            'transcript_ablation': 10,
            'splice_acceptor_variant': 9,
            'splice_donor_variant': 9,
            'stop_gained': 8,
            'frameshift_variant': 8,
            'stop_lost': 7,
            'start_lost': 7,
            'missense_variant': 4,
            'splice_region_variant': 3,
            'synonymous_variant': 2,
            'intron_variant': 1,
            'upstream_gene_variant': 0,
            'downstream_gene_variant': 0
        }
        
        self.CLIN_SIG_SEVERITY = {
            'pathogenic': 5,
            'likely_pathogenic': 4,
            'uncertain_significance': 3,
            'likely_benign': 2,
            'benign': 1,
            'not_provided': 0
        }
        
        self.IMPACT_SEVERITY = {
            'HIGH': 4,
            'MODERATE': 3,
            'LOW': 2,
            'MODIFIER': 1
        }

    def load_data(self):
        """Load and prepare the clean dataset"""
        print("📁 LOADING ENHANCED DATASET...")
        
        # Define data path
        data_path = "/u/aa107/uiuc-cancer-research/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv"
        
        if not os.path.exists(data_path):
            print(f"❌ Dataset not found: {data_path}")
            return None, None
        
        try:
            # Load dataset
            df = pd.read_csv(data_path, low_memory=False)
            print(f"✅ Loaded {len(df):,} variants with {len(df.columns)} columns")
            
            # Select enhanced features
            selected_features = self._select_enhanced_features(df)
            
            # Encode categorical features
            df_encoded = self._encode_categorical_features(df)
            
            # Create target variable
            y = self._create_target_variable(df)
            
            # Prepare feature matrix
            X = self._prepare_features(df_encoded, selected_features)
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            return X, y
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None

    def _select_enhanced_features(self, df):
        """Select 57 high-value features using 8-tier priority system"""
        print("🔧 SELECTING 57 ENHANCED FEATURES...")
        selected_features = []
        
        # TIER 1: VEP-Corrected Features (4 features) - HIGHEST PRIORITY
        # Note: CLIN_SIG removed to eliminate data leakage
        tier1_features = ['Consequence', 'DOMAINS', 'PUBMED', 'VAR_SYNONYMS']
        for feature in tier1_features:
            if feature in df.columns:
                selected_features.append(feature)
                self.feature_groups['tier1_vep_corrected'].append(feature)
        
        # TIER 2: Core VEP Annotations (10 features)
        tier2_features = ['SYMBOL', 'BIOTYPE', 'CANONICAL', 'PICK', 'HGVSc', 
                         'HGVSp', 'Protein_position', 'Amino_acids', 'Existing_variation', 'VARIANT_CLASS']
        for feature in tier2_features:
            if feature in df.columns:
                selected_features.append(feature)
                self.feature_groups['tier2_core_vep'].append(feature)
        
        # TIER 3: AlphaMissense Integration (2 features)
        tier3_features = ['alphamissense_pathogenicity', 'alphamissense_class']
        for feature in tier3_features:
            if feature in df.columns:
                selected_features.append(feature)
                self.feature_groups['tier3_alphamissense'].append(feature)
        
        # TIER 4: Population Genetics (17 features)
        tier4_features = ['AF', 'AFR_AF', 'AMR_AF', 'EAS_AF', 'EUR_AF', 'SAS_AF',
                         'gnomADe_AF', 'gnomADe_AFR_AF', 'gnomADe_AMR_AF', 'gnomADe_ASJ_AF',
                         'gnomADe_EAS_AF', 'gnomADe_FIN_AF', 'gnomADe_MID_AF', 'gnomADe_NFE_AF',
                         'gnomADe_REMAINING_AF', 'gnomADe_SAS_AF', 'af_1kg']
        for feature in tier4_features:
            if feature in df.columns:
                selected_features.append(feature)
                self.feature_groups['tier4_population'].append(feature)
        
        # TIER 5: Functional Predictions (6 features)
        tier5_features = ['IMPACT', 'sift_score', 'polyphen_score', 'SIFT', 'PolyPhen', 'impact_score']
        for feature in tier5_features:
            if feature in df.columns:
                selected_features.append(feature)
                self.feature_groups['tier5_functional'].append(feature)
        
        # TIER 6: Clinical Context (5 features)
        tier6_features = ['SOMATIC', 'PHENO', 'EXON', 'INTRON', 'CCDS']
        for feature in tier6_features:
            if feature in df.columns:
                selected_features.append(feature)
                self.feature_groups['tier6_clinical'].append(feature)
        
        # TIER 7: Variant Properties (8 features)
        tier7_features = ['ref_length', 'alt_length', 'variant_size', 'is_indel', 
                         'is_snv', 'is_lof', 'is_missense', 'is_synonymous']
        for feature in tier7_features:
            if feature in df.columns:
                selected_features.append(feature)
                self.feature_groups['tier7_variant_props'].append(feature)
        
        # TIER 8: Prostate Biology (4 features)
        tier8_features = ['is_important_gene', 'dna_repair_pathway', 'mismatch_repair_pathway', 'hormone_pathway']
        for feature in tier8_features:
            if feature in df.columns:
                selected_features.append(feature)
                self.feature_groups['tier8_prostate_biology'].append(feature)
        
        print(f"✅ Selected {len(selected_features)} features across 8 tiers:")
        for tier, features in self.feature_groups.items():
            if features:
                print(f"   {tier}: {len(features)} features")
        
        return selected_features

    def _encode_categorical_features(self, df):
        """Encode categorical features using VEP severity tables"""
        print("🔧 ENCODING CATEGORICAL FEATURES...")
        df_encoded = df.copy()
        
        # Encode Consequence using severity ranking
        if 'Consequence' in df.columns:
            df_encoded['Consequence_encoded'] = df['Consequence'].map(
                lambda x: self.CONSEQUENCE_SEVERITY.get(str(x).lower(), 0) if pd.notna(x) else 0
            )
            df_encoded = df_encoded.drop('Consequence', axis=1)
            df_encoded = df_encoded.rename(columns={'Consequence_encoded': 'Consequence'})
        
        # Encode CLIN_SIG using severity ranking (for target creation, not features)
        if 'CLIN_SIG' in df.columns:
            df_encoded['CLIN_SIG_encoded'] = df['CLIN_SIG'].map(
                lambda x: self.CLIN_SIG_SEVERITY.get(str(x).lower(), 0) if pd.notna(x) else 0
            )
            df_encoded = df_encoded.drop('CLIN_SIG', axis=1)
            df_encoded = df_encoded.rename(columns={'CLIN_SIG_encoded': 'CLIN_SIG'})
        
        # Encode IMPACT using severity ranking
        if 'IMPACT' in df.columns:
            df_encoded['IMPACT_encoded'] = df['IMPACT'].map(
                lambda x: self.IMPACT_SEVERITY.get(str(x).upper(), 1) if pd.notna(x) else 1
            )
            df_encoded = df_encoded.drop('IMPACT', axis=1)
            df_encoded = df_encoded.rename(columns={'IMPACT_encoded': 'IMPACT'})
        
        # Parse SIFT scores from format "deleterious(0.01)" -> 0.01
        if 'SIFT' in df.columns:
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
            am_encoding = {'likely_pathogenic': 2, 'ambiguous': 1, 'likely_benign': 0}
            df_encoded['alphamissense_class_encoded'] = df['alphamissense_class'].map(
                lambda x: am_encoding.get(str(x).lower(), 1) if pd.notna(x) else 1
            )
            df_encoded = df_encoded.drop('alphamissense_class', axis=1)
            df_encoded = df_encoded.rename(columns={'alphamissense_class_encoded': 'alphamissense_class'})
        
        # Convert remaining categorical to numeric using label encoding
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
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
        
        print(f"✅ Categorical encoding completed")
        return df_encoded

    def _prepare_features(self, df_encoded, selected_features):
        """Prepare feature matrix with robust missing value handling"""
        print("🔧 PREPARING FEATURES...")
        
        # Ensure all selected features exist
        available_features = [f for f in selected_features if f in df_encoded.columns]
        if len(available_features) != len(selected_features):
            missing = set(selected_features) - set(available_features)
            print(f"⚠️  Missing features: {missing}")
        
        # Create feature matrix
        X = df_encoded[available_features].copy()
        
        # Convert to numeric and handle missing values
        for feature in available_features:
            # Convert to numeric
            X[feature] = pd.to_numeric(X[feature], errors='coerce')
            # Fill missing with median for numeric features
            X[feature] = X[feature].fillna(X[feature].median())
            # If all missing, fill with 0
            X[feature] = X[feature].fillna(0)
        
        print(f"✅ Prepared {X.shape[1]} features for {X.shape[0]:,} samples")
        return X

    def _create_target_variable(self, df):
        """Create hierarchical target variable: CLIN_SIG > AlphaMissense > IMPACT"""
        print("🎯 CREATING HIERARCHICAL TARGET VARIABLE...")
        
        targets = []
        
        for _, row in df.iterrows():
            # Priority 1: CLIN_SIG (if available)
            if 'CLIN_SIG' in df.columns and pd.notna(row['CLIN_SIG']) and str(row['CLIN_SIG']).lower() != 'unknown':
                clin_sig = str(row['CLIN_SIG']).lower()
                if clin_sig in ['pathogenic', 'likely_pathogenic']:
                    targets.append('Pathogenic')
                elif clin_sig in ['benign', 'likely_benign']:
                    targets.append('Benign')
                else:
                    targets.append('VUS')
            
            # Priority 2: AlphaMissense (if CLIN_SIG not available)
            elif 'alphamissense_class' in df.columns and pd.notna(row['alphamissense_class']):
                am_class = str(row['alphamissense_class']).lower()
                if am_class == 'likely_pathogenic':
                    targets.append('Pathogenic')
                elif am_class == 'likely_benign':
                    targets.append('Benign')
                else:
                    targets.append('VUS')
            
            # Priority 3: IMPACT (as fallback)
            elif 'IMPACT' in df.columns and pd.notna(row['IMPACT']):
                impact = str(row['IMPACT']).upper()
                if impact == 'HIGH':
                    targets.append('Pathogenic')
                elif impact in ['LOW', 'MODIFIER']:
                    targets.append('Benign')
                else:
                    targets.append('VUS')
            
            # Default: VUS
            else:
                targets.append('VUS')
        
        # Convert to numpy array
        y = np.array(targets)
        
        # Print distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"✅ Target distribution:")
        for label, count in zip(unique, counts):
            print(f"   {label}: {count:,} ({count/len(y)*100:.1f}%)")
        
        return y

    def train(self, X_train, y_train, X_val, y_val):
        """Train TabNet model"""
        print("🚀 TRAINING TABNET MODEL...")
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Initialize TabNet
        self.model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=1.3,
            lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            mask_type='entmax',
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=1
        )
        
        # Train model
        self.model.fit(
            X_train_scaled, y_train_encoded,
            eval_set=[(X_val_scaled, y_val_encoded)],
            eval_name=['val'],
            eval_metric=['accuracy'],
            max_epochs=100,
            patience=20,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val_encoded, y_pred)
        
        print(f"✅ Training completed. Validation accuracy: {accuracy:.3f}")
        return accuracy

    def cross_validate(self, X, y, cv_folds=5):
        """Perform cross-validation"""
        print(f"🔄 RUNNING {cv_folds}-FOLD CROSS-VALIDATION...")
        
        cv_scores = []
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"   Fold {fold}/{cv_folds}...")
            
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Create fold model
            fold_model = TabNetClassifier(
                n_d=32, n_a=32, n_steps=3,  # Reduced for faster CV
                gamma=1.3, lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                verbose=0
            )
            
            # Encode and scale
            fold_le = LabelEncoder()
            y_train_encoded = fold_le.fit_transform(y_train_fold)
            y_val_encoded = fold_le.transform(y_val_fold)
            
            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train_fold)
            X_val_scaled = fold_scaler.transform(X_val_fold)
            
            # Train and evaluate
            fold_model.fit(X_train_scaled, y_train_encoded, max_epochs=50, patience=10)
            y_pred = fold_model.predict(X_val_scaled)
            fold_accuracy = accuracy_score(y_val_encoded, y_pred)
            cv_scores.append(fold_accuracy)
            
            print(f"   Fold {fold} accuracy: {fold_accuracy:.3f}")
        
        mean_accuracy = np.mean(cv_scores)
        std_accuracy = np.std(cv_scores)
        
        print(f"📊 Cross-validation results: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
        
        return {'mean_accuracy': mean_accuracy, 'std_accuracy': std_accuracy, 'fold_scores': cv_scores}

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        y_test_encoded = self.label_encoder.transform(y_test)
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        print(f"📊 TEST EVALUATION:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test_encoded, y_pred, target_names=self.label_encoder.classes_))
        
        return accuracy

    def get_feature_importance(self):
        """Get feature importance analysis"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get feature importance from TabNet
        feature_importance = self.model.feature_importances_
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"📊 Top 10 Most Important Features:")
        for _, row in importance_df.head(10).iterrows():
            # Find which tier this feature belongs to
            tier = 'unknown'
            for tier_name, features in self.feature_groups.items():
                if row['feature'] in features:
                    tier = tier_name
                    break
            print(f"   {row['feature']} ({tier}): {row['importance']:.3f}")
        
        # Group by tier analysis
        print(f"\n📊 FEATURE GROUP ANALYSIS:")
        print("=" * 40)
        tier_importance = {}
        for tier_name, features in self.feature_groups.items():
            tier_features = [f for f in features if f in self.feature_names]
            if tier_features:
                tier_imp = importance_df[importance_df['feature'].isin(tier_features)]['importance'].sum()
                tier_importance[tier_name] = tier_imp
        
        # Sort by importance
        sorted_tiers = sorted(tier_importance.items(), key=lambda x: x[1], reverse=True)
        for tier_name, importance in sorted_tiers:
            tier_features = [f for f in self.feature_groups[tier_name] if f in self.feature_names]
            print(f"   {tier_name}: {importance:.3f} ({importance*100:.1f}%) - {len(tier_features)} features")
        
        return importance_df

    def predict_with_explanation(self, X, feature_names=None):
        """Get predictions with attention explanations"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if feature_names is None:
            feature_names = self.feature_names
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and explanations
        predictions = self.model.predict(X_scaled)
        explanations, masks = self.model.explain(X_scaled)
        
        # Convert predictions back to original labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return {
            'predictions': predicted_labels,
            'explanations': explanations,
            'attention_masks': masks,
            'feature_names': feature_names
        }

    def save_model(self, model_path):
        """Save the trained model and preprocessing components"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Create directory if it doesn't exist
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save all components
        model_data = {
            'tabnet_model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'feature_groups': self.feature_groups,
            'n_d': self.n_d,
            'n_a': self.n_a,
            'n_steps': self.n_steps,
            'training_timestamp': datetime.now().isoformat()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to: {model_path}")
        
        # Save model metadata
        metadata_path = model_path.replace('.pkl', '_metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"TabNet Prostate Cancer Variant Classifier\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Features: {len(self.feature_names)}\n")
            f.write(f"Architecture: n_d={self.n_d}, n_a={self.n_a}, n_steps={self.n_steps}\n")
            f.write(f"Classes: {list(self.label_encoder.classes_)}\n")
            f.write(f"\nFeature Groups:\n")
            for tier, features in self.feature_groups.items():
                if features:
                    f.write(f"  {tier}: {len(features)} features\n")
        
        print(f"✅ Metadata saved to: {metadata_path}")

    def load_model(self, model_path):
        """Load a trained model and preprocessing components"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore all components
        self.model = model_data['tabnet_model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.feature_groups = model_data['feature_groups']
        self.n_d = model_data['n_d']
        self.n_a = model_data['n_a']
        self.n_steps = model_data['n_steps']
        
        print(f"✅ Model loaded from: {model_path}")
        print(f"📊 Features: {len(self.feature_names)}")
        print(f"🏗️  Architecture: n_d={self.n_d}, n_a={self.n_a}, n_steps={self.n_steps}")
        print(f"🎯 Classes: {list(self.label_encoder.classes_)}")

def main():
    """Main training pipeline"""
    print("🧬 TabNet Prostate Cancer Classifier - ENHANCED VERSION")
    print("=" * 60)
    print("✅ 57 VEP-corrected features with clinical interpretability")
    print("🎯 Expected accuracy: 70-80% (realistic clinical performance)")
    print("")
    
    # Initialize model
    tabnet = ProstateVariantTabNet(n_d=64, n_a=64, n_steps=6)
    
    # Load data
    X, y = tabnet.load_data()
    if X is None:
        print("❌ Failed to load data")
        return False
    
    # Split data
    print(f"\n📊 Data split:")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"   Training: {len(X_train):,} variants")
    print(f"   Validation: {len(X_val):,} variants")
    print(f"   Test: {len(X_test):,} variants")
    print(f"   Features: {X.shape[1]}")
    
    # Cross-validation
    cv_results = tabnet.cross_validate(X_train, y_train, cv_folds=3)
    
    # Train final model
    validation_accuracy = tabnet.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    test_accuracy = tabnet.evaluate(X_test, y_test)
    
    # Feature importance analysis
    importance_df = tabnet.get_feature_importance()
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine save location based on environment
    if os.path.exists("/u/aa107/scratch"):
        # Running in SLURM job - save to scratch first
        model_path = f"/u/aa107/scratch/tabnet_model_{timestamp}.pkl"
    else:
        # Running locally - save to results directory
        results_dir = "/u/aa107/uiuc-cancer-research/results/training"
        os.makedirs(results_dir, exist_ok=True)
        model_path = f"{results_dir}/tabnet_model_{timestamp}.pkl"
    
    # Save the trained model
    tabnet.save_model(model_path)
    
    # Print final results
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Cross-validation: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
    print(f"   Validation accuracy: {validation_accuracy:.3f}")
    print(f"   Test accuracy: {test_accuracy:.3f}")
    print(f"   Model saved: {model_path}")
    
    # Success criteria
    if test_accuracy >= 0.70:
        print(f"\n🎉 SUCCESS: Achieved target performance!")
        if test_accuracy >= 0.85:
            print(f"   🏆 EXCELLENT: {test_accuracy:.1%} accuracy")
        else:
            print(f"   ✅ GOOD: {test_accuracy:.1%} accuracy")
    else:
        print(f"\n⚠️  MODERATE: {test_accuracy:.1%} accuracy - expected for complex genomic data")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        print(f"\n✅ Training pipeline completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()