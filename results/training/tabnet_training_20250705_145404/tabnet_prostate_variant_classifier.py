#!/usr/bin/env python3
"""
Enhanced TabNet Prostate Cancer Variant Classification
Simple but highly accurate implementation with 57 VEP-corrected features

Location: /u/aa107/uiuc-cancer-research/src/model/tabnet_prostate_variant_classifier.py
Author: PhD Research Student, University of Illinois
Contact: aa107@illinois.edu

Key Improvements:
- Expanded from 24 to 57 high-value features using 8-tier priority system
- VEP severity encoding tables for proper categorical handling
- Robust string conversion and missing value handling
- Hierarchical target variable creation (CLIN_SIG > AlphaMissense > IMPACT)
- Clinical interpretability with feature group analysis
"""

import pandas as pd
import numpy as np
import re
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

    def _select_enhanced_features(self, df):
        """Select 57 high-value features using 8-tier priority system"""
        print("🔧 SELECTING 57 ENHANCED FEATURES...")
        selected_features = []
        
        # TIER 1: VEP-Corrected Features (5 features) - HIGHEST PRIORITY
        tier1_features = ['Consequence', 'DOMAINS', 'PUBMED', 'VAR_SYNONYMS'] # removed CLIN_SIG as it is creating circular logic reference
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
        
        # Encode CLIN_SIG using severity ranking
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
            
            # Priority 3: IMPACT (fallback)
            elif 'IMPACT' in df.columns and pd.notna(row['IMPACT']):
                impact = str(row['IMPACT']).upper()
                if impact in ['HIGH', 'MODERATE']:
                    targets.append('Pathogenic')
                elif impact in ['LOW', 'MODIFIER']:
                    targets.append('Benign')
                else:
                    targets.append('VUS')
            
            # Default fallback
            else:
                targets.append('VUS')
        
        target_counts = pd.Series(targets).value_counts()
        print(f"✅ Target distribution:")
        for target, count in target_counts.items():
            print(f"   {target}: {count:,} ({count/len(targets)*100:.1f}%)")
        
        return np.array(targets)

    def load_data(self):
        """Load and prepare enhanced dataset"""
        print("📁 LOADING ENHANCED DATASET...")
        
        clean_path = "/u/aa107/uiuc-cancer-research/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv"
        
        if not Path(clean_path).exists():
            raise FileNotFoundError(f"Clean dataset not found: {clean_path}")
        
        df = pd.read_csv(clean_path, low_memory=False)
        print(f"✅ Loaded {len(df):,} variants with {len(df.columns)} columns")
        
        # Validate no data leakage
        leakage_features = ['sift_prediction', 'polyphen_prediction', 'functional_pathogenicity']
        leakage_found = [f for f in leakage_features if f in df.columns]
        if leakage_found:
            raise ValueError(f"❌ Data leakage detected: {leakage_found}")
        
        # Select 57 enhanced features
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
        
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Add feature groups
        feature_groups = []
        for feature in importance_df['feature']:
            group = 'unknown'
            for group_name, group_features in self.feature_groups.items():
                if feature in group_features:
                    group = group_name
                    break
            feature_groups.append(group)
        
        importance_df['group'] = feature_groups
        
        return importance_df

    def analyze_feature_groups(self):
        """Analyze importance by feature groups for clinical interpretation"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        importance_df = self.get_feature_importance()
        
        print("\n📊 FEATURE GROUP ANALYSIS:")
        print("=" * 40)
        
        group_importance = importance_df.groupby('group')['importance'].agg(['sum', 'mean', 'count'])
        group_importance = group_importance.sort_values('sum', ascending=False)
        
        for group in group_importance.index:
            total_imp = group_importance.loc[group, 'sum']
            avg_imp = group_importance.loc[group, 'mean']
            count = int(group_importance.loc[group, 'count'])
            pct = total_imp * 100
            
            print(f"   {group}: {total_imp:.3f} ({pct:.1f}%) - {count} features")
        
        return group_importance


def main():
    """Main training and evaluation pipeline"""
    print("🧬 TabNet Prostate Cancer Classifier - ENHANCED VERSION")
    print("=" * 60)
    print("✅ 57 VEP-corrected features with clinical interpretability")
    print("🎯 Expected accuracy: 70-80% (realistic clinical performance)")
    print()
    
    # Initialize model
    tabnet = ProstateVariantTabNet(n_d=64, n_a=64, n_steps=6)
    
    try:
        # Load enhanced data
        X, y = tabnet.load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        print(f"\n📊 Data split:")
        print(f"   Training: {X_train.shape[0]:,} variants")
        print(f"   Test: {X_test.shape[0]:,} variants")
        print(f"   Features: {X_train.shape[1]}")
        
        # Cross-validation
        cv_results = tabnet.cross_validate(X, y, cv_folds=3)
        
        # Train final model
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        val_accuracy = tabnet.train(X_train_split, y_train_split, X_val_split, y_val_split)
        
        # Final test evaluation
        test_accuracy = tabnet.evaluate(X_test, y_test)
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Cross-validation: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
        print(f"   Test accuracy: {test_accuracy:.3f}")
        
        # Feature importance analysis
        feature_importance = tabnet.get_feature_importance()
        print(f"\n📊 Top 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']} ({row['group']}): {row['importance']:.3f}")
        
        # Feature group analysis
        tabnet.analyze_feature_groups()
        
        return tabnet
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    model = main()