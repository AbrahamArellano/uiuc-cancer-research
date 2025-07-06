#!/usr/bin/env python3
"""
Enhanced TabNet Validation Framework - Updated for 57-Feature Implementation
Updated performance thresholds for realistic clinical expectations (70-80%)

Location: /u/aa107/uiuc-cancer-research/src/model/validation/validate_tabnet.py
Author: PhD Research Student, University of Illinois

Key Updates:
- suspicious_threshold: 0.90 (not 0.95) 
- excellent_threshold: 0.80 (not 0.85)
- good_threshold: 0.70 (not 0.75)
- Uses clean dataset with no data leakage
- Validates 57-feature implementation
"""

import pandas as pd
import numpy as np
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add project root to path
sys.path.append('/u/aa107/uiuc-cancer-research/src')

class EnhancedTabNetValidator:
    def __init__(self):
        self.project_dir = Path("/u/aa107/uiuc-cancer-research")
        self.results_dir = self.project_dir / "results/validation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # UPDATED Performance thresholds (from handover document)
        self.performance_thresholds = {
            'excellent': 0.80,      # Updated from 0.85
            'good': 0.70,           # Updated from 0.75
            'acceptable': 0.60,     # Reasonable baseline
            'suspicious': 0.90      # Updated from 0.95 - above this indicates data leakage
        }
        
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'data_leakage_detected': False,
            'alphamissense_integrated': False,
            'feature_count': 0,
            'summary': {}
        }
    
    def enhanced_dataset_validation(self):
        """Step 1: Enhanced dataset validation using clean dataset (no leakage)"""
        print("\n🔍 STEP 1: ENHANCED DATASET VALIDATION")
        print("=" * 50)
        
        try:
            # Use CLEAN dataset path (critical for no data leakage)
            clean_path = "/u/aa107/uiuc-cancer-research/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv"
            
            if not Path(clean_path).exists():
                print(f"❌ Clean dataset not found: {clean_path}")
                return False
            
            # Load dataset
            df = pd.read_csv(clean_path, low_memory=False)
            print(f"✅ Clean dataset loaded: {df.shape[0]:,} variants × {df.shape[1]} features")
            
            # CRITICAL: Check for data leakage features
            leakage_features = ['sift_prediction', 'polyphen_prediction', 'functional_pathogenicity', 
                               'sift_confidence', 'polyphen_confidence']
            leakage_found = [f for f in leakage_features if f in df.columns]
            
            if leakage_found:
                print(f"❌ CRITICAL: Data leakage features found: {leakage_found}")
                self.validation_results['data_leakage_detected'] = True
                return False
            else:
                print(f"✅ No data leakage features detected")
            
            # Check VEP-corrected features (Tier 1)
            tier1_features = ['Consequence', 'CLIN_SIG', 'DOMAINS', 'PUBMED', 'VAR_SYNONYMS']
            missing_tier1 = [f for f in tier1_features if f not in df.columns]
            
            if missing_tier1:
                print(f"❌ VEP-corrected features missing: {missing_tier1}")
                return False
            else:
                print(f"✅ VEP-corrected features present: {len(tier1_features)}")
            
            # Check AlphaMissense features (Tier 3)
            alphamissense_features = ['alphamissense_pathogenicity', 'alphamissense_class']
            missing_am = [f for f in alphamissense_features if f not in df.columns]
            
            if missing_am:
                print(f"❌ AlphaMissense features missing: {missing_am}")
                self.validation_results['alphamissense_integrated'] = False
                return False
            else:
                print(f"✅ AlphaMissense features present: {len(alphamissense_features)}")
                self.validation_results['alphamissense_integrated'] = True
            
            # Check AlphaMissense coverage
            am_coverage = df['alphamissense_pathogenicity'].notna().sum()
            coverage_rate = am_coverage / len(df) * 100
            print(f"📊 AlphaMissense coverage: {am_coverage:,} variants ({coverage_rate:.1f}%)")
            
            if coverage_rate < 30:
                print(f"⚠️  Low AlphaMissense coverage - expected >30%")
            else:
                print(f"✅ Good AlphaMissense coverage")
            
            # Store dataset info
            self.validation_results['dataset_info'] = {
                'variant_count': int(df.shape[0]),
                'feature_count': int(df.shape[1]),
                'alphamissense_coverage': float(coverage_rate),
                'data_leakage_detected': False
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Dataset validation failed: {e}")
            traceback.print_exc()
            return False
    
    def baseline_validation(self):
        """Step 2: Baseline Random Forest validation to detect data leakage"""
        print("\n🌲 STEP 2: BASELINE RANDOM FOREST VALIDATION")
        print("=" * 50)
        
        try:
            # Load data using enhanced TabNet logic
            from model.tabnet_prostate_variant_classifier import ProstateVariantTabNet
            
            # Initialize TabNet to use its data loading logic
            tabnet = ProstateVariantTabNet()
            X, y = tabnet.load_data()
            
            print(f"📊 Baseline validation data: {X.shape[0]:,} samples × {X.shape[1]} features")
            self.validation_results['feature_count'] = X.shape[1]
            
            # Use subset for quick validation
            if len(X) > 10000:
                print("📊 Using subset for quick baseline validation...")
                indices = np.random.choice(len(X), 10000, replace=False)
                X_subset = X.iloc[indices]
                y_subset = y[indices]
            else:
                X_subset = X
                y_subset = y
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)
            
            # Cross-validation with Random Forest
            cv_scores = []
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            print("🔄 Running 5-fold cross-validation...")
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_subset), 1):
                X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
                y_train_fold, y_val_fold = y_subset[train_idx], y_subset[val_idx]
                
                # Train baseline Random Forest
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_train_fold, y_train_fold)
                
                # Evaluate
                y_pred = rf.predict(X_val_fold)
                fold_accuracy = accuracy_score(y_val_fold, y_pred)
                cv_scores.append(fold_accuracy)
                
                print(f"   Fold {fold} accuracy: {fold_accuracy:.3f}")
            
            mean_accuracy = np.mean(cv_scores)
            std_accuracy = np.std(cv_scores)
            
            print(f"\n🎯 Baseline RF Performance:")
            print(f"   Mean accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
            
            # UPDATED thresholds for data leakage detection
            if mean_accuracy > self.performance_thresholds['suspicious']:
                print("❌ CRITICAL: Suspiciously high baseline accuracy - DATA LEAKAGE DETECTED!")
                self.validation_results['data_leakage_detected'] = True
                return False
            elif mean_accuracy > self.performance_thresholds['excellent']:
                print("⚠️  HIGH: Check features - might indicate leakage")
            elif mean_accuracy > self.performance_thresholds['good']:
                print("✅ EXCELLENT: Target performance achieved")
            elif mean_accuracy > self.performance_thresholds['acceptable']:
                print("✅ GOOD: Realistic performance")
            else:
                print("✅ MODERATE: Expected for complex genomic data")
            
            self.validation_results['baseline_validation'] = {
                'mean_accuracy': float(mean_accuracy),
                'std_accuracy': float(std_accuracy),
                'fold_scores': [float(score) for score in cv_scores]
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Baseline validation failed: {e}")
            traceback.print_exc()
            return False
    
    def tabnet_validation(self):
        """Step 3: TabNet model validation with updated performance expectations"""
        print("\n🔥 STEP 3: TABNET MODEL VALIDATION")
        print("=" * 50)
        
        try:
            from model.tabnet_prostate_variant_classifier import ProstateVariantTabNet
            
            # Initialize TabNet model (smaller for validation)
            tabnet = ProstateVariantTabNet(n_d=32, n_a=32, n_steps=3)
            
            # Load data
            X, y = tabnet.load_data()
            
            if X is None:
                print("❌ Could not load data")
                return False
            
            print(f"📊 TabNet validation data: {X.shape[0]:,} samples × {X.shape[1]} features")
            
            # Use subset for quick validation
            if len(X) > 5000:
                print("📊 Using subset for quick TabNet validation...")
                indices = np.random.choice(len(X), 5000, replace=False)
                X_subset = X.iloc[indices]
                y_subset = y[indices]
            else:
                X_subset = X
                y_subset = y
            
            # Cross-validation
            cv_results = tabnet.cross_validate(X_subset, y_subset, cv_folds=3)
            
            mean_accuracy = cv_results['mean_accuracy']
            std_accuracy = cv_results['std_accuracy']
            
            print(f"\n🎯 TabNet Performance:")
            print(f"   Mean accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
            
            # UPDATED thresholds for performance evaluation
            if mean_accuracy > self.performance_thresholds['suspicious']:
                print("❌ CRITICAL: Suspiciously high TabNet accuracy - DATA LEAKAGE DETECTED!")
                self.validation_results['data_leakage_detected'] = True
                return False
            elif mean_accuracy > self.performance_thresholds['excellent']:
                print("✅ EXCELLENT: Target performance achieved (80%+ accuracy)")
            elif mean_accuracy > self.performance_thresholds['good']:
                print("✅ GOOD: Realistic clinical performance (70-80% range)")
            elif mean_accuracy > self.performance_thresholds['acceptable']:
                print("✅ ACCEPTABLE: Within expected range (60-70%)")
            else:
                print("⚠️  MODERATE: Below target - consider feature engineering")
            
            self.validation_results['tabnet_validation'] = {
                'mean_accuracy': float(mean_accuracy),
                'std_accuracy': float(std_accuracy),
                'fold_scores': [float(score) for score in cv_results['fold_scores']]
            }
            
            return True
            
        except Exception as e:
            print(f"❌ TabNet validation failed: {e}")
            traceback.print_exc()
            return False
    
    def generate_report(self):
        """Generate comprehensive validation report"""
        print("\n📋 GENERATING VALIDATION REPORT")
        print("=" * 40)
        
        # Update summary
        self.validation_results['summary'] = {
            'data_leakage_detected': self.validation_results['data_leakage_detected'],
            'alphamissense_integrated': self.validation_results['alphamissense_integrated'],
            'feature_count': self.validation_results['feature_count'],
            'validation_passed': not self.validation_results['data_leakage_detected']
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"enhanced_tabnet_validation_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"✅ Report saved: {report_file}")
        
        # Print summary
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   Data leakage detected: {'❌ YES' if self.validation_results['data_leakage_detected'] else '✅ NO'}")
        print(f"   AlphaMissense integrated: {'✅ YES' if self.validation_results['alphamissense_integrated'] else '❌ NO'}")
        print(f"   Features validated: {self.validation_results['feature_count']}")
        
        if 'baseline_validation' in self.validation_results:
            baseline_acc = self.validation_results['baseline_validation']['mean_accuracy']
            print(f"   Baseline accuracy: {baseline_acc:.3f}")
        
        if 'tabnet_validation' in self.validation_results:
            tabnet_acc = self.validation_results['tabnet_validation']['mean_accuracy']
            print(f"   TabNet accuracy: {tabnet_acc:.3f}")
        
        print(f"\n🎯 PERFORMANCE THRESHOLDS (Updated):")
        print(f"   Excellent: {self.performance_thresholds['excellent']:.1%} (was 85%)")
        print(f"   Good: {self.performance_thresholds['good']:.1%} (was 75%)")
        print(f"   Suspicious: {self.performance_thresholds['suspicious']:.1%} (was 95%)")
        
        return report_file

def main():
    """Main validation pipeline with updated 70-80% performance expectations"""
    print("🧬 ENHANCED TABNET VALIDATION FRAMEWORK")
    print("=" * 60)
    print("Updated for 57-feature implementation with realistic 70-80% accuracy targets")
    print()
    
    # Initialize validator
    validator = EnhancedTabNetValidator()
    
    print("🧬 Enhanced TabNet Validator Initialized")
    print(f"📁 Results directory: {validator.results_dir}")
    print(f"⚠️  Suspicious accuracy threshold: {validator.performance_thresholds['suspicious']*100:.1f}% (updated)")
    print(f"✅ Target accuracy range: {validator.performance_thresholds['good']*100:.1f}%-{validator.performance_thresholds['excellent']*100:.1f}% (updated)")
    
    # Step 1: Dataset validation
    if not validator.enhanced_dataset_validation():
        print("❌ Dataset validation failed - aborting")
        return False
    
    # Step 2: Baseline validation
    if not validator.baseline_validation():
        print("❌ Baseline validation failed - possible data leakage")
        return False
    
    # Step 3: TabNet validation
    if not validator.tabnet_validation():
        print("❌ TabNet validation failed")
        return False
    
    # Generate report
    report_file = validator.generate_report()
    
    print(f"\n🎉 VALIDATION COMPLETED SUCCESSFULLY!")
    print(f"📋 Report: {report_file}")
    
    if not validator.validation_results['data_leakage_detected']:
        print("✅ NO DATA LEAKAGE DETECTED - Ready for production training!")
        print("\n🎯 Next steps:")
        print("   1. Run full training: python src/model/tabnet_prostate_variant_classifier.py")
        print("   2. Or submit cluster job for full training")
        print("   3. Expected final performance: 75-85% accuracy with clinical interpretability")
    else:
        print("❌ DATA LEAKAGE DETECTED - Fix required before training")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        traceback.print_exc()
        sys.exit(1)