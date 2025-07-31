#!/usr/bin/env python3
"""
🔍 TabNet Model Loading Diagnostic Test
=====================================
Simple independent test to determine if model files are corrupted 
or if the validation timing is the issue.

Run this in a FRESH Python session (not the training session).
"""

import os
import sys
import torch
import traceback
from pathlib import Path

def test_model_loading():
    """Test loading both pickle and TabNet native formats"""
    
    print("🔍 TABNET MODEL LOADING DIAGNOSTIC TEST")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("")
    
    # Model file paths (update with your actual timestamp)
    MODEL_TIMESTAMP = "20250731_004749"  # UPDATE THIS if different
    BASE_PATH = "/u/aa107/scratch"
    
    pickle_path = f"{BASE_PATH}/tabnet_model_{MODEL_TIMESTAMP}.pkl"
    tabnet_path = f"{BASE_PATH}/tabnet_model_{MODEL_TIMESTAMP}_tabnet.zip"
    
    print(f"📁 Testing model files:")
    print(f"  Pickle: {pickle_path}")
    print(f"  TabNet: {tabnet_path}")
    print("")
    
    # Test 1: File existence and sizes
    print("📋 TEST 1: FILE EXISTENCE & SIZES")
    print("-" * 35)
    
    results = {}
    
    for name, path in [("Pickle", pickle_path), ("TabNet", tabnet_path)]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✅ {name}: {size_mb:.1f} MB")
            results[f"{name.lower()}_exists"] = True
            results[f"{name.lower()}_size_mb"] = size_mb
        else:
            print(f"❌ {name}: FILE NOT FOUND")
            results[f"{name.lower()}_exists"] = False
            return results
    
    print("")
    
    # Test 2: Pickle format loading
    print("📋 TEST 2: PICKLE FORMAT LOADING")
    print("-" * 35)
    
    try:
        # Add project directory to Python path
        project_dir = "/u/aa107/uiuc-cancer-research"
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)
            sys.path.insert(0, f"{project_dir}/src")
        
        # Import the custom TabNet class
        from model.tabnet_prostate_variant_classifier import ProstateVariantTabNet
        
        print("✅ Successfully imported ProstateVariantTabNet")
        
        # Test pickle loading
        test_tabnet = ProstateVariantTabNet()
        test_tabnet.load_model(pickle_path)
        
        print("✅ Pickle loading: SUCCESS")
        print(f"  Model type: {type(test_tabnet.model)}")
        
        results['pickle_loading'] = True
        
    except Exception as e:
        print(f"❌ Pickle loading: FAILED")
        print(f"  Error: {e}")
        print(f"  Error type: {type(e).__name__}")
        traceback.print_exc()
        results['pickle_loading'] = False
    
    print("")
    
    # Test 3: TabNet native format loading
    print("📋 TEST 3: TABNET NATIVE FORMAT LOADING")
    print("-" * 40)
    
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        
        # Test TabNet native loading
        test_model = TabNetClassifier()
        test_model.load_model(tabnet_path)
        
        print("✅ TabNet native loading: SUCCESS")
        print(f"  Model type: {type(test_model)}")
        
        results['tabnet_loading'] = True
        
    except Exception as e:
        print(f"❌ TabNet native loading: FAILED")
        print(f"  Error: {e}")
        print(f"  Error type: {type(e).__name__}")
        if "Invalid magic number" in str(e) or "corrupt" in str(e).lower():
            print("  🚨 This is the SAME error from validation!")
        traceback.print_exc()
        results['tabnet_loading'] = False
    
    print("")
    
    # Test 4: File integrity check
    print("📋 TEST 4: FILE INTEGRITY CHECK")
    print("-" * 35)
    
    try:
        # Check if we can read raw bytes
        with open(tabnet_path, 'rb') as f:
            first_bytes = f.read(10)
            print(f"✅ First 10 bytes: {first_bytes}")
            
        # Check if it's a valid zip file
        import zipfile
        with zipfile.ZipFile(tabnet_path, 'r') as zf:
            file_list = zf.namelist()
            print(f"✅ Zip file valid, contains: {file_list}")
            
        results['file_integrity'] = True
        
    except Exception as e:
        print(f"❌ File integrity check: FAILED")
        print(f"  Error: {e}")
        results['file_integrity'] = False
    
    print("")
    
    # Summary
    print("🎯 DIAGNOSTIC SUMMARY")
    print("=" * 25)
    
    if results.get('pickle_loading') and results.get('tabnet_loading'):
        print("✅ RESULT: Both formats load successfully!")
        print("💡 CONCLUSION: Files are NOT corrupted")
        print("🔧 SOLUTION: The validation issue is likely TIMING-related")
        print("   → Add file sync + delay before validation")
        
    elif results.get('file_integrity') and not results.get('tabnet_loading'):
        print("⚠️  RESULT: Files exist and are valid zip, but TabNet loading fails")
        print("💡 CONCLUSION: Library/session state issue")
        print("🔧 SOLUTION: Skip in-session validation or use external process")
        
    elif not results.get('file_integrity'):
        print("❌ RESULT: Files are actually corrupted")
        print("💡 CONCLUSION: Save process has issues")
        print("🔧 SOLUTION: Fix save process with proper file flushing")
        
    else:
        print("🤔 RESULT: Mixed results - need further investigation")
    
    return results

if __name__ == "__main__":
    
    # Check if we're in the right environment
    print("🔧 Environment check...")
    try:
        import pytorch_tabnet
        print("✅ pytorch_tabnet available")
    except ImportError:
        print("❌ pytorch_tabnet not available")
        print("💡 Run: conda activate tabnet-prostate")
        sys.exit(1)
    
    # Run the diagnostic
    results = test_model_loading()
    
    print("\n🏁 Diagnostic completed!")
    print("Copy the output above to share results.")