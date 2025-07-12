#!/usr/bin/env python3
"""
Debug VCF Validation Script - Focus on Consequence and CLIN_SIG
Validates VEP enhancement effectiveness for TabNet ML training
WITH EXTENSIVE DEBUG OUTPUT
"""

import re
import sys
from collections import Counter

def analyze_vcf(vcf_file):
    """Debug analysis focusing on Consequence and CLIN_SIG fields."""
    
    print("🔍 DEBUG VCF VALIDATION - CONSEQUENCE & CLIN_SIG FOCUS")
    print("=" * 60)
    
    # Stats tracking
    total_variants = 0
    annotated_variants = 0
    consequence_data = []
    clin_sig_data = []
    
    # Debug data
    debug_samples = []
    
    # Find CSQ header to map field positions
    csq_fields = []
    
    with open(vcf_file, 'r') as f:
        for line in f:
            # Parse CSQ header
            if line.startswith('##INFO=<ID=CSQ'):
                match = re.search(r'Format: ([^"]+)', line)
                if match:
                    csq_fields = match.group(1).split('|')
                    print(f"✅ Found CSQ header with {len(csq_fields)} fields")
                    
                    # DEBUG: Print all field positions
                    print("\n🔍 DEBUG: CSQ Field Positions:")
                    for i, field in enumerate(csq_fields):
                        print(f"  Position {i:2d}: {field}")
                    
                    # Find Consequence and CLIN_SIG positions
                    consequence_pos = None
                    clin_sig_pos = None
                    
                    for i, field in enumerate(csq_fields):
                        if 'consequence' in field.lower():
                            consequence_pos = i
                            print(f"\n📍 Consequence field at position {i}: {field}")
                        elif 'clin_sig' in field.lower() or 'clinical_significance' in field.lower():
                            clin_sig_pos = i
                            print(f"📍 CLIN_SIG field at position {i}: {field}")
                    
                    if consequence_pos is None:
                        print("\n⚠️  Consequence field not found, using position 1 (typical VEP)")
                        consequence_pos = 1
                    if clin_sig_pos is None:
                        print("⚠️  CLIN_SIG field not found, will search for 'pathogenic' patterns")
                
                break
        
        # Process variants - WITH DEBUG
        print(f"\n🔍 DEBUG: Processing variants with Consequence at pos {consequence_pos}, CLIN_SIG at pos {clin_sig_pos}")
        
        for line in f:
            if line.startswith('#'):
                continue
                
            total_variants += 1
            fields = line.strip().split('\t')
            
            if len(fields) >= 8:
                info_field = fields[7]
                
                # Extract CSQ data
                csq_match = re.search(r'CSQ=([^;]+)', info_field)
                if csq_match:
                    annotated_variants += 1
                    csq_string = csq_match.group(1)
                    
                    # DEBUG: Show first few CSQ strings
                    if annotated_variants <= 5:
                        print(f"\n🔍 DEBUG Variant {annotated_variants}:")
                        print(f"  Full INFO: {info_field[:200]}...")
                        print(f"  CSQ string: {csq_string[:300]}...")
                    
                    # Use first transcript (main annotation)
                    first_transcript = csq_string.split(',')[0]
                    csq_values = first_transcript.split('|')
                    
                    # DEBUG: Show CSQ parsing
                    if annotated_variants <= 5:
                        print(f"  First transcript: {first_transcript[:200]}...")
                        print(f"  CSQ values count: {len(csq_values)}")
                        if len(csq_values) > max(consequence_pos, clin_sig_pos or 0):
                            print(f"  Consequence raw: '{csq_values[consequence_pos]}'")
                            if clin_sig_pos:
                                print(f"  CLIN_SIG raw: '{csq_values[clin_sig_pos]}'")
                    
                    # Extract Consequence
                    if consequence_pos < len(csq_values):
                        consequence = csq_values[consequence_pos]
                        if consequence:
                            consequence_data.append(consequence)
                            
                            # DEBUG: Track samples for detailed analysis
                            if len(debug_samples) < 20:
                                debug_samples.append({
                                    'variant_id': fields[2] if len(fields) > 2 else f"var_{annotated_variants}",
                                    'consequence': consequence,
                                    'clin_sig': csq_values[clin_sig_pos] if clin_sig_pos and clin_sig_pos < len(csq_values) else 'N/A',
                                    'has_consequence_concat': '&' in consequence,
                                    'has_clin_sig_concat': '&' in (csq_values[clin_sig_pos] if clin_sig_pos and clin_sig_pos < len(csq_values) else ''),
                                    'full_csq': first_transcript
                                })
                    
                    # Extract CLIN_SIG
                    if clin_sig_pos and clin_sig_pos < len(csq_values):
                        clin_sig = csq_values[clin_sig_pos]
                        if clin_sig:
                            clin_sig_data.append(clin_sig)
                    else:
                        # Search for pathogenic patterns in the whole CSQ string
                        if 'pathogenic' in csq_string.lower():
                            # Extract the field containing pathogenic
                            for i, value in enumerate(csq_values):
                                if 'pathogenic' in value.lower():
                                    clin_sig_data.append(value)
                                    # DEBUG: Track where pathogenic was found
                                    if annotated_variants <= 5:
                                        print(f"  Found pathogenic in position {i}: '{value}'")
                                    break
    
    # DEBUG: Show sample analysis
    print(f"\n🔍 DEBUG: Sample Analysis (first 20 variants)")
    print("=" * 60)
    for i, sample in enumerate(debug_samples):
        print(f"Sample {i+1}: {sample['variant_id']}")
        print(f"  Consequence: '{sample['consequence']}' (concat: {sample['has_consequence_concat']})")
        print(f"  CLIN_SIG: '{sample['clin_sig']}' (concat: {sample['has_clin_sig_concat']})")
        if sample['has_consequence_concat'] or sample['has_clin_sig_concat']:
            print(f"  Full CSQ: {sample['full_csq'][:150]}...")
        print()
    
    # Analysis Results
    print(f"\n📊 VALIDATION RESULTS")
    print("=" * 30)
    print(f"Total variants: {total_variants:,}")
    print(f"Annotated variants: {annotated_variants:,}")
    print(f"Annotation rate: {(annotated_variants/total_variants)*100:.1f}%")
    
    # Consequence Analysis
    print(f"\n🎯 CONSEQUENCE FIELD ANALYSIS")
    print("=" * 35)
    print(f"Total consequence values: {len(consequence_data):,}")
    
    # Check for concatenation in Consequence
    concatenated_consequences = [c for c in consequence_data if '&' in c]
    print(f"❌ Concatenated consequences: {len(concatenated_consequences):,} ({(len(concatenated_consequences)/len(consequence_data)*100):.1f}%)")
    
    # DEBUG: Show detailed concatenation examples
    if concatenated_consequences:
        print("\n🔍 DEBUG: Concatenated Consequence Examples:")
        unique_patterns = set(concatenated_consequences[:50])  # First 50 unique patterns
        for i, example in enumerate(list(unique_patterns)[:10]):
            parts = example.split('&')
            print(f"  {i+1}. '{example}' -> Parts: {parts}")
    else:
        print("✅ No concatenation found in Consequence field!")
    
    # Top consequences
    consequence_counts = Counter(consequence_data)
    print(f"\n🔍 DEBUG: Top 10 consequences with counts:")
    for i, (cons, count) in enumerate(consequence_counts.most_common(10)):
        has_concat = '&' in cons
        print(f"  {i+1:2d}. {cons} ({count:,} variants) {'[CONCAT]' if has_concat else ''}")
    
    # CLIN_SIG Analysis
    print(f"\n🏥 CLIN_SIG FIELD ANALYSIS")
    print("=" * 30)
    print(f"Total CLIN_SIG values: {len(clin_sig_data):,}")
    
    if clin_sig_data:
        # Check for concatenation in CLIN_SIG
        concatenated_clin_sig = [c for c in clin_sig_data if '&' in c]
        print(f"❌ Concatenated CLIN_SIG: {len(concatenated_clin_sig):,} ({(len(concatenated_clin_sig)/len(clin_sig_data)*100):.1f}%)")
        
        # DEBUG: Show detailed CLIN_SIG concatenation examples
        if concatenated_clin_sig:
            print("\n🔍 DEBUG: Concatenated CLIN_SIG Examples:")
            unique_patterns = set(concatenated_clin_sig[:50])  # First 50 unique patterns
            for i, example in enumerate(list(unique_patterns)[:10]):
                parts = example.split('&')
                print(f"  {i+1}. '{example}' -> Parts: {parts}")
        else:
            print("✅ No concatenation found in CLIN_SIG field!")
        
        # Top CLIN_SIG values
        clin_sig_counts = Counter(clin_sig_data)
        print(f"\n🔍 DEBUG: Top 10 CLIN_SIG values with counts:")
        for i, (clin, count) in enumerate(clin_sig_counts.most_common(10)):
            has_concat = '&' in clin
            print(f"  {i+1:2d}. {clin} ({count:,} variants) {'[CONCAT]' if has_concat else ''}")
    else:
        print("⚠️  No CLIN_SIG data found")
    
    # Other Field Issues Check
    print(f"\n🔍 OTHER FIELD MALFORMATION CHECK")
    print("=" * 40)
    
    # Sample a few variants to check for excessive concatenation
    sample_variants = 0
    heavily_concatenated_fields = []
    
    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if sample_variants >= 100:  # Check first 100 variants
                break
                
            fields = line.strip().split('\t')
            if len(fields) >= 8:
                info_field = fields[7]
                csq_match = re.search(r'CSQ=([^;]+)', info_field)
                if csq_match:
                    csq_string = csq_match.group(1)
                    first_transcript = csq_string.split(',')[0]
                    csq_values = first_transcript.split('|')
                    
                    # Check each field for excessive concatenation
                    for i, value in enumerate(csq_values):
                        if value and '&' in value:
                            amp_count = value.count('&')
                            if amp_count > 10:  # Heavily concatenated
                                field_name = csq_fields[i] if i < len(csq_fields) else f"Field_{i}"
                                heavily_concatenated_fields.append((field_name, amp_count, value[:100]))
                
                sample_variants += 1
    
    if heavily_concatenated_fields:
        print("⚠️  Fields with heavy concatenation (>10 '&' symbols):")
        unique_fields = {}
        for field_name, count, example in heavily_concatenated_fields:
            if field_name not in unique_fields or count > unique_fields[field_name][0]:
                unique_fields[field_name] = (count, example)
        
        for field_name, (max_count, example) in sorted(unique_fields.items(), key=lambda x: x[1][0], reverse=True)[:5]:
            print(f"  - {field_name}: up to {max_count} concatenated values")
            print(f"    Example: {example}...")
    else:
        print("✅ No heavily concatenated fields found")
    
    # DEBUG: Manual comparison with first few variants
    print(f"\n🔍 DEBUG: Manual Comparison Check")
    print("=" * 35)
    print("Let's manually check the first 3 variants from command line:")
    
    # Summary
    print(f"\n🎯 SUMMARY")
    print("=" * 15)
    consequence_ok = len(concatenated_consequences) == 0
    clin_sig_ok = len(concatenated_clin_sig) == 0 if clin_sig_data else True
    
    print(f"Debug findings:")
    print(f"- Using Consequence position: {consequence_pos}")
    print(f"- Using CLIN_SIG position: {clin_sig_pos}")
    print(f"- Found {len(concatenated_consequences)} concatenated consequences")
    print(f"- Found {len(concatenated_clin_sig)} concatenated CLIN_SIG values")
    
    if consequence_ok and clin_sig_ok:
        print("✅ SUCCESS: Critical fields (Consequence, CLIN_SIG) are clean!")
        print("✅ VEP enhancement parameters worked effectively")
        print("✅ Ready for TabNet ML training")
    else:
        print("❌ ISSUES FOUND:")
        if not consequence_ok:
            print(f"  - Consequence field has {len(concatenated_consequences):,} concatenated values")
        if not clin_sig_ok:
            print(f"  - CLIN_SIG field has {len(concatenated_clin_sig):,} concatenated values")
        print("❌ Additional VEP parameter tuning needed")

if __name__ == "__main__":
    #vcf_file = "/u/aa107/uiuc-cancer-research/data/processed/vep/vep_annotated.vcf"
    vcf_file = "/u/aa107/uiuc-cancer-research/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv"
    
    try:
        analyze_vcf(vcf_file)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)