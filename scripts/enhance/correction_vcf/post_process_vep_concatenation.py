#!/usr/bin/env python3
"""
VEP CSV Concatenation Correction Script - FIXED VERSION
Fixes concatenated CLIN_SIG and other fields in TabNet CSV files

Location: /u/aa107/uiuc-cancer-research/scripts/enhance/correction_vcf/post_process_vep_concatenation.py
Usage: python post_process_vep_concatenation.py

Author: PhD Research Student, University of Illinois
Contact: aa107@illinois.edu

CRITICAL FIX: Now works directly on CSV files instead of VCF parsing
"""

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from collections import Counter

# Enhanced severity ranking tables
CLIN_SIG_SEVERITY = {
    'pathogenic': 5,
    'likely_pathogenic': 4,
    'uncertain_significance': 3,
    'likely_benign': 2,
    'benign': 1,
    'not_provided': 0,
    'drug_response': 2,
    'other': 1,
    'association': 2,
    'protective': 2,
    'conflicting_interpretations_of_pathogenicity': 3,
    'affects': 2,
    'risk_factor': 2,
    'confers_sensitivity': 2,
    '': 0,  # Empty values
    'nan': 0,
    'none': 0
}

CONSEQUENCE_SEVERITY = {
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
    'stop_retained_variant': 2,
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
    'intergenic_variant': 0,
    '': 0,  # Empty values
    'nan': 0
}

class CSVConcatenationCleaner:
    """Cleans concatenated fields in TabNet CSV files"""
    
    def __init__(self):
        self.stats = {
            'total_variants': 0,
            'clin_sig_cleaned': 0,
            'consequence_cleaned': 0,
            'domains_cleaned': 0,
            'var_synonyms_cleaned': 0
        }
        
    def log_debug(self, message, level="INFO"):
        """Debug logging with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def normalize_value(self, value):
        """Normalize field values for consistent matching"""
        if pd.isna(value) or value == '':
            return ''
        
        # Convert to string and normalize
        value = str(value).lower().strip()
        
        # Remove leading underscores (artifact from AWK processing)
        value = value.lstrip('_')
        
        # Handle common variations
        value = value.replace(' ', '_')
        value = value.replace('-', '_')
        
        return value
    
    def get_highest_severity_value(self, concatenated_value, severity_table):
        """Select value with highest severity score"""
        if pd.isna(concatenated_value) or concatenated_value == '':
            return ''
        
        # Split by '&' separator
        parts = str(concatenated_value).split('&')
        
        # Normalize and clean parts
        cleaned_parts = []
        for part in parts:
            normalized = self.normalize_value(part)
            if normalized:
                cleaned_parts.append(normalized)
        
        if not cleaned_parts:
            return ''
        
        # Find highest severity value
        max_severity = -1
        best_value = cleaned_parts[0]
        
        for part in cleaned_parts:
            severity = severity_table.get(part, -1)
            if severity > max_severity:
                max_severity = severity
                best_value = part
        
        # Return original case format if possible
        for original_part in str(concatenated_value).split('&'):
            if self.normalize_value(original_part) == best_value:
                return original_part.strip()
        
        return best_value
    
    def clean_clin_sig_column(self, df):
        """Clean CLIN_SIG column concatenation"""
        self.log_debug("🏥 Cleaning CLIN_SIG column...")
        
        if 'CLIN_SIG' not in df.columns:
            self.log_debug("⚠️ CLIN_SIG column not found", "WARNING")
            return df
        
        # Find concatenated values
        concatenated_mask = df['CLIN_SIG'].astype(str).str.contains('&', na=False)
        concatenated_count = concatenated_mask.sum()
        
        self.log_debug(f"Found {concatenated_count:,} concatenated CLIN_SIG values")
        
        if concatenated_count > 0:
            # Show sample patterns before cleaning
            sample_patterns = df.loc[concatenated_mask, 'CLIN_SIG'].value_counts().head(5)
            self.log_debug("Sample concatenated patterns:")
            for pattern, count in sample_patterns.items():
                self.log_debug(f"  {pattern}: {count} variants")
            
            # Apply severity ranking to concatenated values
            df.loc[concatenated_mask, 'CLIN_SIG'] = df.loc[concatenated_mask, 'CLIN_SIG'].apply(
                lambda x: self.get_highest_severity_value(x, CLIN_SIG_SEVERITY)
            )
            
            self.stats['clin_sig_cleaned'] = concatenated_count
            self.log_debug(f"✅ Cleaned {concatenated_count:,} CLIN_SIG values")
            
            # Show results after cleaning
            remaining_concat = df['CLIN_SIG'].astype(str).str.contains('&', na=False).sum()
            self.log_debug(f"Remaining concatenated values: {remaining_concat}")
        
        return df
    
    def clean_consequence_column(self, df):
        """Clean Consequence column concatenation"""
        self.log_debug("🎯 Cleaning Consequence column...")
        
        if 'Consequence' not in df.columns:
            self.log_debug("⚠️ Consequence column not found", "WARNING")
            return df
        
        # Find concatenated values
        concatenated_mask = df['Consequence'].astype(str).str.contains('&', na=False)
        concatenated_count = concatenated_mask.sum()
        
        self.log_debug(f"Found {concatenated_count:,} concatenated Consequence values")
        
        if concatenated_count > 0:
            # Apply severity ranking
            df.loc[concatenated_mask, 'Consequence'] = df.loc[concatenated_mask, 'Consequence'].apply(
                lambda x: self.get_highest_severity_value(x, CONSEQUENCE_SEVERITY)
            )
            
            self.stats['consequence_cleaned'] = concatenated_count
            self.log_debug(f"✅ Cleaned {concatenated_count:,} Consequence values")
        
        return df
    
    def clean_other_concatenated_fields(self, df):
        """Clean other fields with concatenation issues"""
        fields_to_clean = {
            'DOMAINS': 'keep_first',
            'VAR_SYNONYMS': 'keep_shortest',
            'PUBMED': 'count_values'
        }
        
        for field_name, method in fields_to_clean.items():
            if field_name not in df.columns:
                continue
                
            concatenated_mask = df[field_name].astype(str).str.contains('&', na=False)
            concatenated_count = concatenated_mask.sum()
            
            if concatenated_count > 0:
                self.log_debug(f"🔧 Cleaning {field_name}: {concatenated_count:,} values")
                
                if method == 'keep_first':
                    df.loc[concatenated_mask, field_name] = df.loc[concatenated_mask, field_name].apply(
                        lambda x: str(x).split('&')[0].strip() if pd.notna(x) else x
                    )
                elif method == 'keep_shortest':
                    df.loc[concatenated_mask, field_name] = df.loc[concatenated_mask, field_name].apply(
                        lambda x: min(str(x).split('&'), key=len).strip() if pd.notna(x) else x
                    )
                elif method == 'count_values':
                    df.loc[concatenated_mask, field_name] = df.loc[concatenated_mask, field_name].apply(
                        lambda x: len(str(x).split('&')) if pd.notna(x) else 0
                    )
                
                self.stats[f"{field_name.lower()}_cleaned"] = concatenated_count
        
        return df
    
    def process_csv_file(self, input_file, output_file=None):
        """Main CSV processing function"""
        self.log_debug("🧬 CSV Concatenation Post-Processing Started")
        self.log_debug(f"📁 Input: {input_file}")
        
        if output_file is None:
            output_file = input_file.replace('.csv', '_cleaned.csv')
        
        self.log_debug(f"📁 Output: {output_file}")
        
        # Validate input file
        if not os.path.exists(input_file):
            self.log_debug(f"❌ Input file not found: {input_file}", "ERROR")
            return False
        
        try:
            # Load CSV
            self.log_debug("📊 Loading CSV file...")
            df = pd.read_csv(input_file, low_memory=False)
            self.stats['total_variants'] = len(df)
            
            self.log_debug(f"✅ Loaded {len(df):,} variants × {len(df.columns)} features")
            
            # Clean concatenated fields
            df = self.clean_clin_sig_column(df)
            df = self.clean_consequence_column(df)
            df = self.clean_other_concatenated_fields(df)
            
            # Save cleaned CSV
            self.log_debug("💾 Saving cleaned CSV...")
            df.to_csv(output_file, index=False)
            
            self.log_debug("✅ Processing complete!")
            self.print_final_stats()
            
            return True
            
        except Exception as e:
            self.log_debug(f"❌ Error processing CSV: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
            return False
    
    def print_final_stats(self):
        """Print final processing statistics"""
        self.log_debug("\n📊 FINAL PROCESSING STATISTICS")
        self.log_debug("=" * 40)
        self.log_debug(f"Total variants processed: {self.stats['total_variants']:,}")
        self.log_debug(f"CLIN_SIG values cleaned: {self.stats['clin_sig_cleaned']:,}")
        self.log_debug(f"Consequence values cleaned: {self.stats['consequence_cleaned']:,}")
        
        if self.stats['total_variants'] > 0:
            cleanup_rate = (self.stats['clin_sig_cleaned'] / self.stats['total_variants']) * 100
            self.log_debug(f"CLIN_SIG cleanup rate: {cleanup_rate:.1f}%")

def main():
    """Main execution function"""
    # File paths
    input_file = "/u/aa107/uiuc-cancer-research/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv"
    output_file = "/u/aa107/uiuc-cancer-research/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv"  # Overwrite original
    
    print("🧬 VEP CSV CONCATENATION CORRECTION")
    print("=" * 50)
    print(f"Processing: {input_file}")
    print(f"Output: {output_file}")
    print("")
    
    # Create cleaner and process
    cleaner = CSVConcatenationCleaner()
    success = cleaner.process_csv_file(input_file, output_file)
    
    if success:
        print("\n🎉 CSV CONCATENATION CORRECTION COMPLETED SUCCESSFULLY!")
        print("\n📋 NEXT STEPS:")
        print("1. Validate results with:")
        print("   python -c \"import pandas as pd; df = pd.read_csv('data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv'); print(f'CLIN_SIG concatenation: {df[\"CLIN_SIG\"].str.contains(\"&\", na=False).sum():,} ({df[\"CLIN_SIG\"].str.contains(\"&\", na=False).mean()*100:.1f}%)')\"")
        print("\n2. Consider retraining TabNet model for improved performance")
    else:
        print("\n❌ CSV CONCATENATION CORRECTION FAILED!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())