#!/bin/bash
#SBATCH --job-name=vep_annotation
#SBATCH --partition=IllinoisComputes
#SBATCH --account=aa107-ic
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64G
#SBATCH --output=/u/aa107/uiuc-cancer-research/scripts/vep/vep_%j.out
#SBATCH --error=/u/aa107/uiuc-cancer-research/scripts/vep/vep_%j.err

# VEP Annotation Script with FIXED Post-Processing for Concatenated Fields
# Critical Fix: Proper CSQ field parsing for Consequence and CLIN_SIG cleaning

set -e  # Exit on any error

# === CONFIGURATION ===
PROJECT_DIR="/u/aa107/uiuc-cancer-research"
INPUT_VCF="${PROJECT_DIR}/data/processed/merged_vcf/merged_prostate_variants.vcf"

# Use scratch for large files
SCRATCH_VEP="/scratch/aa107/vep_workspace"
CACHE_DIR="${SCRATCH_VEP}/vep_cache"              # Cache in scratch (15GB+)
CONTAINER_DIR="${SCRATCH_VEP}/containers"         # Container in scratch (640MB)
TEMP_DIR="${SCRATCH_VEP}/temp"                    # Temp files in scratch

# Final outputs in project (keep for easy access)
OUTPUT_DIR="${PROJECT_DIR}/data/processed/vep"
LOG_FILE="${OUTPUT_DIR}/vep_annotation.log"

echo "=== VEP WITH FIXED POST-PROCESSING STARTED: $(date) ===" | tee $LOG_FILE
echo "🎯 Target: Fix concatenated fields (Consequence & CLIN_SIG) blocking TabNet training" | tee -a $LOG_FILE
echo "🔧 Solution: Corrected AWK scripts for proper CSQ field parsing" | tee -a $LOG_FILE

# === CREATE DIRECTORIES ===
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH_VEP}"
mkdir -p "${CACHE_DIR}"
mkdir -p "${CONTAINER_DIR}"
mkdir -p "${TEMP_DIR}"

echo "📁 Directories created:" | tee -a $LOG_FILE
echo "  • Cache: ${CACHE_DIR}" | tee -a $LOG_FILE
echo "  • Container: ${CONTAINER_DIR}" | tee -a $LOG_FILE
echo "  • Output: ${OUTPUT_DIR}" | tee -a $LOG_FILE

# === DOWNLOAD BIOCONTAINER VEP TO SCRATCH ===
VEP_CONTAINER="${CONTAINER_DIR}/vep-biocontainer.sif"

if [ ! -f "$VEP_CONTAINER" ]; then
    echo "📦 Downloading BioContainers VEP (114.1)..." | tee -a $LOG_FILE
    cd "${CONTAINER_DIR}"
    apptainer pull vep-biocontainer.sif docker://quay.io/biocontainers/ensembl-vep:114.1--pl5321h2a3209d_0
    echo "✅ BioContainers VEP downloaded" | tee -a $LOG_FILE
else
    echo "✅ BioContainers VEP container already exists" | tee -a $LOG_FILE
fi

# === VALIDATE INPUT ===
echo "📊 Validating input file..." | tee -a $LOG_FILE
if [ ! -f "$INPUT_VCF" ]; then
    echo "❌ ERROR: Input VCF not found: $INPUT_VCF" | tee -a $LOG_FILE
    exit 1
fi

INPUT_COUNT=$(grep -v "^#" $INPUT_VCF | wc -l)
echo "✅ Input variants: $INPUT_COUNT" | tee -a $LOG_FILE

# === INSTALL VEP CACHE IN SCRATCH ===
echo "📚 Setting up VEP cache..." | tee -a $LOG_FILE

if [ ! -d "${CACHE_DIR}/homo_sapiens" ]; then
    echo "Installing VEP cache for human GRCh38..." | tee -a $LOG_FILE
    
    # Bind both project and scratch directories to container
    apptainer exec \
        --bind ${PROJECT_DIR}:${PROJECT_DIR} \
        --bind ${SCRATCH_VEP}:${SCRATCH_VEP} \
        $VEP_CONTAINER \
        vep_install \
        -a cf \
        -s homo_sapiens \
        -y GRCh38 \
        -c ${CACHE_DIR} \
        --CONVERT 2>&1 | tee -a $LOG_FILE
    
    if [ $? -eq 0 ]; then
        echo "✅ VEP cache installation completed" | tee -a $LOG_FILE
        
        # Create symbolic link in project for easy access
        ln -sf ${CACHE_DIR} ${OUTPUT_DIR}/vep_cache_link
        echo "🔗 Cache symlink created: ${OUTPUT_DIR}/vep_cache_link -> ${CACHE_DIR}" | tee -a $LOG_FILE
    else
        echo "❌ VEP cache installation failed, trying manual download..." | tee -a $LOG_FILE
        
        # Fallback: Manual download to scratch
        echo "📥 Downloading cache manually to scratch..." | tee -a $LOG_FILE
        cd ${CACHE_DIR}
        wget -q https://ftp.ensembl.org/pub/release-114/variation/indexed_vep_cache/homo_sapiens_vep_114_GRCh38.tar.gz
        tar -xzf homo_sapiens_vep_114_GRCh38.tar.gz
        rm homo_sapiens_vep_114_GRCh38.tar.gz
        echo "✅ Manual cache download completed" | tee -a $LOG_FILE
    fi
else
    echo "✅ VEP cache already exists" | tee -a $LOG_FILE
fi

# === TEST VEP WITH SMALL SAMPLE ===
echo "🧪 Testing VEP with small sample..." | tee -a $LOG_FILE
TEST_VCF="${TEMP_DIR}/test_sample.vcf"
TEST_OUTPUT="${TEMP_DIR}/test_output.vcf"

head -1000 $INPUT_VCF > $TEST_VCF

apptainer exec \
    --bind ${PROJECT_DIR}:${PROJECT_DIR} \
    --bind ${SCRATCH_VEP}:${SCRATCH_VEP} \
    $VEP_CONTAINER \
    vep \
    --input_file $TEST_VCF \
    --output_file $TEST_OUTPUT \
    --format vcf --vcf --force_overwrite \
    --species homo_sapiens --assembly GRCh38 \
    --cache --dir_cache ${CACHE_DIR} \
    --offline \
    --sift b --polyphen b --symbol --numbers \
    --canonical --protein --biotype \
    --fork 4 2>&1 | tee -a $LOG_FILE

# Check if test worked
if [ -f "$TEST_OUTPUT" ]; then
    TEST_VARIANTS=$(grep -v "^#" $TEST_OUTPUT | wc -l)
    echo "✅ Test successful: $TEST_VARIANTS variants processed" | tee -a $LOG_FILE
    rm -f $TEST_VCF $TEST_OUTPUT
else
    echo "❌ Test failed - check VEP configuration" | tee -a $LOG_FILE
    exit 1
fi

# === RUN FULL VEP ANNOTATION ===
echo "🚀 Running full VEP annotation..." | tee -a $LOG_FILE

# Determine optimal fork count
if [ $INPUT_COUNT -gt 100000 ]; then
    FORK_COUNT=16
    echo "Using 16 forks for large dataset" | tee -a $LOG_FILE
else
    FORK_COUNT=8
    echo "Using 8 forks for standard dataset" | tee -a $LOG_FILE
fi

# Use temp directory in scratch for intermediate processing
TEMP_OUTPUT="${TEMP_DIR}/vep_annotated_temp.vcf"

apptainer exec \
    --bind ${PROJECT_DIR}:${PROJECT_DIR} \
    --bind ${SCRATCH_VEP}:${SCRATCH_VEP} \
    $VEP_CONTAINER \
    vep \
    --input_file $INPUT_VCF \
    --output_file $TEMP_OUTPUT \
    --format vcf --vcf --force_overwrite \
    --species homo_sapiens --assembly GRCh38 \
    --cache --dir_cache ${CACHE_DIR} \
    --offline \
    --sift b --polyphen b --symbol --numbers --biotype \
    --canonical --protein --ccds --uniprot --domains \
    --regulatory --variant_class \
    --af --af_1kg --af_gnomad \
    --pubmed --var_synonyms \
    --pick_allele_gene --pick_order canonical,appris,tsl,biotype,ccds,rank,length --pick --flag_pick --per_gene \
    --fork $FORK_COUNT \
    --buffer_size 5000 \
    --stats_file ${TEMP_DIR}/vep_summary.html 2>&1 | tee -a $LOG_FILE

# === COPY RESULTS TO PROJECT DIRECTORY ===
echo "📋 Copying results to project directory..." | tee -a $LOG_FILE

if [ -f "$TEMP_OUTPUT" ]; then
    # Copy main output
    cp $TEMP_OUTPUT ${OUTPUT_DIR}/vep_annotated.vcf
    
    # Copy summary if exists
    if [ -f "${TEMP_DIR}/vep_summary.html" ]; then
        cp ${TEMP_DIR}/vep_summary.html ${OUTPUT_DIR}/
    fi
    
    echo "✅ Results copied to project directory" | tee -a $LOG_FILE
else
    echo "❌ VEP annotation failed - no output file generated" | tee -a $LOG_FILE
    exit 1
fi

# === INITIAL VALIDATION ===
echo "📊 Initial validation..." | tee -a $LOG_FILE
echo "=== VEP ANNOTATION COMPLETED: $(date) ===" | tee ${OUTPUT_DIR}/validation_report.txt
echo "Generated: $(date)" | tee -a ${OUTPUT_DIR}/validation_report.txt
echo "" | tee -a ${OUTPUT_DIR}/validation_report.txt

FINAL_OUTPUT="${OUTPUT_DIR}/vep_annotated.vcf"
if [ -f "$FINAL_OUTPUT" ]; then
    OUTPUT_COUNT=$(grep -v "^#" $FINAL_OUTPUT | wc -l)
    echo "✅ VEP annotation completed" | tee -a ${OUTPUT_DIR}/validation_report.txt
    echo "Input variants: $INPUT_COUNT" | tee -a ${OUTPUT_DIR}/validation_report.txt
    echo "Output variants: $OUTPUT_COUNT" | tee -a ${OUTPUT_DIR}/validation_report.txt
    
    # Check annotation quality
    if [ $OUTPUT_COUNT -gt 0 ]; then
        CSQ_COUNT=$(grep -c "CSQ=" $FINAL_OUTPUT || echo "0")
        if [ $CSQ_COUNT -gt 0 ]; then
            echo "✅ VEP annotations found: $CSQ_COUNT variants with CSQ data" | tee -a ${OUTPUT_DIR}/validation_report.txt
            
            # Sample annotation check
            SAMPLE_CSQ=$(grep -v "^#" $FINAL_OUTPUT | head -1 | grep -o "CSQ=[^;]*" | head -1)
            echo "Sample annotation: $SAMPLE_CSQ" | tee -a ${OUTPUT_DIR}/validation_report.txt
            
            # Count functional consequences
            MISSENSE_COUNT=$(grep -c "missense_variant" $FINAL_OUTPUT || echo "0")
            STOP_GAINED_COUNT=$(grep -c "stop_gained" $FINAL_OUTPUT || echo "0")
            SPLICE_COUNT=$(grep -c "splice" $FINAL_OUTPUT || echo "0")
            
            echo "Functional consequences found:" | tee -a ${OUTPUT_DIR}/validation_report.txt
            echo "  Missense variants: $MISSENSE_COUNT" | tee -a ${OUTPUT_DIR}/validation_report.txt
            echo "  Stop gained: $STOP_GAINED_COUNT" | tee -a ${OUTPUT_DIR}/validation_report.txt
            echo "  Splice variants: $SPLICE_COUNT" | tee -a ${OUTPUT_DIR}/validation_report.txt
            
        else
            echo "⚠️ No CSQ annotations found - checking for other VEP fields" | tee -a ${OUTPUT_DIR}/validation_report.txt
        fi
    else
        echo "❌ No variants in output file" | tee -a ${OUTPUT_DIR}/validation_report.txt
        exit 1
    fi
    
    OUTPUT_SIZE=$(du -sh $FINAL_OUTPUT | cut -f1)
    echo "Output file size: $OUTPUT_SIZE" | tee -a ${OUTPUT_DIR}/validation_report.txt
    
    # Success rate calculation
    if [ $INPUT_COUNT -gt 0 ]; then
        SUCCESS_RATE=$(echo "scale=1; $OUTPUT_COUNT * 100 / $INPUT_COUNT" | bc -l)
        echo "Success rate: ${SUCCESS_RATE}%" | tee -a ${OUTPUT_DIR}/validation_report.txt
    fi
    
else
    echo "❌ VEP annotation failed - output file not found" | tee -a ${OUTPUT_DIR}/validation_report.txt
    exit 1
fi

# === CRITICAL FIX: PROPER CSQ FIELD POST-PROCESSING ===
echo "🔧 CRITICAL FIX: Cleaning concatenated CSQ fields..." | tee -a $LOG_FILE

# Create temporary file for post-processing
TEMP_CLEANED="${OUTPUT_DIR}/vep_annotated_cleaned.vcf"
cp $FINAL_OUTPUT $TEMP_CLEANED

# Pre-processing validation
echo "📊 Pre-processing validation..." | tee -a $LOG_FILE
PRE_CONSEQUENCE_CONCAT=$(grep -v "^#" $TEMP_CLEANED | grep -o "CSQ=[^;]*" | grep -o "|[^|]*|" | sed -n '2p' | grep -c "&" || echo "0")
PRE_CLINSIG_CONCAT=$(grep -v "^#" $TEMP_CLEANED | grep -o "CSQ=[^;]*" | cut -d',' -f1 | cut -d'|' -f51 | grep -c "&" || echo "0")
echo "Pre-processing: Consequence concatenation samples: $PRE_CONSEQUENCE_CONCAT" | tee -a $LOG_FILE
echo "Pre-processing: CLIN_SIG concatenation samples: $PRE_CLINSIG_CONCAT" | tee -a $LOG_FILE

# FIXED AWK SCRIPT #1: Clean CLIN_SIG field (position 50 in CSQ)
echo "🧹 Cleaning CLIN_SIG field (position 50)..." | tee -a $LOG_FILE
awk '
BEGIN {
    # Clinical significance severity ranking (higher = more severe)
    clin_rank["pathogenic"] = 5
    clin_rank["likely_pathogenic"] = 4
    clin_rank["uncertain_significance"] = 3
    clin_rank["likely_benign"] = 2
    clin_rank["benign"] = 1
    clin_rank["not_provided"] = 0
    
    # Also handle variations with underscores and case differences
    clin_rank["Pathogenic"] = 5
    clin_rank["Likely_pathogenic"] = 4
    clin_rank["Uncertain_significance"] = 3
    clin_rank["Likely_benign"] = 2
    clin_rank["Benign"] = 1
    clin_rank["Not_provided"] = 0
}
/^#/ { print; next }
{
    # Only process lines with CSQ field
    if ($0 ~ /CSQ=/) {
        # Extract CSQ field content
        match($0, /CSQ=([^;]+)/, csq_match)
        if (csq_match[1]) {
            # Split multiple transcripts by comma
            split(csq_match[1], transcripts, ",")
            
            # Process each transcript
            for (t in transcripts) {
                # Split fields by pipe
                split(transcripts[t], fields, "|")
                
                # Process CLIN_SIG field at position 51 (1-indexed, so fields[51])
                if (length(fields) >= 51 && fields[51] != "") {
                    clin_sig = fields[51]
                    
                    # Handle concatenated CLIN_SIG values
                    if (clin_sig ~ /[&\/]/) {
                        # Split by & or / separators
                        gsub(/[&\/]/, " ", clin_sig)
                        split(clin_sig, clin_parts, " ")
                        
                        # Find most severe clinical significance
                        max_rank = -1
                        best_clin = ""
                        
                        for (i in clin_parts) {
                            if (clin_parts[i] != "") {
                                # Clean and normalize
                                clean_clin = clin_parts[i]
                                gsub(/[^a-zA-Z_]/, "", clean_clin)
                                clean_clin = tolower(clean_clin)
                                
                                # Check ranking
                                if (clean_clin in clin_rank && clin_rank[clean_clin] > max_rank) {
                                    max_rank = clin_rank[clean_clin]
                                    best_clin = clin_parts[i]
                                }
                            }
                        }
                        
                        # Replace with best clinical significance
                        if (best_clin != "") {
                            fields[51] = best_clin
                        }
                    }
                }
                
                # Rebuild transcript
                new_transcript = ""
                for (f = 1; f <= length(fields); f++) {
                    new_transcript = new_transcript (f > 1 ? "|" : "") fields[f]
                }
                transcripts[t] = new_transcript
            }
            
            # Rebuild CSQ field
            new_csq = ""
            for (t in transcripts) {
                new_csq = new_csq (t > 1 ? "," : "") transcripts[t]
            }
            
            # Replace CSQ in original line
            gsub(/CSQ=[^;]+/, "CSQ=" new_csq)
        }
    }
    print
}' $TEMP_CLEANED > "${TEMP_CLEANED}.tmp"

# Check if CLIN_SIG cleaning worked
if [ -f "${TEMP_CLEANED}.tmp" ]; then
    mv "${TEMP_CLEANED}.tmp" $TEMP_CLEANED
    echo "✅ CLIN_SIG cleaning completed" | tee -a $LOG_FILE
else
    echo "❌ CLIN_SIG cleaning failed" | tee -a $LOG_FILE
    exit 1
fi

# FIXED AWK SCRIPT #2: Clean Consequence field (position 1 in CSQ)
echo "🧹 Cleaning Consequence field (position 1)..." | tee -a $LOG_FILE
awk '
BEGIN {
    # Consequence severity ranking (higher = more severe)
    cons_rank["transcript_ablation"] = 10
    cons_rank["splice_acceptor_variant"] = 9
    cons_rank["splice_donor_variant"] = 9
    cons_rank["stop_gained"] = 8
    cons_rank["frameshift_variant"] = 8
    cons_rank["stop_lost"] = 7
    cons_rank["start_lost"] = 7
    cons_rank["transcript_amplification"] = 6
    cons_rank["inframe_insertion"] = 5
    cons_rank["inframe_deletion"] = 5
    cons_rank["missense_variant"] = 4
    cons_rank["protein_altering_variant"] = 4
    cons_rank["splice_region_variant"] = 3
    cons_rank["incomplete_terminal_codon_variant"] = 3
    cons_rank["stop_retained_variant"] = 2
    cons_rank["synonymous_variant"] = 2
    cons_rank["coding_sequence_variant"] = 2
    cons_rank["mature_miRNA_variant"] = 2
    cons_rank["5_prime_UTR_variant"] = 1
    cons_rank["3_prime_UTR_variant"] = 1
    cons_rank["non_coding_transcript_exon_variant"] = 1
    cons_rank["intron_variant"] = 1
    cons_rank["NMD_transcript_variant"] = 1
    cons_rank["non_coding_transcript_variant"] = 1
    cons_rank["regulatory_region_variant"] = 1
    cons_rank["upstream_gene_variant"] = 0
    cons_rank["downstream_gene_variant"] = 0
    
    # Handle common splice variants
    cons_rank["splice_polypyrimidine_tract_variant"] = 3
    cons_rank["splice_donor_region_variant"] = 3
    cons_rank["splice_donor_5th_base_variant"] = 3
}
/^#/ { print; next }
{
    # Only process lines with CSQ field
    if ($0 ~ /CSQ=/) {
        # Extract CSQ field content
        match($0, /CSQ=([^;]+)/, csq_match)
        if (csq_match[1]) {
            # Split multiple transcripts by comma
            split(csq_match[1], transcripts, ",")
            
            # Process each transcript
            for (t in transcripts) {
                # Split fields by pipe
                split(transcripts[t], fields, "|")
                
                # Process Consequence field at position 2 (1-indexed, so fields[2])
                if (length(fields) >= 2 && fields[2] != "") {
                    consequence = fields[2]
                    
                    # Handle concatenated consequences
                    if (consequence ~ /&/) {
                        # Split by & separator
                        split(consequence, cons_parts, "&")
                        
                        # Find most severe consequence
                        max_rank = -1
                        best_cons = ""
                        
                        for (i in cons_parts) {
                            if (cons_parts[i] != "") {
                                # Clean consequence
                                clean_cons = cons_parts[i]
                                gsub(/[^a-zA-Z_]/, "", clean_cons)
                                
                                # Check ranking
                                if (clean_cons in cons_rank && cons_rank[clean_cons] > max_rank) {
                                    max_rank = cons_rank[clean_cons]
                                    best_cons = cons_parts[i]
                                }
                            }
                        }
                        
                        # Replace with best consequence
                        if (best_cons != "") {
                            fields[2] = best_cons
                        }
                    }
                }
                
                # Rebuild transcript
                new_transcript = ""
                for (f = 1; f <= length(fields); f++) {
                    new_transcript = new_transcript (f > 1 ? "|" : "") fields[f]
                }
                transcripts[t] = new_transcript
            }
            
            # Rebuild CSQ field
            new_csq = ""
            for (t in transcripts) {
                new_csq = new_csq (t > 1 ? "," : "") transcripts[t]
            }
            
            # Replace CSQ in original line
            gsub(/CSQ=[^;]+/, "CSQ=" new_csq)
        }
    }
    print
}' $TEMP_CLEANED > "${TEMP_CLEANED}.tmp"

# Check if Consequence cleaning worked
if [ -f "${TEMP_CLEANED}.tmp" ]; then
    mv "${TEMP_CLEANED}.tmp" $TEMP_CLEANED
    echo "✅ Consequence cleaning completed" | tee -a $LOG_FILE
else
    echo "❌ Consequence cleaning failed" | tee -a $LOG_FILE
    exit 1
fi

# Replace original with cleaned version
mv $TEMP_CLEANED $FINAL_OUTPUT
echo "✅ Post-processing completed - concatenated fields cleaned" | tee -a $LOG_FILE

# === POST-PROCESSING VALIDATION ===
echo "📊 Post-processing validation..." | tee -a $LOG_FILE
POST_CONSEQUENCE_CONCAT=$(grep -v "^#" $FINAL_OUTPUT | grep -o "CSQ=[^;]*" | grep -o "|[^|]*|" | sed -n '2p' | grep -c "&" || echo "0")
POST_CLINSIG_CONCAT=$(grep -v "^#" $FINAL_OUTPUT | grep -o "CSQ=[^;]*" | cut -d',' -f1 | cut -d'|' -f51 | grep -c "&" || echo "0")
echo "Post-processing: Consequence concatenation samples: $POST_CONSEQUENCE_CONCAT" | tee -a $LOG_FILE
echo "Post-processing: CLIN_SIG concatenation samples: $POST_CLINSIG_CONCAT" | tee -a $LOG_FILE

# === CLEANUP TEMP FILES ===
echo "🧹 Cleaning up temporary files..." | tee -a $LOG_FILE
rm -f ${TEMP_DIR}/vep_annotated_temp.vcf

# === FINAL VALIDATION ===
echo "🔍 Running final validation..." | tee -a $LOG_FILE
echo "Run validation: python /u/aa107/uiuc-cancer-research/scripts/validation/comprehensive_column_audit.py" | tee -a $LOG_FILE

# === SUMMARY ===
echo "" | tee -a ${OUTPUT_DIR}/validation_report.txt
echo "=== STORAGE SUMMARY ===" | tee -a ${OUTPUT_DIR}/validation_report.txt
echo "Cache location: ${CACHE_DIR}" | tee -a ${OUTPUT_DIR}/validation_report.txt
echo "Cache size: $(du -sh ${CACHE_DIR} | cut -f1)" | tee -a ${OUTPUT_DIR}/validation_report.txt
echo "Container location: ${VEP_CONTAINER}" | tee -a ${OUTPUT_DIR}/validation_report.txt
echo "Output location: ${OUTPUT_DIR}" | tee -a ${OUTPUT_DIR}/validation_report.txt
echo "" | tee -a ${OUTPUT_DIR}/validation_report.txt

echo "=== FILES GENERATED ===" | tee -a ${OUTPUT_DIR}/validation_report.txt
ls -lh ${OUTPUT_DIR}/ | tee -a ${OUTPUT_DIR}/validation_report.txt

echo "" | tee -a ${OUTPUT_DIR}/validation_report.txt
echo "=== NEXT STEPS ===" | tee -a ${OUTPUT_DIR}/validation_report.txt
echo "1. Review validation report: ${OUTPUT_DIR}/validation_report.txt" | tee -a ${OUTPUT_DIR}/validation_report.txt
echo "2. Check VEP summary: ${OUTPUT_DIR}/vep_summary.html" | tee -a ${OUTPUT_DIR}/validation_report.txt
echo "3. Run validation: python /u/aa107/uiuc-cancer-research/scripts/validation/comprehensive_column_audit.py" | tee -a ${OUTPUT_DIR}/validation_report.txt
echo "4. Convert VCF to CSV for TabNet: python scripts/vep/vcf_to_tabnet.py" | tee -a ${OUTPUT_DIR}/validation_report.txt
echo "5. Proceed with TabNet training" | tee -a ${OUTPUT_DIR}/validation_report.txt

echo "=== VEP WITH FIXED POST-PROCESSING COMPLETED: $(date) ===" | tee -a $LOG_FILE

# === FINAL STATUS CHECK ===
if [ -f "$FINAL_OUTPUT" ] && [ $OUTPUT_COUNT -gt 0 ]; then
    echo "✅ SUCCESS! VEP annotation completed with $OUTPUT_COUNT variants"
    echo "📁 Main output: $FINAL_OUTPUT"
    echo "📊 Summary: ${OUTPUT_DIR}/vep_summary.html"
    echo "📋 Report: ${OUTPUT_DIR}/validation_report.txt"
    echo "🔗 Cache symlink: ${OUTPUT_DIR}/vep_cache_link -> ${CACHE_DIR}"
    echo "💾 Scratch workspace: ${SCRATCH_VEP}"
    echo ""
    echo "🔧 CRITICAL FIXES APPLIED:"
    echo "  • Fixed CSQ field parsing for proper post-processing"
    echo "  • Corrected field positions (Consequence=2, CLIN_SIG=51)"
    echo "  • Added proper concatenation handling with & and / separators"
    echo "  • Implemented severity ranking for both fields"
    echo "  • Added pre/post processing validation"
    echo ""
    echo "📋 VALIDATION REQUIRED:"
    echo "  Run: python /u/aa107/uiuc-cancer-research/scripts/validation/comprehensive_column_audit.py"
    echo "  Expected: <2% concatenation in both Consequence and CLIN_SIG fields"
else
    echo "❌ FAILED! Check logs and validation report"
    exit 1
fi