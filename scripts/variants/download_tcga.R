#!/usr/bin/env Rscript
# Enhanced TCGA-PRAD Download Script
# Robust version for UIUC Campus Cluster

cat("🧬 Starting TCGA-PRAD Mutation Download\n")
cat("=====================================\n")

# Set up R library path for user installation
user_lib_path <- "~/Rlibs"
if (!dir.exists(user_lib_path)) {
  dir.create(user_lib_path, recursive = TRUE)
  cat("📁 Created R library directory:", user_lib_path, "\n")
}

# Add to library paths
.libPaths(c(user_lib_path, .libPaths()))

# Function to safely install packages
safe_install <- function(package, source = "CRAN") {
  if (!require(package, character.only = TRUE, quietly = TRUE)) {
    cat("📦 Installing", package, "from", source, "\n")
    if (source == "CRAN") {
      install.packages(package, lib = user_lib_path, 
                      repos = "https://cran.r-project.org")
    } else if (source == "GitHub") {
      if (!require("devtools", character.only = TRUE)) {
        install.packages("devtools", lib = user_lib_path,
                        repos = "https://cran.r-project.org")
      }
      devtools::install_github(package, lib = user_lib_path)
    }
    library(package, character.only = TRUE)
  } else {
    cat("✅", package, "already installed\n")
  }
}

# Install required packages
tryCatch({
  safe_install("devtools")

  # Install TCGAmutations from GitHub (package name is just "TCGAmutations")
  if (!require("TCGAmutations", character.only = TRUE, quietly = TRUE)) {
    cat("📦 Installing TCGAmutations from GitHub\n")
    devtools::install_github("PoisonAlien/TCGAmutations")
  } else {
    cat("✅ TCGAmutations already installed\n")
  }

  library(TCGAmutations)
  cat("✅ All packages loaded successfully\n\n")
  
}, error = function(e) {
  cat("❌ Package installation failed:", conditionMessage(e), "\n")
  cat("💡 Try running this on a head node with internet access\n")
  quit(status = 1)
})

# Create output directory (use absolute path based on script location)
script_dir <- tryCatch({
  dirname(sys.frame(1)$ofile)
}, error = function(e) {
  getwd()
})

# Handle case when running via Rscript
if (is.null(script_dir) || script_dir == "") {
  script_dir <- getwd()
}

# Build path relative to project root
output_dir <- file.path(dirname(dirname(script_dir)), "data", "raw", "variants")

# Fallback: if that doesn't exist, use relative to current working directory
if (!grepl("uiuc-cancer-research", output_dir)) {
  output_dir <- file.path(getwd(), "data", "raw", "variants")
}

cat("📁 Output directory:", output_dir, "\n")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("📁 Created output directory\n")
}

# Download TCGA-PRAD mutations
cat("🔄 Downloading TCGA-PRAD mutations from MC3 dataset...\n")
cat("This may take several minutes...\n")

tryCatch({
  # Download the data - tcga_load returns the MAF object directly
  prad_maf <- tcga_load(study = "PRAD", source = "MC3")

  cat("📊 TCGA data loaded successfully\n")
  cat("📊 Object class:", class(prad_maf), "\n")

  # Extract mutation data from MAF object
  if (inherits(prad_maf, "MAF")) {
    prad_mutations <- prad_maf@data
    cat("✅ Extracted data from MAF object\n")
  } else if (is.data.frame(prad_maf)) {
    prad_mutations <- prad_maf
    cat("✅ Data is already a data frame\n")
  } else {
    # Try to access data slot directly
    prad_mutations <- tryCatch({
      slot(prad_maf, "data")
    }, error = function(e) {
      as.data.frame(prad_maf)
    })
    cat("✅ Converted to data frame\n")
  }

  cat("📊 Raw mutations loaded:", nrow(prad_mutations), "mutations\n")
  cat("📊 Columns available:", ncol(prad_mutations), "columns\n")

  # Display first few column names to verify structure
  cat("🔍 First 10 columns:", paste(head(colnames(prad_mutations), 10), collapse = ", "), "\n")

}, error = function(e) {
  cat("❌ TCGA download failed:", conditionMessage(e), "\n")
  cat("💡 Check internet connectivity and TCGA server status\n")
  quit(status = 1)
})

# Select and clean columns for merging
cat("\n🔧 Processing mutation data...\n")

# Define required columns (with fallbacks for different naming)
required_columns <- list(
  gene = c("Hugo_Symbol", "Gene_Symbol", "gene"),
  chromosome = c("Chromosome", "chr", "CHROM"),
  start_pos = c("Start_Position", "Start_position", "pos", "POS"),
  end_pos = c("End_Position", "End_position", "end"),
  ref_allele = c("Reference_Allele", "Ref", "REF"),
  alt_allele = c("Tumor_Seq_Allele2", "Alt", "ALT", "Tumor_Allele"),
  variant_class = c("Variant_Classification", "Consequence", "Effect"),
  variant_type = c("Variant_Type", "Type"),
  sample_id = c("Tumor_Sample_Barcode", "Sample_ID", "sample"),
  protein_change = c("HGVSp_Short", "HGVSp", "Protein_Change"),
  population_freq = c("ExAC_AF", "AF", "gnomAD_AF"),
  impact = c("IMPACT", "Impact", "Severity")
)

# Function to find the correct column name
find_column <- function(col_options, df_cols) {
  for (col in col_options) {
    if (col %in% df_cols) return(col)
  }
  return(NA)
}

# Map column names
available_cols <- colnames(prad_mutations)
column_mapping <- sapply(required_columns, find_column, available_cols)

# Remove NA mappings and create selection
valid_mapping <- column_mapping[!is.na(column_mapping)]
cat("✅ Found", length(valid_mapping), "out of", length(required_columns), "required columns\n")

# Convert to data.frame if it's a data.table (MAF returns data.table)
if (inherits(prad_mutations, "data.table")) {
  prad_mutations <- as.data.frame(prad_mutations)
}

# Select available columns
prad_clean <- prad_mutations[, valid_mapping, drop = FALSE]

# Rename to standard names
names(prad_clean) <- names(valid_mapping)

# Add standardized columns for merging
if ("chromosome" %in% names(prad_clean)) {
  prad_clean$chr <- gsub("chr", "", prad_clean$chromosome)  # Remove 'chr' prefix if present
}
if ("start_pos" %in% names(prad_clean)) {
  prad_clean$pos <- prad_clean$start_pos
}

# Basic data cleaning
cat("🧹 Cleaning data...\n")

# Remove rows with missing essential information
essential_cols <- intersect(c("gene", "chr", "pos"), names(prad_clean))
if (length(essential_cols) > 0) {
  initial_rows <- nrow(prad_clean)
  prad_clean <- prad_clean[complete.cases(prad_clean[, essential_cols]), ]
  cat("📊 Removed", initial_rows - nrow(prad_clean), "rows with missing essential data\n")
}

# Filter for standard chromosomes
if ("chr" %in% names(prad_clean)) {
  standard_chrs <- c(1:22, "X", "Y")
  prad_clean <- prad_clean[prad_clean$chr %in% standard_chrs, ]
  cat("📊 Filtered to standard chromosomes\n")
}

# Add pathway annotations for prostate cancer relevance
cat("🎯 Adding prostate cancer pathway annotations...\n")

# Define prostate cancer gene sets
prostate_gene_sets <- list(
  core_prostate = c("AR", "PTEN", "TP53", "MYC", "ERG", "ETV1", "ETV4", "ETV5", 
                   "TMPRSS2", "SPINK1", "CHD1", "SPOP", "FOXA1", "IDH1"),
  dna_repair = c("BRCA1", "BRCA2", "ATM", "CHEK2", "PALB2", "RAD51D", "BRIP1", 
                "FANCA", "MLH1", "MSH2", "MSH6", "PMS2", "NBN"),
  hormone_pathway = c("CYP17A1", "SRD5A2", "CYP19A1", "ESR1", "ESR2"),
  pi3k_pathway = c("PIK3CA", "PIK3R1", "AKT1", "AKT2", "AKT3", "MTOR", "TSC1", "TSC2")
)

# Add pathway annotations
if ("gene" %in% names(prad_clean)) {
  for (pathway in names(prostate_gene_sets)) {
    prad_clean[[paste0(pathway, "_gene")]] <- 
      as.integer(prad_clean$gene %in% prostate_gene_sets[[pathway]])
  }
  cat("✅ Added pathway annotations\n")
}

# Save to CSV
output_path <- file.path(output_dir, "tcga_prad_mutations.csv")
write.csv(prad_clean, output_path, row.names = FALSE)

# Generate summary
cat("\n📊 TCGA-PRAD Processing Summary\n")
cat("==============================\n")
cat("Total mutations:", nrow(prad_clean), "\n")
if ("gene" %in% names(prad_clean)) {
  cat("Unique genes:", length(unique(prad_clean$gene)), "\n")
  
  # Top mutated genes
  top_genes <- head(sort(table(prad_clean$gene), decreasing = TRUE), 10)
  cat("\nTop 10 mutated genes:\n")
  for (i in 1:length(top_genes)) {
    cat(sprintf("  %s: %d mutations\n", names(top_genes)[i], top_genes[i]))
  }
}

if ("chr" %in% names(prad_clean)) {
  cat("\nChromosome distribution:\n")
  chr_dist <- sort(table(prad_clean$chr))
  for (i in 1:length(chr_dist)) {
    cat(sprintf("  Chr %s: %d mutations\n", names(chr_dist)[i], chr_dist[i]))
  }
}

# Pathway summary
pathway_cols <- grep("_gene$", names(prad_clean), value = TRUE)
if (length(pathway_cols) > 0) {
  cat("\nProstate cancer pathway mutations:\n")
  for (col in pathway_cols) {
    pathway_name <- gsub("_gene$", "", col)
    count <- sum(prad_clean[[col]], na.rm = TRUE)
    cat(sprintf("  %s: %d mutations\n", pathway_name, count))
  }
}

cat("\n✅ TCGA-PRAD data saved to:", output_path, "\n")
cat("🎯 Ready for merging with COSMIC and ClinVar datasets!\n")