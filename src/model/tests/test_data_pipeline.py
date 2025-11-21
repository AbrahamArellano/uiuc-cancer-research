#!/usr/bin/env python3
"""
Test suite for Prostate-VarBench data pipeline.

Tests data integrity, file existence, and basic validation
for each stage of the pipeline.

Usage:
    pytest src/model/tests/test_data_pipeline.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestDataDirectories:
    """Test that required directories exist."""

    def test_project_structure(self):
        """Test basic project structure exists."""
        required_dirs = [
            "data",
            "scripts",
            "src",
            "config",
            "docs",
        ]
        for dir_name in required_dirs:
            dir_path = PROJECT_ROOT / dir_name
            assert dir_path.exists(), f"Required directory missing: {dir_name}"

    def test_scripts_structure(self):
        """Test scripts directory structure."""
        script_dirs = [
            "scripts/download",
            "scripts/variants",
            "scripts/merge",
            "scripts/vep",
            "scripts/enhance",
        ]
        for dir_name in script_dirs:
            dir_path = PROJECT_ROOT / dir_name
            assert dir_path.exists(), f"Script directory missing: {dir_name}"


class TestDownloadScripts:
    """Test that download scripts exist and are valid."""

    def test_cosmic_download_exists(self):
        """Test COSMIC download script exists."""
        script_path = PROJECT_ROOT / "scripts" / "download" / "download_cosmic_http.py"
        assert script_path.exists(), "COSMIC HTTP download script missing"

    def test_clinvar_download_exists(self):
        """Test ClinVar download script exists."""
        script_path = PROJECT_ROOT / "scripts" / "download" / "download_clinvar.py"
        assert script_path.exists(), "ClinVar download script missing"

    def test_tcga_download_exists(self):
        """Test TCGA download script exists."""
        script_path = PROJECT_ROOT / "scripts" / "variants" / "download_tcga.R"
        assert script_path.exists(), "TCGA R download script missing"

    def test_download_all_script_exists(self):
        """Test master download script exists."""
        script_path = PROJECT_ROOT / "scripts" / "download" / "download_all.sh"
        assert script_path.exists(), "Master download script missing"


class TestRawData:
    """Test raw data files (if downloaded)."""

    @pytest.fixture
    def raw_variants_dir(self):
        return PROJECT_ROOT / "data" / "raw" / "variants"

    def test_cosmic_data_exists(self, raw_variants_dir):
        """Test COSMIC data file exists (skip if not downloaded)."""
        cosmic_files = list(raw_variants_dir.glob("Cosmic_MutantCensus*.tsv*"))
        if not cosmic_files:
            pytest.skip("COSMIC data not downloaded yet")
        assert len(cosmic_files) > 0, "COSMIC TSV file not found"

    def test_clinvar_data_exists(self, raw_variants_dir):
        """Test ClinVar data file exists (skip if not downloaded)."""
        clinvar_files = list(raw_variants_dir.glob("clinvar*.vcf*"))
        if not clinvar_files:
            pytest.skip("ClinVar data not downloaded yet")
        assert len(clinvar_files) > 0, "ClinVar VCF file not found"

    def test_tcga_data_exists(self, raw_variants_dir):
        """Test TCGA-PRAD data file exists (skip if not downloaded)."""
        tcga_file = raw_variants_dir / "tcga_prad_mutations.csv"
        if not tcga_file.exists():
            pytest.skip("TCGA-PRAD data not downloaded yet")
        assert tcga_file.exists(), "TCGA-PRAD CSV file not found"


class TestProcessedData:
    """Test processed data files (if pipeline has run)."""

    @pytest.fixture
    def processed_dir(self):
        return PROJECT_ROOT / "data" / "processed"

    def test_cosmic_filtered_exists(self, processed_dir):
        """Test filtered COSMIC data exists."""
        cosmic_dir = processed_dir / "cosmic_prostate"
        if not cosmic_dir.exists():
            pytest.skip("COSMIC filtering not run yet")
        cosmic_file = cosmic_dir / "cosmic_prostate.csv"
        assert cosmic_file.exists(), "Filtered COSMIC file not found"

    def test_clinvar_filtered_exists(self, processed_dir):
        """Test filtered ClinVar data exists."""
        clinvar_dir = processed_dir / "clinvar_prostate"
        if not clinvar_dir.exists():
            pytest.skip("ClinVar filtering not run yet")
        clinvar_file = clinvar_dir / "clinvar_prostate.csv"
        assert clinvar_file.exists(), "Filtered ClinVar file not found"

    def test_merged_data_exists(self, processed_dir):
        """Test merged dataset exists."""
        merged_dir = processed_dir / "merged"
        if not merged_dir.exists():
            pytest.skip("Dataset merging not run yet")
        merged_file = merged_dir / "merged_prostate_variants.csv"
        assert merged_file.exists(), "Merged CSV file not found"

    def test_tabnet_csv_exists(self, processed_dir):
        """Test TabNet training CSV exists."""
        tabnet_dir = processed_dir / "tabnet_csv"
        if not tabnet_dir.exists():
            pytest.skip("VCF to CSV conversion not run yet")
        tabnet_files = list(tabnet_dir.glob("prostate_variants_tabnet*.csv"))
        assert len(tabnet_files) > 0, "TabNet CSV file not found"


class TestDataIntegrity:
    """Test data integrity and format."""

    @pytest.fixture
    def raw_variants_dir(self):
        return PROJECT_ROOT / "data" / "raw" / "variants"

    def test_cosmic_file_not_empty(self, raw_variants_dir):
        """Test COSMIC file has content."""
        cosmic_files = list(raw_variants_dir.glob("Cosmic_MutantCensus*.tsv"))
        if not cosmic_files:
            pytest.skip("COSMIC data not downloaded")
        cosmic_file = cosmic_files[0]
        assert cosmic_file.stat().st_size > 1000, "COSMIC file appears empty"

    def test_clinvar_vcf_valid(self, raw_variants_dir):
        """Test ClinVar VCF has valid header."""
        clinvar_file = raw_variants_dir / "clinvar_GRCh38.vcf"
        if not clinvar_file.exists():
            pytest.skip("ClinVar data not downloaded")

        with open(clinvar_file, 'r') as f:
            first_line = f.readline()
        assert first_line.startswith("##fileformat=VCF"), "Invalid VCF header"

    def test_tcga_csv_valid(self, raw_variants_dir):
        """Test TCGA CSV has expected columns."""
        tcga_file = raw_variants_dir / "tcga_prad_mutations.csv"
        if not tcga_file.exists():
            pytest.skip("TCGA data not downloaded")

        import pandas as pd
        df = pd.read_csv(tcga_file, nrows=5)
        expected_cols = ["gene", "chromosome", "start_pos"]
        for col in expected_cols:
            assert col in df.columns, f"Missing expected column: {col}"


class TestConfiguration:
    """Test configuration files."""

    def test_env_example_exists(self):
        """Test .env.example template exists."""
        env_example = PROJECT_ROOT / ".env.example"
        assert env_example.exists(), ".env.example file missing"

    def test_pipeline_config_template_exists(self):
        """Test pipeline config template exists."""
        config_template = PROJECT_ROOT / "config" / "pipeline_config_template.py"
        assert config_template.exists(), "Pipeline config template missing"

    def test_environment_yml_exists(self):
        """Test conda environment file exists."""
        env_yml = PROJECT_ROOT / "environment.yml"
        assert env_yml.exists(), "environment.yml file missing"


class TestDocumentation:
    """Test documentation files."""

    def test_readme_exists(self):
        """Test README.md exists."""
        readme = PROJECT_ROOT / "README.md"
        assert readme.exists(), "README.md missing"

    def test_setup_docs_exist(self):
        """Test setup documentation exists."""
        setup_doc = PROJECT_ROOT / "docs" / "SETUP.md"
        assert setup_doc.exists(), "docs/SETUP.md missing"

    def test_troubleshooting_exists(self):
        """Test troubleshooting guide exists."""
        troubleshooting = PROJECT_ROOT / "docs" / "TROUBLESHOOTING.md"
        assert troubleshooting.exists(), "docs/TROUBLESHOOTING.md missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
