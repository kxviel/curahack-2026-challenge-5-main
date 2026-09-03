"""Prepare metadata and OTU tables for the Gut Aging Index calculator.

Inputs
------
* GGMP metadata: Supplementary Table S4
* AGP metadata: Supplementary Table S6
* GGMP and AGP OTU data: BIOM files

Outputs
-------
* meta.tsv: ``id``, ``age``, and ``health`` (``y`` or ``n``)
* otu.tsv: ``id`` followed by OTU abundance columns

OTUs present in fewer than 10% of the filtered samples are excluded.

Requirements
------------
``pandas``, ``numpy``, ``biom-format``, ``h5py``, ``scipy``, and ``openpyxl``
"""

from collections.abc import Collection
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
from biom import Table, load_table

# Paths

BASE_DIR: Final = Path(__file__).resolve().parent
DATA_DIR: Final = BASE_DIR / "Downloaded-Data"
SUPP_DIR: Final = BASE_DIR / "Supplementary-Material"
OUTPUT_DIR: Final = BASE_DIR / "Processed-Data"

SUPP_EXCEL: Final = SUPP_DIR / "41598_2024_82418_MOESM2_ESM.xlsx"
GGMP_BIOM_PATH: Final = DATA_DIR / "GCMP" / "GGMP-feces.biom"
AGP_BIOM_PATH: Final = DATA_DIR / "AGP" / "AGP-feces.biom"

OTU_PREVALENCE_THRESHOLD: Final = 0.10


# GGMP health columns

GGMP_HEART_STROKE_COLS: Final = (
    "heart_bypass_surgery",
    "heart_stent_surgery",
    "heart_angina_pectoris",
    "heart_aspirin",
    "heart_statins",
    "stroke_ischemic",
    "stroke_hemorrhagic",
)

GGMP_RESPIRATORY_COLS: Final = (
    "copd",
    "asthma",
)

GGMP_GENERAL_DISEASE_COLS: Final = (
    "osteoarticular_disease",
    "waist_neck_disease",
    "digestive_system_disease",
    "urinary_system_disease",
)

GGMP_SPECIFIC_DISEASE_COLS: Final = (
    "dis_T1DM",
    "dis_T2DM",
    "dis_fatty_liver",
    "dis_psoriasis",
    "dis_AD",
    "dis_PD",
    "dis_ASD",
    "dis_MS",
    "dis_atherosclerosis",
    "dis_LE",
    "dis_ARDS",
    "dis_gastritis",
    "dis_hepatic_calculus",
    "dis_cholecystitis",
    "dis_colitis",
    "dis_IBS",
    "dis_kidneyStone",
    "dis_gout",
    "dis_AS",
    "dis_RA",
    "dis_neurosis",
    "dis_CFS",
    "dis_constipation_symptom",
    "dis_diarrhea_symptom",
)

# ``malignant_tumor_disease`` is handled separately because ``a`` means absent.
GGMP_ALL_YN_DISEASE_COLS: Final = (
    GGMP_HEART_STROKE_COLS
    + GGMP_RESPIRATORY_COLS
    + GGMP_GENERAL_DISEASE_COLS
    + GGMP_SPECIFIC_DISEASE_COLS
    + ("MetS",)
)

GGMP_BIOM_PREFIX: Final = "11757."
GGMP_BIOM_SUFFIX: Final = ".56280"

BiomAxis = Literal["sample", "observation"]


# BIOM helpers


def biom_length(table: Table, axis: BiomAxis) -> int:
    """Return the number of entries on a BIOM axis."""
    return table.length(axis=axis)


def biom_ids(table: Table, axis: BiomAxis) -> set[str]:
    """Return the IDs on a BIOM axis as a set."""
    return set(table.ids(axis=axis))


def biom_id_list(table: Table, axis: BiomAxis) -> list[str]:
    """Return the IDs on a BIOM axis in their existing order."""
    return [str(value) for value in table.ids(axis=axis)]


def filter_biom_samples(table: Table, sample_ids: Collection[str]) -> Table:
    """Return a copy of a BIOM table containing only the requested samples."""
    return table.filter(sample_ids, axis="sample", inplace=False)


def biom_to_filtered_dataframe(
    biom_path: Path,
    sample_ids: Collection[str],
    prevalence_threshold: float,
) -> pd.DataFrame:
    """Load, sample-filter, prevalence-filter, and tabularize a BIOM file."""
    if not 0.0 <= prevalence_threshold <= 1.0:
        raise ValueError(
            f"prevalence_threshold must be between 0 and 1, got {prevalence_threshold}"
        )

    print(f"  Loading BIOM: {biom_path.name}")
    table = load_table(str(biom_path))

    sample_count = biom_length(table, "sample")
    otu_count = biom_length(table, "observation")
    print(f"  Raw BIOM: {otu_count:,} OTUs x {sample_count:,} samples")

    requested_ids = set(sample_ids)
    common_ids = sorted(requested_ids & biom_ids(table, "sample"))
    missing_count = len(requested_ids) - len(common_ids)

    if missing_count:
        print(f"  NOTE: {missing_count:,} requested sample IDs not found in BIOM")
    if not common_ids:
        raise ValueError(f"No requested sample IDs were found in BIOM file {biom_path}")

    table = filter_biom_samples(table, common_ids)
    sample_count = biom_length(table, "sample")
    otu_count = biom_length(table, "observation")
    print(f"  After sample filter: {otu_count:,} OTUs x {sample_count:,} samples")

    # BIOM matrices are observations x samples.
    otu_presence = table.matrix_data.getnnz(axis=1)
    prevalence = otu_presence.astype(np.float64) / sample_count
    keep_mask = prevalence >= prevalence_threshold
    keep_otu_ids = [
        otu_id
        for otu_id, keep in zip(
            biom_id_list(table, "observation"), keep_mask, strict=True
        )
        if bool(keep)
    ]

    if not keep_otu_ids:
        raise ValueError(
            "No OTUs remain after applying prevalence threshold "
            f"{prevalence_threshold:.3f}"
        )

    table = table.filter(keep_otu_ids, axis="observation", inplace=False)
    sample_count = biom_length(table, "sample")
    otu_count = biom_length(table, "observation")
    print(
        f"  After OTU prevalence filter (>= {prevalence_threshold * 100:.0f}%): "
        f"{otu_count:,} OTUs x {sample_count:,} samples"
    )

    otu_df = pd.DataFrame(
        table.matrix_data.toarray().T,
        index=biom_id_list(table, "sample"),
        columns=biom_id_list(table, "observation"),
    )
    otu_df.index.name = "id"
    return otu_df


# Reporting


def print_cohort_stats(
    meta_df: pd.DataFrame,
    label: str,
    paper_total: int,
    paper_healthy: int,
    paper_nonhealthy: int,
) -> None:
    """Print cohort statistics beside the paper-reported values."""
    healthy = meta_df["health"] == "y"
    nonhealthy = meta_df["health"] == "n"

    total_count = len(meta_df)
    healthy_count = healthy.sum()
    nonhealthy_count = nonhealthy.sum()
    healthy_age = meta_df.loc[healthy, "age"]
    nonhealthy_age = meta_df.loc[nonhealthy, "age"]

    healthy_pct = 100 * healthy_count / max(total_count, 1)
    nonhealthy_pct = 100 * nonhealthy_count / max(total_count, 1)

    print(f"\n  {label} Cohort Summary:")
    print(f"    Total samples:  {total_count:>6,}  (paper: {paper_total:,})")
    # Retained to keep the original console output unchanged.
    print(f"  Total samples: {total_count:,}  (paper: {paper_total:,})")
    print(
        f"  Healthy:      {healthy_count:>6,}  (paper: {paper_healthy:,}) "
        f"[{healthy_pct:.1f}%]"
    )
    print(
        f"  Non-healthy:  {nonhealthy_count:>6,}  (paper: {paper_nonhealthy:,}) "
        f"[{nonhealthy_pct:.1f}%]"
    )

    if not healthy_age.empty:
        paper_age = "45.97 +/- 16.38" if label == "GGMP" else "45.43 +/- 14.91"
        print(
            f"  Healthy mean age:     {healthy_age.mean():.2f} "
            f"+/- {healthy_age.std():.2f} (paper: {paper_age})"
        )

    if not nonhealthy_age.empty:
        paper_age = "54.05 +/- 14.01" if label == "GGMP" else "49.57 +/- 14.15"
        print(
            f"  Non-healthy mean age: {nonhealthy_age.mean():.2f} "
            f"+/- {nonhealthy_age.std():.2f} (paper: {paper_age})"
        )


# Cohort processing


def process_ggmp() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process the Guangdong Gut Microbiome Project dataset."""
    print("=" * 65)
    print("  GGMP Dataset Processing")
    print("=" * 65)

    print("\n[1/4] Loading metadata from Supplementary Table S4...")
    meta_df = pd.read_excel(SUPP_EXCEL, sheet_name="Sup Table 4", header=1)
    meta_df["SampleID"] = meta_df["SampleID"].astype(str)
    print(f"  Table S4: {meta_df.shape[0]:,} samples x {meta_df.shape[1]} columns")

    numeric_columns = ["age", "anthrop_BMI", "biochem_FBG"]
    meta_df[numeric_columns] = meta_df[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    print("\n[2/4] Filtering for complete phenotypic data...")
    disease_columns = [
        column for column in GGMP_ALL_YN_DISEASE_COLS if column in meta_df.columns
    ]
    print(f"  Disease columns (y/n): {len(disease_columns)}")

    complete = (
        meta_df["age"].notna()
        & (meta_df["age"] >= 18)
        & meta_df["anthrop_BMI"].notna()
        & meta_df["biochem_FBG"].notna()
        & meta_df["antibiotics"].isin(["y", "n"])
        & meta_df["malignant_tumor_disease"].notna()
        & meta_df[disease_columns].isin(["y", "n"]).all(axis=1)
    )
    meta_df = meta_df.loc[complete].copy()
    print(f"  Samples with complete data: {len(meta_df):,}")

    print("\n[3/4] Defining health status...")
    is_healthy = (
        (meta_df[disease_columns] == "n").all(axis=1)
        & (meta_df["malignant_tumor_disease"] == "a")
        & (meta_df["biochem_FBG"] < 6.1)
        & (meta_df["anthrop_BMI"] < 24)
        & (meta_df["antibiotics"] == "n")
    )
    meta_df["health"] = np.where(is_healthy, "y", "n")

    print_cohort_stats(
        meta_df,
        "GGMP",
        paper_total=6014,
        paper_healthy=1133,
        paper_nonhealthy=4881,
    )

    print("\n[4/4] Loading BIOM and filtering OTUs...")
    # Table S4: G440205594 -> BIOM: 11757.G440205594.56280
    biom_to_sample_id = {
        f"{GGMP_BIOM_PREFIX}{sample_id}{GGMP_BIOM_SUFFIX}": sample_id
        for sample_id in meta_df["SampleID"]
    }
    otu_df = biom_to_filtered_dataframe(
        GGMP_BIOM_PATH,
        list(biom_to_sample_id),
        OTU_PREVALENCE_THRESHOLD,
    )
    print("  (Paper reports: 942 OTUs after filtering)")

    common_biom_ids = set(otu_df.index)
    sample_to_biom_id = {
        sample_id: biom_id
        for biom_id, sample_id in biom_to_sample_id.items()
        if biom_id in common_biom_ids
    }

    metadata_by_id = meta_df.set_index("SampleID")
    sample_ids = list(sample_to_biom_id)
    meta_out = metadata_by_id.loc[sample_ids, ["age", "health"]].copy()
    meta_out["age"] = meta_out["age"].astype(int)
    meta_out.index = pd.Index(
        [sample_to_biom_id[sample_id] for sample_id in meta_out.index],
        name="id",
    )

    otu_out = otu_df.loc[meta_out.index].copy()
    print(
        f"\n  Final GGMP output: {len(meta_out):,} samples, {otu_out.shape[1]:,} OTUs"
    )
    return meta_out, otu_out


def process_agp() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process the American Gut Project dataset."""
    print("\n" + "=" * 65)
    print("  AGP Dataset Processing")
    print("=" * 65)

    print("\n[1/3] Loading metadata from Supplementary Table S6...")
    meta_df = pd.read_excel(SUPP_EXCEL, sheet_name="Sup Table 6", header=1)
    meta_df["SampleID"] = meta_df["SampleID"].astype(str)
    print(f"  Table S6: {meta_df.shape[0]:,} samples x {meta_df.shape[1]} columns")

    meta_df["age"] = pd.to_numeric(meta_df["age"], errors="coerce")
    print(f"  Health column values: {meta_df['health'].value_counts().to_dict()}")

    valid = meta_df["age"].notna() & meta_df["health"].isin(["y", "n"])
    meta_df = meta_df.loc[valid].copy()
    print_cohort_stats(
        meta_df,
        "AGP",
        paper_total=5966,
        paper_healthy=1852,
        paper_nonhealthy=4114,
    )

    print("\n[2/3] Loading BIOM and filtering OTUs...")
    otu_df = biom_to_filtered_dataframe(
        AGP_BIOM_PATH,
        meta_df["SampleID"].tolist(),
        OTU_PREVALENCE_THRESHOLD,
    )

    print("\n[3/3] Aligning metadata and OTU data...")
    metadata_by_id = meta_df.set_index("SampleID")
    common_ids = otu_df.index.intersection(metadata_by_id.index)
    meta_out = metadata_by_id.loc[common_ids, ["age", "health"]].copy()
    meta_out["age"] = meta_out["age"].astype(float)
    meta_out.index.name = "id"

    otu_out = otu_df.loc[common_ids].copy()
    otu_out.index.name = "id"

    print(f"\n  Final AGP output: {len(meta_out):,} samples, {otu_out.shape[1]:,} OTUs")
    return meta_out, otu_out


# Optional output helpers


def save_dataset(
    output_dir: Path,
    meta_df: pd.DataFrame,
    otu_df: pd.DataFrame,
) -> None:
    """Save one processed cohort in the calculator's expected format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "meta.tsv"
    otu_path = output_dir / "otu.tsv"

    meta_df.to_csv(meta_path, sep="\t")
    otu_df.to_csv(otu_path, sep="\t")

    print(f"\n  Saved: {meta_path}  ({meta_df.shape[0]} x {meta_df.shape[1]})")
    print(f"  Saved: {otu_path}   ({otu_df.shape[0]} x {otu_df.shape[1]})")


def print_completion(ggmp_dir: Path, agp_dir: Path) -> None:
    """Print the original pipeline completion instructions."""
    print("\n" + "=" * 65)
    print("  Data preparation complete!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print()
    print("  To run the GAI pipeline:")
    print(f"    python gai_cal.py {ggmp_dir}/meta.tsv {ggmp_dir}/otu.tsv")
    print(f"    python gai_cal.py {agp_dir}/meta.tsv {agp_dir}/otu.tsv")
    print("=" * 65)


def main() -> None:
    """Run the same active GGMP preparation step as the original script."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ggmp_dir = OUTPUT_DIR / "GGMP"
    ggmp_dir.mkdir(parents=True, exist_ok=True)

    ggmp_meta, ggmp_otu = process_ggmp()

    # These steps remain disabled, exactly as they were in the original script:
    save_dataset(ggmp_dir, ggmp_meta, ggmp_otu)
    agp_dir = OUTPUT_DIR / "AGP"
    agp_meta, agp_otu = process_agp()
    save_dataset(agp_dir, agp_meta, agp_otu)
    print_completion(ggmp_dir, agp_dir)


# The original entry point was disabled, so it remains disabled here.
if __name__ == "__main__":
    main()
