import datetime
from pathlib import Path
from typing import Final

import pandas as pd
import pycaret.regression as pyc

AGE_RANGES: Final[tuple[tuple[int, int], ...]] = (
    (18, 20),
    (20, 25),
    (25, 30),
    (30, 35),
    (35, 40),
    (40, 45),
    (45, 50),
    (50, 55),
    (55, 60),
    (60, 65),
    (65, 70),
    (70, 75),
    (75, 100),
)


def split_otu_by_health(
    meta_path: Path, otu_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Read meta.tsv and otu.tsv
    meta_df = pd.read_csv(meta_path, sep="\t")
    meta_df = meta_df.set_index("id")

    otu_df = pd.read_csv(otu_path, sep="\t")
    otu_df = otu_df.set_index("id")

    # Split otu_df based on the 'health' column in meta_df
    healthy_otu_df = otu_df[meta_df["health"] == "y"]
    # nonhealthy_otu_df = otu_df[meta_df["health"] == "n"]

    predicted_age_df = pd.merge(
        healthy_otu_df, meta_df["age"], left_index=True, right_index=True, how="inner"
    )

    return predicted_age_df, meta_df, otu_df


def model_health_ages(
    predicted_age_df: pd.DataFrame,
    otu_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    # Use pycaret to model healthy otu_df and predict the physiological age of the samples
    pyc.setup(
        data=predicted_age_df,
        target="age",
        session_id=123,
        use_gpu=True,
    )

    # Compare regression models and return the best model.
    best_model = pyc.compare_models(
        # exclude=["lightgbm"],
        # sort="MAE",
        # errors="raise",
    )

    # pull() returns the comparison leaderboard as a DataFrame.
    compare_result = pyc.pull()
    compare_result.to_csv(
        output_dir / "compare_models.tsv",
        sep="\t",
        index=True,
    )

    # Tune the best model.
    tuned_model = pyc.tune_model(
        best_model,
        # optimize="MAE",
        # n_iter=50,
    )

    # pull() now returns the tuning results.
    tune_result = pyc.pull()
    tune_result.to_csv(
        output_dir / "tuned_best_model.tsv",
        sep="\t",
        index=True,
    )

    # Refit tuned model on the complete training dataset.
    final_best_model = pyc.finalize_model(tuned_model)

    # Predict physiological age.
    prediction_result = pyc.predict_model(
        final_best_model,
        data=otu_df,
    )
    age_predictions = prediction_result["prediction_label"]

    # Save final model.
    current_date = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d")
    pyc.save_model(
        final_best_model,
        str(output_dir / f"final_best_model_{current_date}"),
    )

    return age_predictions.to_frame()


def calculate_raw_gai(
    meta_df: pd.DataFrame, age_predictions: pd.DataFrame
) -> pd.DataFrame:
    """Add predicted age minus chronological age as the raw GAI."""
    meta_df["raw GAI"] = age_predictions["prediction_label"] - meta_df["age"]
    return meta_df


def calculate_adjust_value(meta_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Calculate and assign the healthy-cohort adjustment for each age range."""
    healthy_raw_gai = meta_df.loc[meta_df["health"] == "y", "raw GAI"]

    adjust_values: list[float] = []
    for start_age, end_age in AGE_RANGES:
        in_age_range = (meta_df["age"] >= start_age) & (meta_df["age"] < end_age)
        adjust_values.append(healthy_raw_gai[in_age_range].mean())

    pd.DataFrame({"age_range": AGE_RANGES, "adjust_value": adjust_values}).to_csv(
        output_dir / "adjust_values.tsv", sep="\t", index=False
    )

    for (start_age, end_age), adjust_value in zip(
        AGE_RANGES,
        adjust_values,
        strict=True,
    ):
        in_age_range = (meta_df["age"] >= start_age) & (meta_df["age"] < end_age)
        meta_df.loc[in_age_range, "adjust value"] = adjust_value

    return meta_df


def calculate_corrected_gai(meta_df):
    # Calculate corrected GAI by subtracting adjust value from raw GAI
    meta_df["corrected GAI"] = meta_df["raw GAI"] - meta_df["adjust value"]

    return meta_df


def save_result(meta_df: pd.DataFrame, result_path: Path) -> None:
    """Save the completed results table as a TSV file."""
    meta_df.to_csv(result_path, sep="\t", index=True)
    print(f"Saved result as {result_path}")


def main(meta_path: Path, otu_path: Path, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split otu.tsv into healthy and get predicted age dataframe
    predicted_age_df, meta_df, otu_df = split_otu_by_health(meta_path, otu_path)

    # Model healthy otu dataframe and predict ages
    age_predictions = model_health_ages(predicted_age_df, otu_df, output_dir)

    # Calculate raw GAI for all samples and add it to meta_df
    meta_df = calculate_raw_gai(meta_df, age_predictions)

    # Calculate adjust values based on age ranges and add them to meta_df
    meta_df = calculate_adjust_value(meta_df, output_dir)

    # Calculate corrected GAI and add it to meta_df
    meta_df = calculate_corrected_gai(meta_df)

    # Save final result as result.tsv
    save_result(meta_df, output_dir / "result.tsv")
