# ml/data_cleaning/clean_care_capability_data.py
"""
Care Capability Data Cleaning Module for PawCare+ ML pipeline.
Cleans owner care capability training dataset by handling missing values,
standardizing categories, and validating target ranges.
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Valid categorical values for each field
VALID_CATEGORIES = {
    "owner_experience": ["novice", "experienced", "expert", "unknown"],
    "vet_access": ["regular", "emergency only", "limited", "none", "unknown"],
    "owner_commitment": ["casual", "dedicated", "obsessive", "unknown"]
}

# Numerical column bounds
NUMERICAL_BOUNDS = {
    "care_capability_score": (0.0, 100.0)  # Target variable range
}

# Required columns in the dataset
REQUIRED_COLUMNS = [
    "owner_experience", "vet_access", "owner_commitment", "care_capability_score"
]

# Common variations to map to standard values
VARIATION_MAPPINGS = {
    "owner_experience": {
        "beginner": "novice",
        "new": "novice",
        "first time": "novice",
        "intermediate": "experienced",
        "advanced": "expert",
        "professional": "expert"
    },
    "vet_access": {
        "24/7": "regular",
        "24-7": "regular",
        "always": "regular",
        "sometimes": "emergency only",
        "rarely": "limited",
        "never": "none"
    },
    "owner_commitment": {
        "low": "casual",
        "medium": "dedicated",
        "high": "obsessive",
        "very high": "obsessive",
        "extreme": "obsessive"
    }
}


def clean_care_capability_dataset(input_file: str, output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Clean owner care capability training dataset.
    
    This function performs the following cleaning operations:
    1. Removes rows with NaN values (complete case analysis)
    2. Standardizes categorical values (lowercase, valid values, mode imputation)
    3. Validates and clips care_capability_score to [0, 100] range
    
    Args:
        input_file: Path to raw owner_care_capability_training.csv
        output_dir: Optional directory to save cleaned data
        
    Returns:
        Pandas DataFrame with cleaned data
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If required columns are missing
    """
    logger.info("=" * 60)
    logger.info("CLEANING OWNER CARE CAPABILITY DATASET")
    logger.info("=" * 60)
    logger.info(f"Input file: {input_file}")
    if output_dir:
        logger.info(f"Output directory: {output_dir}")
    
    # ==========================================
    # STEP 1: LOAD DATA
    # ==========================================
    logger.info("\n📂 Loading dataset...")
    try:
        df = pd.read_csv(input_file)
        original_rows, original_cols = df.shape
        logger.info(f"Loaded {original_rows:,} rows, {original_cols} columns")
        logger.info(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        logger.error(f"File not found: {input_file}")
        raise
    
    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Store initial statistics
    initial_stats = {
        "total_rows": original_rows,
        "missing_values": df[REQUIRED_COLUMNS].isnull().sum().to_dict(),
        "target_range": {
            "min": float(df["care_capability_score"].min()) if not df["care_capability_score"].isnull().all() else None,
            "max": float(df["care_capability_score"].max()) if not df["care_capability_score"].isnull().all() else None,
            "mean": float(df["care_capability_score"].mean()) if not df["care_capability_score"].isnull().all() else None
        }
    }
    
    rows_before = len(df)
    
    # ==========================================
    # STEP 2: REMOVE ROWS WITH NaN VALUES
    # ==========================================
    logger.info("\n🔍 Removing rows with missing values (complete case analysis)...")
    
    # Check for NaN values
    nan_counts = df[REQUIRED_COLUMNS].isnull().sum()
    if nan_counts.sum() > 0:
        logger.info(f"  NaN counts: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Remove rows with any NaN in required columns
        df = df.dropna(subset=REQUIRED_COLUMNS)
        
        rows_after_nan = len(df)
        removed_nan = rows_before - rows_after_nan
        logger.info(f"  Removed {removed_nan} rows with NaN values")
    else:
        removed_nan = 0
        logger.info("  No NaN values found")
    
    # ==========================================
    # STEP 3: STANDARDIZE CATEGORICAL VALUES
    # ==========================================
    logger.info("\n🏷️ Standardizing categorical values...")
    
    categorical_fixes = {}
    
    for col, valid_values in VALID_CATEGORIES.items():
        if col in df.columns:
            logger.info(f"  Processing {col}...")
            
            # Convert to lowercase and strip whitespace
            df[col] = df[col].astype(str).str.lower().str.strip()
            
            # Apply variation mappings
            if col in VARIATION_MAPPINGS:
                df[col] = df[col].replace(VARIATION_MAPPINGS[col])
            
            # Replace common invalid values with 'unknown'
            df[col] = df[col].replace({
                r'^\s*$': 'unknown',  # Empty strings
                'null': 'unknown',
                'none': 'unknown',
                'n/a': 'unknown',
                'na': 'unknown'
            }, regex=True)
            
            # Count invalid values before correction
            invalid_mask = ~df[col].isin(valid_values)
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                # Find mode of valid values for imputation
                valid_mask = df[col].isin(valid_values)
                if valid_mask.any():
                    mode_value = df.loc[valid_mask, col].mode()[0]
                else:
                    mode_value = 'unknown'
                
                # Replace invalid values with mode
                df.loc[invalid_mask, col] = mode_value
                
                logger.info(f"    Replaced {invalid_count} invalid values with '{mode_value}'")
                categorical_fixes[col] = invalid_count
    
    # ==========================================
    # STEP 4: VALIDATE AND CLIP CARE CAPABILITY SCORE
    # ==========================================
    logger.info("\n🎯 Validating care capability score range [0, 100]...")
    
    target_col = "care_capability_score"
    min_val, max_val = NUMERICAL_BOUNDS[target_col]
    
    # Check for values outside range
    below_min = (df[target_col] < min_val).sum()
    above_max = (df[target_col] > max_val).sum()
    
    logger.info(f"  Values below {min_val}: {below_min}")
    logger.info(f"  Values above {max_val}: {above_max}")
    
    # Clip values to valid range
    df[target_col] = df[target_col].clip(min_val, max_val)
    
    # Get final range statistics
    target_stats = {
        "clipped_below": int(below_min),
        "clipped_above": int(above_max),
        "final_min": float(df[target_col].min()),
        "final_max": float(df[target_col].max()),
        "final_mean": float(df[target_col].mean()),
        "final_median": float(df[target_col].median())
    }
    
    # ==========================================
    # STEP 5: RESET INDEX
    # ==========================================
    df = df.reset_index(drop=True)
    
    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    cleaned_rows = len(df)
    removed_rows = original_rows - cleaned_rows
    
    logger.info("\n" + "=" * 40)
    logger.info("CLEANING SUMMARY")
    logger.info("=" * 40)
    logger.info(f"Original rows: {original_rows:,}")
    logger.info(f"Cleaned rows: {cleaned_rows:,}")
    logger.info(f"Rows removed: {removed_rows:,} ({removed_rows/original_rows*100:.1f}%)")
    logger.info(f"Target score range: [{target_stats['final_min']:.1f}, {target_stats['final_max']:.1f}]")
    logger.info(f"Target score mean: {target_stats['final_mean']:.1f}")
    
    if categorical_fixes:
        logger.info(f"\nCategorical fixes:")
        for col, count in categorical_fixes.items():
            logger.info(f"  {col}: fixed {count} invalid values")
    
    if removed_nan > 0:
        logger.info(f"\nNaN removal: removed {removed_nan} rows")
    
    if below_min > 0 or above_max > 0:
        logger.info(f"\nTarget clipping: {below_min} below 0, {above_max} above 100")
    
    # ==========================================
    # SAVE CLEANED DATA IF OUTPUT DIR PROVIDED
    # ==========================================
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "owner_care_capability_clean.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"\n💾 Cleaned data saved to: {output_file}")
    
    logger.info("\n✅ Care capability dataset cleaning completed successfully")
    
    return df


def validate_care_capability_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that cleaned data meets quality standards.
    
    Args:
        df: Cleaned DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "valid": True,
        "checks": {},
        "warnings": []
    }
    
    # Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        validation_results["valid"] = False
        validation_results["checks"]["missing_columns"] = missing_cols
    
    # Check for any remaining NaN values
    nan_counts = df[REQUIRED_COLUMNS].isnull().sum()
    if nan_counts.sum() > 0:
        validation_results["valid"] = False
        validation_results["checks"]["nan_values"] = nan_counts[nan_counts > 0].to_dict()
    
    # Check categorical values are valid
    for col, valid_values in VALID_CATEGORIES.items():
        if col in df.columns:
            invalid = ~df[col].isin(valid_values)
            if invalid.any():
                validation_results["warnings"].append(f"{col}: {invalid.sum()} invalid values found")
    
    # Check target range
    target_col = "care_capability_score"
    min_val, max_val = NUMERICAL_BOUNDS[target_col]
    
    out_of_range = (df[target_col] < min_val) | (df[target_col] > max_val)
    if out_of_range.any():
        validation_results["warnings"].append(
            f"{target_col}: {out_of_range.sum()} values outside [{min_val}, {max_val}]"
        )
    
    # Check for unrealistic exact values (e.g., all same score)
    unique_scores = df[target_col].nunique()
    if unique_scores < 10 and len(df) > 100:
        validation_results["warnings"].append(
            f"Target has only {unique_scores} unique values - possible data quality issue"
        )
    
    return validation_results


def analyze_categorical_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze distribution of categorical features.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        Dictionary with distribution analysis
    """
    analysis = {}
    
    for col in VALID_CATEGORIES.keys():
        if col in df.columns:
            value_counts = df[col].value_counts()
            percentages = (value_counts / len(df) * 100).round(2)
            
            analysis[col] = {
                "counts": value_counts.to_dict(),
                "percentages": percentages.to_dict(),
                "mode": value_counts.index[0] if not value_counts.empty else None,
                "mode_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                "unknown_percentage": float(percentages.get('unknown', 0))
            }
    
    return analysis


# ==========================================
# COMMAND LINE INTERFACE
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean owner care capability dataset")
    parser.add_argument("input_file", help="Path to raw owner_care_capability_training.csv")
    parser.add_argument("--output", "-o", help="Output file path to save cleaned data (optional)")
    
    args = parser.parse_args()
    
    # Run cleaning
    cleaned_df = clean_care_capability_dataset(args.input_file)
    
    # Save if output specified
    if args.output:
        cleaned_df.to_csv(args.output, index=False)
        print(f"\n💾 Cleaned data saved to: {args.output}")
    
    # Validate
    validation = validate_care_capability_data(cleaned_df)
    if validation["valid"]:
        print("\n✅ Data validation passed")
    else:
        print("\n⚠️ Data validation had issues:")
        print(validation["checks"])
    
    # Show categorical distribution
    print("\n📊 Categorical Distribution:")
    analysis = analyze_categorical_distribution(cleaned_df)
    for col, dist in analysis.items():
        print(f"\n{col}:")
        for value, pct in list(dist.get("percentages", {}).items())[:3]:
            print(f"  • {value}: {pct}%")