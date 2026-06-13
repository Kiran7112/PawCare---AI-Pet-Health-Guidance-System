# # ml/data_cleaning/clean_health_risk_data.py
# """
# Health Risk Data Cleaning Module for PawCare+ ML pipeline.
# Cleans pet health risk training dataset by standardizing categories,
# handling missing values, and removing outliers.
# """

# import pandas as pd
# import numpy as np
# import logging
# import os
# from typing import Dict, Any, Optional, Tuple

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Valid categorical values for each field
# VALID_CATEGORIES = {
#     "pet_species": ["dog", "cat", "rabbit", "bird", "reptile", "other", "unknown"],
#     "weight_status": ["underweight", "normal", "overweight", "obese", "unknown"],
#     "living_situation": ["apartment", "house", "farm", "outdoor", "mixed", "unknown"],
#     "exercise_level": ["sedentary", "light", "moderate", "very active", "unknown"]
# }

# # Numerical column bounds
# NUMERICAL_BOUNDS = {
#     "age_years": (0, 30),  # Reasonable age range for pets
#     "conditions_count": (0, 20),  # Maximum reasonable number of conditions
#     "allergies_count": (0, 10),  # Maximum reasonable number of allergies
#     "health_risk_score": (0.0, 1.0)  # Target variable bounds
# }

# # Required columns in the dataset
# REQUIRED_COLUMNS = [
#     "pet_species", "weight_status", "living_situation", "exercise_level",
#     "age_years", "conditions_count", "allergies_count", "health_risk_score"
# ]


# def clean_health_risk_dataset(input_file: str, output_dir: Optional[str] = None) -> pd.DataFrame:
#     """
#     Clean pet health risk training dataset.
    
#     This function performs the following cleaning operations:
#     1. Standardizes categorical values (lowercase, valid values, mode imputation)
#     2. Removes rows with NaN in numerical columns
#     3. Removes extreme outliers using 3×IQR method
#     4. Clips numerical values to valid bounds
    
#     Args:
#         input_file: Path to raw pet_health_risk_training.csv
#         output_dir: Optional directory to save cleaned data
        
#     Returns:
#         Pandas DataFrame with cleaned data
        
#     Raises:
#         FileNotFoundError: If input file doesn't exist
#         ValueError: If required columns are missing
#     """
#     logger.info("=" * 60)
#     logger.info("CLEANING PET HEALTH RISK DATASET")
#     logger.info("=" * 60)
#     logger.info(f"Input file: {input_file}")
#     if output_dir:
#         logger.info(f"Output directory: {output_dir}")
    
#     # ==========================================
#     # STEP 1: LOAD DATA
#     # ==========================================
#     logger.info("\n📂 Loading dataset...")
#     try:
#         df = pd.read_csv(input_file)
#         original_rows, original_cols = df.shape
#         logger.info(f"Loaded {original_rows:,} rows, {original_cols} columns")
#         logger.info(f"Columns: {list(df.columns)}")
#     except FileNotFoundError:
#         logger.error(f"File not found: {input_file}")
#         raise
    
#     # Validate required columns
#     missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
#     if missing_cols:
#         error_msg = f"Missing required columns: {missing_cols}"
#         logger.error(error_msg)
#         raise ValueError(error_msg)
    
#     # Store original row count for reporting
#     rows_before = len(df)
    
#     # ==========================================
#     # STEP 2: STANDARDIZE CATEGORICAL VALUES
#     # ==========================================
#     logger.info("\n🏷️ Standardizing categorical values...")
    
#     categorical_fixes = {}
    
#     for col, valid_values in VALID_CATEGORIES.items():
#         if col in df.columns:
#             logger.info(f"  Processing {col}...")
            
#             # Convert to lowercase and strip whitespace
#             df[col] = df[col].astype(str).str.lower().str.strip()
            
#             # Replace common invalid values with 'unknown'
#             df[col] = df[col].replace({
#                 r'^\s*$': 'unknown',  # Empty strings
#                 'null': 'unknown',
#                 'none': 'unknown',
#                 'n/a': 'unknown',
#                 'na': 'unknown'
#             }, regex=True)
            
#             # Count invalid values before correction
#             invalid_mask = ~df[col].isin(valid_values)
#             invalid_count = invalid_mask.sum()
            
#             if invalid_count > 0:
#                 # Find mode of valid values for imputation
#                 valid_mask = df[col].isin(valid_values)
#                 if valid_mask.any():
#                     mode_value = df.loc[valid_mask, col].mode()[0]
#                 else:
#                     mode_value = 'unknown'
                
#                 # Replace invalid values with mode
#                 df.loc[invalid_mask, col] = mode_value
                
#                 logger.info(f"    Replaced {invalid_count} invalid values with '{mode_value}'")
#                 categorical_fixes[col] = invalid_count
    
#     # ==========================================
#     # STEP 3: REMOVE ROWS WITH NaN IN NUMERICAL COLUMNS
#     # ==========================================
#     logger.info("\n🔍 Removing rows with missing numerical values...")
    
#     numerical_cols = ["age_years", "conditions_count", "allergies_count", "health_risk_score"]
    
#     # Check for NaN values
#     nan_counts = df[numerical_cols].isnull().sum()
#     if nan_counts.sum() > 0:
#         logger.info(f"  NaN counts: {nan_counts[nan_counts > 0].to_dict()}")
        
#         # Remove rows with any NaN in numerical columns
#         df = df.dropna(subset=numerical_cols)
        
#         rows_after_nan = len(df)
#         removed_nan = rows_before - rows_after_nan
#         logger.info(f"  Removed {removed_nan} rows with NaN values")
#     else:
#         removed_nan = 0
#         logger.info("  No NaN values found")
    
#     rows_before_outlier = len(df)
    
#     # ==========================================
#     # STEP 4: REMOVE EXTREME OUTLIERS USING 3×IQR METHOD
#     # ==========================================
#     logger.info("\n📊 Removing extreme outliers using 3×IQR method...")
    
#     outlier_cols = ["age_years", "conditions_count"]
#     outlier_stats = {}
#     total_removed_outlier = 0
    
#     for col in outlier_cols:
#         if col in df.columns:
#             # Calculate quartiles and IQR
#             Q1 = df[col].quantile(0.25)
#             Q3 = df[col].quantile(0.75)
#             IQR = Q3 - Q1
            
#             # Define bounds (3×IQR for extreme outliers)
#             lower_bound = Q1 - 3 * IQR
#             upper_bound = Q3 + 3 * IQR
            
#             # Count outliers
#             outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
#             outlier_count = len(outliers)
            
#             if outlier_count > 0:
#                 logger.info(f"  {col}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
#                 logger.info(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
#                 logger.info(f"    Found {outlier_count} outliers ({outlier_count/len(df)*100:.1f}%)")
                
#                 # Remove outliers
#                 df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
#                 total_removed_outlier += outlier_count
            
#             outlier_stats[col] = {
#                 "Q1": Q1,
#                 "Q3": Q3,
#                 "IQR": IQR,
#                 "lower_bound": lower_bound,
#                 "upper_bound": upper_bound,
#                 "outliers_removed": outlier_count
#             }
    
#     logger.info(f"  Total removed outliers: {total_removed_outlier}")
    
#     # ==========================================
#     # STEP 5: CLIP VALUES TO VALID BOUNDS
#     # ==========================================
#     logger.info("\n🎯 Clipping values to valid bounds...")
    
#     clip_stats = {}
#     for col, (min_val, max_val) in NUMERICAL_BOUNDS.items():
#         if col in df.columns:
#             before_clip_min = (df[col] < min_val).sum()
#             before_clip_max = (df[col] > max_val).sum()
            
#             # Clip values
#             df[col] = df[col].clip(min_val, max_val)
            
#             if before_clip_min > 0 or before_clip_max > 0:
#                 logger.info(f"  {col}: Clipped {before_clip_min} below {min_val}, {before_clip_max} above {max_val}")
            
#             clip_stats[col] = {
#                 "clipped_below": int(before_clip_min),
#                 "clipped_above": int(before_clip_max)
#             }
    
#     # ==========================================
#     # STEP 6: RESET INDEX
#     # ==========================================
#     df = df.reset_index(drop=True)
    
#     # ==========================================
#     # FINAL SUMMARY
#     # ==========================================
#     cleaned_rows = len(df)
#     removed_rows = original_rows - cleaned_rows
    
#     logger.info("\n" + "=" * 40)
#     logger.info("CLEANING SUMMARY")
#     logger.info("=" * 40)
#     logger.info(f"Original rows: {original_rows:,}")
#     logger.info(f"Cleaned rows: {cleaned_rows:,}")
#     logger.info(f"Rows removed: {removed_rows:,} ({removed_rows/original_rows*100:.1f}%)")
    
#     if categorical_fixes:
#         logger.info(f"\nCategorical fixes:")
#         for col, count in categorical_fixes.items():
#             logger.info(f"  {col}: fixed {count} invalid values")
    
#     if removed_nan > 0:
#         logger.info(f"\nNaN removal: removed {removed_nan} rows")
    
#     if total_removed_outlier > 0:
#         logger.info(f"\nOutlier removal: removed {total_removed_outlier} rows")
    
#     # ==========================================
#     # SAVE CLEANED DATA IF OUTPUT DIR PROVIDED
#     # ==========================================
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#         output_file = os.path.join(output_dir, "pet_health_risk_clean.csv")
#         df.to_csv(output_file, index=False)
#         logger.info(f"\n💾 Cleaned data saved to: {output_file}")
    
#     logger.info("\n✅ Health risk dataset cleaning completed successfully")
    
#     return df


# def validate_health_risk_data(df: pd.DataFrame) -> Dict[str, Any]:
#     """
#     Validate that cleaned data meets quality standards.
    
#     Args:
#         df: Cleaned DataFrame to validate
        
#     Returns:
#         Dictionary with validation results
#     """
#     validation_results = {
#         "valid": True,
#         "checks": {},
#         "warnings": []
#     }
    
#     # Check required columns
#     missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
#     if missing_cols:
#         validation_results["valid"] = False
#         validation_results["checks"]["missing_columns"] = missing_cols
    
#     # Check for any remaining NaN values
#     nan_counts = df[REQUIRED_COLUMNS].isnull().sum()
#     if nan_counts.sum() > 0:
#         validation_results["valid"] = False
#         validation_results["checks"]["nan_values"] = nan_counts[nan_counts > 0].to_dict()
    
#     # Check categorical values are valid
#     for col, valid_values in VALID_CATEGORIES.items():
#         if col in df.columns:
#             invalid = ~df[col].isin(valid_values)
#             if invalid.any():
#                 validation_results["warnings"].append(f"{col}: {invalid.sum()} invalid values found")
    
#     # Check numerical ranges
#     for col, (min_val, max_val) in NUMERICAL_BOUNDS.items():
#         if col in df.columns:
#             out_of_range = (df[col] < min_val) | (df[col] > max_val)
#             if out_of_range.any():
#                 validation_results["warnings"].append(
#                     f"{col}: {out_of_range.sum()} values outside [{min_val}, {max_val}]"
#                 )
    
#     return validation_results


# # ==========================================
# # COMMAND LINE INTERFACE
# # ==========================================

# if __name__ == "__main__":
#     import sys
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Clean pet health risk dataset")
#     parser.add_argument("input_file", help="Path to raw pet_health_risk_training.csv")
#     parser.add_argument("--output", "-o", help="Output file path to save cleaned data (optional)")
    
#     args = parser.parse_args()
    
#     # Run cleaning
#     cleaned_df = clean_health_risk_dataset(args.input_file)
    
#     # Save if output specified
#     if args.output:
#         cleaned_df.to_csv(args.output, index=False)
#         print(f"\n💾 Cleaned data saved to: {args.output}")
    
#     # Validate
#     validation = validate_health_risk_data(cleaned_df)
#     if validation["valid"]:
#         print("\n✅ Data validation passed")
#     else:
#         print("\n⚠️ Data validation had issues:")
#         print(validation["checks"])


# ml/data_cleaning/clean_health_risk_data.py
# ml/data_cleaning/clean_health_risk_data.py
"""
Health Risk Data Cleaning Module for PawCare+ ML pipeline.
Cleans pet health risk training dataset by standardizing categories,
handling missing values, and removing outliers.
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
    "pet_species": ["dog", "cat", "rabbit", "bird", "reptile", "other", "unknown"],
    "weight_status": ["underweight", "normal", "overweight", "obese", "unknown"],
    "living_situation": ["apartment", "house", "farm", "outdoor", "mixed", "unknown"],
    "exercise_level": ["sedentary", "light", "moderate", "very active", "unknown"]
}

# Numerical column bounds
NUMERICAL_BOUNDS = {
    "age_years": (0, 30),
    "conditions_count": (0, 20),
    "allergies_count": (0, 10),
    "health_risk_score": (0.0, 1.0)
}

# Required columns in the dataset (NO SYMPTOM COLUMNS - training data doesn't have them)
REQUIRED_COLUMNS = [
    "pet_species", "weight_status", "living_situation", "exercise_level",
    "age_years", "conditions_count", "allergies_count", "health_risk_score"
]


def clean_health_risk_dataset(input_file: str, output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Clean pet health risk training dataset.
    
    This function performs:
    1. Standardizes categorical values (lowercase, valid values, mode imputation)
    2. Removes rows with NaN in numerical columns
    3. Removes extreme outliers using 3×IQR method
    4. Clips numerical values to valid bounds
    """
    logger.info("=" * 60)
    logger.info("CLEANING PET HEALTH RISK DATASET")
    logger.info("=" * 60)
    logger.info(f"Input file: {input_file}")
    if output_dir:
        logger.info(f"Output directory: {output_dir}")
    
    # STEP 1: LOAD DATA
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
    
    rows_before = len(df)
    
    # STEP 2: STANDARDIZE CATEGORICAL VALUES
    logger.info("\n🏷️ Standardizing categorical values...")
    
    categorical_fixes = {}
    
    for col, valid_values in VALID_CATEGORIES.items():
        if col in df.columns:
            logger.info(f"  Processing {col}...")
            
            df[col] = df[col].astype(str).str.lower().str.strip()
            
            df[col] = df[col].replace({
                r'^\s*$': 'unknown',
                'null': 'unknown',
                'none': 'unknown',
                'n/a': 'unknown',
                'na': 'unknown'
            }, regex=True)
            
            invalid_mask = ~df[col].isin(valid_values)
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                valid_mask = df[col].isin(valid_values)
                if valid_mask.any():
                    mode_value = df.loc[valid_mask, col].mode()[0]
                else:
                    mode_value = 'unknown'
                
                df.loc[invalid_mask, col] = mode_value
                logger.info(f"    Replaced {invalid_count} invalid values with '{mode_value}'")
                categorical_fixes[col] = invalid_count
    
    # STEP 3: REMOVE ROWS WITH NaN IN NUMERICAL COLUMNS
    logger.info("\n🔍 Removing rows with missing numerical values...")
    
    numerical_cols = ["age_years", "conditions_count", "allergies_count", "health_risk_score"]
    
    nan_counts = df[numerical_cols].isnull().sum()
    if nan_counts.sum() > 0:
        logger.info(f"  NaN counts: {nan_counts[nan_counts > 0].to_dict()}")
        df = df.dropna(subset=numerical_cols)
        removed_nan = rows_before - len(df)
        logger.info(f"  Removed {removed_nan} rows with NaN values")
    else:
        removed_nan = 0
        logger.info("  No NaN values found")
    
    rows_before_outlier = len(df)
    
    # STEP 4: REMOVE EXTREME OUTLIERS USING 3×IQR METHOD
    logger.info("\n📊 Removing extreme outliers using 3×IQR method...")
    
    outlier_cols = ["age_years", "conditions_count"]
    total_removed_outlier = 0
    
    for col in outlier_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                logger.info(f"  {col}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
                logger.info(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                logger.info(f"    Found {outlier_count} outliers ({outlier_count/len(df)*100:.1f}%)")
                
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                total_removed_outlier += outlier_count
    
    logger.info(f"  Total removed outliers: {total_removed_outlier}")
    
    # STEP 5: CLIP VALUES TO VALID BOUNDS
    logger.info("\n🎯 Clipping values to valid bounds...")
    
    for col, (min_val, max_val) in NUMERICAL_BOUNDS.items():
        if col in df.columns:
            before_clip_min = (df[col] < min_val).sum()
            before_clip_max = (df[col] > max_val).sum()
            
            df[col] = df[col].clip(min_val, max_val)
            
            if before_clip_min > 0 or before_clip_max > 0:
                logger.info(f"  {col}: Clipped {before_clip_min} below {min_val}, {before_clip_max} above {max_val}")
    
    # STEP 6: RESET INDEX
    df = df.reset_index(drop=True)
    
    # FINAL SUMMARY
    cleaned_rows = len(df)
    removed_rows = original_rows - cleaned_rows
    
    logger.info("\n" + "=" * 40)
    logger.info("CLEANING SUMMARY")
    logger.info("=" * 40)
    logger.info(f"Original rows: {original_rows:,}")
    logger.info(f"Cleaned rows: {cleaned_rows:,}")
    logger.info(f"Rows removed: {removed_rows:,} ({removed_rows/original_rows*100:.1f}%)")
    
    if categorical_fixes:
        logger.info(f"\nCategorical fixes:")
        for col, count in categorical_fixes.items():
            logger.info(f"  {col}: fixed {count} invalid values")
    
    if removed_nan > 0:
        logger.info(f"\nNaN removal: removed {removed_nan} rows")
    
    if total_removed_outlier > 0:
        logger.info(f"\nOutlier removal: removed {total_removed_outlier} rows")
    
    # SAVE CLEANED DATA IF OUTPUT DIR PROVIDED
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "pet_health_risk_clean.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"\n💾 Cleaned data saved to: {output_file}")
    
    logger.info("\n✅ Health risk dataset cleaning completed successfully")
    
    return df


def validate_health_risk_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate that cleaned data meets quality standards."""
    validation_results = {
        "valid": True,
        "checks": {},
        "warnings": []
    }
    
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        validation_results["valid"] = False
        validation_results["checks"]["missing_columns"] = missing_cols
    
    nan_counts = df[REQUIRED_COLUMNS].isnull().sum()
    if nan_counts.sum() > 0:
        validation_results["valid"] = False
        validation_results["checks"]["nan_values"] = nan_counts[nan_counts > 0].to_dict()
    
    for col, valid_values in VALID_CATEGORIES.items():
        if col in df.columns:
            invalid = ~df[col].isin(valid_values)
            if invalid.any():
                validation_results["warnings"].append(f"{col}: {invalid.sum()} invalid values found")
    
    for col, (min_val, max_val) in NUMERICAL_BOUNDS.items():
        if col in df.columns:
            out_of_range = (df[col] < min_val) | (df[col] > max_val)
            if out_of_range.any():
                validation_results["warnings"].append(
                    f"{col}: {out_of_range.sum()} values outside [{min_val}, {max_val}]"
                )
    
    return validation_results


# ==========================================
# COMMAND LINE INTERFACE
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean pet health risk dataset")
    parser.add_argument("input_file", help="Path to raw pet_health_risk_training.csv")
    parser.add_argument("--output", "-o", help="Output file path to save cleaned data (optional)")
    
    args = parser.parse_args()
    
    cleaned_df = clean_health_risk_dataset(args.input_file)
    
    if args.output:
        cleaned_df.to_csv(args.output, index=False)
        print(f"\n💾 Cleaned data saved to: {args.output}")
    
    validation = validate_health_risk_data(cleaned_df)
    if validation["valid"]:
        print("\n✅ Data validation passed")
    else:
        print("\n⚠️ Data validation had issues:")
        print(validation["checks"])