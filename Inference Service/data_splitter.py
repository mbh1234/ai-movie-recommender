"""
Data splitting utilities for the movie recommendation system.

Provides reusable temporal splitting logic so we can avoid data leakage
while producing auxiliary cold-start sets for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import logging
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TemporalSplitResult:
    """Container for temporal split outputs."""

    train: pd.DataFrame
    warm_test: pd.DataFrame
    cutoff_timestamp: pd.Timestamp
    cold_user_test: pd.DataFrame
    cold_item_test: pd.DataFrame
    metadata: Dict[str, int]


@dataclass
class TemporalSplitResult3Way:
    """Container for three-way temporal split outputs (train/val/test)."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_cutoff_timestamp: pd.Timestamp
    val_cutoff_timestamp: pd.Timestamp
    cold_user_val: pd.DataFrame
    cold_item_val: pd.DataFrame
    cold_user_test: pd.DataFrame
    cold_item_test: pd.DataFrame
    metadata: Dict[str, int]


def temporal_train_test_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    train_fraction: float = 0.8,
    ensure_warm_start: bool = True,
) -> TemporalSplitResult:
    """
    Split interactions using a temporal cutoff to prevent label leakage.

    Args:
        df: Interactions dataframe containing timestamp information.
        timestamp_col: Column name with interaction timestamps.
        train_fraction: Fraction of data to keep in the training set (0 < f < 1).
        ensure_warm_start: If True, drop test rows with unseen users/items from the warm split.

    Returns:
        TemporalSplitResult containing warm and cold-start splits along with metadata.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in dataframe for temporal split")

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1 (exclusive)")

    working_df = df.copy()
    working_df[timestamp_col] = pd.to_datetime(working_df[timestamp_col], errors="coerce")
    working_df = working_df.dropna(subset=[timestamp_col])

    if len(working_df) < 2:
        raise ValueError("Not enough valid timestamped interactions to perform temporal split")

    working_df = working_df.sort_values(timestamp_col).reset_index(drop=True)

    cutoff_idx = max(int(len(working_df) * train_fraction), 1)
    # Ensure both splits contain at least one interaction
    if cutoff_idx >= len(working_df):
        cutoff_idx = len(working_df) - 1

    train = working_df.iloc[:cutoff_idx].copy()
    test = working_df.iloc[cutoff_idx:].copy()

    cutoff_timestamp = train.iloc[-1][timestamp_col]

    cold_user_test = pd.DataFrame(columns=test.columns)
    cold_item_test = pd.DataFrame(columns=test.columns)
    warm_test = test.copy()
    metadata: Dict[str, int] = {
        "initial_test_rows": len(test),
        "dropped_for_cold_start": 0,
        "cold_user_rows": 0,
        "cold_item_rows": 0,
    }

    if ensure_warm_start:
        known_users = set(train["user_id"].unique())
        known_items = set(train["movie_id"].unique())

        cold_user_mask = ~warm_test["user_id"].isin(known_users)
        cold_item_mask = warm_test["user_id"].isin(known_users) & ~warm_test["movie_id"].isin(known_items)

        cold_user_test = warm_test[cold_user_mask].copy()
        cold_item_test = warm_test[cold_item_mask].copy()

        warm_mask = ~cold_user_mask & ~cold_item_mask
        warm_test = warm_test[warm_mask].copy()

        metadata["cold_user_rows"] = len(cold_user_test)
        metadata["cold_item_rows"] = len(cold_item_test)
        metadata["dropped_for_cold_start"] = len(test) - len(warm_test)

    logger.info(
        "Temporal split created | train=%s | warm_test=%s | cold_user=%s | cold_item=%s | cutoff=%s",
        len(train),
        len(warm_test),
        len(cold_user_test),
        len(cold_item_test),
        cutoff_timestamp,
    )

    return TemporalSplitResult(
        train=train,
        warm_test=warm_test,
        cutoff_timestamp=cutoff_timestamp,
        cold_user_test=cold_user_test,
        cold_item_test=cold_item_test,
        metadata=metadata,
    )


def compute_temporal_cutoff(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    train_fraction: float = 0.8,
) -> Optional[pd.Timestamp]:
    """
    Convenience helper to compute the timestamp cutoff used for temporal splits.

    Returns None if the column is missing or no valid timestamps are available.
    """
    if timestamp_col not in df.columns:
        return None

    working_df = df.copy()
    working_df[timestamp_col] = pd.to_datetime(working_df[timestamp_col], errors="coerce")
    working_df = working_df.dropna(subset=[timestamp_col])

    if working_df.empty:
        return None

    working_df = working_df.sort_values(timestamp_col).reset_index(drop=True)
    cutoff_idx = max(int(len(working_df) * train_fraction), 1)
    if cutoff_idx >= len(working_df):
        cutoff_idx = len(working_df) - 1

    return working_df.iloc[cutoff_idx - 1][timestamp_col]


def temporal_train_val_test_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    ensure_warm_start: bool = True,
) -> TemporalSplitResult3Way:
    """
    Split interactions into train/validation/test using temporal cutoffs (60/20/20).
    
    This prevents data leakage and allows proper hyperparameter tuning:
    - Train set: used to train models with different hyperparameters
    - Validation set: used to select best hyperparameters
    - Test set: final evaluation only, never touched during development
    
    Args:
        df: Interactions dataframe containing timestamp information.
        timestamp_col: Column name with interaction timestamps.
        train_fraction: Fraction of data for training (default: 0.6).
        val_fraction: Fraction of data for validation (default: 0.2).
        ensure_warm_start: If True, drop validation/test rows with unseen users/items.
        
    Returns:
        TemporalSplitResult3Way containing train/val/test splits and cold-start data.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in dataframe for temporal split")
    
    test_fraction = 1.0 - train_fraction - val_fraction
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and test_fraction > 0):
        raise ValueError(f"Invalid fractions: train={train_fraction}, val={val_fraction}, test={test_fraction}")
    
    if abs((train_fraction + val_fraction + test_fraction) - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0: train={train_fraction}, val={val_fraction}, test={test_fraction}")
    
    working_df = df.copy()
    working_df[timestamp_col] = pd.to_datetime(working_df[timestamp_col], errors="coerce")
    working_df = working_df.dropna(subset=[timestamp_col])
    
    if len(working_df) < 3:
        raise ValueError("Not enough valid timestamped interactions for 3-way temporal split")
    
    working_df = working_df.sort_values(timestamp_col).reset_index(drop=True)
    
    # Calculate split indices
    train_cutoff_idx = max(int(len(working_df) * train_fraction), 1)
    val_cutoff_idx = max(int(len(working_df) * (train_fraction + val_fraction)), train_cutoff_idx + 1)
    
    # Ensure all splits have at least one row
    if val_cutoff_idx >= len(working_df):
        val_cutoff_idx = len(working_df) - 1
    
    # Create splits
    train = working_df.iloc[:train_cutoff_idx].copy()
    validation = working_df.iloc[train_cutoff_idx:val_cutoff_idx].copy()
    test = working_df.iloc[val_cutoff_idx:].copy()
    
    train_cutoff_timestamp = train.iloc[-1][timestamp_col]
    val_cutoff_timestamp = validation.iloc[-1][timestamp_col] if len(validation) > 0 else train_cutoff_timestamp
    
    # Initialize cold-start dataframes
    cold_user_val = pd.DataFrame(columns=validation.columns)
    cold_item_val = pd.DataFrame(columns=validation.columns)
    cold_user_test = pd.DataFrame(columns=test.columns)
    cold_item_test = pd.DataFrame(columns=test.columns)
    
    warm_validation = validation.copy()
    warm_test = test.copy()
    
    metadata: Dict[str, int] = {
        "initial_val_rows": len(validation),
        "initial_test_rows": len(test),
        "dropped_val_for_cold_start": 0,
        "dropped_test_for_cold_start": 0,
        "cold_user_val_rows": 0,
        "cold_item_val_rows": 0,
        "cold_user_test_rows": 0,
        "cold_item_test_rows": 0,
    }
    
    if ensure_warm_start:
        known_users = set(train["user_id"].unique())
        known_items = set(train["movie_id"].unique())
        
        # Process validation set
        cold_user_val_mask = ~warm_validation["user_id"].isin(known_users)
        cold_item_val_mask = warm_validation["user_id"].isin(known_users) & ~warm_validation["movie_id"].isin(known_items)
        
        cold_user_val = warm_validation[cold_user_val_mask].copy()
        cold_item_val = warm_validation[cold_item_val_mask].copy()
        
        warm_val_mask = ~cold_user_val_mask & ~cold_item_val_mask
        warm_validation = warm_validation[warm_val_mask].copy()
        
        metadata["cold_user_val_rows"] = len(cold_user_val)
        metadata["cold_item_val_rows"] = len(cold_item_val)
        metadata["dropped_val_for_cold_start"] = len(validation) - len(warm_validation)
        
        # Process test set (relative to train only, not train+val)
        cold_user_test_mask = ~warm_test["user_id"].isin(known_users)
        cold_item_test_mask = warm_test["user_id"].isin(known_users) & ~warm_test["movie_id"].isin(known_items)
        
        cold_user_test = warm_test[cold_user_test_mask].copy()
        cold_item_test = warm_test[cold_item_test_mask].copy()
        
        warm_test_mask = ~cold_user_test_mask & ~cold_item_test_mask
        warm_test = warm_test[warm_test_mask].copy()
        
        metadata["cold_user_test_rows"] = len(cold_user_test)
        metadata["cold_item_test_rows"] = len(cold_item_test)
        metadata["dropped_test_for_cold_start"] = len(test) - len(warm_test)
    
    logger.info(
        "3-way temporal split | train=%s | val=%s | test=%s | train_cutoff=%s | val_cutoff=%s",
        len(train),
        len(warm_validation),
        len(warm_test),
        train_cutoff_timestamp,
        val_cutoff_timestamp,
    )
    
    return TemporalSplitResult3Way(
        train=train,
        validation=warm_validation,
        test=warm_test,
        train_cutoff_timestamp=train_cutoff_timestamp,
        val_cutoff_timestamp=val_cutoff_timestamp,
        cold_user_val=cold_user_val,
        cold_item_val=cold_item_val,
        cold_user_test=cold_user_test,
        cold_item_test=cold_item_test,
        metadata=metadata,
    )

