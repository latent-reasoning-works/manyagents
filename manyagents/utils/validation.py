"""Generic validation and filtering utilities for items.

This module provides reusable utilities for:
- Filtering items using custom validation functions
- Applying multiple filter criteria
- Flagging vs. rejecting items

These functions can be used:
1. By model adapters internally (e.g., filtering model outputs)
2. Directly in workflows for validation operations
3. In domain-specific examples (e.g., drug safety validation)
"""

import asyncio
import logging
import importlib
from typing import Dict, Any, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)


async def validate_and_filter(
    data: pd.DataFrame,
    filter_function: str,
    filter_params: Optional[Dict[str, Any]] = None,
    filter_mode: str = "flag"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate and filter items using a custom filter function.
    
    This is the main entry point for validation/filtering operations.
    
    Args:
        data: Input DataFrame to validate
        filter_function: Python path "module:function"
        filter_params: Parameters passed to filter function
        filter_mode: 
            - "strict": Reject items with any flag
            - "flag": Keep all items, just add flag columns
            
    Returns:
        Tuple of (passed_df, flagged_df)
        - In "strict" mode: flagged_df contains rejected items
        - In "flag" mode: flagged_df is empty (all in passed_df with flags)
        
    Example:
        passed, flagged = await validate_and_filter(
            data=candidates_df,
            filter_function="myproject.validation:check_safety",
            filter_params={"checks": ["toxicity", "ddi"]},
            filter_mode="strict"
        )
    """
    log.info(f"Validating {len(data)} items using {filter_function}")
    
    # Apply filter function
    validated_df = await apply_filter_function(
        data, filter_function, filter_params or {}
    )
    
    # Find flag columns
    flag_columns = [col for col in validated_df.columns if col.endswith("_flag")]
    
    if not flag_columns:
        raise ValueError(f"{filter_function} must return at least one flag column (ending in '_flag')")
    
    log.info(f"Found flag columns: {flag_columns}")
    
    # Apply filtering based on mode
    if filter_mode == "strict":
        # Reject items with any flag = True
        any_flag = validated_df[flag_columns].any(axis=1)
        passed_df = validated_df[~any_flag].copy()
        flagged_df = validated_df[any_flag].copy()
    else:  # flag mode
        # Keep all items with flags visible
        passed_df = validated_df.copy()
        flagged_df = pd.DataFrame()
    
    log.info(f"Validation complete: {len(passed_df)} passed, {len(flagged_df)} flagged")
    
    return passed_df, flagged_df


async def apply_filter_function(
    data: pd.DataFrame,
    filter_function: str,
    filter_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply a custom filter function to data.
    
    The filter function should have signature:
        def filter_fn(data: pd.DataFrame, **params) -> pd.DataFrame
    
    And should return a DataFrame with flag columns added (e.g., toxicity_flag).
    Flag columns should be boolean, with True indicating a problem.
    
    Args:
        data: Input DataFrame
        filter_function: Python path "module:function"
        filter_params: Parameters to pass to function
        
    Returns:
        DataFrame with flag columns added
        
    Example:
        validated = await apply_filter_function(
            data=df,
            filter_function="manyagents.examples.drug_discovery.validation:validate_safety",
            filter_params={"checks": ["toxicity", "ddi"], "strict": True}
        )
    """
    if not filter_function:
        raise ValueError("filter_function is required; choose an explicit validator")
    
    # Parse module path
    if ":" not in filter_function:
        raise ValueError(
            f"Invalid filter_function format: {filter_function}. "
            f"Expected 'module:function_name'"
        )
    
    module_path, func_name = filter_function.split(":", 1)
    
    try:
        # Import and call function
        module = importlib.import_module(module_path)
        filter_func = getattr(module, func_name)
        
        log.info(f"Calling {module_path}:{func_name} with params: {filter_params}")
        
        result = filter_func(data=data, **filter_params)
        
        if asyncio.iscoroutine(result):
            validated_df = await result
        else:
            validated_df = result
        
        if not isinstance(validated_df, pd.DataFrame):
            raise TypeError(
                f"{filter_function} returned {type(validated_df)}, expected pd.DataFrame"
            )
        
        return validated_df
        
    except ImportError as e:
        raise ImportError(
            f"Could not import {module_path}: {e}. "
            f"Make sure the module is in your PYTHONPATH."
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Function {func_name} not found in {module_path}: {e}"
        ) from e


def filter_by_column(
    data: pd.DataFrame,
    column: str,
    condition: str,
    value: Any
) -> pd.DataFrame:
    """
    Simple filtering by column value (no custom filter function).
    
    Args:
        data: Input DataFrame
        column: Column name to filter on
        condition: "equals", "greater_than", "less_than", "contains"
        value: Value to compare against
        
    Returns:
        Filtered DataFrame
        
    Example:
        high_confidence = filter_by_column(df, "confidence", "greater_than", 0.8)
        kinases = filter_by_column(df, "target_type", "equals", "Kinase")
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if condition == "equals":
        return data[data[column] == value]
    elif condition == "greater_than":
        return data[data[column] > value]
    elif condition == "less_than":
        return data[data[column] < value]
    elif condition == "contains":
        return data[data[column].str.contains(str(value), na=False)]
    else:
        raise ValueError(f"Unknown condition: {condition}")
