"""Generic data operations for DataFrame transformations and integration.

This module provides reusable utilities for:
- Loading data from various sources via custom functions
- Merging and transforming DataFrames
- Data validation and preprocessing

These functions can be used:
1. By model adapters internally (e.g., ManyLatentsAdapter preprocessing data)
2. Directly in workflows as standalone operations
3. In domain-specific examples (e.g., drug discovery data integration)
"""

import asyncio
import logging
import importlib
from typing import Dict, Any, Optional, List, Union

import pandas as pd

log = logging.getLogger(__name__)


async def load_and_merge_dataframes(
    sources: List[Dict[str, Any]],
    merge_strategy: str = "concat",
    merge_on: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Load data from multiple sources and merge into single DataFrame.
    
    This is the main entry point for data integration operations.
    
    Args:
        sources: List of source configs, each containing:
            - transform_function: Python path "module:function"
            - params: Dict of parameters for the function
        merge_strategy: "inner_join" | "outer_join" | "concat"
        merge_on: Column name(s) for joining (if applicable)
        
    Returns:
        Merged DataFrame
        
    Example:
        sources = [
            {
                "transform_function": "myproject.data:load_chembl",
                "params": {"query": "kinase_inhibitors"}
            },
            {
                "transform_function": "myproject.data:load_expression", 
                "params": {"dataset": "lincs"}
            }
        ]
        df = await load_and_merge_dataframes(sources, merge_strategy="inner_join")
    """
    log.info(f"Loading {len(sources)} data sources with strategy: {merge_strategy}")
    
    # Load all sources
    dataframes = []
    for i, source_config in enumerate(sources):
        df = await load_source(source_config, index=i)
        dataframes.append(df)
        log.info(f"Source {i+1}: Loaded {len(df)} records, {len(df.columns)} columns")
    
    # Merge dataframes
    if len(dataframes) == 0:
        return pd.DataFrame()
    elif len(dataframes) == 1:
        return dataframes[0]
    else:
        return merge_dataframes(dataframes, merge_strategy, merge_on)


async def load_source(
    source_config: Dict[str, Any],
    index: int = 0
) -> pd.DataFrame:
    """
    Load data from a single source using a custom transform function.
    
    The transform function is dynamically imported and called. This allows
    domain-specific data loading logic to be external to manyagents core.
    
    Args:
        source_config: Config with:
            - transform_function: "module.path:function_name"
            - params: Dict passed to function
        index: Source index for error reporting
        
    Returns:
        DataFrame loaded by the transform function
        
    Example:
        config = {
            "transform_function": "manyagents.examples.drug_discovery.data:fetch_chembl",
            "params": {"query": "kinase_inhibitors", "limit": 500}
        }
        df = await load_source(config)
    """
    transform_path = source_config.get("transform_function")
    params = source_config.get("params", {})
    
    if not transform_path:
        raise ValueError(f"Source {index}: 'transform_function' is required")
    
    # Parse module path and function name
    if ":" not in transform_path:
        raise ValueError(
            f"Invalid transform_function format: {transform_path}. "
            f"Expected 'module:function_name'"
        )
    
    module_path, func_name = transform_path.split(":", 1)
    
    try:
        # Dynamically import module and get function
        module = importlib.import_module(module_path)
        transform_func = getattr(module, func_name)
        
        log.info(f"Calling {module_path}:{func_name} with params: {params}")
        
        # Call function (support both sync and async)
        result = transform_func(**params)
        if asyncio.iscoroutine(result):
            df = await result
        else:
            df = result
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"{transform_path} returned {type(df)}, expected pd.DataFrame"
            )
        
        return df
        
    except ImportError as e:
        raise ImportError(
            f"Could not import {module_path}: {e}. "
            f"Make sure the module is in your PYTHONPATH."
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Function {func_name} not found in {module_path}: {e}"
        ) from e


def merge_dataframes(
    dataframes: List[pd.DataFrame],
    strategy: str = "concat",
    merge_on: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Merge multiple DataFrames using specified strategy.
    
    Args:
        dataframes: List of DataFrames to merge
        strategy: Merge strategy:
            - "inner_join": Join on common columns (inner)
            - "outer_join": Join on common columns (outer)
            - "concat": Stack vertically (ignore index)
        merge_on: Column name(s) for joining (required for joins)
        
    Returns:
        Merged DataFrame
        
    Example:
        df1 = pd.DataFrame({"id": [1,2], "val": [10,20]})
        df2 = pd.DataFrame({"id": [1,3], "score": [0.5,0.8]})
        merged = merge_dataframes([df1, df2], "inner_join", merge_on="id")
    """
    if strategy == "inner_join":
        if not merge_on:
            # Auto-detect common columns
            common_cols = set(dataframes[0].columns)
            for df in dataframes[1:]:
                common_cols &= set(df.columns)
            if common_cols:
                merge_on = list(common_cols)[0]
                log.info(f"Auto-detected merge column: {merge_on}")
            else:
                log.warning("No common columns found, using concat instead")
                return pd.concat(dataframes, axis=0, ignore_index=True)
        
        result = dataframes[0]
        for df in dataframes[1:]:
            result = pd.merge(result, df, on=merge_on, how="inner")
        return result
    
    elif strategy == "outer_join":
        if not merge_on:
            raise ValueError("outer_join requires 'merge_on' parameter")
        
        result = dataframes[0]
        for df in dataframes[1:]:
            result = pd.merge(result, df, on=merge_on, how="outer")
        return result
    
    elif strategy == "concat":
        return pd.concat(dataframes, axis=0, ignore_index=True)
    
    else:
        log.warning(f"Unknown strategy '{strategy}', using concat")
        return pd.concat(dataframes, axis=0, ignore_index=True)


def transform_dataframe(
    data: pd.DataFrame,
    transform_function: str,
    params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Apply a custom transformation function to a DataFrame.
    
    Args:
        data: Input DataFrame
        transform_function: Python path "module:function"
        params: Optional parameters for function
        
    Returns:
        Transformed DataFrame
        
    Example:
        df = transform_dataframe(
            data=raw_df,
            transform_function="myproject.transforms:normalize_features",
            params={"method": "zscore"}
        )
    """
    if ":" not in transform_function:
        raise ValueError(
            f"Invalid transform_function format: {transform_function}. "
            f"Expected 'module:function_name'"
        )
    
    module_path, func_name = transform_function.split(":", 1)
    params = params or {}
    
    try:
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        
        result = func(data, **params)
        
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"{transform_function} returned {type(result)}, expected pd.DataFrame"
            )
        
        return result
        
    except ImportError as e:
        raise ImportError(f"Could not import {module_path}: {e}") from e
    except AttributeError as e:
        raise AttributeError(f"Function {func_name} not found in {module_path}: {e}") from e
