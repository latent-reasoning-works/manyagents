"""Generic scoring and ranking utilities for items.

This module provides reusable utilities for:
- Scoring items using custom functions
- Ranking by multiple criteria
- Applying thresholds and filters

These functions can be used:
1. By model adapters internally (e.g., ranking model outputs)
2. Directly in workflows for ranking operations
3. In domain-specific examples (e.g., drug repurposing candidate scoring)
"""

import asyncio
import logging
import importlib
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


async def score_and_rank(
    data: pd.DataFrame,
    scoring_function: str,
    scoring_params: Optional[Dict[str, Any]] = None,
    embeddings: Optional[np.ndarray] = None,
    threshold: float = 0.0,
    max_results: Optional[int] = None,
    sort_ascending: bool = False
) -> pd.DataFrame:
    """
    Score and rank items using a custom scoring function.
    
    This is the main entry point for ranking operations.
    
    Args:
        data: Input DataFrame to score
        scoring_function: Python path "module:function"
        scoring_params: Parameters passed to scoring function
        embeddings: Optional embeddings array from previous step
        threshold: Minimum score to include (0-1)
        max_results: Maximum number of items to return
        sort_ascending: Whether to sort ascending (default: False = highest first)
        
    Returns:
        DataFrame with 'score' column added, filtered and ranked
        
    Example:
        scored = await score_and_rank(
            data=candidates_df,
            scoring_function="myproject.scoring:score_candidates",
            scoring_params={"method": "combined"},
            threshold=0.6,
            max_results=100
        )
    """
    log.info(f"Scoring {len(data)} items using {scoring_function}")
    
    # Apply scoring function
    scored_df = await apply_scoring_function(
        data, scoring_function, scoring_params or {}, embeddings
    )
    
    # Filter by threshold
    if "score" in scored_df.columns:
        scored_df = scored_df[scored_df["score"] >= threshold]
        log.info(f"After threshold {threshold}: {len(scored_df)} items remain")
    
    # Sort
    if "score" in scored_df.columns:
        scored_df = scored_df.sort_values("score", ascending=sort_ascending)
    
    # Limit results
    if max_results is not None:
        scored_df = scored_df.head(max_results)
    
    log.info(f"Final ranking: {len(scored_df)} items")
    
    return scored_df


async def apply_scoring_function(
    data: pd.DataFrame,
    scoring_function: str,
    scoring_params: Dict[str, Any],
    embeddings: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Apply a custom scoring function to data.
    
    The scoring function should have signature:
        def score_fn(data: pd.DataFrame, embeddings: Optional[np.ndarray], **params) -> pd.DataFrame
    
    And must return a DataFrame with a 'score' column added.
    
    Args:
        data: Input DataFrame
        scoring_function: Python path "module:function"
        scoring_params: Parameters to pass to function
        embeddings: Optional embeddings array
        
    Returns:
        DataFrame with 'score' column
        
    Example:
        scored = await apply_scoring_function(
            data=df,
            scoring_function="manyagents.examples.drug_discovery.scoring:score_repurposing",
            scoring_params={"method": "combined", "failure_mode": None},
            embeddings=embedding_array
        )
    """
    if not scoring_function:
        log.warning("No scoring function provided, using default (random scores)")
        result = data.copy()
        result["score"] = np.random.random(len(data))
        return result
    
    # Parse module path
    if ":" not in scoring_function:
        raise ValueError(
            f"Invalid scoring_function format: {scoring_function}. "
            f"Expected 'module:function_name'"
        )
    
    module_path, func_name = scoring_function.split(":", 1)
    
    try:
        # Import and call function
        module = importlib.import_module(module_path)
        scoring_func = getattr(module, func_name)
        
        log.info(f"Calling {module_path}:{func_name} with params: {scoring_params}")
        
        result = scoring_func(
            data=data,
            embeddings=embeddings,
            **scoring_params
        )
        
        if asyncio.iscoroutine(result):
            scored_df = await result
        else:
            scored_df = result
        
        if not isinstance(scored_df, pd.DataFrame):
            raise TypeError(
                f"{scoring_function} returned {type(scored_df)}, expected pd.DataFrame"
            )
        
        if "score" not in scored_df.columns:
            raise ValueError(
                f"{scoring_function} must return DataFrame with 'score' column"
            )
        
        return scored_df
        
    except ImportError as e:
        raise ImportError(
            f"Could not import {module_path}: {e}. "
            f"Make sure the module is in your PYTHONPATH."
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Function {func_name} not found in {module_path}: {e}"
        ) from e


def rank_by_column(
    data: pd.DataFrame,
    column: str,
    ascending: bool = False,
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Simple ranking by a single column (no custom scoring function).
    
    Args:
        data: Input DataFrame
        column: Column name to rank by
        ascending: Sort order (False = highest first)
        top_n: Limit to top N results
        
    Returns:
        Sorted DataFrame
        
    Example:
        top_drugs = rank_by_column(df, "confidence", ascending=False, top_n=50)
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result = data.sort_values(column, ascending=ascending)
    
    if top_n is not None:
        result = result.head(top_n)
    
    return result
