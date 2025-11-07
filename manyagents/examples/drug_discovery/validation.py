"""Drug discovery validation functions.

These functions demonstrate how to use manyagents generic utils for
validating drug candidates against safety criteria. They can be called
by the generic validation utilities.

Validation functions must have signature:
    def validate_fn(data: pd.DataFrame, **params) -> pd.DataFrame

And must return DataFrame with flag columns added (e.g., toxicity_flag).
"""

import logging
import pandas as pd
import numpy as np
from typing import List

log = logging.getLogger(__name__)


def validate_safety(
    data: pd.DataFrame,
    checks: List[str] = None,
    confidence_boost: float = 0.0
) -> pd.DataFrame:
    """
    Validate drug candidates against safety criteria.
    
    This is a MOCK implementation. In production, this would query:
    - ToxCast: https://www.epa.gov/chemical-research/toxicity-forecasting
    - DrugBank DDI: https://go.drugbank.com/drug-drug-interactions
    - FDA FAERS: https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers
    - ClinicalTrials.gov
    
    Args:
        data: DataFrame with drug candidates
        checks: List of safety checks to perform:
            - "toxicity": Check against toxicity databases
            - "ddi": Drug-drug interaction potential
            - "adverse_events": Known adverse events
            - "approval_status": Regulatory approval status
        confidence_boost: Adjust confidence for approved drugs (0-1)
        
    Returns:
        DataFrame with flag columns added:
            - toxicity_flag (bool)
            - ddi_flag (bool)
            - adverse_event_flag (bool)
            - approval_status (str: "Approved", "Investigational", "Experimental")
            
    Example:
        from manyagents.examples.drug_discovery.validation import validate_safety
        validated_df = validate_safety(
            data=candidates_df,
            checks=["toxicity", "ddi", "adverse_events"],
            confidence_boost=0.1
        )
    """
    checks = checks or ["toxicity", "ddi"]
    log.info(f"Running safety validation with checks: {checks}")
    
    result = data.copy()
    
    # Initialize flag columns
    result["toxicity_flag"] = False
    result["ddi_flag"] = False
    result["adverse_event_flag"] = False
    result["approval_status"] = "Unknown"
    
    # Run each check
    if "toxicity" in checks:
        # Mock toxicity check (20% failure rate)
        result["toxicity_flag"] = np.random.choice(
            [True, False], len(result), p=[0.2, 0.8]
        )
        n_flagged = result["toxicity_flag"].sum()
        log.info(f"Toxicity check: {n_flagged}/{len(result)} flagged")
    
    if "ddi" in checks:
        # Mock drug-drug interaction check (15% failure rate)
        result["ddi_flag"] = np.random.choice(
            [True, False], len(result), p=[0.15, 0.85]
        )
        n_flagged = result["ddi_flag"].sum()
        log.info(f"DDI check: {n_flagged}/{len(result)} flagged")
    
    if "adverse_events" in checks:
        # Mock adverse event check (10% failure rate)
        result["adverse_event_flag"] = np.random.choice(
            [True, False], len(result), p=[0.1, 0.9]
        )
        n_flagged = result["adverse_event_flag"].sum()
        log.info(f"Adverse events: {n_flagged}/{len(result)} flagged")
    
    if "approval_status" in checks:
        # Mock approval status
        result["approval_status"] = np.random.choice(
            ["Approved", "Investigational", "Experimental"],
            len(result),
            p=[0.3, 0.4, 0.3]
        )
        
        # Boost confidence for approved drugs (if score column exists)
        if "score" in result.columns and confidence_boost > 0:
            approved_mask = result["approval_status"] == "Approved"
            result.loc[approved_mask, "score"] += confidence_boost
            result["score"] = result["score"].clip(0, 1)
            n_boosted = approved_mask.sum()
            log.info(f"Confidence boost: {n_boosted} approved drugs boosted by {confidence_boost}")
    
    return result


def validate_strict(data: pd.DataFrame, max_flags: int = 0) -> pd.DataFrame:
    """
    Strict validation - reject candidates with any safety flags.
    
    This is a convenience function that runs all checks and is intended
    to be used with filter_mode="strict" to reject flagged items.
    
    Args:
        data: DataFrame with drug candidates
        max_flags: Maximum number of flags allowed (default: 0 = no flags)
        
    Returns:
        DataFrame with all flag columns added
        
    Example:
        from manyagents.examples.drug_discovery.validation import validate_strict
        validated_df = validate_strict(data=candidates_df, max_flags=0)
        # Use with: filter_mode="strict" to reject any with flags
    """
    return validate_safety(
        data,
        checks=["toxicity", "ddi", "adverse_events", "approval_status"],
        confidence_boost=0.0
    )


def validate_permissive(data: pd.DataFrame) -> pd.DataFrame:
    """
    Permissive validation - flag but don't reject.
    
    This runs basic checks but is intended to be used with filter_mode="flag"
    to keep all items with flags visible for user review.
    
    Args:
        data: DataFrame with drug candidates
        
    Returns:
        DataFrame with flag columns added
        
    Example:
        from manyagents.examples.drug_discovery.validation import validate_permissive
        validated_df = validate_permissive(data=candidates_df)
        # Use with: filter_mode="flag" to keep all items
    """
    return validate_safety(
        data,
        checks=["toxicity"],
        confidence_boost=0.1
    )
