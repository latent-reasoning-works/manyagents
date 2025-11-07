"""Drug discovery scoring functions.

These functions demonstrate how to use manyagents generic utils for
ranking drug repurposing candidates. They can be called by the generic
scoring utilities.

Scoring functions must have signature:
    def score_fn(data: pd.DataFrame, embeddings: Optional[np.ndarray], **params) -> pd.DataFrame

And must return DataFrame with 'score' column added.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)


def score_repurposing(
    data: pd.DataFrame,
    embeddings: Optional[np.ndarray] = None,
    method: str = "combined",
    failure_mode: Optional[str] = None
) -> pd.DataFrame:
    """
    Score drugs for repurposing potential.
    
    This is a DEMONSTRATION implementation showing both success and failure modes.
    In production, this would use:
    - Graph neural networks for drug-target prediction
    - Causal inference for mechanism identification
    - Literature mining from PubMed
    
    Args:
        data: DataFrame with drug-target interaction data
        embeddings: Optional embeddings from manylatents
        method: Scoring method:
            - "embedding_similarity": Use embedding distances
            - "target_overlap": Score by target similarity
            - "combined": Combine multiple signals
        failure_mode: For educational demos:
            - None: Normal scoring
            - "overfit": Artificially high confidence, low diversity
            - "target_bias": Bias toward well-studied targets (kinases, GPCRs)
            
    Returns:
        DataFrame with 'score' column added (0-1 range)
        
    Example:
        from manyagents.examples.drug_discovery.scoring import score_repurposing
        scored_df = score_repurposing(
            data=candidates_df,
            embeddings=embedding_array,
            method="combined",
            failure_mode=None
        )
    """
    if failure_mode == "overfit":
        return _score_with_overfitting(data)
    elif failure_mode == "target_bias":
        return _score_with_target_bias(data)
    else:
        return _score_normally(data, embeddings, method)


def _score_normally(
    data: pd.DataFrame,
    embeddings: Optional[np.ndarray],
    method: str
) -> pd.DataFrame:
    """Normal scoring without failure modes."""
    log.info(f"Scoring {len(data)} candidates using method: {method}")
    
    result = data.copy()
    
    # Mock scoring based on method
    if method == "embedding_similarity" and embeddings is not None:
        # Use embedding distances as proxy for repurposing potential
        scores = np.random.beta(2, 5, len(data))  # Skewed toward lower scores
    elif method == "target_overlap":
        # Score based on target similarity
        scores = np.random.beta(3, 3, len(data))  # More uniform
    else:  # combined
        scores = np.random.beta(2.5, 4, len(data))
    
    result["score"] = scores
    
    # Add additional metadata
    result["original_indication"] = np.random.choice(
        ["Cancer", "Diabetes", "Hypertension", "Inflammation"],
        len(data)
    )
    result["proposed_indication"] = np.random.choice(
        ["Alzheimer's", "Parkinson's", "ALS", "MS"],
        len(data)
    )
    result["evidence_count"] = np.random.randint(1, 50, len(data))
    
    return result


def _score_with_overfitting(data: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstrate overfitting failure mode.
    
    Characteristics:
    - Artificially high confidence scores
    - Limited diversity in mechanisms
    - Skewed toward common targets
    """
    log.warning("Scoring with OVERFIT failure mode for demonstration")
    
    result = data.copy()
    
    # Overconfident scores (heavily skewed toward 1.0)
    result["score"] = np.random.beta(8, 2, len(data))
    
    # Low diversity
    result["original_indication"] = "Cancer"  # Same for all
    result["proposed_indication"] = np.random.choice(
        ["Alzheimer's", "Parkinson's"],
        len(data),
        p=[0.9, 0.1]  # Heavily biased
    )
    result["mechanism"] = [f"Target_{i%5}" for i in range(len(data))]  # Only 5 targets
    result["evidence_count"] = np.random.randint(10, 100, len(data))  # Inflated
    
    return result


def _score_with_target_bias(data: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstrate target class bias failure mode.
    
    Characteristics:
    - Favor kinases and GPCRs (most studied)
    - Higher scores for well-documented targets
    - Lower scores for novel target classes
    """
    log.warning("Scoring with TARGET_BIAS failure mode for demonstration")
    
    result = data.copy()
    
    # Assign target types with bias
    target_types = np.random.choice(
        ["Kinase", "GPCR", "Ion_Channel", "Enzyme"],
        len(data),
        p=[0.5, 0.3, 0.1, 0.1]  # Heavy bias toward kinases/GPCRs
    )
    result["mechanism"] = target_types
    
    # Higher scores for biased targets
    result["score"] = np.where(
        np.isin(target_types, ["Kinase", "GPCR"]),
        np.random.beta(5, 2, len(data)),  # High confidence
        np.random.beta(2, 5, len(data))   # Low confidence
    )
    
    # More evidence for biased targets
    result["evidence_count"] = np.where(
        np.isin(target_types, ["Kinase", "GPCR"]),
        np.random.randint(50, 200, len(data)),  # Many papers
        np.random.randint(1, 10, len(data))     # Few papers
    )
    
    result["original_indication"] = np.random.choice(
        ["Cancer", "Diabetes", "Hypertension"],
        len(data)
    )
    result["proposed_indication"] = np.random.choice(
        ["Cancer", "Diabetes"],  # Biased toward same indications
        len(data),
        p=[0.8, 0.2]
    )
    
    return result
