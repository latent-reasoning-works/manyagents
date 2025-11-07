"""Drug discovery data loading functions.

These functions demonstrate how to use manyagents generic utils for
drug discovery workflows. They provide domain-specific data loading
that can be called by the generic data_ops utilities.

In production, these would call real APIs:
- ChEMBL: https://chembl.gitbook.io/chembl-interface-documentation/
- DrugBank: https://www.drugbank.ca/
- LINCS L1000: https://clue.io/
- Open Targets: https://platform.opentargets.org/
"""

import logging
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


def fetch_chembl(query: str = "kinase_inhibitors", limit: int = 500) -> pd.DataFrame:
    """
    Fetch drug-target interaction data from ChEMBL.
    
    This is a MOCK implementation. In production, this would call:
    https://www.ebi.ac.uk/chembl/api/data/

    Args:
        query: Search query (e.g., "kinase_inhibitors", "approved_drugs")
        limit: Maximum number of records to return
        
    Returns:
        DataFrame with columns: chembl_id, molecule_name, target_chembl_id,
                               target_name, pchembl_value, assay_type
                               
    Example:
        from manyagents.examples.drug_discovery.data import fetch_chembl
        df = fetch_chembl(query="kinase_inhibitors", limit=100)
    """
    log.info(f"[MOCK] Fetching ChEMBL data: query={query}, limit={limit}")
    
    # Mock ChEMBL drug-target interaction data
    return pd.DataFrame({
        "chembl_id": [f"CHEMBL{1000+i}" for i in range(limit)],
        "molecule_name": [f"Drug_{query}_{i}" for i in range(limit)],
        "target_chembl_id": [f"CHEMBL{2000+i%50}" for i in range(limit)],
        "target_name": [f"Kinase_{i%50}" for i in range(limit)],
        "pchembl_value": np.random.uniform(5, 10, limit),  # Binding affinity
        "assay_type": np.random.choice(["B", "F"], limit),  # Binding or Functional
        "drug_id": [f"DRUG_{i}" for i in range(limit)]  # Common key for joining
    })


def fetch_expression(dataset: str = "lincs_l1000", limit: int = 500) -> pd.DataFrame:
    """
    Fetch gene expression data from LINCS L1000.
    
    This is a MOCK implementation. In production, this would call:
    https://clue.io/api
    
    Args:
        dataset: Dataset name (e.g., "lincs_l1000")
        limit: Maximum number of records to return
        
    Returns:
        DataFrame with columns: pert_id, pert_name, cell_line, gene_symbol,
                               z_score, drug_id
                               
    Example:
        from manyagents.examples.drug_discovery.data import fetch_expression
        df = fetch_expression(dataset="lincs_l1000", limit=100)
    """
    log.info(f"[MOCK] Fetching expression data: dataset={dataset}, limit={limit}")
    
    # Mock LINCS L1000 expression signature data
    return pd.DataFrame({
        "pert_id": [f"PERT_{i}" for i in range(limit)],
        "pert_name": [f"Drug_{i}" for i in range(limit)],
        "cell_line": np.random.choice(["A549", "MCF7", "VCAP"], limit),
        "gene_symbol": [f"GENE{i%100}" for i in range(limit)],
        "z_score": np.random.randn(limit) * 2,  # Differential expression
        "drug_id": [f"DRUG_{i}" for i in range(limit)]  # Common key for joining
    })


def fetch_drugbank(limit: int = 500) -> pd.DataFrame:
    """
    Fetch approved drug information from DrugBank.
    
    This is a MOCK implementation. In production, this would call:
    https://go.drugbank.com/api
    
    Args:
        limit: Maximum number of records to return
        
    Returns:
        DataFrame with columns: drugbank_id, name, indication, mechanism,
                               approved, drug_id
                               
    Example:
        from manyagents.examples.drug_discovery.data import fetch_drugbank
        df = fetch_drugbank(limit=100)
    """
    log.info(f"[MOCK] Fetching DrugBank data: limit={limit}")
    
    # Mock DrugBank approved drug data
    return pd.DataFrame({
        "drugbank_id": [f"DB{10000+i:05d}" for i in range(limit)],
        "name": [f"DrugName_{i}" for i in range(limit)],
        "indication": np.random.choice(
            ["Cancer", "Diabetes", "Hypertension", "Inflammation"],
            limit
        ),
        "mechanism": [f"Mechanism_{i%20}" for i in range(limit)],
        "approved": np.random.choice([True, False], limit, p=[0.7, 0.3]),
        "drug_id": [f"DRUG_{i}" for i in range(limit)]  # Common key for joining
    })


async def fetch_all_sources(chembl_limit: int = 500, expression_limit: int = 500) -> pd.DataFrame:
    """
    Example of loading and merging multiple data sources.
    
    This demonstrates how to use manyagents.utils.data_ops for integration.
    
    Args:
        chembl_limit: ChEMBL records to fetch
        expression_limit: Expression records to fetch
        
    Returns:
        Merged DataFrame with drug-target and expression data
        
    Example:
        from manyagents.examples.drug_discovery.data import fetch_all_sources
        df = await fetch_all_sources(chembl_limit=100, expression_limit=100)
    """
    from manyagents.utils.data_ops import load_and_merge_dataframes
    
    sources = [
        {
            "transform_function": "manyagents.examples.drug_discovery.data:fetch_chembl",
            "params": {"query": "kinase_inhibitors", "limit": chembl_limit}
        },
        {
            "transform_function": "manyagents.examples.drug_discovery.data:fetch_expression",
            "params": {"dataset": "lincs_l1000", "limit": expression_limit}
        }
    ]
    
    return await load_and_merge_dataframes(
        sources=sources,
        merge_strategy="inner_join",
        merge_on="drug_id"
    )
