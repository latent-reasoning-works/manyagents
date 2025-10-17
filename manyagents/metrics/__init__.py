"""
Metrics computation feature for manyAgents.

IMPORTANT: Only use MetricComputer for workflows that do NOT use manylatents.
When using ManyLatentsAdapter, metrics are computed natively within manylatents.
"""

from .compute import MetricComputer

__all__ = ["MetricComputer"]
