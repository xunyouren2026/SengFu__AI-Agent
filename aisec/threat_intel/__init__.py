"""
AISEC Threat Intelligence Module
==================================
Threat feed aggregation, STIX/CVE parsing, indicator extraction,
event correlation, confidence scoring, and feed management.
"""

from .feed_aggregator import (
    FeedAggregator,
    STIXParser,
    CVEParser,
    IndicatorExtractor,
    EventCorrelator,
    ConfidenceScorer,
    FeedManager,
    ThreatIndicator,
    FeedSource,
    CorrelationMatch,
    IndicatorType,
    ThreatLevel,
    FeedFormat,
    FeedStatus,
)

__all__ = [
    "FeedAggregator",
    "STIXParser",
    "CVEParser",
    "IndicatorExtractor",
    "EventCorrelator",
    "ConfidenceScorer",
    "FeedManager",
    "ThreatIndicator",
    "FeedSource",
    "CorrelationMatch",
    "IndicatorType",
    "ThreatLevel",
    "FeedFormat",
    "FeedStatus",
]
