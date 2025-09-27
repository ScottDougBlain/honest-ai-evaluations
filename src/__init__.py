"""
Honest AI Evaluations: Non-Sycophantic Response Framework

A comprehensive evaluation framework for preventing harmful sycophancy in LLMs,
bridging clinical psychology insights about therapeutic honesty with AI safety.
"""

from .sycophancy_evaluator import (
    SycophancyEvaluator,
    SycophancyMetrics,
    SycophancyTestCase,
    EvaluationResult,
    SycophancyType,
    EvaluationDomain,
    ResponseQuality
)

from .test_case_generator import TestCaseGenerator

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # Core evaluation classes
    "SycophancyEvaluator",
    "SycophancyMetrics",
    "SycophancyTestCase",
    "EvaluationResult",

    # Enums
    "SycophancyType",
    "EvaluationDomain",
    "ResponseQuality",

    # Test case generation
    "TestCaseGenerator"
]