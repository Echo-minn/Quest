"""
Quest: core package init.

Kept intentionally lightweight to avoid heavy side effects and circular
imports. Import models and utilities explicitly from `quest.models`
and `quest.utils` instead of relying on top-level re-exports.
"""

__version__ = "0.0.1"

__all__ = [
    "__version__",
]