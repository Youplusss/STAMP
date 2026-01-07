"""LLM-based anomaly explanation modules for STAMP.

This package is designed to be *optional*:
- If `transformers` is installed, we can call a local HuggingFace causal LLM to generate a narrative report.
- If not, we fall back to a deterministic, template-based explanation.

Main entry:
- `explain.pipeline.generate_explanations(...)`
"""
