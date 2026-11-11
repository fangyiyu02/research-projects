"""idea_b — Idea B experiments. Separate from lossy_verify (Idea A); no lossy_verify file is modified.

It reuses, read-only, these pieces of lossy_verify: the DSpark evaluator subclass, gsm8k answers and
grading, the mismatch/position logger, and analyze.py / speedup.py for tables.
Expected layout:  /workspace/DeepSpec   /workspace/lossy_verify   /workspace/idea_b
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_HERE)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

import lossy_verify  # noqa: E402,F401  (puts DeepSpec on sys.path)
