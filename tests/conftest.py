"""
Test-session setup.

macOS-local workaround: faiss-cpu and torch each ship their own copy of
libomp. When both are loaded in one process (which test_pipeline.py does —
it builds a real e5 + FAISS index) the duplicated OpenMP runtime segfaults
the interpreter on the first torch op that follows a FAISS call. Pinning the
OpenMP thread count before either library is imported avoids it.

This only affects local test runs on a Mac; Colab (the actual target) has a
single libomp and is unaffected. It must run before torch/faiss are imported,
which is why it lives at conftest import time rather than in a fixture.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
