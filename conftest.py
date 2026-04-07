"""Root conftest — set NOOPT=1 before tinygrad imports.

tinygrad reads NOOPT at import time and caches it. If benchmark tests
(which don't set NOOPT) run before compiler tests (which need NOOPT=1),
the env var set inside test files is too late. Setting it here ensures
all tests see NOOPT=1 regardless of collection order.
"""

import os

os.environ.setdefault("NOOPT", "1")
