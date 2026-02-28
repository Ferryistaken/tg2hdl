"""Utilities for inspecting tinygrad UOps."""

from tinygrad import UOp
from typing import List

from typing import List
from textwrap import shorten

def pretty_print_uops(uops: List["UOp"], full_width: bool = False) -> None:
    """Print a linearized UOp list in a readable table format.

    Each row: index, op name, dtype, arg, sources (by index).
    If full_width is True, columns expand to fit content without truncation.
    """

    # ---- 1. Collect raw string data first ----
    rows = []
    idx = {id(u): i for i, u in enumerate(uops)}

    # Header row
    rows.append({
        "idx": "#",
        "op": "op",
        "dtype": "dtype",
        "arg": "arg",
        "src": "src",
    })

    for i, u in enumerate(uops):
        src_ids = "[" + ", ".join(f"%{idx[id(s)]}" for s in u.src) + "]"
        rows.append({
            "idx": str(i),
            "op": str(u.op),
            "dtype": str(u.dtype),
            "arg": repr(u.arg),
            "src": src_ids,
        })

    # ---- 2. Determine column widths ----
    if full_width:
        # Use max length of actual content for each column
        W_IDX   = max(len(r["idx"])   for r in rows)
        W_OP    = max(len(r["op"])    for r in rows)
        W_DTYPE = max(len(r["dtype"]) for r in rows)
        W_ARG   = max(len(r["arg"])   for r in rows)
        W_SRC   = max(len(r["src"])   for r in rows)
        truncate = False
    else:
        # Fixed widths with truncation
        W_IDX   = 3
        W_OP    = 18
        W_DTYPE = 20
        W_ARG   = 20
        W_SRC   = 10
        truncate = True

    def fmt_cell(val: str, width: int, align_left: bool = True) -> str:
        if truncate:
            # Reserve space for ellipsis if we truncate
            val = shorten(val, width=width, placeholder="…")
        if align_left:
            return f"{val:<{width}}"
        else:
            return f"{val:>{width}}"

    # ---- 3. Print header and rows ----
    header = rows[0]
    header_line = (
        f"{fmt_cell(header['idx'],   W_IDX,   align_left=False)}  "
        f"{fmt_cell(header['op'],    W_OP)}  "
        f"{fmt_cell(header['dtype'], W_DTYPE)}  "
        f"{fmt_cell(header['arg'],   W_ARG)}  "
        f"{fmt_cell(header['src'],   W_SRC)}"
    )
    print(header_line)
    print("-" * len(header_line))

    for r in rows[1:]:
        line = (
            f"{fmt_cell(r['idx'],   W_IDX,   align_left=False)}  "
            f"{fmt_cell(r['op'],    W_OP)}  "
            f"{fmt_cell(r['dtype'], W_DTYPE)}  "
            f"{fmt_cell(r['arg'],   W_ARG)}  "
            f"{fmt_cell(r['src'],   W_SRC)}"
        )
        print(line)

