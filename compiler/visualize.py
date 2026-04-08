from __future__ import annotations

from dataclasses import dataclass
from tinygrad.uop.ops import Ops

from .backend import HDLRenderer, _get_uops, compile_model, uops_to_kernel_ir
from .utils import format_uops


@dataclass
class KernelView:
    index: int
    metadata: tuple
    buffer_count: int
    uops: list
    kernel_ir: object


@dataclass
class PipelineView:
    kernel_views: list[KernelView]
    connections: list[tuple[int, int, int, int]]
    topo_order: list[int]
    execution_kernel_map: list[int]
    execution_connections: list[tuple[int, int, int, int]]
    execution_states: list[str]
    copy_groups: dict[int, list[list[tuple[int, int, int, int]]]]

    def to_text(self, *, include_uops: bool = True, include_kernel_ir: bool = True, full_width_uops: bool = False) -> str:
        lines = []
        lines.append("PIPELINE")
        lines.append(f"  kernels: {len(self.kernel_views)}")
        lines.append(f"  source_kernel_order: {[kv.index for kv in self.kernel_views]}")
        lines.append(f"  topo_order: {self.topo_order}")
        if self.connections:
            lines.append("  dependency_edges:")
            for src_k, src_buf, dst_k, dst_buf in self.connections:
                lines.append(f"    - K{src_k}.buf{src_buf} -> K{dst_k}.buf{dst_buf}")
        else:
            lines.append("  dependency_edges: none")

        lines.append("")
        lines.append("EXECUTION_PLAN")
        lines.append(f"  execution_kernel_order: {self.execution_kernel_map}")
        if self.execution_connections:
            lines.append("  execution_edges:")
            for src_k, src_buf, dst_k, dst_buf in self.execution_connections:
                src_orig = self.execution_kernel_map[src_k]
                dst_orig = self.execution_kernel_map[dst_k]
                lines.append(
                    f"    - exec K{src_k} (src K{src_orig}).buf{src_buf} -> "
                    f"exec K{dst_k} (src K{dst_orig}).buf{dst_buf}"
                )
        else:
            lines.append("  execution_edges: none")

        if self.copy_groups:
            lines.append("  copy_groups:")
            for exec_k, groups in sorted(self.copy_groups.items()):
                src_orig = self.execution_kernel_map[exec_k]
                for gi, group in enumerate(groups):
                    src_buf = group[0][1]
                    dests = ", ".join(
                        f"K{dst_k}.buf{dst_buf} (src K{self.execution_kernel_map[dst_k]})"
                        for _src_k, _src_buf, dst_k, dst_buf in group
                    )
                    lines.append(
                        f"    - exec K{exec_k} (src K{src_orig}) group {gi}: "
                        f"buf{src_buf} -> {dests}"
                    )
        else:
            lines.append("  copy_groups: none")

        lines.append("  fsm_states:")
        for state in self.execution_states:
            lines.append(f"    - {state}")

        for kv in self.kernel_views:
            lines.append("")
            lines.append(f"KERNEL K{kv.index}")
            lines.append(f"  metadata: {kv.metadata}")
            lines.append(f"  buffers: {kv.buffer_count}")
            if include_uops:
                lines.append("  uops:")
                lines.extend(f"    {line}" for line in format_uops(kv.uops, full_width=full_width_uops).splitlines())
            if include_kernel_ir:
                lines.append("  kernel_ir:")
                lines.extend(f"    {line}" for line in kv.kernel_ir.format(kv.kernel_ir).splitlines())
        return "\n".join(lines)

    def to_dot(self) -> str:
        lines = [
            "digraph tinygrad_pipeline {",
            '  rankdir="LR";',
            '  node [shape=box, style="rounded"];',
        ]
        for kv in self.kernel_views:
            meta = ", ".join(str(m.name if hasattr(m, "name") else m) for m in kv.metadata) if kv.metadata else "no-metadata"
            label = f"K{kv.index}\\n{meta}\\n{len(kv.uops)} uops"
            lines.append(f'  K{kv.index} [label="{_escape_dot(label)}"];')
        for src_k, src_buf, dst_k, dst_buf in self.connections:
            edge_label = f"buf{src_buf} -> buf{dst_buf}"
            lines.append(f'  K{src_k} -> K{dst_k} [label="{_escape_dot(edge_label)}"];')
        lines.append("}")
        return "\n".join(lines)

    def execution_dot(self) -> str:
        lines = [
            "digraph topmodule_execution {",
            '  rankdir="LR";',
            '  node [shape=box, style="rounded"];',
        ]
        for exec_k, orig_k in enumerate(self.execution_kernel_map):
            lines.append(f'  K{exec_k} [label="exec K{exec_k}\\nsrc K{orig_k}"];')
        for src_k, src_buf, dst_k, dst_buf in self.execution_connections:
            edge_label = f"buf{src_buf} -> buf{dst_buf}"
            lines.append(f'  K{src_k} -> K{dst_k} [label="{_escape_dot(edge_label)}"];')
        lines.append("}")
        return "\n".join(lines)


def _escape_dot(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _detect_connections(compute_items) -> list[tuple[int, int, int, int]]:
    output_buf_ids = {}
    for k_idx, si in enumerate(compute_items):
        if si.bufs:
            output_buf_ids[id(si.bufs[0])] = k_idx

    connections = []
    for k_idx, si in enumerate(compute_items):
        for buf_pos, buf in enumerate(si.bufs[1:], start=1):
            if buf is not None and id(buf) in output_buf_ids:
                src_k = output_buf_ids[id(buf)]
                if src_k != k_idx:
                    connections.append((src_k, 0, k_idx, buf_pos))
    return sorted(connections)


def _toposort_kernels(num_kernels: int, connections: list[tuple[int, int, int, int]]) -> list[int]:
    indegree = [0] * num_kernels
    outgoing = {k_idx: set() for k_idx in range(num_kernels)}
    for src_k, _src_buf, dst_k, _dst_buf in connections:
        if dst_k not in outgoing[src_k]:
            outgoing[src_k].add(dst_k)
            indegree[dst_k] += 1

    ready = [k_idx for k_idx in range(num_kernels) if indegree[k_idx] == 0]
    order = []
    while ready:
        cur = ready.pop(0)
        order.append(cur)
        for dst_k in sorted(outgoing[cur]):
            indegree[dst_k] -= 1
            if indegree[dst_k] == 0:
                ready.append(dst_k)
    return order


def _build_copy_groups(connections: list[tuple[int, int, int, int]], num_kernels: int):
    groups_by_src = {k_idx: [] for k_idx in range(num_kernels)}
    grouped = {}
    for src_k, src_buf, dst_k, dst_buf in connections:
        key = (src_k, src_buf)
        grouped.setdefault(key, []).append((src_k, src_buf, dst_k, dst_buf))
    for (src_k, _src_buf), conns in grouped.items():
        groups_by_src[src_k].append(sorted(conns))
    for src_k in groups_by_src:
        groups_by_src[src_k].sort(key=lambda grp: (grp[0][1], grp[0][2], grp[0][3]))
    return {src_k: groups for src_k, groups in groups_by_src.items() if groups}


def _build_execution_states(num_kernels: int, copy_groups: dict[int, list[list[tuple[int, int, int, int]]]]) -> list[str]:
    states = ["IDLE"]
    for exec_k in range(num_kernels):
        states.append(f"K{exec_k}_RUN")
        states.append(f"K{exec_k}_WAIT")
        for gi, _group in enumerate(copy_groups.get(exec_k, [])):
            states.append(f"K{exec_k}_COPY_{gi}")
    return states


def analyze_schedule(schedule) -> PipelineView:
    """Build a developer-facing view of the tinygrad -> HDL pipeline.

    This intentionally keeps two graph levels separate:
      - inter-kernel graph: the schedule dependency DAG
      - intra-kernel representation: linearized UOps and KernelIR
    """
    renderer = HDLRenderer()
    compute_items = [si for si in schedule if si.ast.op == Ops.SINK]
    connections = _detect_connections(compute_items)
    topo_order = _toposort_kernels(len(compute_items), connections)
    execution_kernel_map = topo_order
    exec_index = {orig_k: exec_k for exec_k, orig_k in enumerate(execution_kernel_map)}
    execution_connections = sorted(
        (exec_index[src_k], src_buf, exec_index[dst_k], dst_buf)
        for src_k, src_buf, dst_k, dst_buf in connections
    )
    copy_groups = _build_copy_groups(execution_connections, len(compute_items))
    execution_states = _build_execution_states(len(compute_items), copy_groups)

    kernel_views = []
    for k_idx, si in enumerate(compute_items):
        uops = _get_uops(si.ast, renderer)
        kernel_ir, _ = uops_to_kernel_ir(uops)
        kernel_views.append(KernelView(
            index=k_idx,
            metadata=getattr(si, "metadata", ()),
            buffer_count=len(getattr(si, "bufs", [])),
            uops=uops,
            kernel_ir=kernel_ir,
        ))

    return PipelineView(
        kernel_views=kernel_views,
        connections=connections,
        topo_order=topo_order,
        execution_kernel_map=execution_kernel_map,
        execution_connections=execution_connections,
        execution_states=execution_states,
        copy_groups=copy_groups,
    )


def analyze_tensor(tensor) -> PipelineView:
    """Convenience wrapper for Tensor.schedule()."""
    return analyze_schedule(tensor.schedule())


def analyze_manual_kernels(kernels, connections, *, original_kernel_ids: list[int] | None = None) -> PipelineView:
    """Build a pipeline view for a manually assembled top-level execution graph.

    This is useful when tinygrad fuses a high-level expression into fewer kernels
    than the executor-level example intends to demonstrate.
    """
    if original_kernel_ids is None:
        original_kernel_ids = list(range(len(kernels)))

    kernel_views = []
    for orig_k, kernel in zip(original_kernel_ids, kernels):
        kernel_views.append(KernelView(
            index=orig_k,
            metadata=("manual_top",),
            buffer_count=len(getattr(kernel, "buf_infos", [])),
            uops=[],
            kernel_ir=kernel.kernel_ir,
        ))

    topo_order = _toposort_kernels(len(kernels), connections)
    execution_kernel_map = topo_order
    exec_index = {orig_k: exec_k for exec_k, orig_k in enumerate(execution_kernel_map)}
    execution_connections = sorted(
        (exec_index[src_k], src_buf, exec_index[dst_k], dst_buf)
        for src_k, src_buf, dst_k, dst_buf in connections
    )
    copy_groups = _build_copy_groups(execution_connections, len(kernels))
    execution_states = _build_execution_states(len(kernels), copy_groups)

    return PipelineView(
        kernel_views=kernel_views,
        connections=sorted(connections),
        topo_order=topo_order,
        execution_kernel_map=execution_kernel_map,
        execution_connections=execution_connections,
        execution_states=execution_states,
        copy_groups=copy_groups,
    )
