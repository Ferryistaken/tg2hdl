"""Benchmark: CPU vs FPGA estimates vs GPU references for MNIST inference.

Measures real CPU time, computes FPGA cycle estimates for various configurations,
and compares against published GPU numbers.
"""

import math
import time
import numpy as np
from tinygrad.nn.state import safe_load
from tinygrad.nn.datasets import mnist


# --- FPGA cycle model (validated by simulation) ---

def gemv_cycles(m, k, num_macs=1):
    """Compute cycles for one GEMV: ceil(K / num_macs) compute + 1 emit per row.

    FIXME: This is a simplified analytical cycle model that assumes perfect
    pipelining and no memory stalls. Real hardware would have additional
    latency from memory arbitration, pipeline bubbles, and FSM overhead.
    """
    return m * (math.ceil(k / num_macs) + 1)


def full_inference_cycles(num_macs=1):
    """Total cycles for 2-layer MNIST: layer1 (128x784) + layer2 (10x128)."""
    c1 = gemv_cycles(128, 784, num_macs)
    c2 = gemv_cycles(10, 128, num_macs)
    return c1, c2, c1 + c2


# --- FPGA resource estimates ---

def mac_resources(num_macs):
    """Rough DSP/LUT estimates per MAC configuration.
    One INT8 MAC ≈ 1 DSP48 slice on Xilinx (or ~200 LUTs without DSP).

    FIXME: These resource estimates are rough approximations. Actual DSP
    and BRAM usage depends on the target FPGA family, synthesis tool
    optimizations, and packing efficiency. The BRAM calculation assumes
    a fixed MNIST architecture and does not account for control logic or
    routing overhead.
    """
    return {
        "dsps": num_macs,
        "bram_kb": (128 * 784 + 784 + 128 * 10 + 128) * 1 / 1024,  # weight+vec storage
    }


def main():
    # --- Load weights, prepare input ---
    state = safe_load("mnist_weights.safetensors")
    w1 = state["l1.weight"].numpy()  # (128, 784)
    b1 = state["l1.bias"].numpy()
    w2 = state["l2.weight"].numpy()  # (10, 128)
    b2 = state["l2.bias"].numpy()
    _, _, X_test, y_test = mnist()
    x = X_test[0].numpy().reshape(784).astype(np.float32) / 255.0

    total_macs = 128 * 784 + 10 * 128  # multiply-accumulate operations
    print("=" * 70)
    print("MNIST 2-Layer MLP Inference Benchmark")
    print(f"Architecture: Linear(784→128) + ReLU + Linear(128→10)")
    print(f"Total MAC operations: {total_macs:,}")
    print("=" * 70)

    # =================================================================
    # 1. CPU Benchmark (numpy, float32)
    # =================================================================
    print("\n--- CPU (numpy float32) ---\n")

    # Warm up
    for _ in range(500):
        h = np.maximum(x @ w1.T + b1, 0)
        logits = h @ w2.T + b2

    # Benchmark single inference
    N = 50_000
    t0 = time.perf_counter()
    for _ in range(N):
        h = np.maximum(x @ w1.T + b1, 0)
        logits = h @ w2.T + b2
    cpu_time_us = (time.perf_counter() - t0) / N * 1e6

    # Batch benchmark (CPU SIMD advantage)
    X_batch = X_test[:256].numpy().reshape(256, 784).astype(np.float32) / 255.0
    N_batch = 2000
    t0 = time.perf_counter()
    for _ in range(N_batch):
        h = np.maximum(X_batch @ w1.T + b1, 0)
        logits = h @ w2.T + b2
    cpu_batch_us = (time.perf_counter() - t0) / N_batch / 256 * 1e6

    cpu_gops = total_macs * 2 / (cpu_time_us * 1e-6) / 1e9  # 2 ops per MAC (mul+add)

    print(f"  Single inference:  {cpu_time_us:.1f} us")
    print(f"  Batched (256):     {cpu_batch_us:.2f} us/image")
    print(f"  Throughput:        {1e6/cpu_time_us:,.0f} inferences/sec (single)")
    print(f"  Effective:         {cpu_gops:.2f} GOPS")

    # =================================================================
    # 2. FPGA Estimates
    # =================================================================
    print("\n--- FPGA Estimates (INT8) ---\n")

    mac_configs = [1, 4, 8, 16, 32, 64, 128]
    # FIXME: FPGA clock speeds are hardcoded typical values, not measured.
    # Real achievable Fmax depends on the specific design, utilization,
    # and place-and-route results. Use synthesis_stats() for actual Fmax.
    fpga_clocks = {
        "iCE40 (budget)":      25,   # MHz - Lattice iCE40
        "ECP5 (mid)":          100,  # MHz - Lattice ECP5
        "Artix-7 (mid)":      200,  # MHz - Xilinx Artix-7
        "Kintex-7 (high)":    300,  # MHz - Xilinx Kintex-7
    }

    # Table header
    print(f"  {'MACs':>4}  {'DSPs':>4}  {'Cycles':>9}  ", end="")
    for name in fpga_clocks:
        print(f"  {name:>18}", end="")
    print()
    print(f"  {'':>4}  {'':>4}  {'':>9}  ", end="")
    for mhz in fpga_clocks.values():
        print(f"  {'(' + str(mhz) + ' MHz)':>18}", end="")
    print()
    print("  " + "-" * (4 + 2 + 4 + 2 + 9 + len(fpga_clocks) * 20))

    for nm in mac_configs:
        _, _, total = full_inference_cycles(nm)
        res = mac_resources(nm)
        print(f"  {nm:>4}  {res['dsps']:>4}  {total:>9,}  ", end="")
        for mhz in fpga_clocks.values():
            time_us = total / mhz  # cycles / MHz = microseconds
            if time_us >= 1000:
                print(f"  {time_us/1000:>15.2f} ms", end="")
            else:
                print(f"  {time_us:>15.1f} us", end="")
        print()

    # =================================================================
    # 3. GPU References
    # =================================================================
    print("\n--- GPU References (published/estimated) ---\n")
    print("  Note: GPU latency for a SINGLE small inference is dominated by")
    print("  kernel launch overhead (~5-10 us), not compute. GPUs shine on")
    print("  large batches. These are rough estimates for this tiny model.\n")

    # FIXME: GPU latency numbers are rough published estimates, not measured
    # on actual hardware. Real latency varies with driver version, CUDA
    # toolkit, kernel launch overhead, and system configuration.
    gpus = [
        # (name, single_inference_us, note)
        ("GTX 1650 (Turing)",       "~20-50 us",     "launch overhead dominates"),
        ("RTX 3060 (Ampere)",       "~15-40 us",     "launch overhead dominates"),
        ("RTX 4090 (Ada)",          "~10-30 us",     "launch overhead dominates"),
        ("A100 (Ampere, datactr)",  "~10-25 us",     "launch overhead dominates"),
        ("RTX 4090 batch=256",      "~0.1 us/img",   "amortized, ~26 us total"),
        ("A100 batch=256",          "~0.05 us/img",  "amortized, ~13 us total"),
    ]

    print(f"  {'GPU':<28}  {'Latency':>14}  Note")
    print("  " + "-" * 70)
    for name, lat, note in gpus:
        print(f"  {name:<28}  {lat:>14}  {note}")

    # =================================================================
    # 4. Power Efficiency Comparison
    # =================================================================
    print("\n--- Power Efficiency ---\n")

    # FIXME: All power draw values below are rough estimates based on
    # published TDP ratings and typical board-level measurements. Actual
    # power consumption varies with workload, voltage regulation efficiency,
    # ambient temperature, and specific device stepping.
    platforms = [
        # (name, power_watts, time_us, batched, note)
        ("Your CPU (single)",    15.0,  cpu_time_us,      False, "~15W package TDP estimate"),
        ("Your CPU (batched)",   15.0,  cpu_batch_us,     True,  "same power, higher throughput"),
        ("iCE40 (1 MAC, 25MHz)", 0.05,  full_inference_cycles(1)[2] / 25, False, "~50 mW total board"),
        ("ECP5 (16 MAC, 100MHz)", 0.5,  full_inference_cycles(16)[2] / 100, False, "~500 mW"),
        ("Artix-7 (64 MAC, 200MHz)", 1.5, full_inference_cycles(64)[2] / 200, False, "~1.5W"),
        ("Kintex-7 (128 MAC, 300MHz)", 3.0, full_inference_cycles(128)[2] / 300, False, "~3W"),
        ("RTX 4090 (single)",   350.0, 30.0,             False, "~350W TDP"),
        ("RTX 4090 (batch=256)", 350.0, 0.1,             True,  "amortized per image"),
        ("A100 (batch=256)",    300.0, 0.05,              True,  "amortized per image"),
    ]

    total_ops = total_macs * 2  # mul + add = 2 ops per MAC

    print(f"  {'Platform':<32}  {'Time':>10}  {'Power':>7}  {'Energy/inf':>12}  {'Eff (GOPS/W)':>13}  {'vs CPU':>8}")
    print("  " + "-" * 95)

    cpu_energy = cpu_time_us * 1e-6 * 15.0  # joules per inference (CPU baseline)

    for name, watts, time_us_val, batched, note in platforms:
        energy_j = time_us_val * 1e-6 * watts
        energy_uj = energy_j * 1e6
        gops_per_w = total_ops / (time_us_val * 1e-6) / 1e9 / watts
        energy_ratio = cpu_energy / energy_j if energy_j > 0 else 0

        time_str = f"{time_us_val:.1f} us" if time_us_val >= 1 else f"{time_us_val*1000:.0f} ns"
        if time_us_val >= 1000:
            time_str = f"{time_us_val/1000:.1f} ms"

        print(f"  {name:<32}  {time_str:>10}  {watts:>5.1f}W  {energy_uj:>9.1f} uJ  {gops_per_w:>10.2f}    {energy_ratio:>6.1f}x")

    # =================================================================
    # 5. Scaling: What happens with bigger models?
    # =================================================================
    print("\n--- What About Bigger Models? ---\n")
    print("  This model is tiny (101K MACs). Here's how platforms scale:\n")

    models = [
        ("MNIST MLP (this)",     101_632,        "784→128→10"),
        ("Small CNN (CIFAR)",    5_000_000,      "3 conv + 2 FC"),
        ("MobileNet-v2",         300_000_000,    "image classifier"),
        ("ResNet-50",            4_000_000_000,  "image classifier"),
        ("BERT-base (1 token)",  22_000_000_000, "NLP transformer"),
    ]

    print(f"  {'Model':<26}  {'MACs':>14}  {'CPU @15W':>12}  {'FPGA 64MAC':>12}  {'A100 batch':>12}")
    print("  " + "-" * 80)

    for name, macs, desc in models:
        # FIXME: CPU scaling assumes linear throughput extrapolation from
        # measured GOPS, which breaks down for cache-bound or memory-bound models
        cpu_us = macs * 2 / (cpu_gops * 1e9) * 1e6 if macs > 200_000 else cpu_time_us
        # FIXME: FPGA GOPS estimate assumes perfect MAC utilization at 200 MHz
        # with no memory stalls or pipeline bubbles
        fpga_gops = 64 * 2 * 200e6 / 1e9  # 25.6 GOPS
        fpga_us = macs * 2 / (fpga_gops * 1e9) * 1e6
        # FIXME: A100 utilization of 30% is a rough guess for small models;
        # actual utilization depends on batch size, kernel launch overhead,
        # and memory access patterns
        a100_tops = 312 * 0.3  # effective TOPS
        a100_us = macs * 2 / (a100_tops * 1e12) * 1e6

        def fmt(us):
            if us >= 1e6: return f"{us/1e6:.0f} s"
            if us >= 1000: return f"{us/1000:.1f} ms"
            if us >= 1: return f"{us:.1f} us"
            return f"{us*1000:.0f} ns"

        print(f"  {name:<26}  {macs:>14,}  {fmt(cpu_us):>12}  {fmt(fpga_us):>12}  {fmt(a100_us):>12}")

    print()
    print("  Takeaway: FPGAs become competitive on speed at MobileNet scale")
    print("  and above, while using 100x less power than a GPU. The sweet")
    print("  spot is edge inference — single images, tight latency budgets,")
    print("  and power constraints (drones, cameras, medical devices).")

    # =================================================================
    # 6. Final Summary
    # =================================================================
    _, _, cycles_1mac = full_inference_cycles(1)
    _, _, cycles_64mac = full_inference_cycles(64)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("  SPEED (this tiny model):")
    print(f"    Your CPU wins at {cpu_time_us:.0f} us — numpy BLAS is hard to beat")
    print(f"    FPGA 64 MAC @ 200 MHz: {cycles_64mac/200:.0f} us (competitive)")
    print(f"    GPU: 20-50 us single (launch overhead), 0.05 us batched")
    print()
    print("  EFFICIENCY (energy per inference):")
    cpu_uj = cpu_time_us * 1e-6 * 15.0 * 1e6
    ice40_uj = full_inference_cycles(1)[2] / 25 * 1e-6 * 0.05 * 1e6
    artix_uj = full_inference_cycles(64)[2] / 200 * 1e-6 * 1.5 * 1e6
    gpu_uj = 30 * 1e-6 * 350 * 1e6
    print(f"    Your CPU:               {cpu_uj:>8.1f} uJ")
    print(f"    iCE40 (1 MAC, 50mW):    {ice40_uj:>8.1f} uJ  ({cpu_uj/ice40_uj:.0f}x better)")
    print(f"    Artix-7 (64 MAC, 1.5W): {artix_uj:>8.1f} uJ  ({cpu_uj/artix_uj:.0f}x better)")
    print(f"    RTX 4090 (single):      {gpu_uj:>8.1f} uJ  ({cpu_uj/gpu_uj:.1f}x {'better' if cpu_uj > gpu_uj else 'worse'})")
    print()
    print("  FPGAs don't win on speed for tiny models.")
    print("  They win on energy: 10-1000x less power per inference.")
    print("  That's why they're used in edge/embedded AI, not datacenters.")


if __name__ == "__main__":
    main()
