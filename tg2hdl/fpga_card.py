"""FPGA board card: load hardware specs from JSON and derive timing models.

An FPGA card is a JSON file under ``fpga_cards/`` that captures every
board-level parameter the toolchain needs — fabric resources, BRAM
geometry, DSP counts, PCIe bandwidth, power draw, and synthesis-tool
flags.  All values come from vendor datasheets (see each card's
``notes`` and ``datasheet_url`` fields).

Usage::

    from tg2hdl.fpga_card import load_card, FPGACard

    card = load_card("lattice_ecp5_45k_cabga381")
    print(card.pcie_xfer_s(4096))   # one-direction transfer time
    print(card.bram_block_bytes)     # data bytes per BRAM block
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


_CARDS_DIR = Path(__file__).resolve().parent.parent / "fpga_cards"

# Default card used when no explicit card is specified.
DEFAULT_CARD = "lattice_ecp5_45k_cabga381"


@dataclass(frozen=True)
class FPGACard:
    """Immutable snapshot of an FPGA board's specifications.

    Every field is sourced from the card JSON.  Derived helpers
    (``pcie_xfer_s``, ``bram_block_bytes``, etc.) are computed from
    these fields — no hardcoded magic numbers.
    """

    # -- identity --
    name: str
    family: str
    part_number: str
    device: str
    package: str
    technology_nm: int
    datasheet: str
    datasheet_url: str

    # -- fabric --
    luts: int
    ffs: int
    logic_cells: int

    # -- BRAM --
    bram_type: str
    bram_blocks: int
    bram_bits_per_block: int
    bram_data_bits_per_block: int
    bram_total_kbits: int

    # -- DSP --
    dsp_type: str
    dsp_slices: int
    dsp_multiplier_width: int
    dsp_multipliers_18x18: int
    dsp_max_frequency_mhz: float

    # -- clocking --
    plls: int

    # -- I/O --
    user_ios: int
    io_banks: int

    # -- SERDES --
    serdes_channels: int
    serdes_max_rate_gbps: float

    # -- PCIe --
    pcie_gen: int
    pcie_lanes: int
    pcie_link_rate_gts: float
    pcie_theoretical_bw_bytes_s: float
    pcie_practical_bw_bytes_s: float
    pcie_practical_efficiency: float
    pcie_dma_latency_s: float
    pcie_max_payload_bytes: int
    pcie_max_read_request_bytes: int

    # -- power --
    static_power_w: float
    typical_dynamic_power_w: float
    typical_total_power_w: float

    # -- synthesis toolchain --
    synth_toolchain: str
    synth_nextpnr_binary: str
    synth_device_flag: str
    synth_package_flag: str
    synth_yosys_target: str
    synth_resource_types: dict[str, str]
    synth_typical_fmax_mhz: float

    # ---------------------------------------------------------------
    # Derived helpers
    # ---------------------------------------------------------------

    @property
    def bram_block_bytes(self) -> int:
        """Data bytes per BRAM block (excluding parity)."""
        return self.bram_data_bits_per_block // 8

    @property
    def bram_total_bytes(self) -> int:
        """Total data BRAM capacity in bytes."""
        return self.bram_blocks * self.bram_block_bytes

    def pcie_xfer_s(self, nbytes: int) -> float:
        """One-direction PCIe transfer time: DMA latency + bytes / bandwidth."""
        return self.pcie_dma_latency_s + nbytes / self.pcie_practical_bw_bytes_s

    def pcie_label(self) -> str:
        """Human-readable PCIe config string for reports."""
        bw_gbs = self.pcie_practical_bw_bytes_s / 1e9
        lat_us = self.pcie_dma_latency_s * 1e6
        return (
            f"PCIe Gen{self.pcie_gen} x{self.pcie_lanes} — "
            f"{bw_gbs:.2f} GB/s practical, "
            f"{lat_us:.1f} µs per-direction DMA latency"
        )

    def fpga_target_label(self) -> str:
        """Human-readable FPGA target string (e.g. 'Lattice ECP5 45K-CABGA381')."""
        return self.name

    def bram_blocks_for_bits(self, total_bits: int) -> int:
        """Number of BRAM blocks needed to store *total_bits* of data."""
        import math
        return math.ceil(total_bits / self.bram_data_bits_per_block)

    def __repr__(self) -> str:
        return f"FPGACard({self.name!r})"


def _parse_card(data: dict) -> FPGACard:
    """Build an FPGACard from parsed JSON data."""
    ident = data["identity"]
    fab = data["fabric"]
    bram = data["bram"]
    dsp = data["dsp"]
    clk = data["clocking"]
    io = data["io"]
    serdes = data["serdes"]
    pcie = data["pcie"]
    pwr = data["power"]
    synth = data["synthesis"]

    return FPGACard(
        name=data["name"],
        family=ident["family"],
        part_number=ident["part_number"],
        device=ident["device"],
        package=ident["package"],
        technology_nm=ident["technology_nm"],
        datasheet=ident["datasheet"],
        datasheet_url=ident["datasheet_url"],

        luts=fab["luts"],
        ffs=fab["ffs"],
        logic_cells=fab["logic_cells"],

        bram_type=bram["type"],
        bram_blocks=bram["blocks"],
        bram_bits_per_block=bram["bits_per_block"],
        bram_data_bits_per_block=bram["data_bits_per_block"],
        bram_total_kbits=bram["total_kbits"],

        dsp_type=dsp["type"],
        dsp_slices=dsp["slices"],
        dsp_multiplier_width=dsp["multiplier_width"],
        dsp_multipliers_18x18=dsp["multipliers_18x18"],
        dsp_max_frequency_mhz=dsp["max_frequency_mhz"],

        plls=clk["plls"],

        user_ios=io["user_ios"],
        io_banks=io["io_banks"],

        serdes_channels=serdes["channels"],
        serdes_max_rate_gbps=serdes["max_rate_gbps"],

        pcie_gen=pcie["gen"],
        pcie_lanes=pcie["lanes"],
        pcie_link_rate_gts=pcie["link_rate_gts"],
        pcie_theoretical_bw_bytes_s=pcie["theoretical_bw_bytes_s"],
        pcie_practical_bw_bytes_s=pcie["practical_bw_bytes_s"],
        pcie_practical_efficiency=pcie["practical_efficiency"],
        pcie_dma_latency_s=pcie["dma_latency_s"],
        pcie_max_payload_bytes=pcie["max_payload_bytes"],
        pcie_max_read_request_bytes=pcie["max_read_request_bytes"],

        static_power_w=pwr["static_power_w"],
        typical_dynamic_power_w=pwr["typical_dynamic_power_w"],
        typical_total_power_w=pwr["typical_total_power_w"],

        synth_toolchain=synth["toolchain"],
        synth_nextpnr_binary=synth["nextpnr_binary"],
        synth_device_flag=synth["device_flag"],
        synth_package_flag=synth["package_flag"],
        synth_yosys_target=synth["yosys_target"],
        synth_resource_types=synth["resource_types"],
        synth_typical_fmax_mhz=synth["typical_fmax_mhz"],
    )


def load_card(card_name: str | None = None) -> FPGACard:
    """Load an FPGA card by name (filename stem without ``.json``).

    Parameters
    ----------
    card_name : str or None
        Name of the card file (e.g. ``"lattice_ecp5_45k_cabga381"``).
        If *None*, the default card is used.

    Returns
    -------
    FPGACard
    """
    card_name = card_name or DEFAULT_CARD
    path = _CARDS_DIR / f"{card_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"FPGA card {card_name!r} not found at {path}. "
            f"Available cards: {list_cards()}"
        )
    with open(path) as f:
        data = json.load(f)
    return _parse_card(data)


def load_card_from_path(path: str | Path) -> FPGACard:
    """Load an FPGA card from an arbitrary JSON file path.

    Parameters
    ----------
    path : str or Path
        Path to the card JSON file.

    Returns
    -------
    FPGACard
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return _parse_card(data)


def list_cards() -> list[str]:
    """Return names of all available FPGA cards."""
    if not _CARDS_DIR.exists():
        return []
    return sorted(p.stem for p in _CARDS_DIR.glob("*.json"))
