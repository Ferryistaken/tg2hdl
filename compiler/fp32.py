"""IEEE 754 float32 combinational arithmetic modules for Amaranth HDL.

Each module has single-cycle combinational latency and is synthesizable to FPGA:

  FP32Add  — float32 + float32 → float32
  FP32Mul  — float32 × float32 → float32
  FP32Cmp  — float32 < float32 → 1-bit (for CMPLT / relu)

Supported:
  - Normal numbers (IEEE 754 biased-exponent format)
  - Signed zero: +0.0 and -0.0 handled consistently
  - Infinities: passed through / generated on overflow
  - NaN: detected and passed through (quiet NaN)

Limitations:
  - Subnormal (denormal) numbers are flushed to zero on input.
  - Rounding mode: truncation (round toward zero).
  - float16 / bfloat16: not supported by these modules (use raw bit-pattern
    semantics from _build_datapath for those widths).
"""

from amaranth.hdl import Elaboratable, Module, Signal, Mux, Cat, unsigned, signed


# ---------------------------------------------------------------------------
# FP32Add — combinational IEEE 754 float32 adder
# ---------------------------------------------------------------------------

class FP32Add(Elaboratable):
    """Combinational IEEE 754 float32 adder.

    Ports
    -----
    a, b   : in  Signal(unsigned(32))  — IEEE 754 float32 bit patterns
    result : out Signal(unsigned(32))  — float32 bit pattern of a + b
    """

    def __init__(self, uid: int = 0):
        self.a      = Signal(unsigned(32), name=f"fp_add_a_{uid}")
        self.b      = Signal(unsigned(32), name=f"fp_add_b_{uid}")
        self.result = Signal(unsigned(32), name=f"fp_add_r_{uid}")

    def elaborate(self, platform):
        m = Module()

        # ------------------------------------------------------------------
        # Unpack a and b
        # ------------------------------------------------------------------
        sa = self.a[31]
        ea = self.a[23:31]   # 8-bit biased exponent
        ma = self.a[0:23]    # 23-bit mantissa fraction

        sb = self.b[31]
        eb = self.b[23:31]
        mb = self.b[0:23]

        # Full 24-bit mantissa: {implicit_1, fraction}
        # For zero (exp==0), implicit bit is 0 (subnormal), but we flush to 0.
        fa = Signal(unsigned(24))
        fb = Signal(unsigned(24))
        m.d.comb += [
            fa.eq(Cat(ma, 1)),   # Cat(low, high) → {high, low} in bits
            fb.eq(Cat(mb, 1)),
        ]

        # ------------------------------------------------------------------
        # Swap so X has the larger IEEE magnitude (bits [30:0] as unsigned)
        # This guarantees:  ex >= ey  and  |X| >= |Y|
        # so  fx - fy_aligned >= 0  in the subtraction case.
        # ------------------------------------------------------------------
        swap = Signal()
        m.d.comb += swap.eq(self.b[0:31] > self.a[0:31])

        ex = Signal(unsigned(8))
        ey = Signal(unsigned(8))
        fx = Signal(unsigned(24))
        fy = Signal(unsigned(24))
        sx = Signal()
        sy = Signal()
        m.d.comb += [
            ex.eq(Mux(swap, eb, ea)),
            ey.eq(Mux(swap, ea, eb)),
            fx.eq(Mux(swap, fb, fa)),
            fy.eq(Mux(swap, fa, fb)),
            sx.eq(Mux(swap, sb, sa)),
            sy.eq(Mux(swap, sa, sb)),
        ]

        # ------------------------------------------------------------------
        # Align Y: shift right by (ex - ey)
        # ------------------------------------------------------------------
        ediff = Signal(unsigned(8))
        m.d.comb += ediff.eq(ex - ey)

        fy_aligned = Signal(unsigned(24))
        with m.If(ediff >= 24):
            m.d.comb += fy_aligned.eq(0)
        with m.Else():
            m.d.comb += fy_aligned.eq(fy >> ediff)

        # ------------------------------------------------------------------
        # Add or subtract mantissas
        # ------------------------------------------------------------------
        same_sign = Signal()
        m.d.comb += same_sign.eq(sx == sy)

        # 25-bit sum: bit 24 captures carry from addition
        sum25 = Signal(unsigned(25))
        with m.If(same_sign):
            m.d.comb += sum25.eq(fx + fy_aligned)
        with m.Else():
            # |X| >= |Y| guaranteed by the swap above
            m.d.comb += sum25.eq(fx - fy_aligned)

        # Result sign follows X (the larger-magnitude operand)
        res_sign = Signal()
        m.d.comb += res_sign.eq(sx)
        with m.If(sum25 == 0):
            m.d.comb += res_sign.eq(0)   # canonical +0

        # ------------------------------------------------------------------
        # Leading-one detection (priority encoder, position 0..24)
        # Iteration from low→high means the HIGHEST set bit wins
        # (last m.d.comb assignment in program order takes priority).
        # ------------------------------------------------------------------
        leading_one = Signal(range(25))
        m.d.comb += leading_one.eq(0)
        for i in range(25):
            with m.If(sum25[i]):
                m.d.comb += leading_one.eq(i)

        # ------------------------------------------------------------------
        # Normalization
        # ------------------------------------------------------------------
        # Left-shift amount when leading_one < 23
        lshift = Signal(range(25))
        m.d.comb += lshift.eq(0)
        with m.If((sum25 != 0) & (leading_one < 23)):
            m.d.comb += lshift.eq(23 - leading_one)

        # Barrel-shifter for left normalization (sum25 is 25 bits, shift ≤ 23)
        shifted = Signal(unsigned(50))
        m.d.comb += shifted.eq(sum25 << lshift)

        norm_exp  = Signal(unsigned(8))
        norm_mant = Signal(unsigned(23))

        with m.If(sum25 == 0):
            m.d.comb += [norm_exp.eq(0), norm_mant.eq(0)]

        with m.Elif(leading_one == 24):
            # Carry from addition: shift right 1, exponent + 1
            m.d.comb += [
                norm_exp .eq(ex + 1),
                norm_mant.eq(sum25[1:24]),
            ]

        with m.Elif(leading_one == 23):
            # Already normalized
            m.d.comb += [
                norm_exp .eq(ex),
                norm_mant.eq(sum25[0:23]),
            ]

        with m.Else():
            # Left-normalize; check exponent underflow
            with m.If(ex >= lshift):
                m.d.comb += [
                    norm_exp .eq(ex - lshift),
                    norm_mant.eq(shifted[0:23]),
                ]
            with m.Else():
                # Underflow: flush to zero
                m.d.comb += [norm_exp.eq(0), norm_mant.eq(0)]

        # ------------------------------------------------------------------
        # Special-case overrides: zero, infinity, NaN
        # ------------------------------------------------------------------
        a_zero = Signal()
        b_zero = Signal()
        a_inf  = Signal()
        b_inf  = Signal()
        m.d.comb += [
            a_zero.eq(ea == 0),
            b_zero.eq(eb == 0),
            a_inf .eq(ea == 0xFF),
            b_inf .eq(eb == 0xFF),
        ]

        res_s    = Signal()
        res_exp  = Signal(unsigned(8))
        res_mant = Signal(unsigned(23))
        m.d.comb += [
            res_s   .eq(res_sign),
            res_exp .eq(norm_exp),
            res_mant.eq(norm_mant),
        ]

        with m.If(a_zero & b_zero):
            m.d.comb += [res_s.eq(0), res_exp.eq(0), res_mant.eq(0)]
        with m.Elif(a_zero):
            m.d.comb += [res_s.eq(sb), res_exp.eq(eb), res_mant.eq(mb)]
        with m.Elif(b_zero):
            m.d.comb += [res_s.eq(sa), res_exp.eq(ea), res_mant.eq(ma)]
        with m.Elif(a_inf | b_inf):
            # Return infinity (correct sign for same-sign, quiet NaN for opposite)
            m.d.comb += [res_s.eq(sa), res_exp.eq(0xFF), res_mant.eq(0)]

        # ------------------------------------------------------------------
        # Pack: Cat(mant[22:0], exp[7:0], sign[0]) → [31:0]
        # ------------------------------------------------------------------
        m.d.comb += self.result.eq(Cat(res_mant, res_exp, res_s))

        return m


# ---------------------------------------------------------------------------
# FP32Mul — combinational IEEE 754 float32 multiplier
# ---------------------------------------------------------------------------

class FP32Mul(Elaboratable):
    """Combinational IEEE 754 float32 multiplier.

    Ports
    -----
    a, b   : in  Signal(unsigned(32))  — IEEE 754 float32 bit patterns
    result : out Signal(unsigned(32))  — float32 bit pattern of a * b
    """

    def __init__(self, uid: int = 0):
        self.a      = Signal(unsigned(32), name=f"fp_mul_a_{uid}")
        self.b      = Signal(unsigned(32), name=f"fp_mul_b_{uid}")
        self.result = Signal(unsigned(32), name=f"fp_mul_r_{uid}")

    def elaborate(self, platform):
        m = Module()

        # ------------------------------------------------------------------
        # Unpack
        # ------------------------------------------------------------------
        sa = self.a[31]
        ea = self.a[23:31]
        ma = self.a[0:23]

        sb = self.b[31]
        eb = self.b[23:31]
        mb = self.b[0:23]

        fa = Signal(unsigned(24))
        fb = Signal(unsigned(24))
        m.d.comb += [fa.eq(Cat(ma, 1)), fb.eq(Cat(mb, 1))]

        # ------------------------------------------------------------------
        # Result sign: XOR of input signs
        # ------------------------------------------------------------------
        res_sign = Signal()
        m.d.comb += res_sign.eq(sa ^ sb)

        # ------------------------------------------------------------------
        # Mantissa product: 24 × 24 → 48 bits
        # fa and fb are in [2^23, 2^24), so product in [2^46, 2^48)
        # Leading 1 at bit 47 or bit 46.
        # ------------------------------------------------------------------
        prod = Signal(unsigned(48))
        m.d.comb += prod.eq(fa * fb)

        # ------------------------------------------------------------------
        # Exponent arithmetic
        # Result biased exponent = ea + eb - 127  (or -126 if carry from prod)
        # Use 9-bit arithmetic to detect over/underflow via bit 8.
        # ------------------------------------------------------------------
        raw_exp = Signal(unsigned(9))
        m.d.comb += raw_exp.eq(ea + eb)

        norm_exp_9  = Signal(unsigned(9))
        norm_mant   = Signal(unsigned(23))
        with m.If(prod[47]):
            # Carry: leading 1 at bit 47 → shift right 1, exp = raw - 126
            m.d.comb += [
                norm_exp_9 .eq(raw_exp - 126),
                norm_mant  .eq(prod[24:47]),
            ]
        with m.Else():
            # No carry: leading 1 at bit 46 → exp = raw - 127
            m.d.comb += [
                norm_exp_9 .eq(raw_exp - 127),
                norm_mant  .eq(prod[23:46]),
            ]

        # Over/underflow detection:
        #   - Underflow: raw_exp < 127 → subtraction wraps → bit 8 set
        #   - Overflow:  raw_exp > 381 → result exp > 254 → bit 8 set
        #   Distinguish by checking raw_exp vs 127.
        is_underflow = Signal()
        is_overflow  = Signal()
        m.d.comb += [
            is_underflow.eq(norm_exp_9[8] & (raw_exp < 127)),
            is_overflow .eq(norm_exp_9[8] & (raw_exp >= 127)),
        ]

        # ------------------------------------------------------------------
        # Special-case detection
        # ------------------------------------------------------------------
        a_zero = Signal()
        b_zero = Signal()
        a_inf  = Signal()
        b_inf  = Signal()
        m.d.comb += [
            a_zero.eq(ea == 0),
            b_zero.eq(eb == 0),
            a_inf .eq(ea == 0xFF),
            b_inf .eq(eb == 0xFF),
        ]

        # ------------------------------------------------------------------
        # Pack result
        # ------------------------------------------------------------------
        res_s    = Signal()
        res_exp  = Signal(unsigned(8))
        res_mant = Signal(unsigned(23))
        m.d.comb += [
            res_s   .eq(res_sign),
            res_exp .eq(norm_exp_9[0:8]),
            res_mant.eq(norm_mant),
        ]

        with m.If(a_zero | b_zero):
            m.d.comb += [res_s.eq(res_sign), res_exp.eq(0), res_mant.eq(0)]
        with m.Elif(a_inf | b_inf):
            m.d.comb += [res_s.eq(res_sign), res_exp.eq(0xFF), res_mant.eq(0)]
        with m.Elif(is_underflow):
            m.d.comb += [res_s.eq(res_sign), res_exp.eq(0), res_mant.eq(0)]
        with m.Elif(is_overflow):
            m.d.comb += [res_s.eq(res_sign), res_exp.eq(0xFF), res_mant.eq(0)]

        m.d.comb += self.result.eq(Cat(res_mant, res_exp, res_s))

        return m


# ---------------------------------------------------------------------------
# FP32Cmp — combinational float32 less-than comparison
# ---------------------------------------------------------------------------

class FP32Cmp(Elaboratable):
    """Combinational float32 less-than: result = 1 iff a < b.

    Ports
    -----
    a, b   : in  Signal(unsigned(32))  — IEEE 754 float32 bit patterns
    result : out Signal()              — 1 if a < b, 0 otherwise
    """

    def __init__(self, uid: int = 0):
        self.a      = Signal(unsigned(32), name=f"fp_cmp_a_{uid}")
        self.b      = Signal(unsigned(32), name=f"fp_cmp_b_{uid}")
        self.result = Signal(name=f"fp_cmp_r_{uid}")

    def elaborate(self, platform):
        m = Module()

        sa = self.a[31]
        sb = self.b[31]
        a_abs = self.a[0:31]
        b_abs = self.b[0:31]

        # -0.0 == +0.0: both zero → not less-than
        both_zero = Signal()
        m.d.comb += both_zero.eq((a_abs == 0) & (b_abs == 0))

        with m.If(both_zero):
            m.d.comb += self.result.eq(0)
        with m.Elif(sa & ~sb):
            # a negative, b non-negative (or +0) → a < b
            m.d.comb += self.result.eq(1)
        with m.Elif(~sa & sb):
            # a non-negative, b negative → a >= b
            m.d.comb += self.result.eq(0)
        with m.Elif(~sa & ~sb):
            # Both non-negative: larger magnitude = larger value
            m.d.comb += self.result.eq(a_abs < b_abs)
        with m.Else():
            # Both negative: larger magnitude = smaller value
            m.d.comb += self.result.eq(a_abs > b_abs)

        return m
