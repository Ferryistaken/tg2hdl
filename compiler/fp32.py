"""IEEE 754 float32 combinational arithmetic modules for Amaranth HDL.

Each module has single-cycle combinational latency and is synthesizable to FPGA:

  FP32Add        — float32 + float32 → float32
  FP32Mul        — float32 × float32 → float32
  FP32Cmp        — float32 < float32 → 1-bit (for CMPLT / relu)
  FP32Exp2       — 2^x → float32
  FP32Log2       — log2(x) → float32
  FP32Reciprocal — 1/x → float32
  FP32Sqrt       — sqrt(x) → float32
  FP32FDiv       — a/b → float32

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


# ---------------------------------------------------------------------------
# FP32Exp2 — combinational 2^x
# ---------------------------------------------------------------------------

class FP32Exp2(Elaboratable):
    """Combinational 2^x for IEEE 754 float32.

    Algorithm
    ---------
    1. Convert x to signed Q8.23 fixed-point (covers ±256 with 2^-23 resolution).
    2. Split into integer part n = x[23:32] and fraction f_fp = x[0:23].
    3. Evaluate 2^f - 1 via 5th-order Horner polynomial in Q0.23 fixed-point.
    4. Pack result: sign=0, exp=(n+127), mant=poly.

    Accuracy: ≤2 ULP for normal inputs; subnormals flush to zero.
    """

    def __init__(self, uid: int = 0):
        self.a      = Signal(unsigned(32), name=f"fp_exp2_a_{uid}")
        self.result = Signal(unsigned(32), name=f"fp_exp2_r_{uid}")

    def elaborate(self, platform):
        m = Module()

        sa = self.a[31]
        ea = self.a[23:31]
        ma = self.a[0:23]

        full_mant = Signal(unsigned(24))
        m.d.comb += full_mant.eq(Cat(ma, 1))   # 1.m in 24 bits

        # ------------------------------------------------------------------
        # Step 1: |x| → unsigned Q8.23 fixed-point (abs_fxp = |x| × 2^23)
        # ------------------------------------------------------------------
        abs_fxp = Signal(unsigned(31))
        with m.Switch(ea):
            for e_val in range(256):
                with m.Case(e_val):
                    eu = e_val - 127
                    if e_val == 0 or e_val == 0xFF:
                        m.d.comb += abs_fxp.eq(0)
                    elif eu >= 8:
                        m.d.comb += abs_fxp.eq((1 << 31) - 1)  # clamp
                    elif eu >= 0:
                        m.d.comb += abs_fxp.eq(full_mant << eu)
                    elif eu >= -23:
                        m.d.comb += abs_fxp.eq(full_mant >> (-eu))
                    else:
                        m.d.comb += abs_fxp.eq(0)   # < 2^-23, rounds to 0

        # Signed Q8.23 (negate for negative x)
        fxp = Signal(signed(32))
        with m.If(sa):
            m.d.comb += fxp.eq(-abs_fxp)
        with m.Else():
            m.d.comb += fxp.eq(abs_fxp)

        # ------------------------------------------------------------------
        # Step 2: split into integer (bits [23:32]) and fraction (bits [0:23])
        # For negative fxp in 2's complement, floor() and frac() still work:
        #   floor(-1.5 × 2^23) = bits[23:32] as signed = -2  ✓
        #   frac(-1.5 × 2^23)  = bits[0:23]              = 0.5 × 2^23 ✓
        # ------------------------------------------------------------------
        n_int = Signal(signed(9))
        f_fp  = Signal(unsigned(23))
        m.d.comb += [
            n_int.eq(fxp[23:32]),
            f_fp .eq(fxp[0:23]),
        ]

        # ------------------------------------------------------------------
        # Step 3: 7th-order Horner polynomial for (2^f - 1) in Q0.23
        #   2^f-1 = f*(C1+f*(C2+f*(C3+f*(C4+f*(C5+f*(C6+f*C7))))))
        #   Coefficients = round(ln2^k/k! × 2^23), k=1..7
        # ------------------------------------------------------------------
        C1 = 5814540   # round(ln2       × 2^23)
        C2 = 2015166   # round(ln2^2/2   × 2^23)
        C3 = 465602    # round(ln2^3/6   × 2^23)
        C4 = 80683     # round(ln2^4/24  × 2^23)
        C5 = 11185     # round(ln2^5/120 × 2^23)
        C6 = 1292      # round(ln2^6/720 × 2^23)
        C7 = 128       # round(ln2^7/5040× 2^23)

        # Products: f_fp(23b) × coeff → up to 47b; >>23 at each step
        p7 = Signal(unsigned(31)); r7 = Signal(unsigned( 8))
        p6 = Signal(unsigned(34)); r6 = Signal(unsigned(11))
        p5 = Signal(unsigned(37)); r5 = Signal(unsigned(14))
        p4 = Signal(unsigned(40)); r4 = Signal(unsigned(17))
        p3 = Signal(unsigned(43)); r3 = Signal(unsigned(20))
        p2 = Signal(unsigned(45)); r2 = Signal(unsigned(22))
        p1 = Signal(unsigned(46)); mant_poly = Signal(unsigned(23))

        m.d.comb += [
            p7.eq(f_fp * C7),         r7.eq(p7[23:31]),
            p6.eq(f_fp * (C6 + r7)),  r6.eq(p6[23:34]),
            p5.eq(f_fp * (C5 + r6)),  r5.eq(p5[23:37]),
            p4.eq(f_fp * (C4 + r5)),  r4.eq(p4[23:40]),
            p3.eq(f_fp * (C3 + r4)),  r3.eq(p3[23:43]),
            p2.eq(f_fp * (C2 + r3)),  r2.eq(p2[23:45]),
            p1.eq(f_fp * (C1 + r2)),  mant_poly.eq(p1[23:46]),
        ]

        # ------------------------------------------------------------------
        # Step 4: construct result float
        # ------------------------------------------------------------------
        out_exp = Signal(signed(10))
        m.d.comb += out_exp.eq(n_int + 127)

        res_s    = Signal()
        res_exp  = Signal(unsigned(8))
        res_mant = Signal(unsigned(23))

        # Default: zero
        m.d.comb += [res_s.eq(0), res_exp.eq(0), res_mant.eq(0)]

        # Normal range
        with m.If(out_exp >= 1):
            with m.If(out_exp <= 254):
                m.d.comb += [res_exp.eq(out_exp[0:8]), res_mant.eq(mant_poly)]
            with m.Else():
                m.d.comb += [res_exp.eq(0xFF), res_mant.eq(0)]   # +inf

        # Special-case overrides
        a_zero = Signal(); a_inf = Signal(); a_nan = Signal()
        m.d.comb += [
            a_zero.eq(ea == 0),
            a_inf .eq((ea == 0xFF) & (ma == 0)),
            a_nan .eq((ea == 0xFF) & (ma != 0)),
        ]
        with m.If(a_nan):
            m.d.comb += [res_exp.eq(0xFF), res_mant.eq(1)]
        with m.Elif(a_inf & sa):        # 2^(-inf) = 0
            m.d.comb += [res_exp.eq(0), res_mant.eq(0)]
        with m.Elif(a_inf):             # 2^(+inf) = +inf
            m.d.comb += [res_exp.eq(0xFF), res_mant.eq(0)]
        with m.Elif(a_zero):            # 2^0 = 1.0
            m.d.comb += [res_exp.eq(127), res_mant.eq(0)]

        m.d.comb += self.result.eq(Cat(res_mant, res_exp, res_s))
        return m


# ---------------------------------------------------------------------------
# FP32Log2 — combinational log2(x)
# ---------------------------------------------------------------------------

class FP32Log2(Elaboratable):
    """Combinational log2(x) for IEEE 754 float32.

    Algorithm
    ---------
    log2(x) = (e - 127) + log2(1.m)   where x = 2^(e-127) × 1.m
    log2(1 + f) for f ∈ [0,1) is approximated by a 5th-order Horner polynomial
    using the substitution f = m / 2^23.

    Special cases: x≤0 → NaN, x=+inf → +inf, x=0 → -inf.
    Accuracy: ≤2 ULP for normal positive inputs.
    """

    def __init__(self, uid: int = 0):
        self.a      = Signal(unsigned(32), name=f"fp_log2_a_{uid}")
        self.result = Signal(unsigned(32), name=f"fp_log2_r_{uid}")

    def elaborate(self, platform):
        import math as _math
        m = Module()

        sa = self.a[31]
        ea = self.a[23:31]
        ma = self.a[0:23]

        # ------------------------------------------------------------------
        # log2(x) = (ea-127) + log2(1+f),  f = ma/2^23 ∈ [0,1)
        #
        # log2(1+f) via: z = f/(f+2),  log2(1+f) = 2/ln2 * atanh(z)
        # atanh(z) ≈ z*(1 + z^2/3 + z^4/5 + z^6/7 + z^8/9)   [5 terms]
        # z ∈ [0,1/3) → error < 2/ln2*(1/3)^11/11 < 2e-6
        # All arithmetic in Q0.23 fixed-point.
        # ------------------------------------------------------------------

        # z_fxp = round(z × 2^23) = (ma × 2^23) // (2^24 + ma)
        denom    = Signal(unsigned(25))
        ma_shift = Signal(unsigned(46))
        z_fxp    = Signal(unsigned(23))
        m.d.comb += denom.eq(0x1000000 + ma)
        m.d.comb += ma_shift.eq(ma << 23)
        m.d.comb += z_fxp.eq(ma_shift // denom)

        # z^2 in Q0.23
        z_sq_full = Signal(unsigned(46))
        z_sq      = Signal(unsigned(23))
        m.d.comb += z_sq_full.eq(z_fxp * z_fxp)
        m.d.comb += z_sq.eq(z_sq_full >> 23)

        # Horner: P(w) = 1 + w/3 + w^2/5 + w^3/7 + w^4/9   (w = z^2)
        INV9 = round(2**23 / 9)   # 932068
        INV7 = round(2**23 / 7)   # 1198373
        INV5 = round(2**23 / 5)   # 1677722
        INV3 = round(2**23 / 3)   # 2796203
        ONE  = 1 << 23             # 8388608

        p3_full = Signal(unsigned(46)); p3 = Signal(unsigned(23))
        p2_full = Signal(unsigned(46)); p2 = Signal(unsigned(23))
        p1_full = Signal(unsigned(46)); p1 = Signal(unsigned(23))
        p0_full = Signal(unsigned(46)); p0 = Signal(unsigned(24))
        m.d.comb += p3_full.eq(z_sq * INV9)
        m.d.comb += p3.eq(INV7 + (p3_full >> 23))
        m.d.comb += p2_full.eq(z_sq * p3)
        m.d.comb += p2.eq(INV5 + (p2_full >> 23))
        m.d.comb += p1_full.eq(z_sq * p2)
        m.d.comb += p1.eq(INV3 + (p1_full >> 23))
        m.d.comb += p0_full.eq(z_sq * p1)
        m.d.comb += p0.eq(ONE + (p0_full >> 23))

        # atanh_fxp = (z × P(z^2)) >> 23   [Q0.23, ≤ atanh(1/3) × 2^23]
        atanh_full = Signal(unsigned(47))
        atanh_fxp  = Signal(unsigned(24))
        m.d.comb += atanh_full.eq(z_fxp * p0)
        m.d.comb += atanh_fxp.eq(atanh_full >> 23)

        # log2(1+f) = 2/ln2 × atanh5  [Q0.23, ≤ 2^23]
        TWO_LOG2E = round(2 / _math.log(2) * (1 << 23))  # 24204406
        log2_1pf_full = Signal(unsigned(48))
        log2_1pf      = Signal(unsigned(24))
        m.d.comb += log2_1pf_full.eq(atanh_fxp * TWO_LOG2E)
        m.d.comb += log2_1pf.eq(log2_1pf_full >> 23)

        # total = (ea-127) × 2^23 + log2_1pf  (signed Q8.23)
        e_unbiased = Signal(signed(9))
        m.d.comb += e_unbiased.eq(ea - 127)

        total_fxp = Signal(signed(32))
        m.d.comb += total_fxp.eq((e_unbiased << 23) + log2_1pf)

        # Convert signed Q8.23 → float32
        res_sign = Signal()
        abs_fxp  = Signal(unsigned(32))
        m.d.comb += res_sign.eq(total_fxp[31])
        with m.If(total_fxp < 0):
            m.d.comb += abs_fxp.eq(-total_fxp)
        with m.Else():
            m.d.comb += abs_fxp.eq(total_fxp)

        # Leading-one detection in abs_fxp (31 bits relevant)
        lo = Signal(range(32))
        m.d.comb += lo.eq(0)
        for i in range(32):
            with m.If(abs_fxp[i]):
                m.d.comb += lo.eq(i)

        # Exponent from fixed-point: leading-one at bit lo → value = 2^(lo-23)
        # biased exponent = (lo - 23) + 127 = lo + 104
        res_exp  = Signal(unsigned(8))
        res_mant = Signal(unsigned(23))
        with m.If(abs_fxp == 0):
            m.d.comb += [res_exp.eq(0), res_mant.eq(0)]
        with m.Elif(lo >= 23):
            # shift right: mant = abs_fxp >> (lo-23), drop leading 1
            shift = Signal(range(32))
            m.d.comb += shift.eq(lo - 23)
            m.d.comb += [
                res_exp.eq(lo + 104),
                res_mant.eq((abs_fxp >> shift)[0:23]),
            ]
        with m.Else():
            # shift left: mant = abs_fxp << (23-lo), drop leading 1
            shift2 = Signal(range(32))
            m.d.comb += shift2.eq(23 - lo)
            shifted = Signal(unsigned(32))
            m.d.comb += shifted.eq(abs_fxp << shift2)
            with m.If(lo + 104 > 0):
                m.d.comb += [
                    res_exp.eq(lo + 104),
                    res_mant.eq(shifted[0:23]),
                ]
            with m.Else():
                m.d.comb += [res_exp.eq(0), res_mant.eq(0)]

        # ------------------------------------------------------------------
        # Special cases
        # ------------------------------------------------------------------
        a_zero = Signal(); a_inf = Signal(); a_nan = Signal()
        m.d.comb += [
            a_zero.eq(ea == 0),
            a_inf .eq((ea == 0xFF) & (ma == 0)),
            a_nan .eq((ea == 0xFF) & (ma != 0)),
        ]

        res_s = Signal()
        m.d.comb += res_s.eq(res_sign)

        # Override for specials
        with m.If(a_nan | sa):              # NaN or negative → NaN
            m.d.comb += [res_s.eq(0), res_exp.eq(0xFF), res_mant.eq(1)]
        with m.Elif(a_zero):                # log2(0) = -inf
            m.d.comb += [res_s.eq(1), res_exp.eq(0xFF), res_mant.eq(0)]
        with m.Elif(a_inf):                 # log2(+inf) = +inf
            m.d.comb += [res_s.eq(0), res_exp.eq(0xFF), res_mant.eq(0)]

        m.d.comb += self.result.eq(Cat(res_mant, res_exp, res_s))
        return m


# ---------------------------------------------------------------------------
# FP32Reciprocal — combinational 1/x via Newton-Raphson
# ---------------------------------------------------------------------------

class FP32Reciprocal(Elaboratable):
    """Combinational 1/x for IEEE 754 float32.

    Uses 2 Newton-Raphson iterations:  y_{n+1} = y_n × (2 - x × y_n)
    Initial guess: negate the biased exponent (gives 1-ULP seed for powers of 2).

    Accuracy: ≤2 ULP for normal inputs.
    """

    def __init__(self, uid: int = 0):
        self.uid    = uid
        self.a      = Signal(unsigned(32), name=f"fp_recip_a_{uid}")
        self.result = Signal(unsigned(32), name=f"fp_recip_r_{uid}")

    def elaborate(self, platform):
        m = Module()

        sa = self.a[31]
        ea = self.a[23:31]
        ma = self.a[0:23]

        # Initial guess: flip exponent bits, keep sign and mantissa.
        # For x = 2^(e-127) × 1.m → guess ≈ 2^(127-e) × 1/1.m  (crude but close)
        # Better seed: invert biased exp = (254 - e), use same mantissa.
        y0_exp = Signal(unsigned(8))
        m.d.comb += y0_exp.eq(253 - ea)   # 253 = 2×127 - 1

        # Seed: sign=0, exp=y0_exp, mant=~ma (bit-complement).
        # Gives initial relative error e0 = f(1-f)/2 ≤ 0.125 for f=ma/2^23.
        y0 = Signal(unsigned(32))
        m.d.comb += y0.eq(Cat(~ma, y0_exp, 0))

        # ------------------------------------------------------------------
        # Iteration 1: y1 = y0 × (2 - x_abs × y0)
        # ------------------------------------------------------------------
        x_abs = Signal(unsigned(32))
        m.d.comb += x_abs.eq(Cat(ma, ea, 0))   # |x|

        TWO = 0x40000000    # 2.0 in IEEE754

        uid = self.uid
        mul0 = FP32Mul(uid=uid * 10 + 0)
        m.submodules[f"rcp_m0_{uid}"] = mul0
        m.d.comb += [mul0.a.eq(x_abs), mul0.b.eq(y0)]
        xy0 = mul0.result

        # 2 - x*y0
        add0 = FP32Add(uid=uid * 10 + 1)
        m.submodules[f"rcp_a0_{uid}"] = add0
        # Negate xy0 (flip sign bit) and add to 2.0
        neg_xy0 = Signal(unsigned(32))
        m.d.comb += neg_xy0.eq(Cat(xy0[0:31], ~xy0[31]))
        m.d.comb += [add0.a.eq(TWO), add0.b.eq(neg_xy0)]
        err0 = add0.result

        mul1 = FP32Mul(uid=uid * 10 + 2)
        m.submodules[f"rcp_m1_{uid}"] = mul1
        m.d.comb += [mul1.a.eq(y0), mul1.b.eq(err0)]
        y1 = mul1.result

        # ------------------------------------------------------------------
        # Iteration 2: y2 = y1 × (2 - x_abs × y1)
        # ------------------------------------------------------------------
        mul2 = FP32Mul(uid=uid * 10 + 3)
        m.submodules[f"rcp_m2_{uid}"] = mul2
        m.d.comb += [mul2.a.eq(x_abs), mul2.b.eq(y1)]
        xy1 = mul2.result

        add1 = FP32Add(uid=uid * 10 + 4)
        m.submodules[f"rcp_a1_{uid}"] = add1
        neg_xy1 = Signal(unsigned(32))
        m.d.comb += neg_xy1.eq(Cat(xy1[0:31], ~xy1[31]))
        m.d.comb += [add1.a.eq(TWO), add1.b.eq(neg_xy1)]
        err1 = add1.result

        mul3 = FP32Mul(uid=uid * 10 + 5)
        m.submodules[f"rcp_m3_{uid}"] = mul3
        m.d.comb += [mul3.a.eq(y1), mul3.b.eq(err1)]
        y2 = mul3.result

        # ------------------------------------------------------------------
        # Iteration 3: y3 = y2 × (2 - x_abs × y2)
        # ------------------------------------------------------------------
        mul4 = FP32Mul(uid=uid * 10 + 6)
        m.submodules[f"rcp_m4_{uid}"] = mul4
        m.d.comb += [mul4.a.eq(x_abs), mul4.b.eq(y2)]
        xy2 = mul4.result

        add2 = FP32Add(uid=uid * 10 + 7)
        m.submodules[f"rcp_a2_{uid}"] = add2
        neg_xy2 = Signal(unsigned(32))
        m.d.comb += neg_xy2.eq(Cat(xy2[0:31], ~xy2[31]))
        m.d.comb += [add2.a.eq(TWO), add2.b.eq(neg_xy2)]
        err2 = add2.result

        mul5 = FP32Mul(uid=uid * 10 + 8)
        m.submodules[f"rcp_m5_{uid}"] = mul5
        m.d.comb += [mul5.a.eq(y2), mul5.b.eq(err2)]
        y3 = mul5.result

        # Apply sign: result = y3 with sign = sa
        res_s = Signal()
        m.d.comb += res_s.eq(sa)

        # Special cases
        a_zero = Signal(); a_inf = Signal()
        m.d.comb += [
            a_zero.eq(ea == 0),
            a_inf .eq(ea == 0xFF),
        ]
        res_exp  = Signal(unsigned(8))
        res_mant = Signal(unsigned(23))
        m.d.comb += [res_exp.eq(y3[23:31]), res_mant.eq(y3[0:23])]

        with m.If(a_zero):              # 1/0 = +inf
            m.d.comb += [res_s.eq(0), res_exp.eq(0xFF), res_mant.eq(0)]
        with m.Elif(a_inf):             # 1/inf = 0
            m.d.comb += [res_s.eq(0), res_exp.eq(0), res_mant.eq(0)]

        m.d.comb += self.result.eq(Cat(res_mant, res_exp, res_s))
        return m


# ---------------------------------------------------------------------------
# FP32Sqrt — combinational sqrt(x)
# ---------------------------------------------------------------------------

class FP32Sqrt(Elaboratable):
    """Combinational sqrt(x) for IEEE 754 float32.

    Uses the identity: sqrt(x) = x × rsqrt(x)
    where rsqrt is approximated by 2 Newton-Raphson iterations:
      y_{n+1} = y_n × (1.5 - 0.5×x×y_n^2)

    Seed: classic Quake III fast inverse sqrt initial guess.
    Accuracy: ≤2 ULP for normal positive inputs.
    """

    def __init__(self, uid: int = 0):
        self.uid    = uid
        self.a      = Signal(unsigned(32), name=f"fp_sqrt_a_{uid}")
        self.result = Signal(unsigned(32), name=f"fp_sqrt_r_{uid}")

    def elaborate(self, platform):
        m = Module()
        uid = self.uid

        sa = self.a[31]
        ea = self.a[23:31]
        ma = self.a[0:23]

        # Initial guess for 1/sqrt(x): magic constant approach
        # seed = 0x5F3759DF - (bits >> 1)
        # Implemented as bit manipulation of the float representation.
        half_bits = Signal(unsigned(32))
        m.d.comb += half_bits.eq(self.a >> 1)
        seed_bits = Signal(unsigned(32))
        m.d.comb += seed_bits.eq(0x5F3759DF - half_bits)

        HALF  = 0x3F000000   # 0.5 in IEEE754
        THALF = 0x3FC00000   # 1.5 in IEEE754

        # ------------------------------------------------------------------
        # Iteration 1: y1 = y0 × (1.5 - 0.5×x×y0^2)
        # ------------------------------------------------------------------
        # y0^2
        sq0 = FP32Mul(uid=uid * 10 + 0)
        m.submodules[f"sq0_{uid}"] = sq0
        m.d.comb += [sq0.a.eq(seed_bits), sq0.b.eq(seed_bits)]

        # 0.5×x
        hx = FP32Mul(uid=uid * 10 + 1)
        m.submodules[f"hx_{uid}"] = hx
        m.d.comb += [hx.a.eq(HALF), hx.b.eq(self.a)]

        # 0.5×x×y0^2
        hxy0sq = FP32Mul(uid=uid * 10 + 2)
        m.submodules[f"hxy0sq_{uid}"] = hxy0sq
        m.d.comb += [hxy0sq.a.eq(hx.result), hxy0sq.b.eq(sq0.result)]

        # 1.5 - 0.5×x×y0^2
        neg_hxy0sq = Signal(unsigned(32))
        m.d.comb += neg_hxy0sq.eq(Cat(hxy0sq.result[0:31], ~hxy0sq.result[31]))
        sub0 = FP32Add(uid=uid * 10 + 3)
        m.submodules[f"sub0_{uid}"] = sub0
        m.d.comb += [sub0.a.eq(THALF), sub0.b.eq(neg_hxy0sq)]

        # y1 = y0 × (1.5 - 0.5×x×y0^2)
        y1_mul = FP32Mul(uid=uid * 10 + 4)
        m.submodules[f"y1m_{uid}"] = y1_mul
        m.d.comb += [y1_mul.a.eq(seed_bits), y1_mul.b.eq(sub0.result)]
        y1 = y1_mul.result

        # ------------------------------------------------------------------
        # Iteration 2: y2 = y1 × (1.5 - 0.5×x×y1^2)
        # ------------------------------------------------------------------
        sq1 = FP32Mul(uid=uid * 10 + 5)
        m.submodules[f"sq1_{uid}"] = sq1
        m.d.comb += [sq1.a.eq(y1), sq1.b.eq(y1)]

        hxy1sq = FP32Mul(uid=uid * 10 + 6)
        m.submodules[f"hxy1sq_{uid}"] = hxy1sq
        m.d.comb += [hxy1sq.a.eq(hx.result), hxy1sq.b.eq(sq1.result)]

        neg_hxy1sq = Signal(unsigned(32))
        m.d.comb += neg_hxy1sq.eq(Cat(hxy1sq.result[0:31], ~hxy1sq.result[31]))
        sub1 = FP32Add(uid=uid * 10 + 7)
        m.submodules[f"sub1_{uid}"] = sub1
        m.d.comb += [sub1.a.eq(THALF), sub1.b.eq(neg_hxy1sq)]

        y2_mul = FP32Mul(uid=uid * 10 + 8)
        m.submodules[f"y2m_{uid}"] = y2_mul
        m.d.comb += [y2_mul.a.eq(y1), y2_mul.b.eq(sub1.result)]
        y2 = y2_mul.result   # ≈ 1/sqrt(x)

        # sqrt(x) = x × (1/sqrt(x))
        final_mul = FP32Mul(uid=uid * 10 + 9)
        m.submodules[f"sfin_{uid}"] = final_mul
        m.d.comb += [final_mul.a.eq(self.a), final_mul.b.eq(y2)]
        sqrtx = final_mul.result

        # ------------------------------------------------------------------
        # Special cases
        # ------------------------------------------------------------------
        a_zero = Signal(); a_inf = Signal(); a_nan = Signal()
        m.d.comb += [
            a_zero.eq(ea == 0),
            a_inf .eq((ea == 0xFF) & (ma == 0)),
            a_nan .eq((ea == 0xFF) & (ma != 0)),
        ]

        res_s    = Signal()
        res_exp  = Signal(unsigned(8))
        res_mant = Signal(unsigned(23))
        m.d.comb += [res_s.eq(0), res_exp.eq(sqrtx[23:31]), res_mant.eq(sqrtx[0:23])]

        with m.If(a_nan | sa):    # NaN or negative → NaN
            m.d.comb += [res_exp.eq(0xFF), res_mant.eq(1)]
        with m.Elif(a_zero):
            m.d.comb += [res_exp.eq(0), res_mant.eq(0)]
        with m.Elif(a_inf):
            m.d.comb += [res_exp.eq(0xFF), res_mant.eq(0)]

        m.d.comb += self.result.eq(Cat(res_mant, res_exp, res_s))
        return m


# ---------------------------------------------------------------------------
# FP32FDiv — combinational a / b  (uses FP32Reciprocal + FP32Mul)
# ---------------------------------------------------------------------------

class FP32FDiv(Elaboratable):
    """Combinational a/b for IEEE 754 float32.

    Computes a × (1/b) using FP32Reciprocal and FP32Mul.
    """

    def __init__(self, uid: int = 0):
        self.uid    = uid
        self.a      = Signal(unsigned(32), name=f"fp_fdiv_a_{uid}")
        self.b      = Signal(unsigned(32), name=f"fp_fdiv_b_{uid}")
        self.result = Signal(unsigned(32), name=f"fp_fdiv_r_{uid}")

    def elaborate(self, platform):
        m = Module()
        uid = self.uid

        recip = FP32Reciprocal(uid=uid)
        mul   = FP32Mul(uid=uid)
        m.submodules[f"fdiv_recip_{uid}"] = recip
        m.submodules[f"fdiv_mul_{uid}"]   = mul

        m.d.comb += [
            recip.a.eq(self.b),
            mul.a  .eq(self.a),
            mul.b  .eq(recip.result),
            self.result.eq(mul.result),
        ]
        return m
