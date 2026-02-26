from amaranth.hdl import Elaboratable, Module, Signal, signed


class ReLU(Elaboratable):
    """Combinational ReLU: output = max(0, input) on signed INT32."""

    def __init__(self, width=32):
        self.width = width
        self.inp = Signal(signed(width))
        self.out = Signal(signed(width))

    def elaborate(self, platform):
        m = Module()
        with m.If(self.inp[-1]):  # negative (MSB set)
            m.d.comb += self.out.eq(0)
        with m.Else():
            m.d.comb += self.out.eq(self.inp)
        return m
