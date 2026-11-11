"""
trojan_loop.py
--------------
Python model of the "Heuristic Iterative Development Loop" used to design,
simulate, and optimize three hardware-trojan variants inserted into an AES core.

Each trojan class below mirrors the RTL trigger/payload logic in
  Trojan/Trojan{1,2,3}/aes_core.v
so the trigger conditions can be exercised without a Verilog simulator.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import random

MASK_128  = (1 << 128) - 1
MASK_64   = (1 << 64)  - 1
MASK_32   = (1 << 32)  - 1
MASK_16   = (1 << 16)  - 1
MASK_8    = (1 << 8)   - 1

# ---------------------------------------------------------------------------
# PPA data (synthesised with Yosys on SKY130 PDK – from area_report.txt files)
# ---------------------------------------------------------------------------

@dataclass
class PPAMetrics:
    name: str
    total_area:      float
    sequential_area: float
    timing_slack:    float

    @property
    def area_overhead_pct(self) -> float:
        golden_area = 151_356.412
        return (self.total_area - golden_area) / golden_area * 100.0

GOLDEN = PPAMetrics("Golden (Clean)", 151_356.41, 61_789.26, 80.2326)

PPA_TABLE: List[PPAMetrics] = [
    GOLDEN,
    PPAMetrics("Trojan 1 (Key Exfil – simple trigger)",     152_846.59, 61_814.28, 80.4417),
    PPAMetrics("Trojan 2 (Key Exfil – counter trigger)",    153_146.88, 62_014.48, 80.5315),
    PPAMetrics("Trojan 3 (DoS – 3-stage FSM, obfuscated)", 151_979.51, 61_864.33, 80.1413),
]


# ---------------------------------------------------------------------------
# Trojan 1 – single-cycle key-exfiltration trigger
#   RTL: next && block[15:0] == 16'h6566
#   Payload: output key[255:128] instead of enc_new_block
# ---------------------------------------------------------------------------

class Trojan1:
    """
    Single-condition trigger.
    Trigger:  block bits [15:0] == 0x6566  while 'next' is asserted.
    Payload:  replace ciphertext output with key[255:128].
    Stealth:  low – any block ending in 0x6566 fires it immediately.
    """

    TRIGGER_PATTERN = 0x6566  # block[15:0]

    def __init__(self) -> None:
        self.trojan_active = False

    def reset(self) -> None:
        self.trojan_active = False

    def clock_edge(self, next_sig: bool, block: int) -> None:
        if not next_sig:
            return
        if (block & MASK_16) == self.TRIGGER_PATTERN:
            self.trojan_active = True

    def result(self, enc_new_block: int, key: int) -> int:
        if self.trojan_active:
            return (key >> 128) & MASK_128   # leak key[255:128]
        return enc_new_block

    @staticmethod
    def trigger_probability() -> float:
        return 1 / (2 ** 16)   # ≈ 1.5 × 10^-5  (low stealth)


# ---------------------------------------------------------------------------
# Trojan 2 – counter-based key-exfiltration trigger
#   RTL: count 'next' pulses; when counter reaches 0xFF leak key[31:0]
#   Payload: replace result[31:0] with key[31:0]
#   Stealth: medium – fires after exactly 255 encryptions
# ---------------------------------------------------------------------------

class Trojan2:
    """
    8-bit saturation counter trigger.
    Trigger:  255th consecutive encryption operation.
    Payload:  replace result[31:0] with key[31:0].
    Stealth:  medium – timing-based, hard to catch in short test vectors.
    """

    COUNTER_MAX = 0xFF

    def __init__(self) -> None:
        self.cnt: int  = 0
        self.triggered = False

    def reset(self) -> None:
        self.cnt       = 0
        self.triggered = False

    def clock_edge(self, next_sig: bool, ready: bool) -> None:
        if next_sig and ready and not self.triggered:
            if self.cnt < self.COUNTER_MAX:
                self.cnt += 1
            else:
                self.triggered = True

    def result(self, enc_new_block: int, key: int) -> int:
        upper = enc_new_block & ~MASK_32       # keep bits [127:32]
        lower = (key & MASK_32) if self.triggered else (enc_new_block & MASK_32)
        return upper | lower

    @staticmethod
    def trigger_probability() -> float:
        return 1 / 255   # fires after a fixed number of ops


# ---------------------------------------------------------------------------
# Trojan 3 – 3-stage FSM Denial-of-Service with obfuscated triggers
#
#   Stage 0 → 1:  block matches two 64-bit halves
#   Stage 1 → 2:  XOR of the two 64-bit halves matches a constant
#   Stage 2 → 3:  reduction-AND of (block[127:96] XOR 0xDEADBEEF)
#                  AND block[31:0] == 0xFACEBABE
#
#   Payload: permanently assert dos_active → force ready=0 forever (DoS)
#   Stealth: high – three sequential blocks all required; obfuscated comparisons
# ---------------------------------------------------------------------------

class Trojan3:
    """
    3-stage sequence-triggered Denial-of-Service trojan.

    The three trigger blocks must arrive in order within consecutive 'next'
    pulses.  Any out-of-sequence block resets the FSM back to stage 0.
    """

    # Stage-0 trigger constants
    S0_UPPER = 0x6666_7777_8888_9999   # block[127:64]
    S0_LOWER = 0xAAAA_BBBB_CCCC_DDDD   # block[63:0]

    # Stage-1 trigger: XOR of the two 64-bit halves
    S1_XOR   = 0x0000_0000_FFFF_FFFF

    # Stage-2 trigger constants (obfuscated)
    S2_XOR_MASK = 0xDEAD_BEEF          # block[127:96] XOR mask (reduction-AND must be 1)
    S2_LOWER    = 0xFACE_BABE          # block[31:0]

    def __init__(self) -> None:
        self.state:      int  = 0   # FSM: 0,1,2,3
        self.dos_active: bool = False

    def reset(self) -> None:
        self.state      = 0
        self.dos_active = False

    # --- individual stage match predicates (mirror the Verilog 'wire match_N') ---

    def _match1(self, block: int) -> bool:
        upper = (block >> 64) & MASK_64
        lower =  block        & MASK_64
        return upper == self.S0_UPPER and lower == self.S0_LOWER

    def _match2(self, block: int) -> bool:
        upper = (block >> 64) & MASK_64
        lower =  block        & MASK_64
        return (upper ^ lower) == self.S1_XOR

    def _match3(self, block: int) -> bool:
        seg_96_127 = (block >> 96) & MASK_32
        low_32     =  block        & MASK_32
        # reduction-AND: all bits of (seg XOR mask) must be 1
        xored      = seg_96_127 ^ self.S2_XOR_MASK
        all_ones   = (xored == MASK_32)
        return all_ones and (low_32 == self.S2_LOWER)

    def clock_edge(self, next_sig: bool, ready: bool, block: int) -> None:
        if not (next_sig and ready):
            return
        if self.state == 0:
            self.state = 1 if self._match1(block) else 0
        elif self.state == 1:
            self.state = 2 if self._match2(block) else 0
        elif self.state == 2:
            if self._match3(block):
                self.state      = 3
                self.dos_active = True
            else:
                self.state      = 0
        elif self.state == 3:
            self.dos_active = True  # locked

    def is_ready(self, hw_ready: bool) -> bool:
        return False if self.dos_active else hw_ready

    @staticmethod
    def trigger_probability() -> float:
        # Three independent random blocks must match sequentially
        p1 = 1 / (2 ** 128)   # exact 128-bit match
        p2 = 1 / (2 ** 64)    # XOR constraint (64-bit degrees of freedom)
        p3 = 1 / (2 ** 32)    # 32-bit lower + all-ones XOR reduction
        return p1 * p2 * p3   # astronomically small ≈ 10^-69


# ---------------------------------------------------------------------------
# Iterative design loop
# ---------------------------------------------------------------------------

def run_iteration(
    trojan,
    name: str,
    ppa: PPAMetrics,
    test_blocks: List[int],
    key: int,
) -> dict:
    """
    Simulate one design iteration:
      1. Reset the trojan FSM.
      2. Feed test_blocks through the trojan's clock_edge logic.
      3. Check whether the trojan fired.
      4. Report PPA metrics.
    Returns a summary dict for the iteration report.
    """
    trojan.reset()
    triggered = False
    normal_outputs = []
    trojan_outputs = []

    ready = True   # simplified model – assume core always ready between ops

    for block in test_blocks:
        if isinstance(trojan, Trojan1):
            trojan.clock_edge(next_sig=True, block=block)
        elif isinstance(trojan, Trojan2):
            trojan.clock_edge(next_sig=True, ready=ready)
        elif isinstance(trojan, Trojan3):
            trojan.clock_edge(next_sig=True, ready=ready, block=block)

        # Simulate normal ciphertext (placeholder XOR for illustration)
        enc_out = (block ^ key) & MASK_128

        if isinstance(trojan, Trojan1):
            out = trojan.result(enc_out, key)
            if trojan.trojan_active:
                triggered = True
        elif isinstance(trojan, Trojan2):
            out = trojan.result(enc_out, key)
            if trojan.triggered:
                triggered = True
        elif isinstance(trojan, Trojan3):
            out = enc_out
            if trojan.dos_active:
                triggered = True

        normal_outputs.append(enc_out)
        trojan_outputs.append(out)

    return {
        "name":           name,
        "triggered":      triggered,
        "total_area":     ppa.total_area,
        "area_overhead":  ppa.area_overhead_pct,
        "timing_slack":   ppa.timing_slack,
        "normal_outputs": normal_outputs,
        "trojan_outputs": trojan_outputs,
    }


def generate_trigger_blocks_t3() -> List[int]:
    """Construct the exact 3-block sequence that fires Trojan 3."""
    # Block 1: match1 condition
    b1 = (Trojan3.S0_UPPER << 64) | Trojan3.S0_LOWER

    # Block 2: match2 condition — any block where upper XOR lower == S1_XOR
    # Choose upper = S0_UPPER, then lower = upper XOR S1_XOR
    upper2 = Trojan3.S0_UPPER
    lower2 = upper2 ^ Trojan3.S1_XOR
    b2 = (upper2 << 64) | lower2

    # Block 3: match3 condition
    # block[127:96] XOR 0xDEADBEEF must be all-ones → block[127:96] = 0xDEADBEEF XOR 0xFFFFFFFF = 0x21524110
    seg_96_127 = Trojan3.S2_XOR_MASK ^ MASK_32  # = 0x21524110
    # block[95:32] can be anything
    middle = 0xCAFE_BABE_DEAD_C0DE
    b3 = (seg_96_127 << 96) | (middle << 32) | Trojan3.S2_LOWER
    return [b1, b2, b3]


def print_ppa_table() -> None:
    print("\n" + "=" * 72)
    print("  PPA IMPACT ANALYSIS: Golden vs. Trojan Iterations")
    print("=" * 72)
    fmt = "{:<42} {:>12} {:>10} {:>10}"
    print(fmt.format("Design", "Total Area", "Overhead", "Slack"))
    print("-" * 72)
    for m in PPA_TABLE:
        oh = "0.00%" if m.name.startswith("Golden") else f"+{m.area_overhead_pct:.2f}%"
        print(fmt.format(m.name, f"{m.total_area:,.2f}", oh, f"{m.timing_slack:.4f}"))
    print("=" * 72)


def heuristic_iterative_loop(seed: int = 42) -> None:
    """
    Main entry point: runs all four phases for each trojan iteration,
    printing a report that mirrors the README methodology.
    """
    rng = random.Random(seed)

    key   = rng.getrandbits(256)
    nist_blocks = [rng.getrandbits(128) for _ in range(10)]  # clean NIST-like vectors

    trojans = [
        (Trojan1(), "Trojan 1", PPA_TABLE[1]),
        (Trojan2(), "Trojan 2", PPA_TABLE[2]),
        (Trojan3(), "Trojan 3", PPA_TABLE[3]),
    ]

    print()
    print("=" * 72)
    print("  HEURISTIC ITERATIVE HARDWARE TROJAN DEVELOPMENT LOOP")
    print("  AES Core (aes_core.v)  —  Multi-Stage Sequence-Triggered DoS")
    print("=" * 72)

    for iteration, (trojan, name, ppa) in enumerate(trojans, start=1):
        print(f"\n{'─' * 72}")
        print(f"  ITERATION {iteration}: {name}")
        print(f"{'─' * 72}")

        # Phase 1 – Conceptualization
        print(f"\n  [Phase 1] Conceptualization")
        if isinstance(trojan, Trojan1):
            print("    Type    : Single-condition key-exfiltration trigger")
            print("    Trigger : block[15:0] == 0x6566  while next=1")
            print("    Payload : Output key[255:128] instead of ciphertext")
        elif isinstance(trojan, Trojan2):
            print("    Type    : Counter-based key-exfiltration trigger")
            print("    Trigger : 255th consecutive encryption operation")
            print("    Payload : Replace result[31:0] with key[31:0]")
        elif isinstance(trojan, Trojan3):
            print("    Type    : 3-stage FSM Denial-of-Service (obfuscated)")
            print("    Trigger : 3 specific blocks in sequence")
            print("    Payload : Permanently lock ready=0 (core unusable)")

        # Phase 2 – Generate trigger test blocks
        print(f"\n  [Phase 2] Logic Integration / Trigger Block Generation")
        if isinstance(trojan, Trojan3):
            test_blocks = generate_trigger_blocks_t3() + nist_blocks
            print(f"    Injected 3 crafted trigger blocks + {len(nist_blocks)} NIST vectors")
        elif isinstance(trojan, Trojan1):
            # Craft one block ending in 0x6566
            craft = (rng.getrandbits(112) << 16) | Trojan1.TRIGGER_PATTERN
            test_blocks = nist_blocks[:5] + [craft] + nist_blocks[5:]
            print(f"    Injected 1 crafted trigger block (block[15:0]=0x6566) among NIST vectors")
        else:  # Trojan2
            # Feed 260 blocks to ensure the counter crosses 255
            test_blocks = [rng.getrandbits(128) for _ in range(260)]
            print(f"    Feeding {len(test_blocks)} random blocks to saturate 8-bit counter")

        # Phase 3 – Simulation
        print(f"\n  [Phase 3] Simulation-Driven Verification")
        result = run_iteration(trojan, name, ppa, test_blocks, key)
        status = "TRIGGERED" if result["triggered"] else "NOT triggered"
        print(f"    Simulation result : {status}")
        print(f"    Trigger prob.     : {trojan.trigger_probability():.2e}")

        if result["triggered"]:
            print("    First differing output sample:")
            for i, (n, t) in enumerate(
                zip(result["normal_outputs"], result["trojan_outputs"])
            ):
                if n != t:
                    print(f"      block[{i}]  normal : {n:#034x}")
                    print(f"      block[{i}]  trojan : {t:#034x}")
                    break

        # Phase 4 – PPA feedback
        print(f"\n  [Phase 4] PPA Feedback & Optimization Decision")
        print(f"    Total area        : {ppa.total_area:>12,.2f}  ({ppa.area_overhead_pct:+.2f}% overhead)")
        print(f"    Sequential area   : {ppa.sequential_area:>12,.2f}")
        print(f"    Timing slack      : {ppa.timing_slack:.4f} ns  (positive → timing MET)")

        if isinstance(trojan, Trojan1):
            print("    Decision: area overhead +0.98% acceptable but trigger too obvious.")
            print("    → Iterate: switch to counter-based trigger for lower detectability.")
        elif isinstance(trojan, Trojan2):
            print("    Decision: counter adds registers (+sequential area +0.37%).")
            print("    → Iterate: replace with FSM + bitwise-obfuscated constants to cut area.")
        elif isinstance(trojan, Trojan3):
            print("    Decision: overhead reduced to +0.41% — within synthesis-tool variance.")
            print("    → ACCEPT: stealth target achieved. No further iteration required.")

    # Final PPA comparison
    print_ppa_table()

    print("\n  Key Observations:")
    print("  • Trojan 3 area overhead (+0.41%) is indistinguishable from tool variance.")
    print("  • All trojans maintain positive timing slack — operational frequency unaffected.")
    print("  • Bitwise XOR obfuscation (Trojan 3) removed ~573 area units vs. Trojan 2.")
    print("  • 3-stage trigger sequence probability ≈ 10⁻⁶⁹ — safe against NIST vectors.\n")


if __name__ == "__main__":
    heuristic_iterative_loop()
