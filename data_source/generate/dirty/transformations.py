"""
Entry variants in the plant-wide ERP and QMS records.

The monitoring units, presses and the cell's QMS forms use coded fields. The
plant-wide work orders and scrap entries do not: part numbers are typed from
travelers and scrap codes are typed into a notes field half the time. Each
function takes the coded value and returns the value as it was entered.
"""
import random

_rng = random.Random(7)

_PART_FORMATS = [
    lambda p: p,                              # P-12341    canonical
    lambda p: p.replace("-", ""),             # P12341
    lambda p: p.lower(),                      # p-12341
    lambda p: "PN-" + p.split("-")[1],        # PN-12341
    lambda p: p.replace("-", " "),            # P 12341
]
_PART_WEIGHTS = [0.62, 0.18, 0.08, 0.08, 0.04]

DEFECT_VARIANTS = {
    "short_shot":    ["short_shot", "Short Shot", "SHORT", "short", "Non-fill", "non fill", "Short shot"],
    "flash":         ["flash", "Flash", "FLASH", "flashing", "Flsh"],
    "sink":          ["sink", "Sink", "Sink Mark", "sink marks", "SINK"],
    "void":          ["void", "Void", "VOID", "bubble", "Voids"],
    "splay":         ["splay", "Splay", "silver streaks", "Splay/Moisture", "SPLAY"],
    "burn":          ["burn", "Burn", "Burn Mark", "dieseling", "BURN"],
    "weld_line":     ["weld_line", "Weld Line", "knit line", "Knit Line", "WELD"],
    "warp":          ["warp", "Warp", "WARP", "warpage", "Warped"],
    "dimensional":   ["dimensional", "Dimensional", "DIM", "Dim", "out of tol", "OOT"],
    "black_specks":  ["black_specks", "Black Specks", "specks", "BLACK SPECKS", "Blk Spk"],
    "contamination": ["contamination", "Contamination", "Contam", "foreign material", "CONTAM"],
    "gate_vestige":  ["gate_vestige", "Gate Vestige", "high gate", "Gate", "vestige"],
    "other":         ["other", "Other", "OTHER", "See Notes", "", "misc"],
    "none":          ["none", "None", "", "OK", "Pass"],
}


def dirty_part(part: str) -> str:
    return _rng.choices(_PART_FORMATS, weights=_PART_WEIGHTS)[0](part)


def dirty_defect(code: str) -> str:
    variants = DEFECT_VARIANTS.get(code, [code])
    return variants[0] if _rng.random() < 0.5 else _rng.choice(variants)
