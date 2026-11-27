#!/usr/bin/env python3
"""Checks in SASS that PDL kernels read nothing an earlier kernel writes before pdl_wait.

usage: scripts/pdl_audit.py [--kernels DIR] [--arch sm_120a] [--verbose] [KERNEL...]

Run it in the pixi environment from the workspace root (`pixi run python3
scripts/pdl_audit.py`). It compiles every `.cu` file in the kernel directory with the
flags build.rs uses, disassembles the cubins with cuobjdump (from PATH, or else through
`pixi exec`), and checks each kernel that contains a `griddepcontrol.wait` (SASS
`ACQBULK`):

1. Pre-wait loads. Every global load (LDG, LD, LDGSTS, atomics) that can execute before
   the wait must take its address from a read-only parameter (see READ_ONLY below).
   A load counts as pre-wait when some path from the kernel entry reaches it without
   passing a wait. A conditional branch around a wait counts as waiting: the kernels
   that wait under a flag (`if (pdl) pdl_wait();`) get that flag set exactly when they
   are launched with PDL.
2. Non-coherent loads. No `.CONSTANT` (ld.global.nc) load anywhere in the kernel may
   take its address from a parameter that some kernel writes, since ptxas may move such
   loads above the wait.

The address of each load is traced back through the instructions before it to the
kernel parameters (constant bank 0 offsets) it derives from. Kernels without a wait are
listed; they must not be launched with PDL unless they read only read-only data.

Exits with status 1 if any kernel fails a check.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# Parameters that no kernel writes during a forward pass: weights, scales and other
# tensors loaded at startup. Matched by exact name; per-kernel additions below. Pointers
# into the same buffer under other names are fine as long as no kernel writes them.
READ_ONLY = {
    "weight", "scales", "global_scale", "router", "bias", "norm_weight", "conv_weight",
    "q_scale", "r_scale", "k_weight", "v_weight", "k_static", "v_static", "table",
    "w13", "s13", "g13", "w2", "s2", "g2", "forget", "gain",
    # Depth attention: learned queries (copied from the weights at load) and the RMS
    # norm weights of the destinations.
    "queries", "weight0", "weight1", "weight2", "weight3",
    # DSpark draft: the Markov head's state weight and the LM-head fold, computed once at
    # load.
    "rnn_state", "fold_weight",
    # Message gates: one weight set per slot.
    *(f"{n}{i}" for n in ("norm_weight", "forget", "gain", "down_weight", "forget_scale",
                          "gain_scale", "down_scale") for i in range(3)),
}
READ_ONLY_PER_KERNEL: dict[str, set[str]] = {
    # The FP8 gate's down projection weights; `down` there is the output.
    "message_gate_step_fp8_2304": {"down0", "down1", "down2"},
    # The draft's FP8 copy of the fold, its row scales and error bounds, made at load.
    "draft_markov_bound": {"q", "scale", "err"},
    # The host uploads the position before the step; no kernel writes it.
    "decode_step_bf16": {"position"},
    "decode_step_fp8": {"position"},
    "decode_step_fp4": {"position"},
    "attention_post_bf16": {"position"},
    # The embedding norm and the first layer's input norm.
    "decode_prologue_2304": {"embed_norm", "input_norm"},
    # The draft's block input kernel preloads embed_norm / mask_embed and reads embed.
    "draft_block_input": {"embed", "embed_norm", "mask_embed"},
    # The multi-row attention copies its first cache tiles before the wait: the host
    # uploads the start before the pass, and only the kernel's own combining blocks
    # append to the cache, after the wait.
    **{f"attention_rows_{f}_{nt}": {"k_cache", "v_cache", "start"}
       for f in ("bf16", "fp8", "fp4") for nt in (1, 2, 3)},
    **{f"attention_rows_tree_{f}_{nt}": {"k_cache", "v_cache", "start"}
       for f in ("bf16", "fp8", "fp4") for nt in (1, 2, 3)},
    # A batched step's state tables and positions: the host writes them before the step.
    **{f"attention_batch_{f}": {"states", "positions"} for f in ("bf16", "fp8", "fp4")},
    **{f"attention_batch_nosplit_{f}_{nt}": {"states", "positions"}
       for f in ("bf16", "fp8", "fp4") for nt in (1, 2, 3)},
    "attention_post_rows_bf16": {"states", "positions"},
    # Ragged packed prefill: host uploads the per-visit state pointer table, slice metadata,
    # and MoE tile descriptors before the pass; no kernel writes them.
    "attention_post_rows_packed_bf16": {"states", "slices"},
    "short_conv_residual_rows_packed_2304": {"states", "slices"},
    "moe_conv_residual_rows_packed_2304": {"states", "slices", "tiles"},
    "shift_visit_states_packed_2304": {"states", "slices"},
}

# Tables of device addresses, per kernel, that the host writes before a pass: a load whose
# address comes from one of them reads a buffer the table names. The batched attention
# copies its first cache tiles before the wait, from each row's caches in its state table,
# as attention_rows does from k_cache and v_cache.
ADDRESS_TABLES: dict[str, set[str]] = {
    **{f"attention_batch_{f}": {"states"} for f in ("bf16", "fp8", "fp4")},
    **{f"attention_batch_nosplit_{f}_{nt}": {"states"}
       for f in ("bf16", "fp8", "fp4") for nt in (1, 2, 3)},
}

# Pointer fields of struct parameters, by (kernel file, type name): the stride of the
# segment array and the byte offset of each pointer field within one segment. A register
# indexed constant load (c[0x0][R2+0x388]) picks a segment at run time, so fields are
# named by offset modulo the stride.
STRUCT_FIELDS = {
    ("linear.cu", "Batch"): (32, 256, {0: "seg.w", 8: "seg.x", 16: "seg.out"}),
    ("linear_fp8.cu", "Batch"): (40, 320, {0: "seg.w", 8: "seg.scale", 16: "seg.x", 24: "seg.out"}),
    ("linear_nvfp4.cu", "Batch"): (
        48, 384, {0: "seg.w", 8: "seg.scale", 16: "seg.outer", 24: "seg.x", 32: "seg.out"},
    ),
    ("linear_rows.cu", "Batch"): (40, 320, {0: "seg.w", 8: "seg.scale", 16: "seg.x", 24: "seg.out"}),
    ("linear_nvfp4_rows.cu", "Batch"): (
        56, 448, {0: "seg.w", 8: "seg.scale", 16: "seg.outer", 24: "seg.x", 32: "seg.out"},
    ),
    # message_gate_rows_2304: one gate_rows::Gate per gate slot.
    ("misc.cu", "GateRowsArgs"): (88, 264, {
        0: "g.down", 8: "g.down_scale", 16: "g.forget", 24: "g.gain", 32: "g.x_state",
        40: "g.x_message", 48: "g.h", 56: "g.probs", 64: "g.norm_weight", 72: "g.normed",
        80: "g.out"}),
    # message_gate_out_rows_3_2304 & message_gate_norm_rows_3_2304: 3 batched attention-side slots.
    ("misc.cu", "GateOut3Args"): (48, 144, {
        0: "g.down_act", 8: "g.forget", 16: "g.gain", 24: "g.h", 32: "g.probs", 40: "g.out"}),
    ("misc.cu", "GateNorm3Args"): (40, 120, {
        0: "g.out", 8: "g.norm_weight", 16: "g.normed", 24: "g.q_values", 32: "g.q_scales"}),
}
READ_ONLY |= {"batch.seg.w", "batch.seg.scale", "batch.seg.outer", "outer"}
READ_ONLY |= {f"args.g.{n}" for n in ("down", "down_scale", "forget", "gain", "norm_weight")}

# The kernel parameters start at this constant bank 0 offset on sm_90 and later. The
# script reads the real base from EIATTR_PARAM_CBANK when present.
DEFAULT_PARAM_BASE = 0x380

LOAD_OPS = ("LDG", "LD", "LDGSTS", "ATOMG", "ATOM", "RED", "REDG", "LDGMC", "UBLKCP", "UTMALDG")
WAIT_OP = "ACQBULK"


@dataclass
class Instr:
    addr: int
    pred: str | None
    op: str
    operands: list[str]
    text: str

    @property
    def base(self) -> str:
        return self.op.split(".")[0]


@dataclass
class Param:
    ordinal: int
    offset: int
    size: int
    name: str = "?"
    pointer: bool = False
    type: str = ""


@dataclass
class Kernel:
    name: str
    source: str
    instrs: list[Instr] = field(default_factory=list)
    params: list[Param] = field(default_factory=list)
    param_base: int = DEFAULT_PARAM_BASE


def run(cmd: list[str], **kw) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def cuobjdump_cmd() -> list[str]:
    if shutil.which("cuobjdump"):
        return ["cuobjdump"]
    version = re.search(r"release (\d+\.\d+)", run(["nvcc", "--version"]))
    spec = f"cuda-cuobjdump={version.group(1)}" if version else "cuda-cuobjdump"
    return ["pixi", "exec", "--spec", spec, "cuobjdump"]


# --- source signatures -------------------------------------------------------------------


def strip_attributes(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        if text.startswith("__attribute__", i):
            j = text.index("(", i)
            depth = 0
            while True:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def split_top(text: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def kernel_signatures(preprocessed: str) -> dict[str, list[tuple[str, bool, str]]]:
    """Maps each extern "C" __global__ function to its (name, is_pointer, type) parameters."""
    sigs = {}
    for m in re.finditer(r'extern\s+"C"', preprocessed):
        end = len(preprocessed)
        for stop in ("{", ";"):
            k = preprocessed.find(stop, m.end())
            if k != -1:
                end = min(end, k)
        header = preprocessed[m.end():end]
        if "global" not in header:
            continue
        header = strip_attributes(header)
        header = re.sub(r"__launch_bounds__\s*\([^)]*\)", "", header)
        open_paren = header.find("(")
        if open_paren == -1:
            continue
        name = re.findall(r"\w+", header[:open_paren])[-1]
        depth, j = 0, open_paren
        while True:
            if header[j] == "(":
                depth += 1
            elif header[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        params = []
        for p in split_top(header[open_paren + 1:j]):
            p = p.strip()
            if not p or p == "void":
                continue
            ident = re.findall(r"\w+", p)[-1]
            params.append((ident, "*" in p, p[: p.rfind(ident)].strip()))
        sigs[name] = params
    return sigs


# --- SASS parsing --------------------------------------------------------------------------

LINE_RE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*/\*")
FUNC_RE = re.compile(r"^\s*Function : (\S+)")


def parse_sass(text: str) -> dict[str, list[Instr]]:
    funcs: dict[str, list[Instr]] = {}
    cur = None
    for line in text.splitlines():
        fm = FUNC_RE.match(line)
        if fm:
            cur = funcs.setdefault(fm.group(1), [])
            continue
        m = LINE_RE.search(line)
        if not m or cur is None:
            continue
        body = m.group(2).strip()
        pred = None
        if body.startswith("@"):
            pred, body = body.split(None, 1)
        parts = body.split(None, 1)
        op = parts[0]
        operands = [o.strip() for o in split_top(parts[1])] if len(parts) > 1 else []
        cur.append(Instr(int(m.group(1), 16), pred, op, operands, m.group(2).strip()))
    return funcs


def parse_params(elf_text: str) -> dict[str, tuple[int, list[Param]]]:
    """Kernel parameters from cuobjdump -elf: EIATTR_KPARAM_INFO and EIATTR_PARAM_CBANK."""
    out: dict[str, tuple[int, list[Param]]] = {}
    cur = None
    expect_cbank = False
    for line in elf_text.splitlines():
        sm = re.match(r"^\.nv\.info\.(\S+)", line)
        if sm:
            cur = sm.group(1)
            out[cur] = (DEFAULT_PARAM_BASE, [])
            continue
        if line.startswith(".") and not line.startswith(".nv.info."):
            cur = None
        if cur is None:
            continue
        if "EIATTR_PARAM_CBANK" in line:
            expect_cbank = True
            continue
        if expect_cbank and "Value:" in line:
            vals = re.findall(r"0x([0-9a-f]+)", line)
            if len(vals) >= 2:
                packed = int(vals[1], 16)
                out[cur] = (packed & 0xFFFF, out[cur][1])
            expect_cbank = False
            continue
        km = re.search(r"Ordinal : 0x([0-9a-f]+)\s+Offset\s*: 0x([0-9a-f]+)\s+Size\s*: 0x([0-9a-f]+)", line)
        if km:
            out[cur][1].append(Param(int(km.group(1), 16), int(km.group(2), 16), int(km.group(3), 16)))
    for name, (base, params) in out.items():
        params.sort(key=lambda p: p.ordinal)
    return out


# --- analysis ------------------------------------------------------------------------------

REG_RE = re.compile(r"(?<![\w.])(U?R)(\d+)(\.64|\.128)?")
CBANK_RE = re.compile(r"c\[0x0\]\[(?:(U?R\d+)\+)?(0x[0-9a-f]+)\]")
IMM_RE = re.compile(r"^-?0x[0-9a-f]+$")


def is_load(ins: Instr) -> bool:
    return ins.base in LOAD_OPS


def is_wait(ins: Instr) -> bool:
    return ins.base == WAIT_OP


def branch_target(ins: Instr) -> int | None:
    for o in reversed(ins.operands):
        m = re.fullmatch(r"`?\(?(0x[0-9a-f]+)\)?", o.strip())
        if m:
            return int(m.group(1), 16)
    return None


def unconditional(ins: Instr) -> bool:
    """No guard predicate, neither as @P nor as a predicate operand (BRA.U !UP0, target)."""
    guarded = ins.pred not in (None, "@PT") or (
        ins.base in ("BRA", "JMP", "EXIT", "RET")
        and any(re.fullmatch(r"!?U?P[0-6]", o) for o in ins.operands)
    )
    return not guarded


def successors(instrs: list[Instr], i: int, index: dict[int, int], assume_pdl: bool) -> list[int]:
    ins = instrs[i]
    nxt = [i + 1] if i + 1 < len(instrs) else []
    base = ins.base
    if base in ("EXIT", "RET", "BPT", "KILL"):
        return [] if unconditional(ins) else nxt
    if base in ("BRA", "JMP", "CALL", "BRX", "JMX"):
        tgt = branch_target(ins)
        if tgt is None:
            raise ValueError(f"indirect branch at {ins.addr:#x}: {ins.text}")
        t = index[tgt]
        if base == "CALL":
            return [t] + nxt
        if unconditional(ins):
            return [t]
        # A conditional branch that skips straight over a wait is taken only without PDL.
        if assume_pdl and t > i and any(is_wait(x) for x in instrs[i + 1:t]) and not any(
            x.base in ("BRA", "EXIT", "RET") for x in instrs[i + 1:t]
        ):
            return nxt
        return [t] + nxt
    return nxt


def pre_wait(instrs: list[Instr]) -> set[int]:
    """Indices of instructions reachable from the entry without passing a wait."""
    index = {ins.addr: k for k, ins in enumerate(instrs)}
    seen = set()
    stack = [0]
    while stack:
        i = stack.pop()
        if i in seen or i >= len(instrs):
            continue
        seen.add(i)
        if is_wait(instrs[i]):
            continue
        stack.extend(successors(instrs, i, index, assume_pdl=True))
    return {i for i in seen if not is_wait(instrs[i])}


def regs_of(operand: str) -> list[str]:
    regs = []
    for m in REG_RE.finditer(operand):
        kind, num, width = m.group(1), int(m.group(2)), m.group(3)
        if kind == "R" and num == 255:
            continue
        n = {".64": 2, ".128": 4}.get(width or "", 1)
        regs.extend(f"{kind}{num + k}" for k in range(n))
    return regs


def dest_regs(ins: Instr) -> list[str]:
    if ins.base == "STL" and ins.operands:
        return [f"L:{ins.operands[0]}"]
    if not ins.operands or ins.base in ("ST", "STG", "STS", "STL", "BRA", "RED", "REDG", "BAR"):
        return []
    if ins.base in ("LDGSTS",):
        return []
    first = ins.operands[0]
    m = REG_RE.match(first.strip())
    if not m:
        return []
    kind, num = m.group(1), int(m.group(2))
    if kind == "R" and num == 255:
        return []
    width = 1
    for suffix, n in ((".128", 4), (".64", 2), ("WIDE", 2)):
        if suffix in ins.op:
            width = n
            break
    if ins.base == "CS2R":
        width = 2
    return [f"{kind}{num + k}" for k in range(width)]


def address_operand(ins: Instr) -> str:
    for o in ins.operands:
        if o.startswith("desc["):
            return o[o.index("]") + 1:]
    for o in ins.operands:
        if o.startswith("["):
            return o
    for o in ins.operands[1:]:
        if "[" in o:
            return o[o.index("["):]
    return ""


@dataclass
class Origin:
    """Where a value comes from: parameters it is computed from directly, parameters whose
    pointed-to data it was loaded through (an index read from memory, say), and anything
    the trace could not follow."""

    params: frozenset[int] = frozenset()
    loaded: frozenset[int] = frozenset()
    unknown: tuple[str, ...] = ()

    def __or__(self, other: Origin) -> Origin:
        return Origin(self.params | other.params, self.loaded | other.loaded,
                      self.unknown + other.unknown)


class Tracer:
    """Traces register values back to the kernel parameters they derive from.

    Follows the control flow graph backwards from a use to every definition that reaches
    it. A guarded definition (@P0 ...) does not hide the ones before it. Spilled registers
    are followed through the stack slot (STL/LDL).
    """

    def __init__(self, k: Kernel):
        self.k = k
        index = {ins.addr: n for n, ins in enumerate(k.instrs)}
        self.preds: list[list[int]] = [[] for _ in k.instrs]
        for n in range(len(k.instrs)):
            for m in successors(k.instrs, n, index, assume_pdl=False):
                self.preds[m].append(n)
        self.dests = [dest_regs(ins) for ins in k.instrs]
        self.reach_memo: dict[tuple[int, str], tuple[list[int], bool]] = {}

    def reaching(self, pos: int, reg: str) -> tuple[list[int], bool]:
        """Definitions of reg that reach instruction pos, and whether the entry does."""
        key = (pos, reg)
        if key in self.reach_memo:
            return self.reach_memo[key]
        found, entry = [], pos == 0
        guard = self.k.instrs[pos].pred
        seen = set()
        stack = list(self.preds[pos])
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if reg in self.dests[n]:
                found.append(n)
                # A definition under the use's own guard covers the use, as in
                # `@!P0 IMAD.WIDE R8, ...; @!P0 LDG R4, [R8.64]`.
                d = self.k.instrs[n]
                if unconditional(d) or (guard is not None and d.pred == guard):
                    continue
            if n == 0:
                entry = True
            stack.extend(self.preds[n])
        self.reach_memo[key] = (found, entry)
        return found, entry

    def index_values(self, pos: int, reg: str, depth: int = 0) -> set[int] | None:
        """The immediate values reg can hold at pos, through MOV and SEL definitions, or
        None if any definition is something else."""
        if depth > 8 or reg.startswith("UR"):
            return None
        found, entry = self.reaching(pos, reg)
        if entry or not found:
            return None
        values: set[int] = set()
        for j in found:
            d = self.k.instrs[j]
            if d.base == "MOV" and len(d.operands) >= 2:
                srcs = [d.operands[1]]
            elif d.base == "SEL" and len(d.operands) >= 3:
                srcs = d.operands[1:3]
            else:
                return None
            for src in srcs:
                if IMM_RE.match(src):
                    values.add(int(src, 16))
                elif (r := regs_of(src)) and len(r) >= 1 and r[0].startswith("R"):
                    sub = self.index_values(j, r[0], depth + 1)
                    if sub is None:
                        return None
                    values |= sub
                else:
                    return None
        return values

    def regs(self, pos: int, regs: list[str]) -> Origin:
        """Origin of the values of regs at instruction pos: a walk over the definitions
        that reach them, their operands' definitions, and so on."""
        params: set[int] = set()
        loaded: set[int] = set()
        unknown: list[str] = []
        seen_defs: set[tuple[int, bool]] = set()
        # Work items: (position, register, whether the value was read from memory).
        work = [(pos, r, False) for r in regs]
        seen_uses: set[tuple[int, str, bool]] = set()
        while work:
            at, reg, via_load = work.pop()
            if (at, reg, via_load) in seen_uses:
                continue
            seen_uses.add((at, reg, via_load))
            found, entry = self.reaching(at, reg)
            if entry and not reg.startswith("L:"):
                unknown.append(f"{reg} may be undefined at {self.k.instrs[at].addr:#x}")
            for j in found:
                if (j, via_load) in seen_defs:
                    continue
                seen_defs.add((j, via_load))
                d = self.k.instrs[j]
                if d.base == "LDL":
                    work.append((j, f"L:{address_operand(d)}", via_load))
                elif d.base == "STL":
                    work.extend((j, r, via_load) for r in regs_of(d.operands[1]))
                elif d.base in ("LDS", "LDSM", "SHFL"):
                    unknown.append(f"loaded by {d.text} at {d.addr:#x}")
                elif is_load(d):
                    work.extend((j, r, True) for r in regs_of(address_operand(d)))
                else:
                    for o in d.operands[1:]:
                        for index, c in CBANK_RE.findall(o):
                            # A register-indexed load, c[0x0][R44+0x380], may pick one of
                            # several parameters; use the index's values when they are
                            # immediates (MOV and SEL chains), else the base offset.
                            steps = self.index_values(j, index) if index else None
                            for step in steps if steps else [0]:
                                off = int(c, 16) + step - self.k.param_base
                                if off >= 0:
                                    (loaded if via_load else params).add(off)
                        if not o.startswith("desc["):
                            work.extend((j, r, via_load) for r in regs_of(o))
        return Origin(frozenset(params), frozenset(loaded), tuple(dict.fromkeys(unknown)))


def param_at(k: Kernel, off: int) -> Param | None:
    for p in k.params:
        if p.offset <= off < p.offset + p.size:
            struct = next((v for (src, t), v in STRUCT_FIELDS.items()
                           if src == k.source and re.search(rf"\b{t}\b", p.type)), None)
            if struct is None:
                return p
            stride, array, fields = struct
            rel = off - p.offset
            name = fields.get(rel % stride) if rel < array else None
            if name is None:
                return Param(p.ordinal, off, 4, f"{p.name}+{rel:#x}", False)
            return Param(p.ordinal * 1000 + rel, off, 8, f"{p.name}.{name}", True)
    return None


def params_of(k: Kernel, offsets: frozenset[int]) -> list[Param]:
    found = {p.name: p for off in offsets if (p := param_at(k, off)) and p.pointer}
    return [found[n] for n in sorted(found)]


@dataclass
class LoadReport:
    ins: Instr
    params: list[Param]
    indexed: list[Param]
    unknown: tuple[str, ...]
    bad: bool


def audit_kernel(k: Kernel) -> tuple[list[LoadReport], list[LoadReport]]:
    ro = READ_ONLY | READ_ONLY_PER_KERNEL.get(k.name, set())
    tables = ADDRESS_TABLES.get(k.name, set())
    early = pre_wait(k.instrs)
    tracer = Tracer(k)
    pre, nc = [], []
    for i, ins in enumerate(k.instrs):
        if not is_load(ins):
            continue
        origin = tracer.regs(i, regs_of(address_operand(ins)))
        pointers = params_of(k, origin.params)
        indexed = params_of(k, origin.loaded)
        written = [p for p in pointers if p.name not in ro]
        if i in early:
            from_table = bool(indexed) and all(p.name in tables for p in indexed)
            bad = bool(written) or bool(origin.unknown) or not (pointers or from_table)
            pre.append(LoadReport(ins, pointers, indexed, origin.unknown, bad))
        elif ".CONSTANT" in ins.op and written:
            nc.append(LoadReport(ins, pointers, indexed, origin.unknown, True))
    return pre, nc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kernels", default="crates/siphon-cuda/kernels")
    ap.add_argument("--arch", default=os.environ.get("SIPHON_CUDA_ARCH", "sm_120a"))
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--params", action="store_true", help="list each kernel's parameters")
    ap.add_argument("only", nargs="*", help="kernel names to check (default: all)")
    args = ap.parse_args()

    kdir = Path(args.kernels)
    cuobjdump = cuobjdump_cmd()
    def load(src: Path, tmp: str) -> list[Kernel]:
        cubin = Path(tmp) / f"{src.stem}.cubin"
        flags = ["-std=c++20", f"-arch={args.arch}", "-I", str(kdir)]
        run(["nvcc", "-cubin", "-O3", "-lineinfo", *flags, "-o", str(cubin), str(src)])
        sigs = kernel_signatures(run(["nvcc", "-E", *flags, str(src)]))
        funcs = parse_sass(run([*cuobjdump, "-sass", str(cubin)]))
        params = parse_params(run([*cuobjdump, "-elf", str(cubin)]))
        out = []
        for name, instrs in funcs.items():
            base, plist = params.get(name, (DEFAULT_PARAM_BASE, []))
            names = sigs.get(name)
            if names is None or len(names) != len(plist):
                print(f"warning: no matching signature for {name}", file=sys.stderr)
                names = [("?", True, "")] * len(plist)
            for p, (pname, ptr, ptype) in zip(plist, names):
                p.name, p.pointer, p.type = pname, ptr, ptype
            out.append(Kernel(name, src.name, instrs, plist, base))
        return out

    kernels: dict[str, Kernel] = {}
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor() as pool:
        for ks in pool.map(lambda src: load(src, tmp), sorted(kdir.glob("*.cu"))):
            kernels.update((k.name, k) for k in ks)

    failed = 0
    no_wait = []
    for name in sorted(kernels, key=lambda n: (kernels[n].source, n)):
        k = kernels[name]
        if args.only and name not in args.only:
            continue
        if not any(is_wait(x) for x in k.instrs):
            no_wait.append(k)
            continue
        if args.params:
            ro = READ_ONLY | READ_ONLY_PER_KERNEL.get(k.name, set())
            print(f"{k.source}:{name}(" + ", ".join(
                (p.name + ("[ro]" if p.name in ro else "")) for p in k.params if p.pointer) + ")")
            continue
        pre, nc = audit_kernel(k)
        bad = [r for r in pre if r.bad] + nc
        verdict = "FAIL" if bad else "ok"
        failed += bool(bad)
        before = sorted({p.name for r in pre for p in r.params})
        print(f"{verdict:4} {k.source}:{name}: {len(pre)} loads before the wait"
              + (f" (from {', '.join(before)})" if before else ""))
        for r in (pre if args.verbose else []) + bad:
            tag = "nc after wait" if r in nc else "before wait"
            src = ", ".join(p.name for p in r.params) or "no parameter"
            extra = f" indexed by {', '.join(p.name for p in r.indexed)}" if r.indexed else ""
            extra += f"; {'; '.join(r.unknown)}" if r.unknown else ""
            mark = "  !" if r.bad else "   "
            print(f"{mark} {tag}: {r.ins.addr:#06x} {r.ins.text}  <- {src}{extra}")
    print(f"\nkernels without a wait (must not read earlier kernels' output when launched "
          f"with PDL): {', '.join(k.name for k in no_wait)}")
    print(f"\n{failed} kernel(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
