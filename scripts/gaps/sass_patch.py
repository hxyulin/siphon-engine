"""Read and patch the scheduling control bits of SASS instructions in an sm_120a cubin.

usage:
  sass_patch.py dump <in.cubin> <kernel>                 # control fields per instruction
  sass_patch.py patch <in.cubin> <kernel> <out.cubin> <field>=<value> [--range A:B]
  sass_patch.py patch <in.cubin> <kernel> <out.cubin> <field>>=<value> ...  # only raise
  sass_patch.py copy <in.cubin> <out.cubin>              # byte-for-byte rewrite
  sass_patch.py nullify <in.cubin> <out.cubin> <prefix>... # set sh_type 0 on matching sections

Volta and later GPUs encode each instruction in 128 bits. The control bits sit in the top
23 bits of the high 64-bit word, as documented for Volta and Turing by Jia et al.
("Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking", 2018) and used by
CuAssembler:
  bits 41-44 stall cycles, 45 yield, 46-48 write barrier (7 = none),
  49-51 read barrier (7 = none), 52-57 wait mask, 58-61 operand reuse flags.
sm_120 keeps the barrier and wait fields there, but nvdisasm 13.0 decodes stall and yield
through per-opcode tables: a stall of 12 to 15 with bit 45 set is undefined.

What runs depends on the ptxas that built the cubin. sm_100 and sm_120 cubins also carry
a `.nv.capmerc.text.<kernel>` section (capsule Mercury). Under driver 620.73, a cubin from
ptxas 13.0 is finalized again from that section at load time, so edits to `.text` have
no effect; with that section cleared, `.text` runs but any edit to it fails to load.
Cubins from ptxas 13.3 and 13.5 run their `.text` as written, so patches take effect.
`patch` sets one field on the instructions of one kernel's .text section (all of them, or
the index range A:B) and leaves every other byte alone, so the ELF stays valid. With
`field>=value` it only raises a field, which for stall counts keeps the schedule safe.
"""

import struct
import sys

FIELDS = {
    "stall": (41, 4),
    "yield": (45, 1),
    "wbar": (46, 3),
    "rbar": (49, 3),
    "wait": (52, 6),
    "reuse": (58, 4),
}


def headers(data):
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    stroff = struct.unpack_from("<Q", data, shoff + shstrndx * shentsize + 0x18)[0]
    for i in range(shnum):
        off = shoff + i * shentsize
        name = struct.unpack_from("<I", data, off)[0]
        end = data.index(b"\0", stroff + name)
        yield data[stroff + name : end].decode(), off


def sections(data):
    # ELF64 little-endian section headers.
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    hdrs = []
    for i in range(shnum):
        off = shoff + i * shentsize
        name, typ, flags, addr, offset, size = struct.unpack_from("<IIQQQQ", data, off)
        hdrs.append((name, offset, size))
    stroff = hdrs[shstrndx][1]
    out = {}
    for name, offset, size in hdrs:
        end = data.index(b"\0", stroff + name)
        out[data[stroff + name : end].decode()] = (offset, size)
    return out


def text(data, kernel):
    secs = sections(data)
    key = ".text." + kernel
    if key not in secs:
        names = [s[6:] for s in secs if s.startswith(".text.")]
        sys.exit(f"no {key}; kernels: {names}")
    return secs[key]


def fields(hi):
    return {f: (hi >> lo) & ((1 << w) - 1) for f, (lo, w) in FIELDS.items()}


def main():
    cmd = sys.argv[1]
    if cmd == "copy":
        data = open(sys.argv[2], "rb").read()
        open(sys.argv[3], "wb").write(data)
        return
    if cmd == "nullify":
        data = bytearray(open(sys.argv[2], "rb").read())
        hit = 0
        for name, off in headers(data):
            if any(name.startswith(p) for p in sys.argv[4:]):
                struct.pack_into("<I", data, off + 4, 0)
                hit += 1
        open(sys.argv[3], "wb").write(data)
        print(f"nullified {hit} sections")
        return
    data = bytearray(open(sys.argv[2], "rb").read())
    off, size = text(data, sys.argv[3])
    n = size // 16
    if cmd == "dump":
        for i in range(n):
            lo, hi = struct.unpack_from("<QQ", data, off + 16 * i)
            f = fields(hi)
            print(
                f"{i * 16:5x} stall {f['stall']:2d} y {f['yield']} wb {f['wbar']} rb {f['rbar']} "
                f"wait {f['wait']:06b} reuse {f['reuse']:04b}  {hi:016x}{lo:016x}"
            )
        return
    if cmd == "patch":
        out = sys.argv[4]
        spec = sys.argv[5]
        at_least = ">=" in spec
        field, value = spec.split(">=" if at_least else "=")
        value = int(value)
        a, b = 0, n
        if "--range" in sys.argv:
            a, b = (int(x) for x in sys.argv[sys.argv.index("--range") + 1].split(":"))
        bit, width = FIELDS[field]
        mask = ((1 << width) - 1) << bit
        changed = 0
        for i in range(a, min(b, n)):
            lo, hi = struct.unpack_from("<QQ", data, off + 16 * i)
            old = (hi & mask) >> bit
            new = (hi & ~mask) | ((max(old, value) if at_least else value) << bit)
            if new != hi:
                changed += 1
            struct.pack_into("<QQ", data, off + 16 * i, lo, new)
        open(out, "wb").write(data)
        print(f"set {field}={value} on {changed} of {b - a} instructions")
        return
    sys.exit(__doc__)


main()
