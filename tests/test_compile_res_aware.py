"""should_compile resolution awareness — the gate must grow with megapixels WITHOUT moving at all
at the 0.25 MP default.

Why this exists: a real 0.98 MP INT8+compile run on a 32 GB card OOM'd on the first backward at
30.8 GB reserved, because the compile gate was a flat 0.25 MP measurement (20 GB). The fix adds a
resolution term that is algebraically zero at 0.25 MP — these tests pin both halves: the term
engages above 0.25, and the extensively-validated default behaviour is byte-identical.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from fizgig.utils.capabilities import (Capabilities, should_compile,          # noqa: E402
                                       _INT8_COMPILE_PEAK_GB, _NF4_COMPILE_PEAK_GB,
                                       _COMPILE_GB_PER_MP, _HEADROOM_GB)

FAILS = []


def ck(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


CAPS = Capabilities(has_cuda=True, device_name="test", sm=(12, 0),
                    vram_gb=32.0, vram_free_gb=30.0,
                    fp8_matmul=True, int8_matmul_train=True, bitsandbytes=True)
LONG = 100_000  # steps — far past every payback threshold, so only the VRAM gates decide


def sc(kind, vram, mp=None, steps=LONG, swap=0):
    kwargs = {} if mp is None else {"mp": mp}
    return should_compile(steps, kind == "nf4", "bf16" if kind == "int8" else "",
                          swap, vram_gb=vram, caps=CAPS, **kwargs)


# --- 1. the 0.25 MP default is byte-identical to the pre-change behaviour -----------------
# Goldens written against the shipped logic: flat INT8 gate at 21.5 GB, no NF4 gate at all.
for vram, want in ((30.0, True), (21.5, True), (21.4, False), (10.0, False)):
    ok, why = sc("int8", vram)
    ck(f"int8 @0.25MP default, {vram} GB free -> {'compile' if want else 'decline'}",
       ok == want, why)
ok, why = sc("int8", 21.4)
ck("  the decline reason is the flat-peak wording (no MP mention)",
   "peaks near 20 GB and" in why and "MP" not in why, why)
for vram in (30.0, 14.0, 8.0):
    ok, why = sc("nf4", vram)
    ck(f"nf4 @0.25MP default, {vram} GB free -> compile (no VRAM gate, as before)", ok, why)

# mp omitted and mp=0.25 must be the same decision AND the same words — any caller that
# doesn't pass mp keeps exactly the old behaviour.
for kind in ("int8", "nf4"):
    for vram in (30.0, 21.4, 14.0):
        ck(f"{kind} {vram} GB: mp omitted == mp=0.25",
           sc(kind, vram) == sc(kind, vram, mp=0.25))

# --- 2. above 0.25 MP the gate grows ------------------------------------------------------
# The run that motivated this: 0.98 MP, ~30 GB free, INT8 -> must now decline.
ok, why = sc("int8", 30.0, mp=0.98)
ck("int8 @0.98MP, 30 GB free -> DECLINED (the OOM run)", not ok, why)
ck("  reason names the resolution and the fix",
   "0.98 MP" in why and "Target Megapixels" in why, why)
ok, why = sc("int8", 32.5, mp=0.98)
ck("int8 @0.98MP with enough free VRAM still compiles", ok, why)

# NF4 gains a gate ONLY above 0.25 MP (extrapolated slope): a 16 GB card at 1 MP declines,
# a big card at 1 MP still compiles.
ok, why = sc("nf4", 14.0, mp=1.0)
ck("nf4 @1.0MP, 14 GB free -> DECLINED", not ok, why)
ok, why = sc("nf4", 30.0, mp=1.0)
ck("nf4 @1.0MP, 30 GB free -> still compiles", ok, why)

# The thresholds are exactly base + slope*(mp-0.25) + headroom — no hidden constants.
_mp = 0.75
_gate = _INT8_COMPILE_PEAK_GB + _COMPILE_GB_PER_MP * (_mp - 0.25) + _HEADROOM_GB
ck("int8 gate sits exactly at the formula",
   sc("int8", _gate + 0.01, mp=_mp)[0] and not sc("int8", _gate - 0.01, mp=_mp)[0])
_gate = _NF4_COMPILE_PEAK_GB + _COMPILE_GB_PER_MP * (_mp - 0.25) + _HEADROOM_GB
ck("nf4 gate sits exactly at the formula",
   sc("nf4", _gate + 0.01, mp=_mp)[0] and not sc("nf4", _gate - 0.01, mp=_mp)[0])

# --- 2b. batch multiplies tokens exactly as resolution does -------------------------------
# batch 1 must be a pure no-op (same decision AND words); batch 2 at the 0.25 MP default is the
# token load of 0.5 MP and gates accordingly.
for kind in ("int8", "nf4"):
    for vram in (30.0, 21.4, 14.0):
        ck(f"{kind} {vram} GB: batch omitted == batch=1",
           sc(kind, vram, mp=0.25)
           == should_compile(LONG, kind == "nf4", "bf16" if kind == "int8" else "",
                             0, vram_gb=vram, caps=CAPS, mp=0.25, batch=1))
ok, why = should_compile(LONG, False, "bf16", 0, vram_gb=30.0, caps=CAPS, mp=0.25, batch=2)
_b2 = should_compile(LONG, False, "bf16", 0, vram_gb=30.0, caps=CAPS, mp=0.5)
ck("int8 @0.25MP batch 2 gates exactly like 0.5 MP batch 1", ok == _b2[0], why)
ok, why = should_compile(LONG, False, "bf16", 0, vram_gb=22.0, caps=CAPS, mp=0.25, batch=2)
ck("int8 @0.25MP batch 2, 22 GB free -> DECLINED, reason names the batch",
   not ok and "batch 2" in why and "batch size" in why, why)

# --- 2c. the memory strategy prices LoKR's factor -----------------------------------------
# Full-matrix LoKR params scale 1/factor²; factor 4 is ~+4 GB of trainable state the rank-32
# LoRA baseline doesn't carry. Defaults (lora, or lokr at factor >= 16) add exactly zero.
from fizgig.utils.capabilities import (estimate_krea2_peak, _lokr_extra_gb,  # noqa: E402
                                       recommend_krea2_strategy, _NF4_PEAK_GB)

ck("lokr factor 16+ adds nothing", _lokr_extra_gb(16) == 0.0 and _lokr_extra_gb(32) == 0.0)
ck("lokr factor 8 adds ~0.6 GB", abs(_lokr_extra_gb(8) - 0.6) < 0.05, _lokr_extra_gb(8))
ck("lokr factor 4 adds ~4.2 GB", 3.5 < _lokr_extra_gb(4) < 5.0, _lokr_extra_gb(4))
ck("estimate: lora path unchanged by the new params",
   estimate_krea2_peak(16.2, 0.25, 1, 32)
   == estimate_krea2_peak(16.2, 0.25, 1, 32, network_type="lora", lokr_factor=4))
ck("estimate: lokr factor 8 == lora + 0.6",
   abs(estimate_krea2_peak(16.2, 0.25, 1, 32, network_type="lokr", lokr_factor=8)
       - (16.2 + 0.6)) < 0.05)

# The scenario that matters: 16 GB card (~14.5 free), NF4. Factor 8 fits resident;
# factor 4 must NOT be told it fits.
CAPS_NO_INT8 = Capabilities(has_cuda=True, device_name="16gb", sm=(8, 6),
                            vram_gb=16.0, vram_free_gb=14.5,
                            fp8_matmul=False, int8_matmul_train=False, bitsandbytes=True)
p8 = recommend_krea2_strategy(vram_gb=14.5, caps=CAPS_NO_INT8,
                              network_type="lokr", lokr_factor=8)
ck("16 GB + LoKR factor 8 -> NF4 no swap still fits", p8.quant_4bit and p8.blocks_to_swap == 0,
   p8.reason)
p4 = recommend_krea2_strategy(vram_gb=14.5, caps=CAPS_NO_INT8,
                              network_type="lokr", lokr_factor=4)
ck("16 GB + LoKR factor 4 -> NF4-resident no longer claimed to fit",
   not (p4.quant_4bit and p4.blocks_to_swap == 0), p4.reason)

# --- 3. the earlier gates still fire first, untouched by mp -------------------------------
ok, why = sc("int8", 30.0, mp=2.0, swap=8)
ck("block swap still declines before any VRAM math", not ok and "block swap" in why, why)
ok, why = should_compile(LONG, False, "", 0, vram_gb=30.0, caps=CAPS, mp=2.0)
ck("fp8/bf16 still declines regardless of mp", not ok and "quantised paths" in why, why)
ok, why = sc("int8", 30.0, mp=0.25, steps=100)
ck("short runs still decline on payback at the default", not ok and "too short" in why, why)

# --- 4. no C compiler -> decline, never crash (the RunPod InductorError) ------------------
# The pod image ships no toolchain; inductor/triton need a host cc for their runtime stubs, so
# compile-on there died with "Failed to find C compiler" 90 s into the run. The gate must turn
# that into a clean decline. Windows is exempt: cl.exe hides outside PATH and the trainer's
# vcvars import finds it later — a PATH check here would false-negative every Windows box.
from unittest import mock  # noqa: E402
import fizgig.utils.capabilities as _caps_mod  # noqa: E402
from fizgig.utils.capabilities import has_host_c_compiler  # noqa: E402

with mock.patch.object(_caps_mod.shutil, "which", return_value=None):
    ck("posix, no cc/gcc/clang -> no compiler", not has_host_c_compiler(platform="posix"))
    ck("windows exempt even with empty PATH", has_host_c_compiler(platform="nt"))
with mock.patch.object(_caps_mod.shutil, "which",
                       side_effect=lambda c: "/usr/bin/gcc" if c == "gcc" else None):
    ck("posix with gcc on PATH -> ok", has_host_c_compiler(platform="posix"))

with mock.patch.object(_caps_mod, "has_host_c_compiler", return_value=False):
    ok, why = sc("int8", 30.0)  # would compile on every other gate
    ck("should_compile declines when no compiler, with the apt hint",
       not ok and "C compiler" in why and "apt install gcc" in why, why)
with mock.patch.object(_caps_mod, "has_host_c_compiler", return_value=True):
    ok, why = sc("int8", 30.0)
    ck("compiler present -> the decision is unchanged", ok, why)

# The trainer's forced-on path routes through the same check (POSIX branch of
# _find_host_compiler) — pin the wiring so a refactor can't silently drop it.
import inspect  # noqa: E402
from fizgig.krea2.trainer import _find_host_compiler, _compile_blocks  # noqa: E402
ck("_find_host_compiler consults has_host_c_compiler on posix",
   "has_host_c_compiler" in inspect.getsource(_find_host_compiler))
ck("_compile_blocks gates on _find_host_compiler",
   "_find_host_compiler()" in inspect.getsource(_compile_blocks))

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
