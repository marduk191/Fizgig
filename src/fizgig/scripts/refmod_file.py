"""RefMod Toolkit's file writes for the web UI, in a process of their own (the server stays torch-free): a JSON job on
stdin, one "@@R {json}" line out.

  bake     {meta, strength, retention, curve, copies, out}  one mod with its settings folded in (the desktop's Bake)
  reflora  {lora, refs: [[path, tensor_key]], out, name}     a LoRA and its references in one file
"""
import json
import os
import sys


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.abspath(os.path.join(here, "..", "..")), os.path.abspath(os.path.join(here, "..", "..", ".."))):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        op = json.loads(sys.stdin.read())
        if op["op"] == "bake":
            from safetensors import safe_open
            from fizgig.minimax import refmod_apply as ra
            meta = op["meta"]
            with safe_open(meta["path"], framework="pt", device="cpu") as f:
                z = f.get_tensor(str(meta.get("tensor_key", "latent"))).clone()
            baked, tags, _note = ra.bake(z, meta, strength=float(op["strength"]), retention=float(op["retention"]),
                                         curve=tuple(op["curve"]))
            copies = int(op.get("copies") or 1)
            if copies > 1:
                baked = baked.repeat(1, 1, copies, 1, 1)      # copies = the frames repeated (the pack's own multiplier)
                tags.append(f"studio: copies {copies}")
            res = {"path": ra.save_baked(op["out"], baked, meta, tags)}
        elif op["op"] == "reflora":
            from fizgig.minimax.reflora import save_reflora
            res = save_reflora(op["lora"], [tuple(r) for r in op["refs"]], op["out"], name=op.get("name") or "")
        else:
            res = {"error": "unknown job"}
    except Exception as e:
        res = {"error": str(e) or type(e).__name__}
    sys.stdout.write("@@R " + json.dumps(res, default=str) + "\n")


if __name__ == "__main__":
    main()
