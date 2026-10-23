"""The fork's GPU choice in the web server (server.gpu_env, _chosen_gpu) — headless, no GPU touched.

What is pinned (all [fork] in src/fizgig/web/server.py):
  1. gpu_env exports Preferences' cuda_device as CUDA_VISIBLE_DEVICES; a card NUMBER also gets
     CUDA_DEVICE_ORDER=PCI_BUS_ID (CUDA's default order is fastest-first, nvidia-smi's is bus order, and on this
     machine they disagree: NVML 0 = RTX 3060, CUDA 0 = RTX 5090);
  2. a pin naming no card nvidia-smi lists (pulled, or a prefs.json from another machine) is dropped, not exported
     as a mask that leaves the run with no GPU at all;
  3. that check needs a real nvidia-smi listing: with none, or with only ROCm's synthetic "AMD GPU" row (#188, no
     uuid), the pin is exported untouched, as before;
  4. a CUDA_VISIBLE_DEVICES set where Fizgig was started wins over Preferences;
  5. _chosen_gpu, unpinned, picks the card an unpinned run really lands on (_unpinned_gpu_uuid), not the first row;
  6. with a second Fizgig on the other GPU (helper_local_gpu), this one keeps the card that wasn't lent.

list_gpus and _unpinned_gpu_uuid are stubbed: nothing here runs nvidia-smi or torch.

Run: venv/Scripts/python.exe tests/test_gpu_env.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# src only. The repo root holds Fizgig 8's fizgig.py, which would shadow the fizgig package.
sys.path.insert(0, os.path.join(REPO, "src"))
_UD = os.path.join(REPO, "user_data")
_had_ud = os.path.isdir(_UD)
from fizgig.web import server as S  # noqa: E402
if not _had_ud and os.path.isdir(_UD) and not os.listdir(_UD):
    os.rmdir(_UD)         # importing the server makes user_data/ (module-level user_file); never leave it behind

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


A = "GPU-db06bcf2-0000-0000-0000-000000000000"     # the 3060
B = "GPU-f6c1fa4e-0000-0000-0000-000000000000"     # the 5090
NVIDIA = [{"index": 0, "name": "NVIDIA GeForce RTX 3060", "total_gb": 12.0, "used_gb": 1.0, "uuid": A},
          {"index": 1, "name": "NVIDIA GeForce RTX 5090", "total_gb": 31.8, "used_gb": 2.0, "uuid": B}]
AMD = [{"index": 0, "name": "AMD Radeon RX 7900 XT", "total_gb": 20.0, "used_gb": 0.0, "uuid": ""}]

for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
    os.environ.pop(k, None)


def env_for(pin, rows):
    S.list_gpus = lambda: rows
    return S.gpu_env({}, {"cuda_device": pin})


# --- 1. a pin is exported -------------------------------------------------------------------------
e = env_for("1", NVIDIA)
ck("a card number is exported", e.get("CUDA_VISIBLE_DEVICES") == "1", e)
ck("  with nvidia-smi's (bus) order, so 1 means the 5090", e.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", e)
e = env_for(B, NVIDIA)
ck("a uuid is exported as is, and needs no device order",
   e.get("CUDA_VISIBLE_DEVICES") == B and "CUDA_DEVICE_ORDER" not in e, e)
e = env_for("", NVIDIA)
ck("no pin: nothing exported (CUDA's default card)", "CUDA_VISIBLE_DEVICES" not in e, e)

# --- 2. a stale pin is dropped --------------------------------------------------------------------
e = env_for("7", NVIDIA)
ck("a card number nvidia-smi doesn't list is dropped", "CUDA_VISIBLE_DEVICES" not in e, e)
e = env_for("GPU-deadbeef", NVIDIA)
ck("  so is a uuid from another machine", "CUDA_VISIBLE_DEVICES" not in e, e)

# --- 3. no real listing: the user isn't second-guessed ----------------------------------------------
e = env_for("7", [])
ck("nvidia-smi lists nothing: the pin is exported anyway", e.get("CUDA_VISIBLE_DEVICES") == "7", e)
e = env_for("1", AMD)
ck("ROCm's one synthetic row (no uuid) doesn't count as a listing: pin 1 is kept",
   e.get("CUDA_VISIBLE_DEVICES") == "1", e)

# --- 4. the environment wins ----------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
try:
    e = env_for("1", NVIDIA)
    ck("a CUDA_VISIBLE_DEVICES set where Fizgig started wins over Preferences",
       "CUDA_VISIBLE_DEVICES" not in e or e["CUDA_VISIBLE_DEVICES"] == "0", e)
finally:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

# --- 5. _chosen_gpu ---------------------------------------------------------------------------------
S._unpinned_gpu_uuid = lambda: B
ck("unpinned: the card an unpinned run lands on (the 5090), not the first row",
   (S._chosen_gpu(NVIDIA, "") or {}).get("uuid") == B, S._chosen_gpu(NVIDIA, ""))
ck("pinned by number: that card", (S._chosen_gpu(NVIDIA, "0") or {}).get("uuid") == A)
ck("pinned by uuid: that card", (S._chosen_gpu(NVIDIA, A) or {}).get("uuid") == A)
S._unpinned_gpu_uuid = lambda: ""
ck("unpinned and unknown (one card, or torch couldn't say): the first row",
   (S._chosen_gpu(NVIDIA, "") or {}).get("uuid") == A)
ck("no cards: None, not an error", S._chosen_gpu([], "") is None)

# --- 6. a second Fizgig on this computer's other GPU (upstream's helpers.local_start) -------------------
# helpers.local_gpus() lends every card but _chosen_gpu's pick for this Fizgig; unpinned, that pick is the card an
# unpinned run opens (the 5090 here), so it lends the 3060 (index 0). Upstream then pinned this Fizgig to "0" -
# nvidia-smi's first card, the 3060 again - and both Fizgigs would have trained on one card.
S._unpinned_gpu_uuid = lambda: B
S.list_gpus = lambda: NVIDIA
e = S.gpu_env({}, {"cuda_device": "", "helper_local_gpu": 0})
ck("helper on the 3060: this Fizgig keeps the 5090 (by uuid), not nvidia-smi's first card",
   e.get("CUDA_VISIBLE_DEVICES") == B, e)
e = S.gpu_env({}, {"cuda_device": "", "helper_local_gpu": 1})
ck("  helper on the 5090: this Fizgig gets the other card, the 3060", e.get("CUDA_VISIBLE_DEVICES") == A, e)
e = S.gpu_env({}, {"cuda_device": "1", "helper_local_gpu": 0})
ck("  a pin in Preferences still wins", e.get("CUDA_VISIBLE_DEVICES") == "1", e)
S.list_gpus = lambda: []
e = S.gpu_env({}, {"cuda_device": "", "helper_local_gpu": 1})
ck("  no nvidia-smi listing: upstream's first card, as before", e.get("CUDA_VISIBLE_DEVICES") == "0", e)
S._unpinned_gpu_uuid = lambda: ""

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
sys.exit(1 if fails else 0)
