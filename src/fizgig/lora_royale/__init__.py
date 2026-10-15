"""LoRA Royale — render a sequence of LoRA checkpoints (epochs of one training
run) side by side and crossfade between them to find the sweet-spot epoch."""

from fizgig.lora_royale.scan import scan_checkpoints, run_name_for_folder

# Seed travel's one-click recipes: each varies a clearly different knob (reference / anchor / journey length /
# identity lock); none touches the prompt.
SEED_TRAVEL_PRESETS = {
    "Hold subject":   dict(ref_strength="0.4",  ref_mp="1.0", sequential=False, waypoints="2"),
    "Journey":        dict(ref_strength="0.2",  ref_mp="1.0", sequential=False, waypoints="6"),
    "Feedback dream": dict(ref_strength="0.25", ref_mp="1.0", sequential=True,  waypoints="4"),
    # No reference at all — pure noise wandering through many seeds.
    "Free ride":      dict(ref_strength="0.0",  ref_mp="1.0", sequential=False, waypoints="8"),
}


def journey_seeds(start, end, n):
    """Ordered seed list for a journey: [start, …n-2 deterministic mids…, end]. Mids derive from `start`, so the same
    Start seed reproduces the same journey."""
    import random
    n = max(2, int(n))
    start, end = int(start), int(end)
    if n == 2:
        return [start, end]
    rng = random.Random(start * 2654435761 + 12345)
    return [start] + [rng.randint(0, 2 ** 31 - 1) for _ in range(n - 2)] + [end]


__all__ = ["scan_checkpoints", "run_name_for_folder", "SEED_TRAVEL_PRESETS", "journey_seeds"]
