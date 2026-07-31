"""Cache-first HuggingFace loading — offline once warm.

`from_pretrained(repo_id)` phones the Hub for an etag check on EVERY call, even when the files
are fully cached — so a machine with flaky or no internet can't start training (or mid-run
auto-recaption) despite having every byte on disk (issue #26). Asking for the cache first makes
the network a first-run-only event: hit the local cache, and only fall back to the Hub when
something is genuinely missing. Also shaves the round-trip off every warm start.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def from_pretrained_cache_first(cls, repo_id: str, **kwargs):
    """`cls.from_pretrained(repo_id)`, trying the local cache before touching the network.

    Works for tokenizers, processors and configs alike; `repo_id` may also be a local
    directory (local_files_only is a no-op there). A partial or missing cache falls through
    to the normal network path, so first runs behave exactly as before.
    """
    try:
        return cls.from_pretrained(repo_id, local_files_only=True, **kwargs)
    except Exception:
        logger.info(f"{cls.__name__}: {repo_id} not in the local cache — fetching from the Hub "
                    "(one-time; later runs load offline)")
        return cls.from_pretrained(repo_id, **kwargs)
