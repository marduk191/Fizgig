"""RAM as a container sees it: a pod's memory limit, not the host's.

A RunPod pod runs in a container on a big shared host: psutil and /proc/meminfo report the HOST's RAM (251 GB on a
4090 secure pod) while the container is killed at its own cgroup limit (31 GB there). Every RAM check in Fizgig
then thinks it has room it doesn't. install() makes psutil.virtual_memory() report the smaller of the two, so those
checks read the real figure; container_ram() is the same reading for code that reads /proc/meminfo itself.

Linux only, and only when a cgroup memory limit is set below the host's RAM: on Windows, macOS, a desktop Linux
box or a container without a limit, nothing changes.
"""
import os
import sys

_V2 = "/sys/fs/cgroup"
_V1 = "/sys/fs/cgroup/memory"


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _file_cache(stat_path, *keys):
    """Page cache the kernel can drop before it kills anything (inactive and active file pages, as MemAvailable
    counts them - a model file just read sits in the active list): not counted as used."""
    n = 0
    for line in (_read(stat_path) or "").splitlines():
        k, _, v = line.partition(" ")
        if k in keys and v.isdigit():
            n += int(v)
    return n


def container_ram():
    """(limit_bytes, available_bytes) of this process's cgroup, or None when there is no limit to honour."""
    if not sys.platform.startswith("linux"):
        return None
    lim, used = _read(os.path.join(_V2, "memory.max")), _read(os.path.join(_V2, "memory.current"))
    if lim is not None and used is not None:                              # cgroup v2
        if not lim.isdigit() or not used.isdigit():
            return None                                                   # "max": no limit
        cache = _file_cache(os.path.join(_V2, "memory.stat"), "inactive_file", "active_file")
    else:                                                                 # cgroup v1
        lim, used = _read(os.path.join(_V1, "memory.limit_in_bytes")), _read(os.path.join(_V1, "memory.usage_in_bytes"))
        if lim is None or used is None or not lim.isdigit() or not used.isdigit():
            return None
        cache = _file_cache(os.path.join(_V1, "memory.stat"), "total_inactive_file", "total_active_file")
    lim, used = int(lim), int(used)
    if lim >= 1 << 60:                                                    # v1's "unlimited" (page-rounded 2^63)
        return None
    return lim, max(0, lim - max(0, used - cache))


def install():
    """Make psutil.virtual_memory() report the container's limit and room when they are below the host's."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import psutil
    except ImportError:
        return
    real = psutil.virtual_memory
    if getattr(real, "_fizgig_container", False):
        return

    def virtual_memory():
        vm = real()
        c = container_ram()
        if c is None or c[0] >= vm.total:
            return vm
        total, avail = c[0], min(vm.available, c[1])
        return vm._replace(total=total, available=avail, percent=round(100.0 * (total - avail) / total, 1))

    virtual_memory._fizgig_container = True
    psutil.virtual_memory = virtual_memory
