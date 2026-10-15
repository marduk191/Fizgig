"""The file picker's "This computer": every drive (Windows) or mounted disk (Linux, macOS), so a folder anywhere on the
machine is a click away. Nothing here touches a network drive's contents: a disconnected one can stall for seconds."""

import os
import sys

DRIVES = "::drives"           # the picker's path for this listing

# Linux filesystems that are the system's own, not somewhere files live
_SKIP_FS = {"proc", "sysfs", "devtmpfs", "devpts", "tmpfs", "securityfs", "cgroup", "cgroup2", "pstore", "bpf",
            "debugfs", "tracefs", "configfs", "fusectl", "mqueue", "hugetlbfs", "autofs", "binfmt_misc", "efivarfs",
            "rpc_pipefs", "nsfs", "ramfs", "squashfs", "overlay", "fuse.gvfsd-fuse", "fuse.portal", "selinuxfs"}
_SKIP_DIRS = ("/proc", "/sys", "/dev", "/run", "/snap", "/boot", "/var/lib/docker", "/etc")


def _windows():
    import ctypes
    k32 = ctypes.windll.kernel32
    mask = k32.GetLogicalDrives()
    kinds = {2: "Removable", 3: "Local disk", 4: "Network drive", 5: "CD/DVD", 6: "RAM disk"}
    out = []
    for i in range(26):
        if not mask & (1 << i):
            continue
        root = f"{chr(65 + i)}:\\"
        kind = k32.GetDriveTypeW(root)
        if kind <= 1:                           # unknown / no root directory
            continue
        label = ""
        if kind in (2, 3, 6):                   # a local volume's name is instant; a network one's can hang
            buf = ctypes.create_unicode_buffer(261)
            if k32.GetVolumeInformationW(root, buf, 261, None, None, None, None, 0):
                label = buf.value
        name = f"{root[:2]} {label}".strip() if label else f"{root[:2]} ({kinds.get(kind, 'Drive')})"
        out.append({"name": name, "path": root})
    return out


def _linux():
    out, seen = [{"name": "/ (root)", "path": "/"}], {"/"}
    try:
        with open("/proc/mounts", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        lines = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 3:
            continue
        mnt = parts[1].replace("\\040", " ")
        if parts[2] in _SKIP_FS or mnt in seen or any(mnt == d or mnt.startswith(d + "/") for d in _SKIP_DIRS):
            continue
        if os.path.isdir(mnt):
            seen.add(mnt)
            out.append({"name": mnt, "path": mnt})
    for d in (os.path.expanduser("~"), "/media", "/mnt", "/workspace"):     # the usual places, when not mounts
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append({"name": d, "path": d})
    return out


def _mac():
    out = [{"name": "/ (root)", "path": "/"}, {"name": os.path.expanduser("~"), "path": os.path.expanduser("~")}]
    try:
        for n in sorted(os.listdir("/Volumes")):
            p = os.path.join("/Volumes", n)
            if os.path.isdir(p) and not os.path.islink(p):
                out.append({"name": n, "path": p})
    except OSError:
        pass
    return out


def listing():
    try:
        return _windows() if os.name == "nt" else _mac() if sys.platform == "darwin" else _linux()
    except Exception:
        return [{"name": os.path.abspath(os.sep), "path": os.path.abspath(os.sep)}]
