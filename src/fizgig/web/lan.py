"""Fizgig on your other computers: found on the local network by themselves, for the Pods tab.

A Fizgig with a web password (the only kind other devices can open) answers a UDP broadcast on DISCOVERY_PORT with
its name, GPU, port and build. The Pods tab asks: every interface's broadcast address gets one question, and the
answers that come back within a second are the list. Nothing leaves the local network (a router never forwards a
broadcast) and the answer holds nothing the login page wouldn't show; opening one still asks for its password.

Standard library only (psutil, when there, finds every interface's broadcast address). Never raises into the server.
"""
import ipaddress
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time

DISCOVERY_PORT = 47810
ASK = b"FIZGIG?1"
INSTANCE = secrets.token_hex(8)             # this process: its own answer is left out of its own list
_info = [None]                              # () -> dict, the answer's facts (set by start())
_NO_WINDOW = {"creationflags": 0x08000000} if os.name == "nt" else {}


def machine_id(prefs, save) -> str:
    """This computer's lasting id (prefs lan_id), so a saved computer whose address changed is still the same one."""
    mid = str(prefs.get("lan_id") or "")
    if not mid:
        mid = secrets.token_hex(8)
        save({"lan_id": mid})
    return mid


def _local(addr) -> bool:
    try:
        return not ipaddress.ip_address(addr).is_global
    except ValueError:
        return False


def _serve():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", DISCOVERY_PORT))
    except OSError:
        s.close()
        return                              # the port is someone else's: this Fizgig just isn't found by itself
    while True:
        try:
            data, src = s.recvfrom(512)
        except OSError:
            time.sleep(1)
            continue
        if not data.startswith(ASK) or not _local(src[0]):
            continue
        try:
            reply = {"fizgig": 1, "instance": INSTANCE, "nonce": data[len(ASK):].strip().decode("ascii", "ignore")[:32],
                     **(_info[0]() if _info[0] else {})}
            s.sendto(json.dumps(reply).encode("utf-8")[:1400], src)
        except Exception:
            pass


def start(info):
    """Answer other Fizgigs' questions from now on. `info` () -> {id, name, port, gpu, vram_gb, version, commit, busy}."""
    _info[0] = info
    threading.Thread(target=_serve, daemon=True, name="fizgig-lan").start()


def _broadcasts() -> list:
    """Every IPv4 interface's own broadcast address (on Windows 255.255.255.255 leaves by one interface only, so a
    second network card, a VPN or WSL's adapter could hide the real network), then the general one."""
    out = []
    try:
        import psutil
        for addrs in psutil.net_if_addrs().values():
            for a in addrs:
                if a.family != socket.AF_INET or not a.netmask or a.address.startswith(("127.", "169.254.")):
                    continue
                try:
                    net = ipaddress.ip_network(f"{a.address}/{a.netmask}", strict=False)
                except ValueError:
                    continue
                if net.prefixlen < 31 and not net.is_global:
                    out.append(str(net.broadcast_address))
    except Exception:
        pass
    return list(dict.fromkeys(out + ["255.255.255.255"]))


def discover(timeout=1.0) -> list:
    """The Fizgigs that answer on this network: [{url, ip, id, name, port, gpu, vram_gb, version, commit, busy}]."""
    nonce = secrets.token_hex(6)
    found = {}
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.bind(("0.0.0.0", 0))
        for b in _broadcasts():
            try:
                s.sendto(ASK + b" " + nonce.encode(), (b, DISCOVERY_PORT))
            except OSError:
                pass
        end = time.monotonic() + timeout
        while (left := end - time.monotonic()) > 0:
            s.settimeout(left)
            try:
                data, src = s.recvfrom(2048)
            except (socket.timeout, OSError):
                break
            try:
                r = json.loads(data.decode("utf-8"))
                port = int(r.get("port") or 0)
            except (ValueError, TypeError, AttributeError):
                continue
            if r.get("fizgig") != 1 or r.get("nonce") != nonce or r.get("instance") == INSTANCE or not 0 < port < 65536:
                continue
            key = str(r.get("id") or "") + ":" + str(port)
            if key in found:                    # one computer answering on two interfaces: the first answer
                continue
            found[key] = {"url": f"http://{src[0]}:{port}/", "ip": src[0], "port": port,
                          **{k: r.get(k) for k in ("id", "name", "gpu", "vram_gb", "version", "commit", "busy",
                                                 "queued", "queue_held")}}
    except OSError:
        pass
    finally:
        s.close()
    return list(found.values())


# ---- Windows Firewall: what Preferences > Web UI shows, and its Allow on my network button -----------------------
RULE = "Fizgig"                             # our rules' display name (TCP: the web UI's port; UDP: DISCOVERY_PORT)


def _exe() -> str:
    """The program that listens: a venv's python.exe hands over to the Python it was made from, and Windows Firewall
    judges that one."""
    try:
        import psutil
        return psutil.Process().exe()
    except Exception:
        return getattr(sys, "_base_executable", "") or sys.executable


def _pythons() -> list:
    """This Python and its sibling (python.exe / pythonw.exe): the app window runs Fizgig on one, the browser launcher
    on the other, and Windows keeps a block rule for each."""
    exe = _exe()
    d, n = os.path.split(exe)
    other = {"python.exe": "pythonw.exe", "pythonw.exe": "python.exe"}.get(n.lower())
    return [exe] + ([os.path.join(d, other)] if other and os.path.exists(os.path.join(d, other)) else [])


def _ps(script, timeout=20):
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
                       capture_output=True, text=True, timeout=timeout, **_NO_WINDOW)
    return r.returncode, r.stdout.strip()


def _q(s) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def firewall_status(port) -> dict:
    """{windows, networks: [{name, category}], allowed: the home network lets this port and the discovery port in
    (our rules, or the ones Windows made for Fizgig's Python when its own question was answered Allow), blocked: a
    block rule on Fizgig's Python (one Windows makes when its question is cancelled; it beats any allow)}."""
    if os.name != "nt":
        return {"windows": False}
    script = f"""
$ErrorActionPreference = 'SilentlyContinue'
$nets = @(Get-NetConnectionProfile | ForEach-Object {{ @{{ name = [string]$_.Name; category = [string]$_.NetworkCategory;
  alias = [string]$_.InterfaceAlias; ips = @((Get-NetIPAddress -InterfaceIndex $_.InterfaceIndex -AddressFamily IPv4).IPAddress) }} }})
$app = @(Get-NetFirewallApplicationFilter -Program {_q(_exe())} | Get-NetFirewallRule |
  Where-Object {{ $_.Enabled -eq 'True' -and $_.Direction -eq 'Inbound' }})
$rules = @(@(Get-NetFirewallRule -DisplayName {_q(RULE)}) + @($app | Where-Object {{ $_.Action -eq 'Allow' }}) |
  Where-Object {{ $_.Enabled -eq 'True' -and $_.Action -eq 'Allow' }} | ForEach-Object {{
  $f = $_ | Get-NetFirewallPortFilter
  @{{ proto = [string]$f.Protocol; ports = @($f.LocalPort | ForEach-Object {{ [string]$_ }}); profile = [string]$_.Profile }} }})
$blocked = @($app | Where-Object {{ $_.Action -eq 'Block' }}).Count -gt 0
@{{ networks = $nets; rules = $rules; blocked = $blocked }} | ConvertTo-Json -Compress -Depth 4
"""
    try:
        code, out = _ps(script)
        d = json.loads(out or "{}")
    except Exception as e:
        return {"windows": True, "error": f"Couldn't read Windows Firewall ({type(e).__name__})."}
    nets = d.get("networks") or []
    if isinstance(nets, dict):
        nets = [nets]
    from fizgig.web.port import lan_ip
    here = lan_ip()                             # the network other computers reach this one on (a VPN's aside)
    for n in nets:
        ips = n.get("ips") or []
        n["ips"] = [ips] if isinstance(ips, str) else ips
        n["lan"] = bool(here and here in n["ips"])
    rules = d.get("rules") or []
    rules = [rules] if isinstance(rules, dict) else rules
    kinds = {n["category"] for n in nets if n["lan"]} or {n["category"] for n in nets}
    kinds = {{"DomainAuthenticated": "Domain"}.get(k, k) for k in kinds}

    def lets_in(proto, p):                      # on every network this computer is reached on
        def one(r):
            ports = r.get("ports") or []
            ports = [ports] if isinstance(ports, str) else ports
            prof = str(r.get("profile") or "")
            def covers(x):                      # "8081", "8081-8090" or "Any"
                a, _, b = str(x).partition("-")
                try:
                    return x == "Any" or int(a) <= p <= int(b or a)
                except ValueError:
                    return False
            return (r.get("proto") in (proto, "Any") and any(covers(x) for x in ports)
                    and ("Any" in prof or all(k in prof for k in kinds)))
        return any(one(r) for r in rules)
    return {"windows": True, "networks": nets, "lan_ip": here,
            "allowed": bool(kinds) and lets_in("TCP", int(port)) and lets_in("UDP", DISCOVERY_PORT),
            "blocked": bool(d.get("blocked"))}


ALLOW_SPAN = 10      # ports from the saved one up: Fizgig moves up to a free one when it's taken


def firewall_allow(port, base=None) -> dict:
    """Allow Fizgig on private networks: our two rules (this port, the discovery port), and Windows' block rules on
    Fizgig's Python taken off. Windows asks for an administrator's OK first."""
    if os.name != "nt":
        raise ValueError("This is for Windows. On Linux, allow the port with your firewall, e.g. "
                         f"sudo ufw allow {int(port)}/tcp && sudo ufw allow {DISCOVERY_PORT}/udp")
    lo = min(int(port), int(base or port))
    hi = max(int(port), lo + ALLOW_SPAN - 1)
    inner = f"""
$ErrorActionPreference = 'SilentlyContinue'
Remove-NetFirewallRule -DisplayName {_q(RULE)}
foreach ($x in @({", ".join(_q(x) for x in _pythons())})) {{ Get-NetFirewallApplicationFilter -Program $x | Get-NetFirewallRule |
  Where-Object {{ $_.Action -eq 'Block' -and $_.Direction -eq 'Inbound' }} | Remove-NetFirewallRule }}
$ErrorActionPreference = 'Stop'
New-NetFirewallRule -DisplayName {_q(RULE)} -Description 'Fizgig web UI, for your other devices' -Direction Inbound -Action Allow -Protocol TCP -LocalPort {lo}-{hi} -Profile Private | Out-Null
New-NetFirewallRule -DisplayName {_q(RULE)} -Description 'Fizgig, found by your other computers' -Direction Inbound -Action Allow -Protocol UDP -LocalPort {DISCOVERY_PORT} -Profile Private | Out-Null
"""
    import base64
    enc = base64.b64encode(inner.encode("utf-16-le")).decode("ascii")
    outer = ("try { $p = Start-Process powershell -Verb RunAs -WindowStyle Hidden -Wait -PassThru -ArgumentList "
             f"'-NoProfile','-ExecutionPolicy','Bypass','-EncodedCommand','{enc}'; exit $p.ExitCode }} catch {{ exit 1223 }}")
    code, _ = _ps(outer, timeout=300)
    if code == 1223:
        raise ValueError("Windows didn't get an administrator's OK, so nothing changed.")
    if code != 0:
        raise ValueError(f"Windows Firewall refused the change (code {code}).")
    return firewall_status(port)
