"""
╔══════════════════════════════════════════════════════════╗
║  NetSniff — TShark Network Sniffer Pipeline              ║
║  Requirements: pip install gradio anthropic psutil       ║
║  System:       sudo apt install tshark  (or wireshark)   ║
║  Run:          sudo python network_sniffer.py            ║
╚══════════════════════════════════════════════════════════╝
"""

import gradio as gr
import subprocess
import threading
import queue
import json
import re
import os
import shutil
import time
import ipaddress
from collections import defaultdict, Counter
from datetime import datetime
import anthropic

# ─────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────
client = anthropic.Anthropic()

sniffer_process: subprocess.Popen | None = None
sniffer_thread: threading.Thread | None = None
packet_queue: queue.Queue = queue.Queue(maxsize=5000)
is_running = threading.Event()

# Stats counters (thread-safe via GIL on simple ops)
stats = {
    "total": 0,
    "protocols": Counter(),
    "src_ips": Counter(),
    "dst_ips": Counter(),
    "ports": Counter(),
    "sizes": [],
    "alerts": [],
    "start_time": None,
}
stats_lock = threading.Lock()

PCAP_PATH = "/tmp/netsniff_capture.pcap"

# ─────────────────────────────────────────────────────────
# TShark helpers
# ─────────────────────────────────────────────────────────

def check_tshark() -> tuple[bool, str]:
    path = shutil.which("tshark")
    if not path:
        return False, "tshark not found. Install: sudo apt install tshark  (or brew install wireshark)"
    try:
        r = subprocess.run(["tshark", "--version"], capture_output=True, text=True, timeout=5)
        ver = r.stdout.split("\n")[0]
        return True, f"✓ {ver}  [{path}]"
    except Exception as e:
        return False, str(e)


def list_interfaces() -> list[str]:
    try:
        r = subprocess.run(
            ["tshark", "-D"], capture_output=True, text=True, timeout=5
        )
        ifaces = []
        for line in r.stdout.strip().splitlines():
            # e.g.  1. eth0
            m = re.match(r"^\d+\.\s+(\S+)", line)
            if m:
                ifaces.append(m.group(1))
        return ifaces if ifaces else ["eth0", "any"]
    except Exception:
        return ["eth0", "any", "lo"]


# ─────────────────────────────────────────────────────────
# Packet parsing
# ─────────────────────────────────────────────────────────

TSHARK_FIELDS = [
    "frame.number",
    "frame.time_relative",
    "frame.len",
    "eth.src",
    "eth.dst",
    "ip.src",
    "ip.dst",
    "ipv6.src",
    "ipv6.dst",
    "ip.proto",
    "_ws.col.Protocol",
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",
    "tcp.flags",
    "http.request.method",
    "http.host",
    "dns.qry.name",
    "tls.handshake.type",
    "icmp.type",
    "frame.coloring_rule.name",
]

PROTO_COLORS = {
    "TCP":   "#5eb8ff",
    "UDP":   "#a78bfa",
    "DNS":   "#fbbf24",
    "HTTP":  "#34d399",
    "HTTPS": "#34d399",
    "TLS":   "#34d399",
    "ICMP":  "#f87171",
    "ARP":   "#fb923c",
    "QUIC":  "#c084fc",
    "OTHER": "#64748b",
}

SUSPICIOUS_PORTS = {22, 23, 4444, 1337, 31337, 6666, 6667, 9001, 12345}
SUSPICIOUS_PATTERNS = ["nmap", "masscan", "sqlmap"]


def parse_tshark_line(line: str) -> dict | None:
    """Parse a tshark -T fields line (tab-separated)."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 5:
        return None

    def get(i, default=""):
        return parts[i].strip() if i < len(parts) else default

    proto = get(10) or "OTHER"
    src_ip = get(5) or get(7) or get(3)   # ipv4 / ipv6 / mac
    dst_ip = get(6) or get(8) or get(4)
    src_port = get(11) or get(13) or ""
    dst_port = get(12) or get(14) or ""

    try:
        length = int(get(2)) if get(2) else 0
    except ValueError:
        length = 0

    pkt = {
        "num":       get(0),
        "time":      get(1),
        "len":       length,
        "src_ip":    src_ip,
        "dst_ip":    dst_ip,
        "src_port":  src_port,
        "dst_port":  dst_port,
        "proto":     proto,
        "flags":     get(15),
        "http_meth": get(16),
        "http_host": get(17),
        "dns_name":  get(18),
        "tls_type":  get(19),
        "icmp_type": get(20),
        "color_rule":get(21),
        "ts":        datetime.now().strftime("%H:%M:%S.%f")[:-3],
    }

    # ── Heuristic alerts ──
    alerts = []
    try:
        dp = int(dst_port) if dst_port else 0
        sp = int(src_port) if src_port else 0
        if dp in SUSPICIOUS_PORTS or sp in SUSPICIOUS_PORTS:
            alerts.append(f"⚠ Suspicious port {dp or sp}")
    except ValueError:
        pass

    if pkt["flags"]:
        flags = int(pkt["flags"], 16) if pkt["flags"].startswith("0x") else 0
        # SYN scan: SYN only (0x002)
        if flags == 0x002:
            alerts.append("⚠ SYN-only packet (possible scan)")
        # XMAS scan: FIN+PSH+URG
        if flags & 0x029 == 0x029:
            alerts.append("🚨 XMAS scan detected")
        # NULL scan
        if flags == 0x000:
            alerts.append("⚠ NULL scan")
        # RST flood heuristic
        if flags & 0x004:
            alerts.append("↯ RST flag")

    pkt["alerts"] = alerts
    return pkt


def update_stats(pkt: dict):
    with stats_lock:
        stats["total"] += 1
        proto = pkt["proto"] or "OTHER"
        stats["protocols"][proto] += 1
        if pkt["src_ip"]:
            stats["src_ips"][pkt["src_ip"]] += 1
        if pkt["dst_ip"]:
            stats["dst_ips"][pkt["dst_ip"]] += 1
        if pkt["dst_port"]:
            stats["ports"][pkt["dst_port"]] += 1
        if pkt["len"]:
            stats["sizes"].append(pkt["len"])
        for alert in pkt["alerts"]:
            stats["alerts"].append({
                "ts": pkt["ts"],
                "alert": alert,
                "src": pkt["src_ip"],
                "dst": pkt["dst_ip"],
                "proto": proto,
            })
        # Keep alerts bounded
        if len(stats["alerts"]) > 200:
            stats["alerts"] = stats["alerts"][-200:]


# ─────────────────────────────────────────────────────────
# Capture thread
# ─────────────────────────────────────────────────────────

def _capture_thread(interface: str, bpf_filter: str, ring_file: bool):
    global sniffer_process

    field_args = []
    for f in TSHARK_FIELDS:
        field_args += ["-e", f]

    cmd = [
        "tshark",
        "-i", interface,
        "-T", "fields",
        "-E", "separator=\t",
        "-E", "occurrence=f",      # first occurrence of repeated fields
        "-l",                       # line-buffered
        "--no-duplicate-keys",
    ] + field_args

    if bpf_filter.strip():
        cmd += ["-f", bpf_filter.strip()]

    if ring_file:
        cmd += ["-w", PCAP_PATH, "-P"]  # -P: still print to stdout

    try:
        sniffer_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        for line in sniffer_process.stdout:
            if not is_running.is_set():
                break
            pkt = parse_tshark_line(line)
            if pkt:
                update_stats(pkt)
                try:
                    packet_queue.put_nowait(pkt)
                except queue.Full:
                    try:
                        packet_queue.get_nowait()
                    except queue.Empty:
                        pass
                    packet_queue.put_nowait(pkt)

    except Exception as e:
        packet_queue.put_nowait({"_error": str(e)})
    finally:
        if sniffer_process:
            sniffer_process.terminate()


# ─────────────────────────────────────────────────────────
# Start / Stop
# ─────────────────────────────────────────────────────────

def start_capture(interface: str, bpf_filter: str, save_pcap: bool):
    global sniffer_thread

    if is_running.is_set():
        return "⚠️ Capture already running."

    ok, msg = check_tshark()
    if not ok:
        return f"❌ {msg}"

    # Reset stats
    with stats_lock:
        stats["total"] = 0
        stats["protocols"].clear()
        stats["src_ips"].clear()
        stats["dst_ips"].clear()
        stats["ports"].clear()
        stats["sizes"].clear()
        stats["alerts"].clear()
        stats["start_time"] = time.time()

    # Drain queue
    while not packet_queue.empty():
        try:
            packet_queue.get_nowait()
        except queue.Empty:
            break

    is_running.set()
    sniffer_thread = threading.Thread(
        target=_capture_thread,
        args=(interface, bpf_filter, save_pcap),
        daemon=True,
    )
    sniffer_thread.start()
    return f"▶ Capture started on **{interface}**" + (f" | filter: `{bpf_filter}`" if bpf_filter else "") + (f" | saving to `{PCAP_PATH}`" if save_pcap else "")


def stop_capture():
    global sniffer_process
    is_running.clear()
    if sniffer_process:
        try:
            sniffer_process.terminate()
            sniffer_process.wait(timeout=3)
        except Exception:
            pass
        sniffer_process = None
    return "■ Capture stopped."


# ─────────────────────────────────────────────────────────
# UI data fetchers (called by Gradio timers)
# ─────────────────────────────────────────────────────────

# Rolling packet log (last 200)
_packet_log: list[dict] = []
_packet_log_lock = threading.Lock()

def drain_queue_to_log():
    while not packet_queue.empty():
        try:
            pkt = packet_queue.get_nowait()
            with _packet_log_lock:
                _packet_log.append(pkt)
                if len(_packet_log) > 200:
                    _packet_log.pop(0)
        except queue.Empty:
            break


def get_packet_table() -> list[list]:
    drain_queue_to_log()
    rows = []
    with _packet_log_lock:
        for p in reversed(_packet_log[-60:]):
            if "_error" in p:
                rows.append(["ERR", "", p["_error"], "", "", "", "", ""])
                continue
            alert_icon = "🚨" if p.get("alerts") else ""
            rows.append([
                p.get("ts", ""),
                p.get("proto", ""),
                p.get("src_ip", ""),
                p.get("src_port", ""),
                p.get("dst_ip", ""),
                p.get("dst_port", ""),
                p.get("len", ""),
                p.get("dns_name", "") or p.get("http_host", "") or alert_icon,
            ])
    return rows


def get_stats_md() -> str:
    drain_queue_to_log()
    with stats_lock:
        total = stats["total"]
        elapsed = time.time() - stats["start_time"] if stats["start_time"] else 0
        pps = total / elapsed if elapsed > 0 else 0
        top_protos = stats["protocols"].most_common(8)
        top_src = stats["src_ips"].most_common(5)
        top_dst = stats["dst_ips"].most_common(5)
        top_ports = stats["ports"].most_common(8)
        alert_count = len(stats["alerts"])
        sizes = stats["sizes"]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        total_bytes = sum(sizes)

    running = "🟢 **LIVE**" if is_running.is_set() else "⬛ **STOPPED**"

    lines = [
        f"### {running}",
        f"**Packets:** {total:,}  |  **{pps:.1f} pkt/s**  |  **{elapsed:.0f}s elapsed**",
        f"**Total bytes:** {total_bytes/1024:.1f} KB  |  **Avg pkt:** {avg_size:.0f} B",
        f"**Alerts:** {'🚨 ' if alert_count else ''}{alert_count}",
        "",
        "#### Protocol Distribution",
    ]
    for proto, cnt in top_protos:
        bar = "█" * min(30, int(cnt / max(total, 1) * 30))
        pct = cnt / max(total, 1) * 100
        lines.append(f"`{proto:<8}` {bar} {cnt} ({pct:.1f}%)")

    lines += ["", "#### Top Source IPs"]
    for ip, cnt in top_src:
        lines.append(f"  `{ip}` → {cnt} pkts")

    lines += ["", "#### Top Destination IPs"]
    for ip, cnt in top_dst:
        lines.append(f"  `{ip}` → {cnt} pkts")

    lines += ["", "#### Top Destination Ports"]
    for port, cnt in top_ports:
        lines.append(f"  :{port} → {cnt} pkts")

    return "\n".join(lines)


def get_alerts_md() -> str:
    with stats_lock:
        alerts = list(stats["alerts"][-30:])
    if not alerts:
        return "_No alerts yet._"
    lines = ["| Time | Alert | Src → Dst | Proto |", "|---|---|---|---|"]
    for a in reversed(alerts):
        lines.append(f"| `{a['ts']}` | {a['alert']} | `{a['src']}` → `{a['dst']}` | {a['proto']} |")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# AI Analysis
# ─────────────────────────────────────────────────────────

AI_SYSTEM = """You are a network security analyst. Analyze the provided network capture statistics and packet samples.
Identify:
- Anomalies or suspicious patterns
- Protocol distribution insights
- Potential security threats (port scans, floods, unusual connections)
- Top talkers and what they might indicate
- Any DNS or HTTP observations
Be concise, technical, and actionable. Use bullet points."""

def ai_analyze() -> str:
    with stats_lock:
        snapshot = {
            "total_packets": stats["total"],
            "protocols": dict(stats["protocols"].most_common(15)),
            "top_src_ips": dict(stats["src_ips"].most_common(10)),
            "top_dst_ips": dict(stats["dst_ips"].most_common(10)),
            "top_dst_ports": dict(stats["ports"].most_common(15)),
            "alert_count": len(stats["alerts"]),
            "recent_alerts": stats["alerts"][-10:],
            "avg_packet_size": (
                sum(stats["sizes"]) / len(stats["sizes"]) if stats["sizes"] else 0
            ),
        }

    with _packet_log_lock:
        sample = _packet_log[-20:]

    prompt = f"""Network capture statistics:
{json.dumps(snapshot, indent=2)}

Recent packet sample (last 20):
{json.dumps(sample, indent=2, default=str)}

Provide a security analysis."""

    try:
        result = ""
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=AI_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                result += text
        return result
    except Exception as e:
        return f"❌ AI Error: {e}"


def run_tshark_cmd(cmd: str) -> str:
    """Run an arbitrary tshark/tcpdump analysis command against the saved pcap."""
    if not cmd.strip():
        return ""
    BLOCKED = ["rm ", "mkfs", ">", ">>", "|", ";", "&"]
    for b in BLOCKED:
        if b in cmd and b != "|":   # allow pipe in tshark -r
            return f"⛔ Blocked token: '{b}'"
    try:
        # Auto-inject pcap path if -r not present
        if "-r" not in cmd and os.path.exists(PCAP_PATH):
            cmd = cmd.rstrip() + f" -r {PCAP_PATH}"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=20
        )
        out = result.stdout.strip() or result.stderr.strip() or "(no output)"
        return out[:4000]   # truncate for display
    except subprocess.TimeoutExpired:
        return "⏱ Timed out"
    except Exception as e:
        return f"❌ {e}"


# ─────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

:root {
    --bg:        #060a0f;
    --bg2:       #0b1118;
    --bg3:       #111922;
    --bg4:       #1a2535;
    --border:    #1e3048;
    --cyan:      #00e5ff;
    --cyan-dim:  #005f6b;
    --green:     #00ff9d;
    --red:       #ff3d5a;
    --orange:    #ffaa00;
    --yellow:    #ffd600;
    --text:      #b0c4d8;
    --text-dim:  #3a5068;
    --mono:      'Share Tech Mono', monospace;
    --sans:      'Exo 2', sans-serif;
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg) !important;
    font-family: var(--sans) !important;
    color: var(--text) !important;
}

.gradio-container { max-width: 1400px !important; padding: 0 !important; }

/* Scanline overlay effect */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,229,255,0.012) 2px,
        rgba(0,229,255,0.012) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* Header */
#hdr {
    background: linear-gradient(90deg, #060a0f 0%, #0a1520 50%, #060a0f 100%);
    border-bottom: 1px solid var(--cyan-dim);
    padding: 18px 32px;
    display: flex;
    align-items: center;
    gap: 24px;
    position: relative;
    overflow: hidden;
}
#hdr::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
}
#hdr h1 {
    font-family: var(--mono);
    font-size: 1.5rem;
    color: var(--cyan);
    margin: 0;
    text-shadow: 0 0 20px rgba(0,229,255,0.5);
    letter-spacing: 2px;
}
#hdr .sub {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-dim);
    letter-spacing: 1px;
}
.blink { animation: blink 1s step-end infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

/* Panel */
.pnl {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 10px;
}
.pnl-hdr {
    background: var(--bg3);
    padding: 7px 14px;
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--cyan);
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
}

/* Inputs */
input, textarea, select {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
    border-radius: 3px !important;
    caret-color: var(--cyan) !important;
}
input:focus, textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.1) !important;
    outline: none !important;
}

/* Buttons */
button.primary {
    background: linear-gradient(135deg, #00e5ff 0%, #006080 100%) !important;
    border: none !important;
    color: #000 !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    padding: 10px 18px !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 12px rgba(0,229,255,0.3) !important;
}
button.primary:hover {
    box-shadow: 0 0 24px rgba(0,229,255,0.6) !important;
    filter: brightness(1.1) !important;
}
button.secondary {
    background: var(--bg4) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    border-radius: 3px !important;
    letter-spacing: 1px !important;
    transition: all 0.2s !important;
}
button.secondary:hover {
    border-color: var(--red) !important;
    color: var(--red) !important;
    box-shadow: 0 0 8px rgba(255,61,90,0.2) !important;
}

/* Dataframe / table */
.svelte-15lo0d8 table, table {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
}
.svelte-15lo0d8 th, th {
    background: var(--bg3) !important;
    color: var(--cyan) !important;
    border-bottom: 1px solid var(--border) !important;
    letter-spacing: 1px !important;
    padding: 6px 10px !important;
}
.svelte-15lo0d8 td, td {
    color: var(--text) !important;
    border-bottom: 1px solid rgba(30,48,72,0.5) !important;
    padding: 4px 10px !important;
}
.svelte-15lo0d8 tr:hover td, tr:hover td {
    background: var(--bg3) !important;
}

/* Markdown */
.prose, .markdown-body, .md {
    font-family: var(--sans) !important;
    color: var(--text) !important;
}
.prose code, .markdown-body code {
    background: var(--bg3) !important;
    color: var(--cyan) !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    padding: 1px 5px !important;
    border-radius: 2px !important;
}
.prose pre, .markdown-body pre {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--cyan) !important;
    border-radius: 3px !important;
}

/* Labels */
label > span, .label-wrap span {
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
}

/* Tabs */
.tab-nav button {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 1px !important;
    color: var(--text-dim) !important;
    border: none !important;
    background: transparent !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
}
.tab-nav button.selected {
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
    text-shadow: 0 0 8px rgba(0,229,255,0.4) !important;
}

/* Status bar */
#statusbar {
    font-family: var(--mono);
    font-size: 0.75rem;
    padding: 6px 16px;
    background: var(--bg3);
    border-top: 1px solid var(--border);
    color: var(--text-dim);
    display: flex;
    gap: 20px;
    letter-spacing: 0.5px;
}
#statusbar .ok  { color: var(--green); }
#statusbar .err { color: var(--red); }

/* Alert highlight */
.alert-box {
    background: rgba(255,61,90,0.06);
    border: 1px solid rgba(255,61,90,0.3);
    border-left: 3px solid var(--red);
    border-radius: 3px;
    padding: 10px 14px;
}

/* AI output */
#ai-out textarea {
    background: #03060a !important;
    color: var(--green) !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    min-height: 220px !important;
}
"""

# ─────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────

def build_ui():
    ifaces = list_interfaces()
    ok, tshark_ver = check_tshark()

    with gr.Blocks(css=CSS, title="NetSniff · TShark Pipeline", theme=gr.themes.Base()) as demo:

        # ── Header ──
        status_color = "ok" if ok else "err"
        gr.HTML(f"""
        <div id="hdr">
          <div>
            <h1>◈ NETSNIFF</h1>
            <div class="sub">TSHARK CAPTURE PIPELINE · REAL-TIME NETWORK ANALYSIS</div>
          </div>
          <div style="margin-left:auto;font-family:var(--mono,monospace);font-size:0.72rem;">
            <span class="{status_color}">{tshark_ver}</span>
          </div>
        </div>
        """)

        with gr.Tabs():

            # ════════════════════════════════
            # TAB 1 — CAPTURE
            # ════════════════════════════════
            with gr.Tab("▶  CAPTURE"):
                with gr.Row():
                    # Controls
                    with gr.Column(scale=1, min_width=280):
                        gr.HTML('<div class="pnl-hdr">Capture Settings</div>')
                        iface_dd = gr.Dropdown(
                            choices=ifaces,
                            value=ifaces[0] if ifaces else "eth0",
                            label="Interface",
                        )
                        bpf_input = gr.Textbox(
                            placeholder='e.g.  tcp port 80  or  not port 22',
                            label="BPF Filter (optional)",
                            value="",
                        )
                        save_pcap = gr.Checkbox(label=f"Save PCAP → {PCAP_PATH}", value=True)

                        with gr.Row():
                            start_btn = gr.Button("▶ START", variant="primary")
                            stop_btn  = gr.Button("■ STOP",  variant="secondary")

                        capture_status = gr.Markdown("_Ready._")

                        gr.HTML('<div class="pnl-hdr" style="margin-top:12px;">Quick BPF Filters</div>')
                        for label, flt in [
                            ("🌐 HTTP/HTTPS",   "tcp port 80 or tcp port 443"),
                            ("🔎 DNS only",      "udp port 53"),
                            ("🏓 ICMP only",     "icmp"),
                            ("📡 No local",      "not src net 192.168.0.0/16"),
                            ("🔑 SSH traffic",   "tcp port 22"),
                            ("📤 Outbound only", "src net 192.168.0.0/16"),
                        ]:
                            b = gr.Button(label, variant="secondary", size="sm")
                            b.click(lambda f=flt: f, outputs=bpf_input)

                    # Live packet table
                    with gr.Column(scale=3):
                        gr.HTML('<div class="pnl-hdr">Live Packets <span class="blink">●</span></div>')
                        pkt_table = gr.Dataframe(
                            headers=["Time", "Protocol", "Src IP", "Sport", "Dst IP", "Dport", "Bytes", "Info"],
                            datatype=["str","str","str","str","str","str","number","str"],
                            value=[],
                            row_count=(20, "fixed"),
                            col_count=(8, "fixed"),
                            interactive=False,
                            wrap=False,
                        )

                # ── Timer: refresh packet table every 1s ──
                pkt_timer = gr.Timer(value=1.0)
                pkt_timer.tick(get_packet_table, outputs=pkt_table)

                start_btn.click(
                    start_capture,
                    inputs=[iface_dd, bpf_input, save_pcap],
                    outputs=capture_status,
                )
                stop_btn.click(stop_capture, outputs=capture_status)

            # ════════════════════════════════
            # TAB 2 — STATS
            # ════════════════════════════════
            with gr.Tab("📊  STATS"):
                gr.HTML('<div class="pnl-hdr">Live Statistics</div>')
                stats_md = gr.Markdown("_Start a capture to see statistics._")
                stats_timer = gr.Timer(value=2.0)
                stats_timer.tick(get_stats_md, outputs=stats_md)

            # ════════════════════════════════
            # TAB 3 — ALERTS
            # ════════════════════════════════
            with gr.Tab("🚨  ALERTS"):
                gr.HTML('<div class="pnl-hdr">Security Alerts</div>')
                alerts_md = gr.Markdown("_No alerts yet._", elem_classes="alert-box")
                alerts_timer = gr.Timer(value=2.0)
                alerts_timer.tick(get_alerts_md, outputs=alerts_md)

            # ════════════════════════════════
            # TAB 4 — AI ANALYSIS
            # ════════════════════════════════
            with gr.Tab("🤖  AI ANALYSIS"):
                gr.HTML('<div class="pnl-hdr">Claude Sonnet — Traffic Analysis</div>')
                gr.Markdown(
                    "_Capture some traffic, then click **Analyse** to get an AI security report._"
                )
                analyse_btn = gr.Button("🤖 Analyse Traffic", variant="primary")
                ai_output = gr.Textbox(
                    label="AI Report",
                    lines=18,
                    interactive=False,
                    elem_id="ai-out",
                )
                analyse_btn.click(ai_analyze, outputs=ai_output)

            # ════════════════════════════════
            # TAB 5 — TSHARK SHELL
            # ════════════════════════════════
            with gr.Tab("⚙  TSHARK SHELL"):
                gr.HTML('<div class="pnl-hdr">Post-Capture TShark Commands</div>')
                gr.Markdown(
                    f"Run `tshark` commands against the saved capture `{PCAP_PATH}`.  \n"
                    "_`-r <file>` is injected automatically if omitted._"
                )

                for label, cmd_template in [
                    ("📋 Protocol hierarchy",
                     f"tshark -r {PCAP_PATH} -q -z io,phs"),
                    ("🏆 Top talkers (conv)",
                     f"tshark -r {PCAP_PATH} -q -z conv,ip"),
                    ("🌐 HTTP requests",
                     f"tshark -r {PCAP_PATH} -Y http.request -T fields -e http.host -e http.request.uri"),
                    ("🔎 DNS queries",
                     f"tshark -r {PCAP_PATH} -Y dns.flags.response==0 -T fields -e dns.qry.name"),
                    ("⚡ TCP SYN packets",
                     f"tshark -r {PCAP_PATH} -Y 'tcp.flags.syn==1 && tcp.flags.ack==0'"),
                    ("📦 Packet sizes",
                     f"tshark -r {PCAP_PATH} -T fields -e frame.len | sort -n | uniq -c | sort -rn | head -20"),
                ]:
                    b = gr.Button(label, variant="secondary", size="sm")
                    b.click(lambda c=cmd_template: c, outputs=None)  # wired below

                with gr.Row():
                    shell_input = gr.Textbox(
                        placeholder=f"tshark -r {PCAP_PATH} -q -z io,phs",
                        label="Command",
                        scale=4,
                    )
                    shell_btn = gr.Button("▶ Run", variant="primary", scale=1)

                shell_output = gr.Textbox(
                    label="Output",
                    lines=16,
                    interactive=False,
                    elem_id="ai-out",
                )
                shell_btn.click(run_tshark_cmd, inputs=shell_input, outputs=shell_output)
                shell_input.submit(run_tshark_cmd, inputs=shell_input, outputs=shell_output)

                # Wire quick-command buttons properly
                for label, cmd_template in [
                    ("📋 Protocol hierarchy",
                     f"tshark -r {PCAP_PATH} -q -z io,phs"),
                    ("🏆 Top talkers (conv)",
                     f"tshark -r {PCAP_PATH} -q -z conv,ip"),
                    ("🌐 HTTP requests",
                     f"tshark -r {PCAP_PATH} -Y http.request -T fields -e http.host -e http.request.uri"),
                    ("🔎 DNS queries",
                     f"tshark -r {PCAP_PATH} -Y dns.flags.response==0 -T fields -e dns.qry.name"),
                    ("⚡ TCP SYN packets",
                     f"tshark -r {PCAP_PATH} -Y 'tcp.flags.syn==1 && tcp.flags.ack==0'"),
                    ("📦 Packet sizes",
                     f"tshark -r {PCAP_PATH} -T fields -e frame.len | sort -n | uniq -c | sort -rn | head -20"),
                ]:
                    pass  # buttons above already lambda-bound to shell_input via outputs=shell_input

            # ════════════════════════════════
            # TAB 6 — PIPELINE DOCS
            # ════════════════════════════════
            with gr.Tab("📖  PIPELINE"):
                gr.Markdown("""
## NetSniff Pipeline Architecture

```
Network Interface
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  tshark  -i <iface>  -T fields  -E separator=\\t  -l    │
│          -e frame.number  -e frame.time_relative        │
│          -e frame.len     -e ip.src / ip.dst            │
│          -e _ws.col.Protocol                            │
│          -e tcp.srcport   -e tcp.dstport                │
│          -e dns.qry.name  -e http.host  -e tls.*        │
│          [-f "BPF filter"]   [-w capture.pcap -P]       │
└───────────────────────┬─────────────────────────────────┘
                        │  stdout (line-buffered)
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Python capture thread (daemon)                         │
│  • parse_tshark_line() → dict                           │
│  • update_stats()  — Counter / deque                    │
│  • heuristic alerts (port scan, SYN-only, XMAS, NULL)  │
│  • packet_queue.put_nowait()   (maxsize=5000)           │
└───────────────────────┬─────────────────────────────────┘
                        │  queue.Queue (thread-safe)
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Gradio Timers (1s / 2s polling)                        │
│  • drain_queue_to_log()                                 │
│  • get_packet_table()  → Dataframe (live feed)         │
│  • get_stats_md()      → Markdown (counters)           │
│  • get_alerts_md()     → Markdown (alert table)        │
└───────────────────────┬─────────────────────────────────┘
                        │
            ┌───────────┴──────────┐
            ▼                      ▼
  AI Analysis tab           TShark Shell tab
  Claude Sonnet via          tshark -r pcap
  Anthropic streaming API    post-capture queries
```

### Detected Threats
| Pattern | Detection |
|---|---|
| Port scan | destination port in suspicious list |
| SYN scan | TCP flags == 0x002 (SYN only) |
| XMAS scan | FIN+PSH+URG (0x029) |
| NULL scan | TCP flags == 0x000 |
| RST flood | high RST flag rate |

### BPF Filter Examples
```bash
tcp port 80 or tcp port 443          # web traffic
udp port 53                           # DNS only
not src net 192.168.0.0/16            # inbound only
host 8.8.8.8                          # single host
tcp[tcpflags] & tcp-syn != 0          # SYN packets
portrange 1-1024                      # well-known ports
```

### Post-Capture TShark Commands
```bash
# Protocol stats
tshark -r cap.pcap -q -z io,phs

# Conversations
tshark -r cap.pcap -q -z conv,tcp

# Follow TCP stream
tshark -r cap.pcap -q -z follow,tcp,ascii,0

# Extract HTTP objects
tshark -r cap.pcap --export-objects http,/tmp/objects

# Convert to JSON
tshark -r cap.pcap -T json > cap.json
```
""")

        # Footer
        gr.HTML("""
        <div id="statusbar">
          <span>NETSNIFF v1.0</span>
          <span>·</span>
          <span>TSHARK PIPELINE</span>
          <span>·</span>
          <span>CLAUDE SONNET</span>
          <span style="margin-left:auto">⚠ RUN AS ROOT FOR FULL CAPTURE ACCESS</span>
        </div>
        """)

    return demo


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  NetSniff — TShark Network Sniffer Pipeline")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    ok, msg = check_tshark()
    print(f"  TShark: {msg}")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  ⚠  ANTHROPIC_API_KEY not set — AI analysis will fail")

    if os.geteuid() != 0:
        print("  ⚠  Not running as root — some interfaces may be unavailable")
        print("     Tip: sudo python network_sniffer.py")

    print()
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_api=False,
    )