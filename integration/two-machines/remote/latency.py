"""Round-trip time between these two boxes, which bandwidth does not imply.

GPU-120 asks for latency *and* bandwidth, and the reason the two are listed
separately is in ``bench/round_link_cost.py``: under every arm of that bench
there is a floor of about 0.31 s per round which is the dial and the round
trip, and **it does not grow with the link**. A round on a fat slow pipe and a
round on a thin fast one are not the same round, and a single MB/s number
cannot tell them apart.

The bench's default is **200 ms**. This says what the number really is between
the two machines that are rented right now, and it says it three ways because
they answer different questions:

``icmp``
    ``ping``, when it is installed and not filtered. The lower bound: no TCP
    handshake, no userspace. Often unavailable on an overlay network, which is
    why it is not the one anything depends on.

``tcp``
    Connect once, then bounce one byte back and forth on the open socket. This
    is the number the round exchange actually pays per request, because
    ``DeltaExchange.fetch`` opens a connection, sends a fixed-size greeting and
    waits for a fixed-size answer before a single byte of report moves.

``tcp-connect``
    A fresh ``connect()`` per sample, which is what a *fetch* really does — one
    connection per peer per round. On a link where a handshake costs three
    round trips this is visibly worse than ``tcp``, and the difference is the
    part of ``gather_wait_seconds`` that no amount of report-shrinking removes.

Node 0 serves, node 1 measures, and only node 1 prints numbers. Run it on both
boxes at once, like every other phase.

    python latency.py --port 29610 --samples 200
"""

import argparse
import json
import os
import socket
import statistics
import subprocess
import time

PAYLOAD = b"x"


def serve(port, seconds):
    """Bounce single bytes back until the clock runs out."""
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("0.0.0.0", port))
    listener.listen(8)
    listener.settimeout(1.0)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            connection, _ = listener.accept()
        except socket.timeout:
            continue
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.settimeout(5.0)
        try:
            while True:
                blob = connection.recv(1)
                if not blob:
                    break
                connection.sendall(blob)
        except OSError:
            pass
        finally:
            connection.close()
    listener.close()


def bounce(peer, port, samples):
    """One connection, many round trips: the per-request cost."""
    connection = socket.create_connection((peer, port), timeout=10)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    taken = []
    try:
        for _ in range(samples):
            started = time.perf_counter()
            connection.sendall(PAYLOAD)
            if not connection.recv(1):
                break
            taken.append((time.perf_counter() - started) * 1000.0)
    finally:
        connection.close()
    return taken


def reconnect(peer, port, samples):
    """A fresh connection each time: what one fetch pays before any bytes."""
    taken = []
    for _ in range(samples):
        started = time.perf_counter()
        connection = socket.create_connection((peer, port), timeout=10)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            connection.sendall(PAYLOAD)
            connection.recv(1)
        finally:
            connection.close()
        taken.append((time.perf_counter() - started) * 1000.0)
    return taken


def icmp(peer, count):
    """``ping``, if this box has it and the network allows it."""
    try:
        done = subprocess.run(
            ["ping", "-c", str(count), "-i", "0.2", "-q", peer],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in done.stdout.splitlines():
        if "min/avg/max" in line:
            return line.strip()
    return None


def summary(name, taken):
    if not taken:
        return {"name": name, "samples": 0}
    ordered = sorted(taken)
    return {
        "name": name,
        "samples": len(taken),
        "min_ms": round(ordered[0], 3),
        "median_ms": round(statistics.median(ordered), 3),
        "p90_ms": round(ordered[int(len(ordered) * 0.9) - 1], 3),
        "max_ms": round(ordered[-1], 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=29610)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--serve-seconds", type=float, default=90.0)
    args = parser.parse_args()

    rank = os.environ["NODE_RANK"]
    peer = os.environ["PEER_ADDR"]

    if rank == "0":
        print("serving the bounce on :%d for %.0fs" % (args.port, args.serve_seconds))
        serve(args.port, args.serve_seconds)
        print("done serving")
        return

    # Node 1 measures. A moment for the server to be listening; a refused
    # connection here is a phase that says nothing, on a box being billed.
    time.sleep(3)

    results = [
        summary("tcp", bounce(peer, args.port, args.samples)),
        summary("tcp-connect", reconnect(peer, args.port, max(20, args.samples // 5))),
    ]
    line = icmp(peer, 20)

    print()
    print("round trip to %s" % peer)
    for entry in results:
        if not entry["samples"]:
            print("  %-12s no samples" % entry["name"])
            continue
        print(
            "  %-12s min %7.2f ms   median %7.2f ms   p90 %7.2f ms   max %7.2f ms"
            % (
                entry["name"], entry["min_ms"], entry["median_ms"],
                entry["p90_ms"], entry["max_ms"],
            )
        )
    print("  %-12s %s" % ("icmp", line or "unavailable (no ping, or filtered)"))

    # The comparison GPU-120 exists to make, printed rather than left to be
    # done from memory later.
    median = results[0].get("median_ms")
    if median:
        print()
        print(
            "  bench/round_link_cost.py assumes 200 ms; measured %.0f ms "
            "(%.2fx). Its default is %s."
            % (
                median, median / 200.0,
                "about right" if 0.5 <= median / 200.0 <= 2.0 else "WRONG for this pair",
            )
        )

    print()
    print(json.dumps({"peer": peer, "icmp": line, "arms": results}))


if __name__ == "__main__":
    main()
