"""What a round costs when the link is not loopback.

GPU-117, under GPU-113. Everything the outer loop was built out of — the round
arithmetic, the transport with a deadline, the wiring into ``@ravex.train_loop``
— has only ever run over loopback, where the network is free. Every round
measured so far printed ``took 0.0s``, so the one number the whole architecture
is chosen around, **what an exchange costs on the link this exists to
tolerate**, has never been observed.

This is that measurement, and it is deliberately done on one machine. Renting
two boxes in two regions to discover something a container already knows is the
expensive way round; what is needed is a link that is slow on purpose.

**The link is a relay, not ``tc``.** On Linux ``tc netem``/``tbf`` in a
container is the more faithful instrument, and it needs ``NET_ADMIN`` and a
Linux kernel. What is here instead is a TCP relay in front of each node's
listener that forwards at a fixed rate with a fixed one-way delay: clumsier,
and measurable on any machine without privileges, which is what makes it
something anyone can re-run. It models one node's uplink shared by every peer
fetching from it, which is what a rented box has. What it does *not* model is
loss, reordering, or a congestion window ramping up — so these numbers are the
optimistic end of a real link, and a real link cannot beat them.

**Four things it answers**, and three of them are claims already written into
the code that loopback could never exercise:

*What a round costs.* Seconds of network against seconds of compute, which is
the ratio that decides ``outer_inner_steps``. Reported as the H at which the
network is a quarter of the round, so it can be read against GPU-113's
arithmetic — 429 s for a 1B model in bf16 at 7 MB/s.

*Whether the deadline bounds a slow transfer.* ``outer_deadline`` has only ever
fired against peers that were not there. A transfer that *takes* ten minutes is
a different case from one that never starts, and the ``starve`` arm is a
deadline set below the time the bytes need. What matters is not only that it
fires but that the round after it still works: a fetch cut mid-store leaves a
half-written directory on the receiving side.

*What the publish lock costs.* ``DeltaExchange._io_lock`` makes a publish wait
for an in-flight fetch, and the comment on it says that is close to free. It is
close to free on loopback, where a fetch lasts milliseconds. Here a fetch lasts
as long as a model transfer, and ``publish_wait_seconds`` says what that is
worth.

*What the round rendezvous costs.* A node asking for round N while its peer is
still in N-1 waits. The ``skew`` arm makes one node take twice as long, so the
waiting is a real span rather than microseconds.

    python bench/round_link_cost.py
    python bench/round_link_cost.py --params 32 --rates 7 --rounds 5

The default is 7 MB/s — measured on RunPod global networking, nine days apart,
and the number GPU-113's premise rests on — an unthrottled arm to stand next to
it, and 200 ms of round trip that loopback does not have.

**Measured**, 2026-09-07, two processes, 8M parameters (a 15.8 MB report on the
wire), 20 inner steps, 200 ms round trip. Seconds per round:

=========== ======== ========= ======== ======== ========== =======
arm         compute  network   waiting  moving   lock       MB/s
=========== ======== ========= ======== ======== ========== =======
loopback    0.42     0.64      0.31     0.33     0.00       48.2
50 MB/s     0.51     1.71      0.30     1.40     0.00       11.3
7 MB/s      0.51     3.28      0.31     2.97     0.00       **5.3**
asymmetric  0.34     2.11      0.45     1.66     **0.72**   8.0
skew        0.59     3.46      0.39     3.07     0.00       5.3
=========== ======== ========= ======== ======== ========== =======

``asymmetric`` throttles one node only; ``skew`` gives one node twice the inner
steps. The ``starve`` and ``after starve`` arms are below.

**The link is the cost, and the transport gets 76% of it.** 15.8 MB at 7 MB/s
is 2.26 s of bytes; the round spends 2.97 s moving them, so about 0.7 s goes on
decode, the store and the manifest, independent of the rate. Which settles the
question the issue was opened for: extrapolated, a 1B model's bf16 delta —
2 GB — takes about **375 s per round** on that link, against the 429 s per
*step* GPU-113 computed for synchronous training. The premise holds, and the
saving really is H.

**The rendezvous is not the problem, and that is not what was expected.** A
node two times slower than its peer costs the fast one 0.08-0.19 s of extra
waiting, not the 0.3 s its compute differs by: the slow node also *starts*
earlier, because its own fetch finished earlier, and a constant speed
difference is absorbed into the pipeline within a round. The 0.31 s floor under
every arm is the dial and the round trip, and it does not grow with the link.

**The publish lock is the problem.** With both nodes on the same link it is
never contended — neither can get ahead of the other, because each is waiting
for the other's report. Throttle one node only and its publish blocks **2.2 to
2.3 s** waiting for a peer's fetch to drain, against a round whose own transfer
is 0.30 s. See ``DeltaExchange._io_lock``.

**The deadline bounds a slow transfer, and the round after it survives.** The
starve arm cut the fetch at 0.85 s of a ~1 s deadline having moved 3 of
15.8 MB; both nodes closed the round over themselves alone and carried on. The
next round came back to two nodes and re-sent 28.7 MB — the abandoned round's
snapshot as well as the new one — so a cut fetch costs a resend and not a run.

**What the instrument cannot do.** Above roughly 30 MB/s the relay's own pacing
granularity, not the configured rate, is what the numbers measure: the 50 MB/s
arm reaches 11.3 MB/s and means nothing. The bands worth reading here are 7 MB/s
and unthrottled.
"""

import argparse
import json
import logging
import os
import queue
import shutil
import socket
import tempfile
import threading
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ravex._dist.exchange import (
    ADDRESS_KEY,
    SEED_ROUND,
    DeltaExchange,
    adopt_outer_state,
    close_round,
)
from ravex._dist.outer import OuterLoop

MIB = 1 << 20

#: Bytes per relay read. Small enough that pacing has something to pace with:
#: at 7 MB/s a chunk is 4.5 ms, so the rate is enforced in steps well under the
#: latency being modelled rather than in one lump per round.
RELAY_CHUNK = 32 << 10

BATCH = 32


class Link(threading.Thread):
    """A fixed-rate, fixed-latency TCP relay in front of one node's listener.

    Every connection through one ``Link`` shares one pacing queue, because a
    node has one uplink and its peers share it. Rate and latency are settable
    between rounds so a run can walk several bands without re-spawning the
    processes, re-seeding the models, or paying the cold first round again.

    Two threads per direction rather than one: a reader that consumes the link
    at the configured rate, and a writer that holds each chunk for the one-way
    delay before passing it on. One thread doing both would turn latency into a
    bandwidth cap — one chunk per delay — which is a different link entirely.
    """

    def __init__(self, upstream, rate=0.0, latency=0.0):
        super().__init__(daemon=True)
        self.upstream = upstream
        self.rate = float(rate)
        self.latency = float(latency)
        #: Bytes this node has served to its peers, which under N nodes of one
        #: size is also what it downloaded.
        self.relayed = 0
        self._reserved = 0.0
        self._pace = threading.Lock()
        self._stop = threading.Event()
        self._busy = 0
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(64)
        self.listener.settimeout(0.5)
        self.port = self.listener.getsockname()[1]

    def configure(self, rate, latency):
        self.rate = float(rate)
        self.latency = float(latency)
        with self._pace:
            self._reserved = 0.0

    def in_flight(self):
        """How many bytes a link this fast, with this delay, can hold.

        **Not a tuning knob, and getting it wrong hid a result.** A queue of a
        fixed number of chunks let 16 MB sit inside the relay, so the sending
        node finished its half of a transfer while half the store had not left
        the wire — and tearing the relay down at the end of a run then cut a
        receive that the exchange had every reason to believe was delivered. It
        printed as a round closing without a peer that was alive and idle,
        which is a product defect if you believe the instrument.

        Sized at a few times the bandwidth-delay product, which is the other
        half of the same lesson: a window *below* it throttles the link to
        ``window / delay`` regardless of the rate asked for, and the first
        version of this did exactly that — it made the unthrottled arm the
        slowest one in the table. The rate limiter is the reader; this only has
        to be wide enough to stay out of its way.
        """
        if not self.rate:
            return 1024
        product = self.rate * 2 * max(self.latency, 0.005)
        return max(8, int(4 * product / RELAY_CHUNK))

    def take(self, count):
        """Wait until the link has room for ``count`` bytes.

        Reservations are absolute times rather than a bucket refilled on read:
        oversleeping one chunk then costs the next one nothing, so the rate
        holds over a transfer instead of drifting below it by whatever the
        sleep granularity is.
        """
        rate = self.rate
        if not rate:
            return
        with self._pace:
            now = time.monotonic()
            start = self._reserved if self._reserved > now else now
            self._reserved = start + count / rate
            wait = start - now
        if wait > 0:
            time.sleep(wait)

    def run(self):
        while not self._stop.is_set():
            try:
                downstream, _ = self.listener.accept()
            except (socket.timeout, OSError):
                continue
            threading.Thread(
                target=self._relay, args=(downstream,), daemon=True
            ).start()

    def _relay(self, downstream):
        # The dial itself is a round trip the relay would otherwise make free.
        if self.latency:
            time.sleep(self.latency)
        try:
            upstream = socket.create_connection(self.upstream, timeout=30)
        except OSError:
            downstream.close()
            return
        self._busy += 1
        pumps = [
            threading.Thread(
                target=self._pump, args=(downstream, upstream, False), daemon=True
            ),
            threading.Thread(
                target=self._pump, args=(upstream, downstream, True), daemon=True
            ),
        ]
        for pump in pumps:
            pump.start()
        for pump in pumps:
            pump.join()
        self._busy -= 1
        for sock in (downstream, upstream):
            try:
                sock.close()
            except OSError:
                pass

    def _pump(self, source, destination, outbound):
        held = queue.Queue(maxsize=self.in_flight())

        def write():
            while True:
                item = held.get()
                if item is None:
                    try:
                        destination.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass
                    return
                release, block = item
                delay = release - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                try:
                    destination.sendall(block)
                except OSError:
                    return

        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        try:
            # To the end of the connection, never to `_stop`: a relay that
            # drops what it is carrying when it is asked to shut down reports
            # the transfer as lost by the *other* side.
            while True:
                block = source.recv(RELAY_CHUNK)
                if not block:
                    break
                self.take(len(block))
                if outbound:
                    self.relayed += len(block)
                held.put((time.monotonic() + self.latency, block))
        except OSError:
            pass
        held.put(None)
        writer.join()

    def stop(self, drain=30.0):
        """Stop accepting, and let what is already crossing finish."""
        deadline = time.monotonic() + drain
        while self._busy > 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        self._stop.set()
        try:
            self.listener.close()
        except OSError:
            pass


def model_for(millions, seed=3):
    """A stack whose delta is the size asked for, and whose steps cost something.

    Square layers: ``depth * width**2`` parameters, so the bytes on the wire and
    the matmuls per step both follow the one knob. What it learns is not the
    question here — GPU-118 is where convergence is measured — but it has to
    *train*, because a round's cost is only meaningful next to the compute it is
    amortised over.
    """
    torch.manual_seed(seed)
    depth = 4
    width = max(64, int((millions * 1e6 / depth) ** 0.5))
    layers = []
    for _ in range(depth):
        layers += [torch.nn.Linear(width, width, bias=False), torch.nn.Tanh()]
    return torch.nn.Sequential(*layers), width


def train(model, optimizer, width, steps, seed):
    generator = torch.Generator().manual_seed(seed)
    loss_fn = torch.nn.MSELoss()
    for _ in range(steps):
        x = torch.randn(BATCH, width, generator=generator)
        y = torch.randn(BATCH, width, generator=generator) * 0.1
        optimizer.zero_grad()
        loss_fn(model(x), y).backward()
        optimizer.step()


def barrier(store, name, world, timeout=600.0):
    """Everybody through, before the link changes underneath a transfer."""
    store.add(name, 1)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if int(store.get(name)) >= world:
            return
        time.sleep(0.05)
    raise RuntimeError("%s: only %s of %d arrived" % (name, store.get(name), world))


def worker(rank, args, root, results):
    if args.log:
        # Ravex's own account of the round, next to the timings. A round that
        # closes over fewer nodes than are alive says so here and nowhere else.
        logging.basicConfig(
            level=getattr(logging, args.log.upper(), logging.INFO),
            format="%(asctime)s rank" + str(rank) + " %(levelname)s %(message)s",
        )
    # Each process gets its share of the cores. Two processes each taking every
    # core would make the compute half of the ratio a measurement of
    # oversubscription.
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // args.nodes))

    # A rendezvous store is all the exchange wants from torch.distributed: it
    # takes peer addresses and the job token from one, and moves every byte on
    # its own sockets. No process group, so no collective can hide a cost here.
    # (`use_libuv=False` because some CPU wheels are built without it, and the
    # store refuses to start rather than fall back.)
    store = dist.TCPStore(
        "127.0.0.1", args.port, args.nodes, rank == 0,
        timeout=timedelta(seconds=300), use_libuv=False,
    )
    model, width = model_for(args.params)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    peers = [peer for peer in range(args.nodes) if peer != rank]

    exchange = DeltaExchange(
        rank, store, root=os.path.join(root, "rank%d" % rank),
        node="n%d" % rank, patience=120.0,
        save_dtype=args.save_dtype,
    )
    assert exchange.start(), "rank %d could not open the exchange" % rank

    # In front of the listener, and advertised in its place: every fetch from
    # this node now crosses a link with a speed. The exchange itself is
    # untouched — it dials what the store says, which is what the address being
    # a store key is for.
    link = Link(("127.0.0.1", exchange.listener.getsockname()[1]))
    link.start()
    store.set(ADDRESS_KEY % rank, ("127.0.0.1:%d" % link.port).encode("utf-8"))

    loop = OuterLoop(model, inner_steps=args.inner, node="n%d" % rank)

    def record(arm, report, compute, moved, rate, latency):
        results.append(
            {
                "arm": arm,
                "rank": rank,
                "round": report["round"],
                "rate_mb": rate,
                "rtt_ms": latency * 2000.0,
                "nodes": report["nodes"],
                "compute_seconds": compute,
                "delta_seconds": report["delta_seconds"],
                "publish_seconds": report["publish_seconds"],
                "publish_wait_seconds": report["publish_wait_seconds"],
                "gather_seconds": report["gather_seconds"],
                "gather_wait_seconds": report["gather_wait_seconds"],
                "apply_seconds": report["apply_seconds"],
                "bytes": moved,
                "inner": args.inner,
            }
        )

    def rounds(arm, count, rate, latency, deadline, slow=1.0, mine=None):
        link.configure((rate if mine is None else mine) * MIB, latency)
        barrier(store, "arm/%s" % arm, args.nodes)
        steps = max(1, int(args.inner * slow))
        for _ in range(count):
            before = link.relayed
            started = time.perf_counter()
            train(model, optimizer, width, steps, seed=loop.round_number * 97 + rank)
            for _ in range(steps):
                loop.record_step()
            compute = time.perf_counter() - started
            report = close_round(loop, exchange, peers, time.monotonic() + deadline)
            record(arm, report, compute, link.relayed - before, rate, latency)

    try:
        # Everybody starts from one set of outer parameters, which is a
        # model-sized transfer of its own and is what a real run pays before
        # round one. Unthrottled: it is startup, not a round.
        barrier(store, "seed", args.nodes)
        assert adopt_outer_state(
            loop, exchange, 0, rank, time.monotonic() + 300
        ), "rank %d never got the seed state" % rank
        # The seed went out under round 0, so training starts at 1 - the same
        # line `_runtime._build_outer_loop` has, and for the same reason: a
        # first round numbered 0 would ask a peer for the round its starting
        # parameters are already published under.
        loop.round_number = SEED_ROUND + 1

        latency = args.rtt / 2000.0
        for rate in args.rates:
            rounds("%g MB/s" % rate if rate else "loopback",
                   args.rounds, rate, latency, args.deadline)

        # `0` in --rates means unthrottled, so it is not a candidate for the
        # slow band the remaining arms want.
        slowest = min([rate for rate in args.rates if rate] or [0.0])

        if args.nodes > 1:
            # One node on the slow link and the rest unthrottled, which is the
            # only shape where `_io_lock` can contend: the fast node finishes
            # its gather while the slow one is still serving, trains, and
            # publishes the next round into a store a fetch still holds open.
            rounds("asymmetric", args.rounds, slowest, latency, args.deadline,
                   mine=slowest if rank == args.nodes - 1 else 0.0)

        if args.nodes > 1 and args.skew > 1:
            # One node twice the work: its peers reach the round first and wait
            # inside `_await_round`, which on loopback is a window of
            # microseconds and here is a window of seconds.
            rounds("skew", args.rounds, slowest, latency, args.deadline,
                   slow=args.skew if rank == args.nodes - 1 else 1.0)

        if args.starve > 0:
            # A deadline below the time the bytes need. Not an absent peer: a
            # peer that is answering, slowly, and gets cut off mid-store.
            paced = [
                row["gather_seconds"] for row in results
                if row["rank"] == rank and row["arm"] == "%g MB/s" % slowest
            ]
            short = max(0.5, args.starve * (paced[-1] if paced else 10.0))
            rounds("starve", 1, slowest, latency, short)
            # And then the rounds after it, at a deadline that fits, because a
            # fetch cut halfway leaves a directory the next one has to survive.
            rounds("after starve", 2, slowest, latency, args.deadline)
    finally:
        exchange.close(linger=30, expect=peers)
        link.stop()


def summarise(rows, nodes):
    """Per arm, what the round cost and what it implies for H."""
    arms = []
    for row in rows:
        if row["arm"] not in arms:
            arms.append(row["arm"])

    print()
    print(
        "%-14s %6s %8s %8s %8s %8s %7s %7s %8s %7s"
        % ("arm", "nodes", "compute", "network", "waiting", "moving", "pub",
           "lock", "MB/s", "H@25%")
    )
    summary = []
    for arm in arms:
        taken = [row for row in rows if row["arm"] == arm]
        count = len(taken)
        mean = lambda key: sum(row[key] for row in taken) / count  # noqa: E731
        network = mean("gather_seconds")
        waiting = mean("gather_wait_seconds")
        moving = network - waiting
        compute = mean("compute_seconds")
        moved = mean("bytes")
        inner = taken[0]["inner"]
        per_step = compute / inner if inner else 0.0
        # The network is a quarter of the round when the compute is three times
        # it, and the compute is H steps. The whole network, not just the
        # bytes: a node cannot train through a rendezvous either.
        implied = (3 * network / per_step) if per_step > 0 else float("nan")
        # Against the bytes only, which is what a link speed can be compared
        # with. Dividing by the whole gather would charge the link for a peer
        # that had not finished its round.
        achieved = (moved / MIB / moving) if moving > 0 else float("nan")
        print(
            "%-14s %6.1f %8.2f %8.2f %8.2f %8.2f %7.2f %7.2f %8.2f %7.0f"
            % (
                arm,
                mean("nodes"),
                compute,
                network,
                waiting,
                moving,
                mean("publish_seconds"),
                mean("publish_wait_seconds"),
                achieved,
                implied,
            )
        )
        summary.append(
            {
                "arm": arm,
                "rounds": count,
                "nodes": mean("nodes"),
                "compute_seconds": compute,
                "network_seconds": network,
                "waiting_seconds": waiting,
                "moving_seconds": moving,
                "publish_seconds": mean("publish_seconds"),
                "publish_wait_seconds": mean("publish_wait_seconds"),
                "megabytes": moved / MIB,
                "achieved_mb_per_second": achieved,
                "seconds_per_step": per_step,
                "inner_steps_for_a_quarter": implied,
            }
        )
    print()
    print("Seconds per round. network splits into waiting - the dial, the round")
    print("trip and a peer still finishing its own round - and moving, which is")
    print("the only part a link speed explains; MB/s is the bytes over that. pub")
    print("is the publish and lock the part of it spent waiting for an in-flight")
    print("fetch. H@25% is the inner steps at which the whole network would be a")
    print("quarter of the round, at this box's per-step compute and %d node(s)."
          % nodes)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--inner", type=int, default=20,
                        help="local steps per round")
    parser.add_argument("--params", type=float, default=8.0,
                        help="millions of parameters, so 8 is a 32 MB delta")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0, 50.0, 7.0],
                        metavar="MB_S",
                        help="link speeds to walk; 0 leaves it unthrottled")
    parser.add_argument("--rtt", type=float, default=200.0,
                        help="round trip, ms, split evenly between directions")
    parser.add_argument("--deadline", type=float, default=900.0)
    parser.add_argument("--skew", type=float, default=2.0,
                        help="the last node's inner steps, as a multiple")
    parser.add_argument("--starve", type=float, default=0.3,
                        help="the starve arm's deadline, as a fraction of a "
                             "measured gather; 0 skips it")
    parser.add_argument("--save-dtype", type=str, default=None,
                        help="cast the delta on the wire: bf16, fp8")
    parser.add_argument("--log", type=str, default=None,
                        help="ravex log level to print alongside: info, debug")
    parser.add_argument("--port", type=int, default=29841)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    root = tempfile.mkdtemp(prefix="ravex-round-link-")
    manager = mp.Manager()
    results = manager.list()
    try:
        mp.spawn(worker, args=(args, root, results), nprocs=args.nodes, join=True)
        rows = [dict(row) for row in results]
    finally:
        shutil.rmtree(root, ignore_errors=True)

    rows.sort(key=lambda row: (row["round"], row["rank"]))
    print()
    print("%d node(s), %.1fM parameters, %d inner steps, %.0f ms round trip"
          % (args.nodes, args.params, args.inner, args.rtt))
    print()
    print("%-14s %5s %6s %6s %8s %8s %8s %8s %8s %8s"
          % ("arm", "round", "rank", "nodes", "compute", "delta", "publish",
             "waiting", "moving", "MB"))
    for row in rows:
        print(
            "%-14s %5d %6d %6d %8.2f %8.2f %8.2f %8.2f %8.2f %8.1f"
            % (
                row["arm"], row["round"], row["rank"], row["nodes"],
                row["compute_seconds"], row["delta_seconds"],
                row["publish_seconds"], row["gather_wait_seconds"],
                row["gather_seconds"] - row["gather_wait_seconds"],
                row["bytes"] / MIB,
            )
        )

    # The first round of the run is the cold one: the peer directories are
    # empty, so the whole store crosses rather than the round's snapshot. Kept
    # in the per-round table above and out of the summary, where it would be a
    # startup's cost reported as an arm's.
    warm = [row for row in rows if row["round"] > 1]
    summary = summarise(warm, args.nodes)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"rounds": rows, "arms": summary}, handle, indent=2)


if __name__ == "__main__":
    main()
