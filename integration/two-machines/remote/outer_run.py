"""The outer loop, on two real machines, reached the way a user reaches it.

GPU-120, point 7 of GPU-113. Everything the outer loop is built out of has only
ever run on loopback, where the network is free — including the round report
that GPU-117 taught to say where its seconds went, which on loopback said
``0.0`` every time. This is the same loop over a link that costs something.

**Not FSDP, and not a bench.** The training function here mentions no rounds,
no deltas and no peers: it is a plain loop under ``@ravex.train_loop``, which is
the only way anyone outside this repository reaches the outer loop, and the
wiring under test is *when* a round closes. One dense model per node, one GPU
per node — the outer loop averages whole parameters between nodes, so sharding
inside a node is a different question that ``20-correctness.sh`` already asks.

**The round reports are taken from the logger's arguments, not from its text.**
``RavexRuntime._close_outer_round`` logs the split with ``%``-arguments, so
``record.args`` carries the nine numbers exactly as they were measured. Parsing
the rendered line would mean re-deriving numbers that are already there, at
one decimal place, and being wrong about it quietly.

    torchrun ... outer_run.py --rounds 6 --inner 50 --params 3e8
    python outer_run.py --rendezvous 10.1.2.3:29400 --until-round 8 ...

The second form is GPU-129: no torchrun and no process group, the nodes meet at
a ``ravex rendezvous`` and take their numbers there. ``110-rendezvous.sh``
drives it.

What comes home is ``rounds.node<rank>.json``: one object per round, directly
comparable with what ``bench/round_link_cost.py`` prints for the same fields.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time

import torch
import torch.nn as nn

#: The exact template ``_close_outer_round`` logs. Matched rather than the
#: rendered text so the numbers arrive unrounded.
ROUND_LINE = "Outer round %d took %.1fs over %d node(s)"

#: In the order ``_close_outer_round`` passes them. Zipped against the
#: record's args, so a field the runtime adds has to be added here in its
#: place: when GPU-140 put `decide` and `recover` before `apply`, a list that
#: had not moved would have read the decision time as the outer step, with
#: nothing to say so.
FIELDS = (
    "round", "total_seconds", "nodes", "gather_seconds", "gather_wait_seconds",
    "delta_seconds", "publish_seconds", "publish_wait_seconds",
    "decide_seconds", "recover_seconds", "apply_seconds",
)


class Heartbeat:
    """Says where the run is, every few seconds, while it is there.

    **The thing whose absence cost a rented afternoon.** A gather can sit for
    the whole deadline with nothing on stdout, and `on-both.sh` hands its output
    back only when the command ends - so "working" and "wedged" looked
    identical, and the only way to tell was to ssh in and read `ps`. This
    writes one line to a file on the box that `watch.sh` tails from the laptop,
    so the question is answered by looking rather than by guessing.

    A daemon thread, so it can never be the reason the process does not exit.
    """

    def __init__(self, path, rank, every=10.0):
        self.path = path
        self.rank = rank
        self.every = every
        self.what = "starting"
        self.since = time.time()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def say(self, what):
        """Mark a new stage. The heartbeat then reports how long it has lasted."""
        self.what = what
        self.since = time.time()
        self._write("%s" % what)

    def stop(self):
        self._stop.set()

    def _write(self, text):
        line = "%s node%d %s" % (
            time.strftime("%H:%M:%S", time.gmtime()), self.rank, text
        )
        try:
            with open(self.path, "a") as handle:
                handle.write(line + os.linesep)
        except OSError:
            pass
        print("  " + line, flush=True)

    def _run(self):
        while not self._stop.wait(self.every):
            self._write("... %s, %.0fs so far" % (self.what, time.time() - self.since))


class Rounds(logging.Handler):
    """Every round's split, kept as numbers."""

    def __init__(self, heartbeat=None):
        super().__init__()
        self.seen = []
        self.other = []
        self.heartbeat = heartbeat

    def emit(self, record):
        message = record.msg if isinstance(record.msg, str) else ""
        if message.startswith(ROUND_LINE) and record.args:
            if len(record.args) != len(FIELDS):
                raise RuntimeError(
                    "the runtime's round line has %d fields and this reads %d; "
                    "update FIELDS" % (len(record.args), len(FIELDS))
                )
            entry = dict(zip(FIELDS, record.args))
            entry["at"] = time.time()
            self.seen.append(entry)
            if self.heartbeat is not None:
                self.heartbeat.say(
                    "round %s closed over %s node(s) in %.2fs (%.2fs network)"
                    % (entry["round"], entry["nodes"], entry["total_seconds"],
                       entry["gather_seconds"])
                )
        elif record.levelno >= logging.WARNING:
            self.other.append(record.getMessage())
            if self.heartbeat is not None:
                self.heartbeat._write("WARN " + record.getMessage()[:160])


def build_model(target_params, hidden):
    """A stack of square Linears sized to a parameter count.

    The same shape ``train_fsdp_cuda.py`` uses, so a delta measured here is
    comparable with a checkpoint measured there.
    """
    per_layer = hidden * hidden + hidden
    layers = max(2, round(target_params / per_layer))
    blocks = []
    for _ in range(layers):
        blocks += [nn.Linear(hidden, hidden), nn.GELU()]
    return nn.Sequential(*blocks), layers


def make_teacher(hidden, held_out=512):
    """Something to learn, and a held-out set nobody trains on.

    **Why this exists (2026-09-15).** Every phase that ran the outer loop on
    two continents trained on noise — inputs and targets both `randn`,
    unrelated — so it measured rounds, nodes and seconds and could not say
    whether anything was learned. This is the smallest task that can: a fixed
    linear map scaled to unit variance, identical on every node because it is
    drawn from a fixed seed, and a held-out set drawn from the same seed that
    no node ever trains on. Each node's *training* inputs stay its own.
    """
    generator = torch.Generator().manual_seed(1234)
    weight = torch.randn(hidden, hidden, generator=generator) / hidden ** 0.5

    def teacher(x):
        return x @ weight.T

    held_x = torch.randn(held_out, hidden, generator=generator)
    return teacher, held_x, teacher(held_x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--inner", type=int, default=50)
    parser.add_argument("--params", type=float, default=3e8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--deadline", type=int, default=900)
    parser.add_argument("--save-dtype", default="", help="'' or bf16 or fp16")
    parser.add_argument("--root", required=True, help="where the round stores go")
    parser.add_argument("--out", required=True, help="where to write rounds.json")
    parser.add_argument(
        "--die-at-round", type=int, default=0,
        help="stop this rank dead at the start of this round; 0 never",
    )
    parser.add_argument(
        "--die-on-rank", type=int, default=1,
        help="which rank dies, when --die-at-round is set",
    )
    parser.add_argument(
        "--rendezvous", default="",
        help="host:port of a `ravex rendezvous`; no torchrun, no process group",
    )
    parser.add_argument("--job", default="gpu129")
    parser.add_argument("--min-nodes", type=int, default=2)
    parser.add_argument(
        "--node-index", type=int, default=0,
        help="with --rendezvous: this box's index, for file names only",
    )
    parser.add_argument("--name", default="", help="names the rounds file")
    parser.add_argument(
        "--until-round", type=int, default=0,
        help="stop once this round has closed, instead of after --rounds",
    )
    parser.add_argument(
        "--task", default="noise", choices=("noise", "teacher"),
        help="noise: measures the round only; teacher: something to learn, "
        "with the held-out loss and a parameter hash logged at every round",
    )
    args = parser.parse_args()

    import torch.distributed as dist

    if args.rendezvous:
        # GPU-129: no process group at all. The node's number comes from the
        # rendezvous; this index names files and the heartbeat, nothing more.
        rank, world = args.node_index, 0
    else:
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))

        # gloo, not nccl: the outer loop takes only addresses and the job token
        # from the process group and moves every byte on its own sockets, so the
        # group never carries a tensor. A nccl group here would add a second
        # network path to go wrong for no benefit.
        dist.init_process_group("gloo", rank=rank, world_size=world)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    beat = Heartbeat(os.path.join(args.out, "phase.log"), rank).start()
    beat.say(
        "%s, device %s"
        % ("meeting at the rendezvous %s" % args.rendezvous if args.rendezvous
           else "built the process group", device.type)
    )
    rounds = Rounds(heartbeat=beat)

    import ravex

    os.makedirs(args.root, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    # Ravex walks up from the working directory for its config, and the round
    # stores go under --root. Kept apart per rank because on one box two ranks
    # sharing a directory would be measuring a filesystem, not a network.
    cwd = os.path.join(args.root, "cwd")
    os.makedirs(cwd, exist_ok=True)
    os.chdir(cwd)

    started = time.time()
    stopped_early = {"at": None}

    @ravex.train_loop(
        outer_loop=True,
        outer_inner_steps=args.inner,
        outer_root=os.path.join(args.root, "rounds"),
        outer_deadline=args.deadline,
        outer_save_dtype=(args.save_dtype or None),
        outer_rendezvous=(args.rendezvous or None),
        outer_job=args.job,
        outer_min_nodes=args.min_nodes,
        enabled=True,
        resume=False,
        checkpoint_every=10 ** 9,
        checkpoint_on_exit=False,
    )
    def train():
        # **The handler goes on here, inside, and not before the decorator
        # runs.** Ravex configures its own logger when it activates, and a
        # handler attached beforehand does not survive that - the phase then
        # comes home with an empty `rounds` list and nothing that looks like a
        # failure, which is the worst shape a result can have. The level has to
        # be set too: the round split is logged at INFO and a logger left at
        # its default inherits the root's WARNING. Both found here rather than
        # on a box being billed by the second.
        ravex_logger = logging.getLogger("ravex")
        ravex_logger.setLevel(logging.INFO)
        ravex_logger.addHandler(rounds)

        # Built inside the decorated function, which is where Ravex can see it.
        torch.manual_seed(3)  # same weights on both nodes to start
        model, layers = build_model(args.params, args.hidden)
        model = model.to(device)
        if args.task == "teacher":
            # Adam where there is something to learn: SGD at this lr barely
            # moves a deep GELU stack in the handful of rounds a rented box can
            # afford, and the question is whether the loss falls.
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

        dense = sum(p.numel() for p in model.parameters())
        if rank == 0:
            print(
                "model: %.2fB parameters, %d layers, hidden %d; a round moves "
                "about %.1f MB of fp32 delta per node"
                % (dense / 1e9, layers, args.hidden, dense * 4 / 1e6)
            )

        # **A DataLoader, and it is load-bearing rather than realistic
        # decoration.** A round is flagged inside ``optimizer.step()`` and
        # closed at the *next batch boundary*, because closing it mid-step
        # would write the outer parameters into the model underneath an
        # optimizer that has not finished. That boundary comes from iterating a
        # DataLoader - so a training loop over pre-batched tensors has to hand
        # that moment over itself with `ravex.batch_boundary()` (GPU-123), and
        # one that does neither flags rounds nobody ever closes while looking
        # healthy throughout. `tests/test_outer_train_loop.py` covers both
        # shapes; a phase here that called the boundary by hand would be
        # measuring the one a user reaching for a DataLoader does not have.
        #
        # Each node draws its own data, which is the point: identical shards
        # would make the average a no-op, and a round that moved nothing would
        # look like a round that worked.
        generator = torch.Generator().manual_seed(11 + rank)
        samples = args.batch * args.inner * (max(args.rounds, args.until_round + 1) + 1)
        inputs = torch.randn(samples, args.hidden, generator=generator)
        if args.task == "teacher":
            teacher, held_x, held_y = make_teacher(args.hidden)
            targets = teacher(inputs)
        else:
            teacher = held_x = held_y = None
            targets = torch.randn(samples, args.hidden, generator=generator)
        data = torch.utils.data.TensorDataset(inputs, targets)
        loader = torch.utils.data.DataLoader(
            data, batch_size=args.batch, shuffle=False, num_workers=0
        )
        loss_fn = nn.MSELoss()

        evals = []

        def evaluate(label):
            """Held-out loss, and a hash of every parameter, right now.

            Called at the top of an iteration just after a round closed and
            before the next step: the one moment every node must hold the
            same parameters. The loss says whether the run is learning; the
            hash says whether the two continents hold one model — bit for
            bit, which a matching loss alone would not prove.
            """
            was_training = model.training
            model.eval()
            total = 0.0
            with torch.no_grad():
                for start in range(0, len(held_x), 256):
                    xb = held_x[start:start + 256].to(device)
                    yb = held_y[start:start + 256].to(device)
                    total += float(loss_fn(model(xb), yb)) * len(xb)
                digest = hashlib.sha256()
                for parameter in model.parameters():
                    digest.update(parameter.detach().float().cpu().numpy().tobytes())
            model.train(was_training)
            entry = {
                "label": label,
                "heldout_loss": total / len(held_x),
                "params_sha256": digest.hexdigest()[:16],
                "at": time.time(),
            }
            evals.append(entry)
            beat.say("EVAL %s: held-out loss %.6f, params %s"
                     % (label, entry["heldout_loss"], entry["params_sha256"]))

        wanted = args.inner * args.rounds
        if args.until_round:
            # A round number rather than a step count is what lets a node that
            # joined late stop on the same round as the others, and so hold the
            # same model at the end (GPU-129).
            wanted = len(loader)
        from ravex._runtime import get_runtime

        runtime = get_runtime()
        done = 0
        evaluated_at = None
        if teacher is not None:
            evaluate("before training")
        beat.say("training round 0 (%d steps per round)" % args.inner)
        for batch, target in loader:
            if done >= wanted:
                break
            outer = runtime._outer if runtime is not None else None
            # Before the stop check, so the last round closed is evaluated too.
            if teacher is not None and outer and outer.round_number != evaluated_at:
                evaluated_at = outer.round_number
                evaluate("after round %d" % (outer.round_number - 1))
            if args.until_round and outer and outer.round_number > args.until_round:
                break
            round_now = done // args.inner
            if done % args.inner == 0 and done:
                # Said before the round closes, so a long silence after this
                # line is the exchange and not the training.
                beat.say("training round %d" % round_now)
            if (args.die_at_round and (args.rendezvous or rank == args.die_on_rank)
                    and round_now == args.die_at_round
                    and done % args.inner == 0):
                # The way a preempted box goes: no leave, no cleanup, the
                # advertised address still on the store.
                print("rank %d going away at round %d" % (rank, round_now),
                      flush=True)
                stopped_early["at"] = round_now
                sys.stdout.flush()
                os._exit(9)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(batch.to(device)), target.to(device)).backward()
            optimizer.step()
            done += 1
        return evals

    evals = train() or []
    beat.say("done training; writing results")
    beat.stop()

    payload = {
        "rank": rank,
        "world": world,
        "params": args.params,
        "hidden": args.hidden,
        "inner": args.inner,
        "save_dtype": args.save_dtype or None,
        "task": args.task,
        "evals": evals,
        "wall_seconds": round(time.time() - started, 2),
        "rounds": rounds.seen,
        "warnings": rounds.other[-40:],
    }
    path = os.path.join(args.out, "rounds.%s.json" % (args.name or "node%d" % rank))
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)

    print()
    print("round  nodes   total   network  (waiting)   delta  publish  decide  recover   apply")
    for entry in rounds.seen:
        print(
            "%5d  %5d  %6.2f  %8.2f  %9.2f  %6.2f  %7.2f  %6.2f  %7.2f  %6.2f"
            % (
                entry["round"], entry["nodes"], entry["total_seconds"],
                entry["gather_seconds"], entry["gather_wait_seconds"],
                entry["delta_seconds"], entry["publish_seconds"],
                entry["decide_seconds"], entry["recover_seconds"],
                entry["apply_seconds"],
            )
        )
    print()
    print("wrote %s" % path)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
