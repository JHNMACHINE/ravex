"""GPU-115: a whole round, end to end, over real sockets - the tests that run several nodes or ranks (GPU-256).

They start processes, or play nodes that talk over real sockets with
deadlines; see tests/distributed/conftest.py. The rest of this subject,
and the helpers both halves use, are in tests/unit/test_dist_round.py.
"""

import os
import threading
import time

import pytest

torch = pytest.importorskip("torch")

from ravex._dist.exchange import DeltaExchange, close_round  # noqa: E402
from ravex._dist.outer import OuterLoop  # noqa: E402

from test_dist_exchange import FakeStore  # noqa: E402
from test_dist_round import a_model, a_problem, loss_of, train  # noqa: E402


class Node(threading.Thread):
    """One participant: a model, a shard, a listener, and its own clock."""

    def __init__(self, rank, store, shard, rounds, inner, world, root,
                 dies_after=None, save_dtype=None, deadline=30, error_feedback=False):
        super().__init__(daemon=True)
        self.rank = rank
        self.store = store
        self.shard = shard
        self.rounds = rounds
        self.inner = inner
        self.peers = [r for r in range(world) if r != rank]
        self.dies_after = dies_after
        #: Apply this round, then vanish at once: listener shut, no linger -
        #: what `os._exit` right after a round looks like to the peers.
        self.vanish_after = None
        self.vanished = False
        self.model_at_vanish = None
        self.deadline = deadline
        self.error_feedback = error_feedback
        self.model = a_model()
        self.exchange = DeltaExchange(
            rank, store, root=os.path.join(root, "rank%d" % rank),
            node="n%d" % rank, patience=15.0, save_dtype=save_dtype,
        )
        self.loop = None
        self.on_loop = None
        self.reports = []
        self.error = None

    def run(self):
        try:
            assert self.exchange.start()
            self.loop = OuterLoop(self.model, inner_steps=self.inner, node="n%d" % self.rank)
            if self.on_loop is not None:
                self.on_loop(self.loop)
            optimizer = None
            for round_number in range(self.rounds):
                if self.dies_after is not None and round_number >= self.dies_after:
                    return  # the box went away, with no notice and no cleanup
                optimizer = train(
                    self.model, *self.shard, self.inner,
                    round_number * self.inner, optimizer,
                )
                for _ in range(self.inner):
                    self.loop.record_step()
                self.reports.append(
                    close_round(
                        self.loop, self.exchange, self.peers,
                        time.monotonic() + self.deadline,
                        error_feedback=self.error_feedback,
                    )
                )
                if self.vanish_after is not None and round_number >= self.vanish_after:
                    self.model_at_vanish = {
                        k: v.detach().clone() for k, v in self.model.state_dict().items()
                    }
                    self.vanished = True
                    return
        except BaseException as exc:  # surfaced on the test's thread
            self.error = exc
        finally:
            # A node that stops serving the moment it is done takes its last
            # report with it - see DeltaExchange.close.
            if self.vanished:
                self.exchange.close()
            else:
                self.exchange.close(linger=5, expect=self.peers)


def run_nodes(root, world=2, rounds=4, inner=15, dies_after=None):
    store = FakeStore()
    x, y = a_problem()
    size = len(x) // world
    nodes = [
        Node(
            rank, store,
            (x[rank * size : (rank + 1) * size], y[rank * size : (rank + 1) * size]),
            rounds, inner, world, root,
            dies_after=dies_after if rank == world - 1 else None,
        )
        for rank in range(world)
    ]
    for node in nodes:
        node.start()
    for node in nodes:
        node.join(180)
    for node in nodes:
        assert not node.is_alive(), "rank %d never finished" % node.rank
        if node.error is not None:
            raise node.error
    return nodes, (x, y)


def test_two_nodes_train_one_model_over_sockets(tmp_path):
    """The end-to-end claim: real deltas, real sockets, one model at the end."""
    nodes, (x, y) = run_nodes(str(tmp_path), world=2, rounds=4, inner=15)

    assert loss_of(nodes[0].model, x, y) < loss_of(a_model(), x, y) / 2

    left = dict(nodes[0].model.named_parameters())
    for name, right in nodes[1].model.named_parameters():
        assert torch.allclose(left[name], right, atol=1e-5), name

    for node in nodes:
        assert [r["nodes"] for r in node.reports] == [2, 2, 2, 2]


def test_three_nodes_all_see_each_other(tmp_path):
    nodes, _ = run_nodes(str(tmp_path), world=3, rounds=3, inner=10)
    for node in nodes:
        assert [r["nodes"] for r in node.reports] == [3, 3, 3]


def test_a_node_that_dies_mid_run_does_not_stop_the_others(tmp_path):
    """The sentence the whole issue exists for, with a real socket going away.

    Rank 2 stops after two rounds without saying anything - no leave, no
    cleanup, its advertised address still on the store. The survivors have to
    close the rounds after that on their own and keep converging.
    """
    nodes, (x, y) = run_nodes(str(tmp_path), world=3, rounds=4, inner=10, dies_after=2)
    survivors = nodes[:2]

    for node in survivors:
        assert len(node.reports) == 4, "a survivor stopped when the third left"
        assert [r["nodes"] for r in node.reports] == [3, 3, 2, 2]

    left = dict(survivors[0].model.named_parameters())
    for name, right in survivors[1].model.named_parameters():
        assert torch.allclose(left[name], right, atol=1e-5), name
    assert loss_of(survivors[0].model, x, y) < loss_of(a_model(), x, y) / 2


def two_halves(world):
    x, y = a_problem()
    size = len(x) // world
    return [(x[r * size:(r + 1) * size], y[r * size:(r + 1) * size]) for r in range(world)]


def run_all(nodes):
    for node in nodes:
        node.start()
    for node in nodes:
        node.join(180)
        assert not node.is_alive(), "rank %d never finished" % node.rank


def same_model(a, b):
    left = dict(a.model.named_parameters())
    for name, right in b.model.named_parameters():
        assert torch.allclose(left[name], right, atol=1e-5), name


def failing_fetch(node, owner, times=None, stall=0.0):
    """Make ``node``'s direct fetches of ``owner``'s rounds run out of time.

    Through the real ``fetch``, handed a deadline already spent - the branch a
    slow link takes. ``stall`` first, so this node reaches the round's decision
    after the others have made it. Relays go through untouched.
    """
    original = node.exchange.fetch
    failed = []

    def fetch(peer, round_number, expected, deadline, kind=0, owner_=None, **kw):
        relay_owner = kw.pop("owner", owner_)
        if peer == owner and kind == 0 and (times is None or len(failed) < times):
            failed.append(round_number)
            time.sleep(stall)
            return original(peer, round_number, expected, time.monotonic(), kind)
        return original(peer, round_number, expected, deadline, kind, owner=relay_owner)

    node.exchange.fetch = fetch
    return failed


def test_a_one_way_timeout_between_live_nodes_does_not_split_the_model(tmp_path):
    """GPU-140 without anyone dying.

    The rule the round rested on was that a node which did not contribute is
    missing from *everybody's* list at once. That holds for a node that is
    gone; it does not hold for a link that is slow in one direction. Here
    rank 1's fetch of rank 0's first report runs out of time while rank 0
    fetches rank 1's normally, and both are alive for the whole run.

    Before the round's set was decided on the store, rank 0 averaged {0, 1},
    rank 1 averaged {1}, and they held two models from then on - both losses
    falling, nothing raised. Now both apply the one set the store decided.
    """
    store = FakeStore()
    nodes = [Node(r, store, shard, 3, 10, 2, str(tmp_path))
             for r, shard in enumerate(two_halves(2))]
    failed = failing_fetch(nodes[1], owner=0, times=1)

    run_all(nodes)
    for node in nodes:
        if node.error is not None:
            raise node.error

    assert failed, "the one-way timeout was never injected"
    assert [r["nodes"] for r in nodes[0].reports] == [r["nodes"] for r in nodes[1].reports]
    same_model(*nodes)


def test_a_report_the_owner_cannot_hand_over_is_relayed_by_one_that_holds_it(tmp_path):
    """The recovery half: rank 2 never reaches rank 0, rank 1 always does.

    Rank 2 stalls before giving up on rank 0, so the round is decided by a
    node that holds rank 0's report and names it. Rank 2 then has to average a
    report it cannot fetch from its owner, and gets it from a node that has it.
    """
    store = FakeStore()
    nodes = [Node(r, store, shard, 2, 10, 3, str(tmp_path))
             for r, shard in enumerate(two_halves(3))]
    failing_fetch(nodes[2], owner=0, stall=1.0)

    run_all(nodes)
    for node in nodes:
        if node.error is not None:
            raise node.error

    assert [r["nodes"] for r in nodes[2].reports] == [3, 3]
    assert sum(r["recovered"] for r in nodes[2].reports) == 2
    same_model(nodes[0], nodes[2])
    same_model(nodes[1], nodes[2])


def test_a_node_that_cannot_get_the_decided_set_stops_instead_of_diverging(tmp_path):
    """And when nobody can hand it over: out, loudly, rather than a second model.

    Three nodes. Rank 2 reaches nobody for rank 0's report - not rank 0, not a
    relay - and gives up late, so the round is decided by a node that holds it.
    Rank 2 cannot average that set, and the only non-divergent move left to it
    is to leave, which is a departure, and a departure is safe.

    It used to be two nodes, with rank 0 proposing a set that named its own
    report while rank 1 could not fetch it. That can no longer happen: a
    proposer names itself only once every peer it names has its report, so
    there rank 1 proposes {1}, rank 0 applies that, and nobody leaves.
    """
    from ravex._dist.exchange import RoundSplitError
    from ravex._dist.membership import MembershipError

    store = FakeStore()
    nodes = [Node(r, store, shard, 2, 10, 3, str(tmp_path), deadline=6)
             for r, shard in enumerate(two_halves(3))]
    failing_fetch(nodes[2], owner=0, stall=1.0)
    nodes[2].exchange.recover = lambda *args, **kwargs: None

    run_all(nodes)

    assert isinstance(nodes[2].error, RoundSplitError)
    assert isinstance(nodes[2].error, MembershipError), "the runtime must not swallow it"
    assert nodes[2].reports == [], "it applied a round it could not agree on"
    for survivor in nodes[:2]:
        assert survivor.error is None
        assert survivor.reports[0]["nodes"] == 3
    same_model(nodes[0], nodes[1])


def test_a_node_that_cannot_compute_its_delta_still_takes_the_others_step(tmp_path):
    """GPU-142: a failure before the decision used to skip the round on this
    node alone - the others took the step, this one did not, two models. Now
    it offers nothing, is left out of the set, and applies the others'."""
    store = FakeStore()
    nodes = [Node(r, store, shard, 3, 10, 2, str(tmp_path), deadline=5)
             for r, shard in enumerate(two_halves(2))]

    failed = []

    def install():
        original = nodes[1].loop.contribution

        def contribution():
            if not failed:
                failed.append(nodes[1].loop.round_number)
                raise RuntimeError("a bug of ours")
            return original()

        nodes[1].loop.contribution = contribution

    nodes[1].on_loop = lambda loop: install()

    run_all(nodes)
    for node in nodes:
        if node.error is not None:
            raise node.error

    assert failed, "the failure was never injected"
    assert [r["nodes"] for r in nodes[0].reports] == [r["nodes"] for r in nodes[1].reports]
    assert 1 in [r["nodes"] for r in nodes[1].reports], "rank 1 was not left out"
    same_model(*nodes)


def test_the_proposer_vanishing_after_its_round_does_not_strand_the_survivor(tmp_path):
    """Found on the EU+US pair, 2026-09-21, in `110-rendezvous.sh kill0`.

    Node 0 decided round 3 over [0, 1], applied it and died before node 1 had
    taken its report. Nobody alive held that report, so node 1 could neither
    average the decided set nor rejoin - the run's only other member was the
    dead one - and the run ended. With two nodes it did not have to: the other
    model died with its node. The proposer now names itself only once every
    peer it names has its report, so here node 0 waits for node 1's slow fetch
    before proposing, and node 1 carries on alone on the model node 0 left.
    """
    store = FakeStore()
    nodes = [Node(r, store, shard, 4, 10, 2, str(tmp_path), deadline=10)
             for r, shard in enumerate(two_halves(2))]
    nodes[0].vanish_after = 1

    original = nodes[1].exchange.fetch

    def slow_toward_zero(peer, round_number, expected, deadline, kind=0, owner=None):
        if peer == 0 and kind == 0:
            time.sleep(1.5)
        return original(peer, round_number, expected, deadline, kind, owner=owner)

    nodes[1].exchange.fetch = slow_toward_zero

    run_all(nodes)
    for node in nodes:
        if node.error is not None:
            raise node.error

    assert nodes[0].vanished
    survivor = nodes[1]
    assert [r["nodes"] for r in survivor.reports] == [2, 2, 1, 1], survivor.reports
    # Through the round node 0 last applied, they held one model.
    assert len(nodes[0].reports) == 2


def test_nodes_at_different_speeds_still_agree_on_one_model(tmp_path):
    """The heterogeneous round, end to end: different step counts, one model."""
    store = FakeStore()
    x, y = a_problem()
    half = len(x) // 2
    fast = Node(0, store, (x[:half], y[:half]), 3, 20, 2, str(tmp_path))
    slow = Node(1, store, (x[half:], y[half:]), 3, 4, 2, str(tmp_path))

    for node in (fast, slow):
        node.start()
    for node in (fast, slow):
        node.join(180)
        if node.error is not None:
            raise node.error

    assert [r["steps"] for r in fast.reports][0] in ([20, 4], [4, 20])
    for report in fast.reports:
        assert report["slowest"] == 4 and report["fastest"] == 20

    left = dict(fast.model.named_parameters())
    for name, right in slow.model.named_parameters():
        assert torch.allclose(left[name], right, atol=1e-5), name


def test_nodes_that_start_from_different_weights_end_up_on_one_model(tmp_path):
    """The seed round, and the failure it exists for is a silent one.

    An outer round applies the same averaged pseudo-gradient to whatever outer
    parameters each node holds, so a difference in the *starting* ones is never
    touched again by anything: it is a constant added to one node's model for
    the whole run. The loss falls, every round closes over everybody, and the
    two nodes are training two models a fixed distance apart.

    Here they deliberately start from different seeds - which is the stronger
    version of what really happens, where each node's first optimizer step has
    already moved its weights with its own data before the loop is built.
    """
    from ravex._dist.exchange import SEED_ROUND, adopt_outer_state

    store = FakeStore()
    x, y = a_problem()
    half = len(x) // 2
    shards = [(x[:half], y[:half]), (x[half:], y[half:])]

    models = [a_model(seed=3), a_model(seed=11)]
    before = [float(p.detach().sum()) for p in models[0].parameters()]
    other = [float(p.detach().sum()) for p in models[1].parameters()]
    assert before != pytest.approx(other), "the two started the same"

    exchanges, loops, errors = [], [], []
    for rank in range(2):
        exchange = DeltaExchange(
            rank, store, root=os.path.join(str(tmp_path), "r%d" % rank),
            node="n%d" % rank, patience=15.0,
        )
        assert exchange.start()
        exchanges.append(exchange)
        loops.append(OuterLoop(models[rank], inner_steps=10, node="n%d" % rank))

    def seed(rank):
        try:
            assert adopt_outer_state(
                loops[rank], exchanges[rank], 0, rank, time.monotonic() + 30
            )
            loops[rank].round_number = SEED_ROUND + 1
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=seed, args=(rank,), daemon=True)
               for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    if errors:
        raise errors[0]

    try:
        left = dict(models[0].named_parameters())
        for name, right in models[1].named_parameters():
            assert torch.allclose(left[name], right, atol=1e-6), name

        # And one real round on top, to show the agreement survives a round
        # rather than only surviving the seeding.
        def one_round(rank):
            try:
                train(models[rank], *shards[rank], 10, 0, None)
                for _ in range(10):
                    loops[rank].record_step()
                close_round(loops[rank], exchanges[rank], [1 - rank],
                            time.monotonic() + 30)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=one_round, args=(rank,), daemon=True)
                   for rank in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        if errors:
            raise errors[0]

        left = dict(models[0].named_parameters())
        for name, right in models[1].named_parameters():
            assert torch.allclose(left[name], right, atol=1e-5), name
    finally:
        for exchange in exchanges:
            exchange.close()


@pytest.mark.parametrize("feedback", [False, True], ids=["plain", "error-feedback"])
def test_a_block_quantized_round_leaves_every_node_on_one_model(tmp_path, feedback):
    """GPU-139: fp4 with a scale per block, with and without error feedback.

    Error feedback changes what each node *sends* - its delta plus last
    round's residual - but never what each node averages, which is still the
    published report of everyone, itself included. So the invariant is the
    same as for a cast, and just as exact.
    """
    store = FakeStore()
    x, y = a_problem()
    half = len(x) // 2
    nodes = [
        Node(rank, store,
             (x[rank * half : (rank + 1) * half], y[rank * half : (rank + 1) * half]),
             4, 15, 2, str(tmp_path), save_dtype="fp4_block", error_feedback=feedback)
        for rank in range(2)
    ]
    for node in nodes:
        node.start()
    for node in nodes:
        node.join(180)
        if node.error is not None:
            raise node.error

    assert loss_of(nodes[0].model, x, y) < loss_of(a_model(), x, y) / 2
    left = dict(nodes[0].model.named_parameters())
    for name, right in nodes[1].model.named_parameters():
        assert torch.equal(left[name], right), name
    assert bool(nodes[0].loop.residual) is feedback


def test_a_cast_round_still_leaves_every_node_on_one_model(tmp_path):
    """``save_dtype`` shrinks the report without splitting the nodes.

    The wire form of a delta is a cast of it, and each node has to average the
    cast one — including its own — or every node takes a slightly different
    outer step from the same round. That failure is the one
    ``adopt_outer_state`` exists to prevent, arriving through the other door:
    nothing raises, the loss falls, every round closes over both nodes, and the
    models drift apart by a quantization error per round for the rest of the
    run. bf16 rather than fp8 so the tolerance below is a real bound and not a
    formality.
    """
    store = FakeStore()
    x, y = a_problem()
    half = len(x) // 2
    nodes = [
        Node(rank, store,
             (x[rank * half : (rank + 1) * half], y[rank * half : (rank + 1) * half]),
             4, 15, 2, str(tmp_path), save_dtype="bf16")
        for rank in range(2)
    ]

    for node in nodes:
        node.start()
    for node in nodes:
        node.join(180)
        if node.error is not None:
            raise node.error

    assert loss_of(nodes[0].model, x, y) < loss_of(a_model(), x, y) / 2

    left = dict(nodes[0].model.named_parameters())
    for name, right in nodes[1].model.named_parameters():
        assert torch.equal(left[name], right), name


def test_the_round_report_says_where_its_seconds_went(tmp_path):
    """GPU-117: the split is in the report a real run produces.

    Not a bench's business alone. Until this existed the only place a round's
    network cost was ever printed was a loopback run, where it was 0.0 — so the
    number the whole architecture is chosen around had never been looked at.
    """
    nodes, _ = run_nodes(str(tmp_path), world=2, rounds=2, inner=10)

    for report in nodes[0].reports:
        for key in ("delta_seconds", "publish_seconds", "publish_wait_seconds",
                    "gather_seconds", "gather_wait_seconds", "apply_seconds"):
            assert key in report, key
            assert report[key] >= 0.0, key
        # The wait is a part of the gather and not a second span next to it.
        assert report["gather_wait_seconds"] <= report["gather_seconds"] + 1e-6
        assert report["publish_wait_seconds"] <= report["publish_seconds"] + 1e-6


def test_the_seed_round_leaves_one_model_even_when_the_report_is_cast(tmp_path):
    """``save_dtype`` must not put the source node a quantization from the rest.

    The peers adopt what came down the wire, which under a cast is not what the
    source wrote. If the source keeps its own uncast copy, the one node that
    did not go through the wire starts a quantization away from everybody else
    — from the first instant, and never touched again, because an outer round
    only ever applies the *same* averaged gradient to whatever each node holds.
    Which is the exact failure ``adopt_outer_state`` exists to prevent,
    reintroduced by the setting that shrinks the round.

    ``torch.equal`` and not ``allclose``: the whole claim is that they are one
    model, and a tolerance is how the previous version of this passed.
    """
    from ravex._dist.exchange import adopt_outer_state

    store = FakeStore()
    models = [a_model(seed=3), a_model(seed=11)]
    exchanges, errors = [], []
    loops = []
    for rank in range(2):
        exchange = DeltaExchange(
            rank, store, root=os.path.join(str(tmp_path), "r%d" % rank),
            node="n%d" % rank, patience=15.0, save_dtype="bf16",
        )
        assert exchange.start()
        exchanges.append(exchange)
        loops.append(OuterLoop(models[rank], inner_steps=10, node="n%d" % rank))

    def seed(rank):
        try:
            assert adopt_outer_state(
                loops[rank], exchanges[rank], 0, rank, time.monotonic() + 30
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=seed, args=(rank,), daemon=True)
               for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    for exchange in exchanges:
        exchange.close()
    if errors:
        raise errors[0]

    for name, value in loops[0].outer.items():
        assert torch.equal(value, loops[1].outer[name]), name
    left = dict(models[0].named_parameters())
    for name, right in models[1].named_parameters():
        assert torch.equal(left[name], right), name


def _seeded(store, tmp_path, loops, progress_of=lambda loop: (loop.round_number, loop.model_steps)):
    """Run the start of a run on every loop at once: the source decided on the
    store, then its state adopted. Returns each node's source."""
    from ravex._dist.exchange import seed_source
    from ravex._dist.membership import adopt_run_state

    exchanges, sources, errors = [], {}, []
    for rank in range(len(loops)):
        exchange = DeltaExchange(
            rank, store, root=os.path.join(str(tmp_path), "r%d" % rank),
            node="n%d" % rank, patience=15.0,
        )
        assert exchange.start()
        exchanges.append(exchange)

    def start(rank):
        try:
            deadline = time.monotonic() + 30
            source = seed_source(store, exchanges[rank].scope, rank, len(loops),
                                 progress_of(loops[rank]), deadline)
            sources[rank] = source
            assert adopt_run_state(loops[rank], exchanges[rank], source, rank, deadline)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=start, args=(rank,), daemon=True) for rank in range(len(loops))]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        if errors:
            raise errors[0]
    finally:
        for exchange in exchanges:
            exchange.close()
    return sources


def test_a_resumed_run_starts_from_the_node_that_got_furthest(tmp_path):
    """GPU-212: three nodes resume from checkpoints of different ages. They
    all go on from the newest - its outer parameters, its momentum, its round
    and its model's steps - and not from rank 0's, which is the oldest."""
    models = [a_model(seed=3), a_model(seed=11), a_model(seed=19)]
    loops = [OuterLoop(models[rank], inner_steps=10, node="n%d" % rank) for rank in range(3)]
    for rank, (round_number, steps) in enumerate([(2, 200), (4, 400), (3, 300)]):
        loops[rank].round_number = round_number
        loops[rank].model_steps = steps
        loops[rank].steps_this_round = 7
        loops[rank].optimizer.buffers = {
            name: torch.full_like(value, float(rank + 1)) for name, value in loops[rank].outer.items()
        }
    furthest = {name: value.clone() for name, value in loops[1].outer.items()}

    sources = _seeded(FakeStore(), tmp_path, loops)

    assert sources == {0: 1, 1: 1, 2: 1}
    for rank, loop in enumerate(loops):
        assert loop.round_number == 4 and loop.model_steps == 400, rank
        assert loop.steps_this_round == 0, rank
        for name, value in furthest.items():
            assert torch.equal(loop.outer[name], value), (rank, name)
            assert torch.equal(loop.optimizer.buffers[name], torch.full_like(value, 2.0)), (rank, name)
        # And in the model: what a node trained after the last round closed
        # is not the run's.
        for name, parameter in models[rank].named_parameters():
            assert torch.equal(parameter.detach(), furthest[name]), (rank, name)


def test_a_fresh_run_starts_from_rank_0_as_it_always_has(tmp_path):
    models = [a_model(seed=3), a_model(seed=11)]
    loops = [OuterLoop(models[rank], inner_steps=10, node="n%d" % rank) for rank in range(2)]
    first = {name: value.clone() for name, value in loops[0].outer.items()}

    assert _seeded(FakeStore(), tmp_path, loops) == {0: 0, 1: 0}
    for loop in loops:
        assert loop.round_number == 0
        for name, value in first.items():
            assert torch.equal(loop.outer[name], value), name
