# Changelog

## Unreleased

### Added

- **A node can join a run that is already going (GPU-121, under GPU-113).**

  The other half of GPU-113's point 4. The death was proved; the arrival had
  never been made to happen, and the two are not symmetric. A departure is safe
  to notice late — the node that went away is missing from *everybody's* list of
  reports at once. An arrival noticed by one node and not another is two nodes
  applying different averages to the same parameters for the rest of the run,
  with the loss falling on both and nothing raising.

  So membership is now **a pure function of the round number**, in
  `ravex._dist.membership`. A joiner writes one key holding the round it will
  first contribute to; every member reads those keys at every boundary and
  computes the peer set for round R as "the base ranks, plus every joiner whose
  round is at most R". Two members reading the store at completely different
  moments still get the same set for the same round, which a plain lookup does
  not give. `_build_outer_loop` computed `range(world)` once, where a joiner
  could appear nowhere.

  **No source is elected**, which is what makes rank 0 being dead a non-event:
  every member writes its outer parameters down at the boundary while a join is
  pending, and the joiner takes them from whichever answers first. A member
  that is gone is a connection that fails and the next name in the list.

  **The state carries the outer momentum**, not only the parameters — the same
  defect GPU-110 found with Adam's moments, one floor up. A node arriving with
  an empty buffer applies a different update to the same gradient from its very
  first round. Found by `torch.equal` failing on the joiner, by far more than a
  last-place difference.

  **An acknowledgement is what turns the last race into a wasted deadline.** A
  joiner publishes nothing until every base member has recorded that it accepted
  the announcement; short of that it stays silent, every member closes the round
  without it, and they all close the *same* round. A known limit falls out of
  that and is named rather than worked around: a run that has already lost a
  member cannot take a new one, because the lost member never acknowledges.

  The wire greeting gains a `kind` field, so a node can serve two different
  things without a round number having to mean two things.

- **Ravex performs the elastic rebuild itself (GPU-110, under GPU-107).**

  `ravex._dist.elastic` stopped at the group and rendezvous layer, and said in
  its own docstring why: ravex is only ever handed an already-constructed
  model, so it had no factory to rebuild one from. `@ravex.train_loop` wraps
  the function the model is born inside, which *is* that factory, and the half
  that was missing is now `regroup()`.

  It captures the full model **and optimizer** state through the same
  `gather_sharded_state` the `sharded_checkpoints: gather` path already uses —
  one way to unshard state, not two — tears the group down, brings it up at the
  new world size, and loads the state into a freshly built model under the new
  mesh. The Adam moments travelling with the weights are what make it a
  continuation and not a restart from a good place, and the test now fails if
  they come back empty.

  A rank that is **joining** passes no model and needs nothing shipped to it
  out of band: the state reaches it in the broadcast that happens once the new
  group is up, which is why the order is capture, destroy, init, share,
  rebuild. `tests/test_dist_elastic_remesh.py` covers a real 2 → 3 change and
  no longer carries the `multiprocessing.Queue` it used to need.

  Two mistakes that are silent in torch now raise `RegroupError` instead:
  capturing after `destroy_process_group()` (which returns wrong bytes and
  surfaces much later as `narrow unexpectedly changed concrete size`), and
  broadcasting from a source that holds nothing (which would put every rank
  back on random weights with nothing failing).

  `elastic=True` on the decorator still raises: what is missing is no longer
  the rebuild but re-entry — your loop holds `model` and `optimizer` in its own
  locals, and a regroup returns new objects ravex cannot assign into your
  frame. The message it raises says so.

### Fixed

- **A round is over the same parameters, not the same parameter *order*
  (GPU-122).** On any model bigger than a toy, every outer round was abandoned.

  `combine` compared `list(delta)` against `list(delta)`, so two contributions
  covering exactly the same parameters were rejected for disagreeing about
  their order — and they normally do. A node's own delta is built by
  `pseudo_gradient` in `named_parameters()` order and never round-trips;
  a peer's has been written to a moonclip store and read back, and arrives in
  the store's order. On the two-layer model every test and every loopback run
  used, the two coincide. On a 48-layer one they do not, and then the check
  raised every round, `_close_outer_round` caught it, `abandon_round` advanced
  the counter, and **every node trained alone for the rest of the run** — loss
  falling, nothing failing, one warning per round the only sign.

  The check still earns its place, so it now asks the question it meant to ask:
  membership of the set, with the offending parameter names in the message
  rather than only the node's.

  Found by the two-machine bench being written for GPU-120, before any machine
  was rented.

### Changed

- **One report directory per round, and the publish lock is gone (GPU-119).**

  A publish used to wait for a peer's in-flight fetch. One store held every
  round, moonclip renames its manifest into place, and on Windows a rename onto
  a file another handle holds fails — so writing round N+1 while serving round
  N had to be serialised. The lock was correct; what it cost had been argued
  from loopback, where a fetch lasts milliseconds.

  Measured, it cost more than the issue that filed it thought — because that
  number was taken at two nodes, and `_serve` held the lock across the whole
  send, so **two peers fetching from one node were serialised against each
  other**. That cannot show at two nodes, where there is only ever one fetcher.
  Three nodes, 7 MB/s, 200 ms round trip:

  | arm | publish, was | of which lock | publish, is | lock, is |
  | --- | --- | --- | --- | --- |
  | both at 7 MB/s | 2.49 s | **2.40 s** | 0.13 s | **0.00 s** |
  | one node slower | 1.07 s | 0.98 s | 0.10 s | 0.00 s |

  Each round now goes into its own directory, marked complete by a file written
  last. What is served is never what is being written, so a send takes no lock
  at all; retention deletes whole directories and skips whatever is on a socket
  right now, which is the one race a lock still had to cover. Publish and
  network together went from 7.66 s to 5.20 s per round at three nodes. A
  two-node round does not change — there the link is the bound either way, and
  the seconds the lock hid show up as an honest wait on the peer instead.

  It costs nothing on the wire: the single store kept three rounds and the
  transport's skip list is what held the transfer to the newest one, so a
  directory per round sends the same single snapshot with nothing to negotiate.
  Measured at 7.4 MB per round either way, plus 3 ms for the extra manager.

  `publish_wait_seconds` stays in the round report and now measures whatever a
  publish spent *not* writing — near zero, and a sentinel for the day something
  serialises a publish against a reader again.

- **The outer average is assembled in a canonical order (GPU-121).**

  `combine` sorted nothing, so each node summed "mine first, then the peers that
  answered" — a different order on every node for the same set. Floating-point
  addition is not associative, so the averages differed in the last places, and
  the difference is *applied to the parameters* rather than cancelling: it
  accumulates round after round. The invariant the whole outer loop rests on
  could therefore only be checked with a tolerance somebody had to guess.
  Sorting by node name costs nothing at these lengths and makes "every node
  holds one model" provable with `torch.equal`.

- **`agree_on_step` asks the store, not the collectives (GPU-111).**

  The second of the two gathers in that family to move, after
  `all_ranks_agree`. Same question, same answer, and the same difference in the
  same direction: a rank that never votes ends the call with its number in the
  log, where the collective ended it by timing out the whole group.

  Silence collapses the answer to `NOTHING_TO_RESUME` rather than to the
  minimum of whoever replied — a rank left out of that minimum would restore a
  different moment in training than the others, which is the one outcome this
  function exists to prevent. It is the same vote a rank with an empty store
  already casts, so it lands in a branch `_resume.py` already had.

  Not here for the microseconds: this one is asked once per resume.

## 0.1.0 — 2026-09-07

### Added

- **A round says where its seconds went, and `bench/round_link_cost.py`
  measures them on a link that is slow on purpose (GPU-117, under GPU-113).**

  Everything the outer loop is built out of had only ever run on loopback,
  where the network is free: every round printed `took 0.0s`, so the number the
  whole architecture is chosen around had never been observed. The round report
  now carries `delta_seconds`, `publish_seconds`, `publish_wait_seconds`,
  `gather_seconds`, `gather_wait_seconds` and `apply_seconds`, and the runtime
  logs the split rather than a total. The two waits are what a total cannot
  tell apart: a round that took minutes is a link to pay for, a peer to stop
  waiting for, or a lock to stop holding.

  The bench puts a fixed-rate, fixed-latency TCP relay in front of each node's
  listener — no `tc`, no privileges, runnable anywhere — and walks the bands.
  At 7 MB/s and 200 ms of round trip, on a 15.8 MB report:

  - **The transport gets 76% of the link.** 2.26 s of bytes, 2.97 s of round,
    the rest decode and store. Extrapolated, a 1B model's bf16 delta is about
    **375 s per round**, against the 429 s per *step* GPU-113 computed for
    synchronous training. The premise holds and the saving really is H.
  - **The rendezvous is not the problem.** A node twice as slow costs its peer
    0.08–0.19 s of extra waiting, not the 0.3 s their compute differs by: the
    slow node also starts earlier, so a constant speed difference is absorbed
    within a round. Under every arm there is a 0.31 s floor, which is the dial
    and the round trip, and it does not grow with the link.
  - **The publish lock is the problem**, and the comment claiming it was close
    to free was written from loopback timings. With both nodes on one link it
    is never contended. Throttle one node only and its publish blocks
    **2.2–2.3 s** waiting for a peer's fetch to drain, against a round whose
    own transfer is 0.30 s. Recorded on `DeltaExchange._io_lock`, and filed.
  - **The deadline bounds a slow transfer, not just an absent peer**, which is
    a case it had never been shown. Cut at 0.85 s of a 1 s deadline having
    moved 3 of 15.8 MB, both nodes closed the round over themselves alone and
    carried on; the next round came back to two nodes and re-sent the abandoned
    snapshot as well as the new one. A cut fetch costs a resend, not a run.

- **`bench/outer_convergence.py`: what H and `save_dtype` cost the loss
  (GPU-118, under GPU-113).**

  `bench/outer_loop.py` answers "does the mechanism work" on a four-feature
  regression, which converges whatever you do to it and so cannot tell H=20
  from H=500. This is the same outer loop on a byte-level transformer over this
  repository's own prose, with contiguous shards so the nodes really do see
  different data, at equal tokens seen, against single-node baselines. The
  quantization arms go through moonclip rather than through `.to(bfloat16)`,
  so what is measured is what crosses the wire, per-tensor float8 scales
  included; there is an error-feedback arm because quantization error is not
  independent between rounds.

  Held-out loss, 2 nodes, 2048 local steps each, against 1.55 for one node
  given the same wall clock and 1.38 for one node given the same *tokens*:

  | H | none | bf16 | fp8 | fp8+ef |
  | --- | --- | --- | --- | --- |
  | 1 | 2.69 | — | — | — |
  | 8 | 1.51 | 1.51 | 1.52 | 1.49 |
  | 64 | **1.45** | 1.45 | **1.44** | 1.44 |
  | 512 | 1.78 | 1.78 | 1.78 | 1.78 |

  - **The cast costs nothing measurable, at any H.** Every dtype column is
    within ±0.005 of `none` — less than the spread between neighbouring arms —
    and fp8 lands marginally *ahead* at H=64, which is how you know it is
    noise. So the 2.56x of bf16 and the 4.84x of fp8 are available. The
    specific worry does not show either: quantization error is not independent
    between rounds, and at H=8 there are 256 rounds to accumulate over with the
    column still flat. **Error feedback is not needed at these sizes.**
  - **H has an optimum with both sides real**, but the H axis and the
    round-count axis are the same axis at equal tokens seen — H=512 is four
    exchanges in the whole run, so what degraded is *few rounds*, not large H.
    The ceiling this can honestly report is on rounds.
  - **H=1 is the worst arm, not the safest.** An outer step per inner step
    compounds a Nesterov buffer at `outer_lr=0.7` two thousand times and
    overwrites the inner optimizer each time. There is no small-H limit where
    this degrades gracefully into ordinary data parallelism, which is the
    natural assumption and is false.

- **`outer_loop`: training across the internet, reached from `@ravex.train_loop`
  (GPU-116, under GPU-113).**

  The outer loop and its transport were only reachable from a test. Now
  `outer_loop: true` — in `ravex.yaml`, the environment, or the decorator —
  wires them into the runtime, and a training function that mentions neither
  rounds nor deltas nor peers trains with its peers. Proved on two processes
  under a real rendezvous, ending with bit-identical models. Documented in
  `docs/configuration.md`.

  **It never turns itself on.** The rest of Ravex activates by itself because
  the worst it does is write a checkpoint; this changes what the run *trains*.
  An inherited config switching it on quietly would be a run doing something
  other than what its own code says.

  A round is flagged inside `optimizer.step()` and closed at the batch
  boundary, the same discipline `checkpoint_every` already follows and for a
  stronger version of the same reason: closing one where it is noticed would
  write outer parameters into a model mid-step, after spending minutes of
  network inside the optimizer.

  **Two defects the wiring exposed, both silent, both found by two processes
  disagreeing about a model.**

  *The outer parameters have to be adopted, not snapshotted.* A round applies
  the same averaged pseudo-gradient to whatever outer state each node holds, so
  a difference in the *starting* state is never touched again — it is a
  constant added to one node's model for the whole run, while the loss falls
  and every round closes over everybody. It arose because the loop is built at
  the first step that has a model, which is after that step moved the weights
  with each node's own data: measured at 0.044 per parameter sum, unchanged by
  any round after. There is now a seed round in which one node's parameters are
  taken as the truth and the others adopt them — the same call a node joining a
  run in progress needs.

  *A failed round has to advance the round number anyway.* A node whose
  exchange failed stayed on its number while the peers moved on, so from then
  on it asked for rounds they had retired while they asked for one it never
  reached: both running, both logging closed rounds, permanently invisible to
  each other. One failed publish was enough. A round is defined by the local
  training that filled it, not by whether the network agreed about it.

  That failed publish was itself real: on Windows, serving a peer while writing
  the next round raced moonclip's manifest rename and returned "access denied".
  Writing and serving this node's store now exclude each other.

- **The round exchange: everyone's delta collected with a deadline instead of a
  collective (GPU-115, under GPU-113).**

  What makes "a node dies and the training does not stop" true rather than
  arranged. Every node serves its own round report and pulls everyone else's,
  each fetch on its own socket with its own deadline. A collective — on any
  medium — needs every participant to arrive and turns one absent node into a
  group-wide failure; here an absent node is one connection that does not
  answer, the other fetches are untouched, and the round closes with whoever
  replied. Proved on real sockets, with a process that stops mid-run and
  survivors that keep converging and stay bit-identical to each other.

  **A report is a moonclip snapshot**, moved by the same Rust framing the
  replication ring uses, manifest and skip list included — measured: after the
  first round only the new snapshot crosses the link, not the accumulated
  store. The first version of this shipped a hand-rolled wire format, which was
  a second way to write tensors down next to the one the project already owns.
  What using the real one actually buys was not what was expected: **zstd on a
  float delta is 1.08x and worth nothing**, while `save_dtype` is 2.56x at bf16
  and 4.84x at fp8 — GPU-113's quantised round already built, as a constructor
  argument.

  **A deadline needs `SO_RCVTIMEO`, not `settimeout`.** `settimeout` makes the
  descriptor non-blocking with the timeout enforced in Python, and the
  descriptor is then handed to Rust, where the first read returns "would block"
  — measured at 0.00s having moved nothing, against 1.50s for a 1.5s kernel
  bound. `RingLink` answers this with `settimeout(None)`, which means **the
  replication transport has no deadline at all** once bytes start moving. That
  was tolerable there and is not here.

  Two defects the tests found, both silent. A node that finished its last round
  and shut its listener **took its report with it**: the peer still fetching it
  closed that round with one contributor fewer, and the two models ended the
  run different in the fifth decimal — so leaving is now an event the protocol
  carries, bounded by a linger. And a node advertising its own hostname cost
  **16.1 seconds** per dial on a box with virtual interfaces, which for a round
  with a deadline is the round; the address is now resolved by the routing
  table, or configured through `RAVEX_EXCHANGE_ADDRESS` — required whenever the
  nodes are not on one network, because behind NAT nothing a process can ask
  its own kernel returns the address a peer dials.

- **An outer loop: many local steps, then one exchange of parameter deltas
  (GPU-114, under GPU-113).**

  The first piece of training *over the internet* rather than inside a
  datacenter. The link between two rented boxes on different continents
  carries 7 MB/s, measured and reproduced nine days apart, and on that link an
  all-reduce of a one-billion-parameter model's gradients costs **429 seconds
  per step**. Synchronous data parallelism is not slow there, it is out of the
  question.

  So each node trains locally for H steps with its own optimizer, and then the
  nodes exchange the *difference* between the parameters they started the
  round with and the ones they hold now. That difference is signed to point
  the way a gradient points, and an outer optimizer — Nesterov, momentum 0.9 —
  takes one step on the average of everybody's. The bytes per round are
  unchanged; they are paid once per H steps instead of every step.

  `ravex._dist.outer` imports no `torch.distributed` at all: it takes a list of
  contributions and says what to do about them, so every rule in it is provable
  in one process. Who collects them, with what deadline, and what to do about
  a node that never answered, is the transport's business and is not written
  yet.

  **Two things it buys immediately, and both are properties of one line.** A
  node that dies during a round is simply not in the list, and the round closes
  over the rest — there is no collective to hang in. A node that runs five
  times slower reports fewer steps rather than holding anyone back, because the
  round can close on the clock instead of on a count.

  **How contributions are averaged was decided by measurement, and the first
  measurement was blind.** A delta is already proportional to the work behind
  it, so weighting *again* by step count counts a fast node twice. On a bench
  where every node draws from the same distribution, that argument is
  untestable — over-weighting a node biases the average toward data that says
  what everyone else's says — and `step_weighted` duly came out slightly
  *ahead* of the plain mean. Give each node its own target and the order
  reverses: `step_weighted` is **45% worse** (2.92 against 2.01), and worse
  than ignoring heterogeneity entirely. The default is the plain mean, all
  three modes stay selectable, and `bench/outer_loop.py --skew-data` is the arm
  that can tell them apart.

  The outer momentum is what separates this from parameter averaging, and it
  pays: 0.0143 against 0.0466 without it, on the same run.

- **Coordinated emergency checkpoint on SIGTERM, for sharded models split
  across more than one machine (GPU-92).**

  A preempted spot instance gets SIGTERM ~10s before it is killed. For a
  plain-replicated (DDP) job that was already enough: rank 0 holds the whole
  state and writes it without needing anything from anyone. For
  `sharded_checkpoints: per_rank`, it was not — extracting even one rank's own
  shard goes through PyTorch's own `get_state_dict`, which is collective even
  when nothing is gathered across ranks, so a lone preempted rank could not
  produce a checkpoint by itself. Until now the SIGTERM handler simply skipped
  the final checkpoint for every sharded job, unconditionally.

  This adds a minimal coordination channel: the rank that catches SIGTERM
  raises a flag; every rank checks for it at the same cadence
  (`emergency_check_every`, default every step); if any rank has it set, every
  rank enters the same collective save together, on the process group they
  were already going to use. Detection runs on its own short-timeout group,
  isolated from the one carrying gradient synchronization, so a stuck
  detection round fails in seconds rather than however long the training
  job's own process group is configured to wait. See
  `ravex._dist.collectives.emergency_group` / `emergency_signalled`, and
  `RavexRuntime._check_emergency_signal`.

  **Read this before relying on it.** The local handoff alone — no network,
  just copying state off the GPU and handing it to the backend — has been
  measured at 10.6s on 8× RTX 5060 Ti, against a SIGTERM budget of roughly
  10s. That number alone consumes essentially the whole budget before this
  channel's own cost. **This is a best-effort attempt that will often not
  complete, not a guarantee that spot training under FSDP is now reliable.**
  It exists because a save that sometimes lands is worth more than one that
  never does, not because the numbers were made to work.

  The channel is deliberately narrow — it says "save now" and nothing else;
  no rank ever waits for a new node or tears down its process group because
  of it. It is the minimal building block GPU-94 (elastic torchrun) will
  need to generalize for broader multi-rank coordination; it is not a
  substitute for that work; do not duplicate this pattern there.

  New checkpoints written this way carry `emergency: true` in their metadata,
  for recovery analysis — nothing reads it back to make a resume decision.

  Config: `emergency_coordination`, `emergency_check_every`,
  `emergency_timeout`. See
  [Emergency checkpoint on preemption](docs/configuration.md#emergency-checkpoint-on-preemption-sharded-models).

- **The framework adapter seam is connected to the runtime (GPU-69).**

  `FrameworkAdapter` has declared three methods since the adapters were written
  — `should_intercept_step`, `collect_extra_state`, `restore_extra_state` — and
  nothing called any of them. `get_adapter()` had no caller anywhere in the
  package; the only thing the runtime ever did with a detected framework was
  write its name into the checkpoint's metadata. Three signatures and an
  intention.

  They have call points now. `should_intercept_step` is asked once, at
  activation, and its answer is a bool the step hook reads — `on_step` is the
  hottest path Ravex has and a dispatch through an adapter does not belong in
  it. `collect_extra_state` runs inside `checkpoint()`, right after the state
  is collected, and what it returns is written under `extra` — its own key,
  because `models`, `sharded` and `rng` have formats this project's readers
  understand and a framework's own bookkeeping is opaque to everything except
  the adapter that produced it. `restore_extra_state` runs on the way out of a
  resume, once the model, the optimizer, the schedulers and the dataset
  position are back.

  Asking at activation is worth something the autoloader could not have given.
  Detection reads `sys.modules`, and the decorator (GPU-108, above) runs when
  the training function is called — with transformers, or Lightning, or
  DeepSpeed already imported by the script above it. The `.pth` asked the same
  question at interpreter startup, before the script had imported anything, and
  the honest answer then was always "vanilla".

  **The failure modes are the design.** Collection runs on the training thread
  inside the same window as the sharded collective, so an adapter that raises
  there would have taken the weights with it — and on a sharded run it would
  have taken them on one rank while the other seven wrote theirs, which is
  worse than losing the checkpoint. Every one of the three is wrapped: the
  exception is logged, the framework's contribution is treated as absent, and
  the checkpoint, the resume or the run carries on. A `should_intercept_step`
  that cannot answer counts steps, because the alternative is a run that trains
  for hours, writes nothing, and looks fine until it is preempted.

  Reading back: a checkpoint with no `extra` resumes exactly as it always did,
  and one carrying state written under a different framework is skipped with a
  warning that names both — handing Lightning's loop counters to a HuggingFace
  adapter is a crash at best, and ignoring them silently is a resume that lost
  half its state without saying so.

  **No adapter overrides any of the three yet**, and that is what this entry
  is: wiring, with `tests/test_adapter_seam.py` holding it to the contract
  above. `Trainer.state` and Lightning's loop counters are the behaviour it was
  built for and they come next. What has *not* been decided is DeepSpeed:
  `detect_framework()` recognizes it, `_ADAPTERS` has no entry for it, and ZeRO
  state does not pass through the optimizer hooks Ravex installs at all — a
  gap to close or a limit to declare, but not one this seam settles.

### Fixed

- **With `outer_save_dtype` set, no two nodes took the same outer step
  (GPU-118).**

  A cast report is what the *peers* read, and a node was averaging the delta it
  had computed instead — so every node combined a different set from the same
  round. That is not a rounding detail: an outer step applies the same averaged
  pseudo-gradient to whatever outer parameters a node holds, so a difference
  between them is never touched again by anything. The nodes part by a
  quantization error per round, forever, and from the outside the run looks
  healthy — the loss falls and every round closes over everybody. It is the
  same failure the seed round of GPU-116 exists to prevent, arriving through
  another door.

  `DeltaExchange.as_published` now reads back what was written whenever
  `save_dtype` is set, so a node averages its own report in the form its peers
  will see. Without a cast it is the identity and costs nothing. Two nodes over
  real sockets at `save_dtype="bf16"` now end four rounds bit-identical; the
  test fails without the fix.

  **The same mistake was a floor lower, in the seed round**, and turning the
  default on is what surfaced it: the peers adopt what came down the wire,
  which under a cast is not what the source wrote, so the one node that did not
  go through the wire started a quantization away from everyone else — from the
  first instant, never touched again. That is the exact failure
  `adopt_outer_state` exists to prevent, reintroduced by the setting that
  shrinks the round. The source now adopts its own published state too. Found
  by the end-to-end test, which is the only place it could be found: nothing
  raises, and the run trains.

  What the defect was worth, measured (`bench/outer_convergence.py`, H=64,
  fp8): loss 1.4100 against 1.4104 with the fix — indistinguishable, and both
  perfectly healthy models. The spread between the two nodes' parameters:
  **3.4e-02** against **0**. Nothing about the loss, the round reports or the
  models themselves says the nodes have stopped training one model; only the
  invariant does, which is why the tests use `torch.equal` and not `allclose`.

- **A replication round over the collectives crashed on an install without
  NumPy.**

  `RuntimeError: Numpy is not available`, from `exchange_stores` — twice: once
  reading a peer's manifest, once handing each received chunk to the store
  writer. Torch does not require NumPy and Ravex depends on PyYAML and nothing
  else, so an image with neither is a configuration to keep working, and CI's
  is one. Both raised *after* the bytes had already crossed the wire, which
  makes it a crash rather than something a caller can fall back from — the same
  reason `_all_gather_object` replaced torch's object collectives.

  The fix is not a NumPy-free spelling of the same conversion but the absence
  of a conversion: `_wire_buffer` receives into a `bytearray` and hands the
  same allocation to torch as a tensor and to the writer as a buffer. No
  NumPy, and no copy either — where `bytes(tensor.tolist())`, which is what
  `collectives` had to use, builds a Python list per megabyte. A test runs both
  roads with `tensor.numpy()` taken away and asserts they still leave the same
  replica; it fails without the fix, with CI's error.

- **Every transformer got a warning about a buffer that cannot drift.**

  `float_buffers` reported all floating-point buffers, and a causal attention
  mask is one — registered non-persistently, identical on every node by
  construction, and named in a warning about parameters drifting apart. Only
  buffers a checkpoint carries are reported now, which is the module's own way
  of separating state from derived state. A warning that cries wolf on the
  usual case is worse than no warning at all.

- **`@ravex.train_loop(storage={"path": ...})` crashed several frames from the
  mistake.**

  `storage` is the one option whose value is a *section* rather than a scalar,
  and the decorator applied every override with a bare `setattr` — so the dict
  replaced the `StorageConfig` dataclass and the failure surfaced later as
  `AttributeError: 'dict' object has no attribute 'path'` from inside
  `_normalize`, naming neither the option nor the decorator. A mapping is the
  obvious thing to reach for, because it is exactly how `ravex.yaml` spells
  that section.

  Both ways in now go through one function, `RavexConfig.apply_storage`, which
  accepts a mapping or a `StorageConfig` and leaves untouched keys at their
  defaults. The two callers differ in one way, and it is the distinction the
  rest of the module already makes: a config *file* must never stop a training
  run, so a bad section there is recorded in `problems` and the defaults stand;
  a decorator keyword is someone typing at the call site and raises, exactly as
  an unknown top-level option already did.

  Two smaller things fell out. An unknown key inside the section — `pth` for
  `path` — was silently dropped by the YAML path and is now reported with the
  valid names. And the field list is the dataclass's own rather than `hasattr`,
  because `hasattr` accepts `is_remote`, which is a property with no setter:
  `storage: {is_remote: true}` in a config file used to raise `AttributeError`
  from somewhere with nothing to do with configuration.

### Changed

- **`outer_save_dtype` is measured now, and still off (GPU-118).**

  It was `null` because nobody had looked at what a cast costs the loss.
  Somebody has: nothing measurable, within ±0.005 at every H tried, and on the
  link this exists for it takes **a fifth off every round** — 15.8 MB to
  11.7 MB on the wire, 3.28 s to 2.61 s of network. So the reason it still
  ships off is no longer "unmeasured". It is that turning it on is what
  surfaced the seed-round defect below, and that the evidence is one 0.48M
  model on one box, where a release changing what a run puts on the wire wants
  a two-machine run behind it. Set it and take the fifth; `bf16` is the smaller
  bet of the two that measured free.

  What did change: an unusable value now lands in `problems` at load rather
  than inside the `except` that gives up on the outer loop mid-run, and
  `none`, `off` and the empty string all turn it off — `compression` accepts
  the same three, and an option that understands only one of them is how a
  setting gets left on by accident.

- **`all_ranks_agree` asks the rendezvous store instead of the collectives
  (GPU-111).**

  It keeps a resume all-or-nothing — one rank quietly starting from scratch
  while the others restore is a job that stops making progress without failing
  — and it did that by pickling a Python bool across two collectives to move
  one bit. Now every rank writes its answer under a key and reads the others'
  with one `wait` and one `multi_get`. Same question, same answer, and
  `ravex/_dist/agreement.py` is the one primitive underneath: an all-gather of
  a *scalar*, not an `all_reduce`, because none of the things Ravex needs its
  ranks to agree about is a tensor.

  **The difference that matters is not the microseconds.** A collective waits
  for a missing rank and then fails the whole group; here the question has a
  deadline, and running out of it *is* the answer — a rank that never said it
  succeeded did not — and the log names which rank went silent, which a group
  timeout never could.

  **The microseconds, measured, and bounded.** `bench/agreement_cost.py` at 4
  ranks on loopback: 1544 µs over the collectives against 849 µs here. That is
  1.8x, not the twenty a bare store lookup (70–98 µs) would suggest, and the
  gap between those two numbers is worth keeping straight: **a gather cannot
  escape waiting for the slowest rank on any medium**. With 5 ms of skew on one
  rank both roads pay it, 5.9 ms here against 6.9 ms there. What the store
  changes for a gather is the cost of the mechanism, not the cost of waiting.

  **The first implementation was slower than what it replaced**, and it is
  recorded here because the bench is the only reason anyone found out. Polling
  `check` on a ladder of sleeps and then reading each answer with its own `get`
  measured **1592 µs** — worse than the 1088 µs collective — because a rank
  arriving 100 µs late still costs a whole sleep, and four answers were four
  round trips. `store.wait` blocks on the server and wakes on the write;
  `multi_get` reads them together. Three round trips and no sleeping. The
  polling road survives as the fallback for a store without those methods, with
  the number attached to it so nobody mistakes it for the fast one.

  `agreement_transport` (`auto` by default, or `store` / `collectives`) is the
  knob. The collective road is kept for a reason that is not performance: a
  store round decides on its own when a rank goes quiet, and a job that would
  rather fail together than proceed without one rank wants the old behaviour.

  Only this one function moved. `agree_on_step` is next and is the same shape;
  `gather_visible_stores` and `agree_on_run_id` carry structures rather than
  scalars, and they stay where they are until someone decides how a
  peer-written blob should be parsed — that is a security question, not an
  encoding one. `emergency_signalled`, the one that runs every step, is a
  different design and not a port: see GPU-111.

- **Both ends of a replication socket now prove they belong to the job
  (GPU-112).**

  The listening port GPU-109 opened on every rank is a surface the process
  group never had: gloo and NCCL only ever speak to a member, and membership is
  the launcher's to decide. Here anything that can reach the port can knock,
  and a knock is not harmless — `StoreWriter` removes the completion marker
  before the first byte lands, so a connection that arrives and then says
  nothing is enough to stop a *good* replica from being read as one.

  Rank 0 now generates a token and leaves it on the rendezvous store; a rank
  dialling its successor presents it, and the listener answers with a digest of
  the token bound to the rank that asked. Not a new trust boundary — whoever
  can read that store is already inside the job. What it buys is that a
  stranger who merely reaches the port is refused before any writer exists, and
  that a rank never pushes a checkpoint into a port that cannot answer for
  itself. A refusal costs that connection and nothing else: the listener keeps
  waiting for the peer it expects, so a port scan cannot deny a rank its ring.

  **What it is not**, said here because the alternative is someone assuming
  otherwise: a shared secret in the clear on a connection nobody has encrypted.
  It identifies, it does not protect. Against an attacker who can read the
  traffic between two ranks or take over an established connection it does
  nothing, and on a network where that is the threat
  `replication_transport: collectives` is the answer — which is one of the
  reasons that road stays supported.

  **And a rank now advertises an address rather than its name.** Publishing
  `socket.gethostname()` was the obvious thing and the wrong one: measured on a
  Windows box with a Hyper-V interface, `create_connection` to this machine's
  own name took **10.06 s**, because the name resolves to several addresses and
  the first ones are on virtual interfaces that swallow the attempt until it
  times out. By address it is instant. A GPU box with docker, WSL or a VPN has
  exactly that shape, and there the cost is not a slow ring — it is a peer that
  gives up before the right address is ever tried. What gets published is the
  local address the routing table would use to reach the rendezvous, asked with
  a UDP socket that sends nothing, with the hostname kept as the fallback. The
  dial itself is capped at ten seconds rather than the link's whole budget, so
  an address nothing can reach costs one attempt and then the collective road.

- **The replica transport is Rust, and the pre-staging path does not come back
  through Python at all (GPU-109).**

  `store_files`, the manifest encodings, `encoded_size`, `encode_store` and
  `StoreWriter` moved from `ravex/_dist/replication.py` into `src/transport.rs`;
  `ravex/_dist/elastic.py`'s `prestage_send` / `prestage_receive` are now two
  lines each, handing their socket's descriptor to the core. The names, the
  signatures and the module they are imported from are unchanged, and
  `tests/test_dist_replication.py` — 69 tests written against the Python — was
  not touched, which is what makes it the oracle for the Rust rather than a
  description of it.

  **The wire format did not change and could not.** A replica written by one
  version is read back by another, and `exchange_stores` still moves its bytes
  over torch collectives through this same framing: little-endian, a `<I` count,
  then `<HQ` plus the name plus a `<B` skip flag per entry, then the bodies of
  everything not skipped.

  **This was measured before it was written, and the first measurement said
  don't.** GPU-84 had found the transfer stuck at 529 MB/s against a 1462 MB/s
  wire and collapsing to 113 MB/s as the chunk grew — a ceiling that was ours,
  not the network's, and the case for rewriting it. That script was a throwaway
  in a session scratchpad, so the first thing done here was to rebuild it as
  `bench/transport_ceiling.py` and run it: 989 MB/s at a 4 MiB chunk, 735 at
  64 MiB. Most of GPU-84's collapse had already been paid off by two earlier
  copy removals, and what was left was not the wire — the real path ran at 81%
  of the *encoding* arm while the wire itself was 4.6x faster than either.

  So the honest reading was that a Rust socket would attack the half that was
  already fast, and that is written down in GPU-109. It was overruled and the
  port was done anyway. What the same bench says afterwards, 2 GiB in 8 files on
  one box over loopback:

  | arm | 4 MiB | 16 MiB | 64 MiB |
  | --- | --- | --- | --- |
  | `exchange_stores` (torch collectives, Rust framing) | 701 MB/s | 780 | 809 |
  | `prestage_*` (Rust end to end) | 1063 MB/s | 1419 | **1611** |

  Twice the throughput at a 64 MiB chunk, and **rising** with chunk size where
  the old path fell. Two things did it: a cursor over the pending buffer instead
  of `del pending[:take]`, which was a memmove of the tail at every file
  boundary, and a reader thread bounded three chunks ahead so the disk and the
  socket stop taking turns. Neither is a Rust trick — both are things the Python
  could have done — but they are what the rewrite was for, and the number is the
  number.

  **`exchange_stores` has both roads now, and the runtime picks one.** A
  `RingLink` holds one connection to the ring successor and one from the
  predecessor, opened once and reused for every round after — the shape the
  protocol already had, since `prestage_send` leaves the connection open
  precisely so a coarse first pass can be followed by smaller deltas.
  Addressing is the rendezvous store rather than a new mechanism: a rank
  advertises its listener under a key and reads its successor's, exactly as
  `elastic.announce_join` / `pending_join` already do for a joiner that belongs
  to no group yet. A store is not a collective — no group, no device, no
  agreement about who is present.

  `replication_transport` is the knob: `auto` (the default), `sockets`, or
  `collectives`. The collective road is not deprecated and should not be. It
  needs no rank to have an address another rank can reach, so on a cluster
  where they cannot open connections to each other it is the one that works,
  and `auto` falls back to it on its own — once, with the reason, rather than
  retrying a hostname that will not resolve on the next round either.

  `tests/test_dist_replication_sockets.py` is what makes the second road a road
  and not a fork: two ranks, both roads, the same store in the same run, and
  the assertion is that the receiving directory is identical down to the file
  list and the completion marker — because everything downstream reads a
  replica without knowing which way it arrived. It also covers the reverse
  direction, which only the socket road has to think about. The recovery round
  pushes a copy *back* against the ring, and on two ranks the successor and the
  predecessor are the same peer, so the direction is passed in rather than
  inferred from rank numbers. Getting it backwards is both ends pushing into
  each other and neither reading, which is a hang rather than an error.

  **What this does not say.** One box, one platform, page cache warm, loopback.
  Between two rented machines the link was measured at 12 MB/s (GPU-96), which
  is two orders of magnitude below any of these numbers: on a real network none
  of this is what limits a replication round. The gain is real where the link is
  fast — one machine, several ranks — and irrelevant where it is not.

  `exchange_stores` itself still moves its bytes over torch collectives. Giving
  it a socket means answering how ranks find each other outside a process group,
  which is a design question and not a port.

- **Ravex has one way in, and it is `@ravex.train_loop` (GPU-108). This breaks
  the public API.**

  ```python
  import ravex

  @ravex.train_loop(preemption_handler=True)
  def train():
      model = build_model()
      optimizer = torch.optim.Adam(model.parameters())
      for batch in loader:
          ...

  train()
  ```

  Through 0.0.5 there was a second way in, and it was the one the README led
  with: installing the package put `ravex_autoload.pth` in site-packages,
  Python executed it in **every** interpreter on the machine, and any process
  that had a `ravex.yaml` above its working directory got checkpointing
  attached without a line of its own saying so. The argument for it was a real
  one — a platform could enable checkpointing for code it does not own — and it
  is what has been removed.

  The decorator is not that same idea written out longhand. It buys something
  import-time activation could not: Ravex knows where the loop begins and where
  it ends, and it holds the function **before the model exists**. The
  autoloader attached to an interpreter, not to a loop, and by the time it saw
  anything the model was already built. Rebuilding a model under a new device
  mesh is what an elastic regroup is, and that needs a callable which predates
  the model — `ravex/_dist/elastic.py` stops at the rendezvous layer for
  exactly this reason. The decorator is the prerequisite it was missing
  (GPU-110). `elastic=True` is accepted as a keyword today and raises with that
  explanation rather than being quietly ignored.

  **Inside the wrapped function nothing changes.** The patches on module
  construction, `Optimizer.step` and dataloader iteration are still what find
  the model, the optimizer, the scheduler, the AMP scaler and the dataset
  position; the zero-boilerplate half of Ravex is the half worth keeping. What
  changed is when they go in — on entry to the decorated call rather than at
  import — and that they come out again on exit, in a `finally`, so the loop is
  unpatched whether it returns or raises and a second call starts clean instead
  of finding the previous run's runtime. Step counting still advances on
  `optimizer.step()`, which is why gradient accumulation still needs no special
  case. Nesting two decorated functions raises rather than letting two runtimes
  count the same steps.

  **Porting existing code.** A `ravex.activate()` call becomes the decorator
  around the training function. Code that relied on the autoloader has to name
  Ravex now: import it, and decorate the entry point. `ravex enable` and
  `ravex disable` are gone with the file they wrote, and `ravex status` no
  longer reports whether the autoloader is armed, because nothing is armed —
  a run is checkpointed if its entry point is decorated and not otherwise,
  which is a fact you read in the source instead of interrogating the
  environment for. `ravex.deactivate()` survives for the two callers that
  still want the exit half by hand: the test suite, and a notebook stopping
  early on a live interpreter.

  One constraint disappeared with the `.pth`, and it had shaped the code:
  `ravex/__init__.py` did not import `typing`, because the autoloader paid that
  import in every interpreter on the machine — measured at 9.8 ms against the
  module's own 0.4 ms, on `python -c` as much as on a training run. Nothing of
  Ravex runs at import time any more, so the import is back and the comment
  that forbade it is gone rather than left standing as an orphaned optimization
  the next reader would take for a rule.

- **A reshard no longer reads every old checkpoint twice (GPU-101).**

  Resuming onto a different number of ranks is two passes over the old stores:
  one to measure how long each old shard is, one to cut out the slices the
  plan asked for. Two passes is not the waste — it is what holds the memory
  bound, since knowing where a shard starts needs every length before it, and
  a single pass would mean every old snapshot resident at once.

  The waste was that the *measuring* pass loaded them. It only ever wanted
  lengths, and there was no way to ask a store for a shape: `load_step` was
  the only reader, and it materializes every tensor. So a reshard from N ranks
  read N complete checkpoints it needed and N it discarded — tens of gigabytes
  on a real model, and between machines it would have been that over the
  network, which would also have undone the `ceil(N/M)+1` bound the reshard is
  built to hold.

  `CheckpointBackend.describe_step` is the question that was missing: the same
  tree, with each tensor replaced by its shape and dtype and everything that
  was never a tensor — placements included — left as it is. On Moonclip that
  is two small reads of one template entry and no tensors at all.

  Nothing depends on it being available. A backend that cannot describe itself
  returns `None` and gets loaded, which is exactly what happened before; the
  `torch_save` backend does precisely that, having one pickle and no way to
  read a shape out of it short of unpickling the lot. Needs Moonclip >= 0.1.0
  for the fast path.

- **The reshard planner is Rust, and Ravex is no longer a pure Python package
  (GPU-105).**

  `ravex._dist.reshard` — the arithmetic that lines a per-rank checkpoint up
  with a different number of ranks — is now a crate in `src/`, reached through
  PyO3 and built into the wheel by maturin. Same shape as Moonclip: a Rust
  engine with a thin Python surface.

  **The import path did not change.** `from ravex._dist.reshard import
  plan_reshard` works as it always did; that module is now twenty lines of
  re-export over `ravex._core`. Every function takes what it took, returns what
  it returned and raises what it raised, down to the wording of the messages.
  That was the condition of the port rather than a courtesy:
  `tests/test_dist_reshard.py` and `tests/test_dist_reshard_locality.py` were
  written against the Python implementation and were **not touched**, so 408
  tests written for the old code are what the new code is held to. A port whose
  tests had to be edited to pass would have proved nothing.

  `reshard.py` went first because it was the one module with no excuse: no
  torch, no process group, no I/O, and a suite that already covered every
  `N -> M` pair in a range rather than the one pair a GPU box happens to have.
  The planner's own tests now run twice — once in `cargo test`, with no
  interpreter and no torch, and once through pytest as before.

  **What this costs, stated plainly, because it is the whole decision.** Until
  now `pip install ravex` worked everywhere, and only the Moonclip backend was
  tied to Linux x86_64. That constraint has moved onto Ravex itself: wheels are
  built per interpreter for Linux x86_64, and anywhere else pip falls back to
  the source distribution, which needs a Rust toolchain to build. The
  alternative was an optional core with the Python kept as a fallback — two
  implementations of the same arithmetic, drifting quietly, with the exhaustive
  test suite proving only whichever one happened to be loaded. One
  implementation and a narrower install was judged the better trade; if it turns
  out to be the wrong one, the Python is in the history and the boundary is a
  single module.

  Not ported, and not for want of trying: `_patches.py`, `_registry.py` and
  `_runtime.py` live on introspecting live Python objects, and moving them would
  mean crossing the PyO3 boundary on every `optimizer.step()`. `_bootstrap.py`
  stayed pure Python for a different reason — it ran in **every** interpreter on
  the machine once the `.pth` was installed, and a compiled extension loaded
  there is the opposite of what that file was for — and GPU-108 has since
  deleted it, later in this same unreleased cycle. `ravex/__init__.py` still
  imports nothing from the core: it reaches for `ravex._core` inside the
  functions that need it, so importing Ravex costs what it did.

- **The build backend is maturin, and `setup.py` is gone.**

  `ravex_autoload.pth` moved from a custom `build_py` command to a
  `[tool.maturin] include` entry, and then out of the package altogether:
  GPU-108 removed the autoloader later in this same unreleased cycle, so no
  wheel this release publishes contains it, and the release workflow no longer
  asserts anything at the root of the archive.

  The trap that made the custom command necessary is worth keeping written
  down anyway, because it is what will catch whoever puts a `.pth` back:
  `data_files` looks like the answer and installs the file one directory
  *above* site-packages, where it arrives, looks installed, and is never run. A
  wheel built that way is not obviously broken — Ravex simply never activates,
  and there is nothing to see. This paragraph is where `pyproject.toml` sends
  the reader who wonders what used to be in that `include` list.

  The version number is now written twice — `Cargo.toml`, which is what maturin
  builds the wheel from, and `ravex/__init__.py`, which is what `ravex status`
  prints — because maturin has no equivalent of setuptools' `dynamic = {attr}`.
  `tests/test_version_is_single.py` fails if the two ever disagree, and the
  release workflow checks both against the tag.

### Removed

- **The autoloader, and everything that existed to manage it (GPU-108).** Gone:
  `ravex_autoload.pth`, `ravex/_bootstrap.py`, `tests/test_bootstrap.py`, the
  `ravex enable` and `ravex disable` subcommands, the autoloader line in
  `ravex status`, the `[tool.maturin] include` entry that shipped the file, and
  the wheel-layout check in `release.yml` that asserted it at the root of the
  archive. See the entry above for what replaces it and why.

- **`ravex.activate()` (GPU-108).** It was the second of the two ways in, and
  with the decorator it would have become a third form of the same thing. The
  point of this release is that there is one. What it did is now the entry half
  of `train_loop`, as `ravex._activate` — underscored, because the way in is
  the decorator and only the decorator also guarantees the exit.

## 0.0.5 — 2026-08-30

### Added

- **`save_dtype`: what precision each part of a checkpoint is stored at.**

  ```yaml
  save_dtype:
    optimizer: bf16
  ```

  Keys are component names — `model`, `optimizer`, `scheduler`, `scaler`,
  `dataloader` — or raw Moonclip globs for anything they do not cover. **The
  first matching rule wins**, in the order written, so an exception is
  expressed by putting it first: `{model: none, "*": bf16}`. A bare value
  applies to everything. `RAVEX_SAVE_DTYPE=model:none,optimizer:bf16` from the
  environment, ordered left to right.

  It is the largest saving available on a run that checkpoints often. On a
  1.5B model under FSDP2, per rank: the weights are a third of the state and
  delta well, −70%; `exp_avg` and `exp_avg_sq` are the other two thirds and
  delta essentially not at all, −1.1% and −4.0%. With β₁ = 0.9 a tenth of each
  moment is replaced by fresh gradient every step, which moves nearly every
  mantissa bit, so the XOR between two steps is noise. **85% of the bytes
  written are the part that does not compress** — and it is also the part that
  tolerates the least precision, since `exp_avg_sq` reaches Adam through
  `sqrt(v)`, which halves the relative error. `{optimizer: bf16}` halves 85%
  of the volume and leaves the model exactly as it was.

  **Components rather than patterns, and the reason is not tidiness.** An
  optimizer is written in two places: under `ravex/optimizers/…` when it is
  not sharded, and under `ravex/sharded/<key>/optimizer/…` when it is, because
  sharded state has its own collection path. A pattern written by hand as
  `ravex/optimizers/*` is right on one GPU and casts **nothing** on the
  sharded run the setting was chosen for — no error, no warning, a
  full-precision checkpoint. Naming the component expands to both. Raw
  patterns still work and pass through untouched, and there is a test that
  documents the trap rather than pretending it is unreachable.

  **Off by default, and it stays off across an upgrade.** It changes the
  numbers a resumed run gets back, and a library that did that to an in-flight
  run because someone bumped a version would be wrong to. What it does instead
  is log one line per run naming the setting when it is unset — at `INFO`, not
  as a warning, because nothing is wrong; the setting is simply worth a great
  deal and is invisible otherwise.

  Values are checked when the config loads rather than when the checkpoint
  manager is built. Moonclip refuses a bad dtype too, but it does so at
  construction, which under Ravex is mid-run and inside the `except` that
  falls back to `torch.save` — so a typo would have cost the run its Moonclip
  checkpointing and said one line about it. A bad rule is now dropped, that
  component is stored unchanged, and the reason lands in the startup problem
  report with everything else. One bad rule does not take the correctly
  spelled ones with it.

  Needs Moonclip 0.0.9, which is already the floor.

  And it says so rather than crashing into it. Moonclip before 0.0.9
  declares `save_dtype` as a string, so the mapping form reaches it as a
  `TypeError` — which `get_backend` catches along with everything else,
  so one configuration line would have cost the run its Moonclip
  checkpointing entirely, reported as "Moonclip backend unavailable". That
  is the exact failure the 0.0.8 post-mortem named: a version requirement
  wearing compatibility as a disguise. The version is now checked in the
  open, and what degrades is the option rather than the backend —
  checkpointing continues, at the precision the tensors arrived in, and
  the log line says which Moonclip would be needed.

- **`reshard_on_resume`: a `per_rank` checkpoint can be resumed at a different
  world size.** Each rank rebuilds its own shard out of the old ones — eight
  ranks' shards onto four, or four onto three, where no new shard equals any
  old one and each is stitched from two.

  The offsets are *measured*, never recomputed. The old lengths are the shapes
  of the tensors that were saved; the new ones are the shapes the live model is
  holding, shared in one `all_gather_object` of a few integers. Reproducing
  torch's chunking rule to derive them would have been a second implementation
  of a rule with an uneven-tail case, and it would have drifted from the first
  in silence.

  **Off by default**, and not out of caution about the arithmetic: a resume
  that reshards silently is a resume that silently succeeds when the launcher
  started three ranks where the job wants four. The mismatch is detected and
  logged always; acting on it is the opt-in.

  Refused, loudly and by name, when a precondition does not hold: a 2-D mesh
  (FSDP crossed with tensor parallel) is a cartesian problem rather than an
  interval one and is not attempted, and a reshard whose old shards are not all
  reachable — as stores or as *complete* peer copies — stops rather than
  assemble tensors with a band of uninitialised rows in them. Local storage
  only so far; on remote the mismatch is reported and the run starts clean.

  Two things do not reshard, and are documented rather than left to be found.
  The sampler partitions the epoch by world size, so a resumed run continues
  the model and not the run: the position is rescaled to preserve the total
  data consumed, every sample is still seen once per epoch, and the order is
  not the one the original run would have taken. And there were N per-rank RNG
  states where there are now M ranks, with no correct mapping between them, so
  they are not restored and the log says so.

  Peak memory is one old snapshot plus this rank's new shards. The old stores
  are read twice — once to measure, once to cut out the slices the plan asked
  for — because a single pass would mean holding every old snapshot at once,
  which is the whole checkpoint per rank. The global tensor is never
  materialised anywhere.

### Changed

- **A diagnostic, `RAVEX_ASSUME_NO_NUMPY`.** Not a setting: it has no place
  in `ravex.yaml` and does nothing useful in a real run — it exists to make
  something measurable that is otherwise unreachable. See
  [Diagnostics](docs/configuration.md#diagnostics).

  It forces the NumPy-free object-gather path, which no ordinary image
  reaches because they all ship NumPy — without it that code would have
  shipped having never run on a network. It is *meant* to be set on one
  machine and not the other: a rank with NumPy and a rank without is what
  two rented boxes from different images look like.

- **The checkpoint handoff always reports `skew` apart from `drain`, on a
  sharded run with more than one rank.** A barrier now runs before the
  accelerator drain unconditionally in that case, timed apart, separating a
  rank waiting for its own device from a rank waiting for a peer.

  This was `RAVEX_SPLIT_DRAIN`, an opt-in diagnostic added earlier in this
  same release. It has been promoted to the normal path rather than kept
  behind a flag, because what it measured on two machines settled the
  question it was written to ask: `skew 34.6 s / drain 0.000 s` on one rank
  and `skew 103.8 s / drain 0.000 s` on the other — all of what used to be
  called `drain`, and excluded from the reported cost on the theory that it
  was queued compute, was in fact a rank waiting on a peer that was late
  because *it* was checkpointing. See GPU-98.

  `drain` keeps its exclusion — genuine queued device work still is not a
  cost of checkpointing. `skew` does not: it is counted like every other
  phase, which means the cadence warning (`checkpoint_every=... is costing
  N% of wall time`) can now fire on a run where it previously stayed quiet,
  because time that was always being spent is no longer hidden under a name
  that excluded it. That is the point of the change, not a side effect of
  it.

  The guard is `sharded and get_world_size() > 1`, computed identically on
  every rank — the same condition the save's own verdict collective already
  carries. A replicated (non-sharded) job never reaches the barrier, for the
  same reason it never reached the diagnostic's version of it: the non-main
  ranks are gone by then, and a barrier only rank 0 entered is a hang until
  NCCL gives up. `RAVEX_SPLIT_DRAIN` is no longer read; setting it does
  nothing.
- **Python 3.9 and 3.10 are no longer supported.** The floor is 3.11, and the
  CI matrix runs 3.11 through 3.14.

  The list is [the Python devguide's](https://devguide.python.org/versions/)
  rather than a judgement of our own about who is still out there. 3.9 reached
  end-of-life on 31 October 2025, ten months before this release. 3.10 is
  still in security-only maintenance, but it reaches end-of-life in **October
  2026** — two months from now — so stopping there would have meant doing this
  again almost immediately. 3.11 runs to October 2027.

  Dropping 3.9 also removes a real cost rather than a nominal one. On 3.9 a
  clean `pip install torch` cannot `import torch.distributed.checkpoint` at
  all: `torch.distributed.elastic.rendezvous.registry` does `from
  importlib_metadata import entry_points` under a `sys.version_info < (3, 10)`
  guard, and the wheel's metadata declares only filelock, fsspec, jinja2,
  networkx, sympy and typing-extensions. Verified against the CPU wheel on
  `python:3.9-bookworm`: the import fails, and installing that one backport
  fixes it. The sharded-resume tests died there and on no other interpreter.
  Every line of it is gated on being below 3.10, so raising the floor deletes
  the problem instead of carrying a workaround for it.

  3.15 is **not** in the matrix, and was for exactly one day. It went in as a
  release candidate marked `continue-on-error`, on the theory that a prerelease
  is worth watching early and never worth holding a tag for. The theory did not
  survive contact: torch on 3.15.0rc1 segfaults in its own C++ —
  `TensorImpl::incref_pyobject`, reached through `torch.save` — and takes the
  whole pytest process down on the *first* test. So there was no partial signal
  to watch: 246 collected, nothing run. What was left was a permanently red
  square, which is the thing a real red hides behind on the day someone checks
  the matrix before tagging. It comes back when torch survives 3.15.

- **Shard placements are stored as data, not as `str(placement)`.** What went
  into a checkpoint was torch's own short repr — `S(0)`, `R`, `P(sum)` — and
  the only way back from it is a parser for a string nobody promised to keep.
  Resharding has to ask which dimension a tensor was split along, so the answer
  is now written down as an answer. The mesh shape is recorded alongside.

  Checkpoints written before this are still read: both the short form and the
  long `Shard(dim=0)` one are parsed back. Nothing about resuming an existing
  checkpoint at the same world size changes.

- **The Moonclip backend now builds `MoonclipManager` rather than
  `CheckpointManager`.** Ravex owns the topology and hands Moonclip a value;
  it no longer relies on a convenience layer that worked one out.

  `CheckpointManager` read `RANK`/`WORLD_SIZE` from the environment whenever
  it was not told. Under `torchrun` it would then believe it was one rank of
  eight and reject the single-rank save API outright — and this backend caught
  that in the `except` around its own construction and fell back to
  `torch.save`, with one log line to say so. A distributed run lost Moonclip
  checkpointing and kept training. The pinning that avoided it has been in
  place since the failure was found, and a test has guarded it since; what
  changes now is that there is nothing left to pin against.

  `MoonclipManager` is the explicit layer underneath, it never guessed, and it
  already accepts every option this backend passes. Nothing changes about
  where checkpoints land or what is in them.

  Requires Moonclip **0.0.9**, and the floor moves accordingly — not for the
  manager, which has always been there, but for `unflatten_state_dict` below.
  A version requirement stated in `pyproject.toml`, rather than a method call
  wrapped in a guard that answers "nothing here" when the method is missing.

- Loading goes through Moonclip's `unflatten_state_dict`, new in 0.0.9 and the
  public inverse of the `flatten_state_dict` this backend already used to
  write. `CheckpointManager.load` rebuilt the state tree *and* called
  `load_state_dict` on live objects; this backend has no live objects at that
  point and wanted only the first half. Until 0.0.9 the first half had no
  public name, so taking it meant taking the second as well.

- **The checkpoint handoff reports `backpressure` apart from `store`.** They
  were one number, and the sum reads like the writer.

  `store` is the shadow copy — the reason the training loop can carry on
  mutating weights while the writer is still working. It is memory bandwidth,
  it scales with the model, and it comes down by making the state smaller.
  `backpressure` is the wait for the *previous* checkpoint's writer: Moonclip
  allows one save in flight, so a writer that has not drained stops the next
  save before any of its work starts. It scales with the cadence and with the
  storage, and it comes down by checkpointing less often or writing somewhere
  faster.

  Added together they name neither. Watching the combined figure grow is what
  produced a hypothesis about a writer in deficit, several A/B runs, and an
  issue filed against the writer — while the phase was a 2 GiB `memcpy` for
  the whole of it. On a 1.5B model at cadence 2 the wait was 4-12% of what the
  single number showed.

  Now, on a run whose writer is genuinely behind, the log line says which:

  ```
  step 2: flatten 0.001s, store 0.101s, backpressure 1.655s
  ```

  where before it said `store 1.756s` and left the reader to guess. Needs
  Moonclip 0.0.9, whose `last_queue_wait()` is where the number comes from —
  it existed before, but only as a line `MOONCLIP_PROFILE=1` printed to
  stderr, which is not somewhere a program can read it. The cadence alarm
  counts the new phase like every other one: it is wall time the training loop
  pays, and it was already in the total.

### Fixed

- **Torch without NumPy broke every collective on the resume path.** Ravex
  depends on PyYAML alone, and torch does not require NumPy either, so an
  install with neither is a supported way to run this. But
  `dist.all_gather_object` decodes what it gathered with
  `tensor.numpy().tobytes()`, so on that install every agreement between ranks
  raised `RuntimeError: Numpy is not available` — not at import and not at
  startup, but at the first collective of a resume, after the wire work had
  already happened and with nothing left for the caller to fall back to.

  The five object collectives — the step every rank holds, the stores each
  machine can see, the run id, the shared-storage probe and the all-or-nothing
  resume flag — now go through one wrapper. Where the conversion works it is
  still torch's own call. Where it does not, the wrapper posts the same two
  `all_gather` calls in the same order and with the same shapes and dtypes,
  and decodes with `bytes(tensor.tolist())`. Matching torch on the wire is the
  point of that care: a job whose ranks disagree about NumPy still meets,
  which is checked with two gloo ranks, one of each.

  The probe is `tensor.numpy()` itself, asked once, rather than `import
  numpy` — torch initialises NumPy on its own, and a version it was not built
  against imports perfectly well and then fails the conversion.

  Found by CI rather than by a user: the unit job installs torch and pytest
  and nothing else, which is exactly this configuration.

- **A store recovered over the network stayed marked as a copy.** A rank that
  came up without its own store and took one back from a peer was left holding
  `.ravex-replica-ok` in its own store directory. Promoting a copy off this
  machine's own disk removed it; the two roads home ended differently.

  The cause is that `StoreWriter` writes the marker into whatever directory it
  filled, which is right — from where it stands it is always building a copy,
  and while the bytes are arriving the absence of that file is the only thing
  separating a half-built store from a whole one. What was missing is the
  other end of the same act: `promote_copy` removes it deliberately once the
  store is whole, and the network path did not.

  No known consequence today — `replica_is_complete()` is asked about replica
  paths, not about a rank's own store, so nothing read the file where it
  should not have been. It is worth correcting because the claim goes false
  immediately: it says a copy is complete about a directory that is, from the
  next save onward, a live store being rewritten. True for one instant and
  left where a later reader would believe it.

  Found on the first pair of real machines this project has run on (two RunPod
  pods, 2026-08-21). Nothing compared the two paths, which is why a divergence
  this small reached rented hardware; a test now holds them side by side.

- **A shard rebuilt on an older torch could come back with a global shape
  nobody asked for.** `DTensor.from_local` is given `shape=`/`stride=` so
  uneven shards survive; a torch whose signature predates those arguments
  raises `TypeError`, and Ravex fell back to calling it without them. That
  fallback infers the global shape as local × mesh size — correct when the
  tensor divides evenly across the mesh, and wrong by exactly the remainder
  when it does not. A 10-row tensor over 4 ranks came back claiming 12.

  Nothing failed at the point of the mistake. Every shard was individually
  valid, `set_state_dict` accepted them, and the first symptom would have been
  somewhere else entirely — which is the same shape as GPU-59 and GPU-79, and
  the reason this is worth a release note rather than a line in a diff.

  The fallback is now taken only after checking that the shape it would infer
  is the shape that is actually there, and refuses with both numbers when it
  is not. Even splits are unaffected, which is most of them. Every rank
  reaches the same verdict — with an uneven split the ranks holding a full
  chunk infer too much and the short one too little, so none of them agrees —
  so no rank is left inside a collective while another raises.

### Removed

- **Four compatibility guards against Moonclip builds the floor already
  excludes.** GPU-88 asks for this sweep before every tag, and names the shape
  to look for: *every guard around a backend call is a version requirement in
  disguise*. Each one below was dated against the release that introduced what
  it guarded, and every one of them was unreachable.

  - `restore_from_remote` was wrapped in `except AttributeError` returning
    `False`. This is the exact case GPU-88 was written about. The method has
    existed since Moonclip 0.0.8 and the floor has said `>=0.0.8` since — the
    guard simply outlived it. It never crashed; it answered *"nothing to
    restore"*, which is indistinguishable from an empty bucket. A rank whose
    disk had been replaced would have started from scratch with its own data
    sitting in the remote, and taken every other rank back with it, because a
    resume is agreed at the oldest step everyone holds.
  - `flatten_state_dict(as_tensors=)` was feature-detected, with the byte path
    as a fallback and a log line saying checkpoints would block the training
    loop about five times longer. The parameter shipped in 0.0.4.
  - `keep_base_in_memory` was feature-detected, warning that the base would be
    retained anyway when a run had explicitly asked otherwise. It shipped in
    0.0.4 too.
  - `_accepts`, the introspection helper the last two used, with nothing left
    to ask.

  None of the removed branches had a test. That is not a coincidence: a branch
  that only runs against a build the packaging forbids cannot be exercised
  without installing one.

  One guard of that shape survives, because it is not dead: the
  `except TypeError` around `DTensor.from_local(shape=, stride=)`. Torch is
  not a declared dependency — Ravex attaches to whatever build is already
  installed — so there is no floor to date it against. What it *did* do
  silently is fixed below.

## 0.0.4 — 2026-08-21

Jobs spanning more than one machine. Everything below exists because of one
question asked on 2026-08-18 — two boxes of eight GPUs, does this work? — and
the answer turned out to be "partly, and it does not tell you which part".

On 2026-08-21 the question was finally put to two machines that were actually
two machines, on a network that could actually fail. The replication holds:
a box that arrived with an empty disk took its store back from its peer and
resumed where the run had stopped, and a copy caught half way was refused
rather than trusted. What that day changed in this release is the last two
entries below — both are things a single machine could not have shown, because
on loopback the number they hide is zero.

### Changed

- **The handoff breakdown names the replication, and the cadence observation
  counts it.** `Checkpoint at step N handed off in 186.362s (drain 34.898s,
  collect 0.228s, flatten 0.001s, store 0.030s)` — a line that declares 186
  seconds and explains 35. The missing 151 were the copy going to the other
  machine: timed inside the handoff, named by nothing, and therefore invisible
  to the observation that exists to say *"checkpointing is eating your wall
  time"*. That observation sums the phases this line names, so it computed
  0.26s against a 291s interval, called it 0.09%, and stayed silent while the
  run spent 64% of its wall time stopped.

  Over loopback the copy costs milliseconds, which is why a missing phase hid
  a number that was always zero. It took two machines with 100 Mbps between
  them — measured at 7 MB/s each way — for it to become the whole handoff.
  Now there is a `replicate` entry, the phases add up to the total, and the
  same situation reports **42% of wall time** and says so. `drain` is still
  excluded, and for the reason it always was: it is the training loop's own
  queued work coming due, not a cost of checkpointing. Replication is not that
  — without it the time would not exist at all.

- **A copy caught mid-transfer is now named rather than called absent.** Losing
  a machine while a replica is in flight left the survivor saying *"No store
  anywhere for rank(s) 1 - starting from scratch. Either the run that wrote
  these had fewer ranks, or the machines holding the last ones are gone"*.
  Neither was true: the machine printing it was powered on with 680 MB of that
  rank's store on its own disk, disqualified because `StoreWriter` removes the
  completeness marker before the first byte lands and the transfer never
  finished. The refusal is correct — a torn copy is not a checkpoint — but the
  explanation sent the reader looking for hardware that was fine.

  The machine holding the copy now says so. It is the only one that can tell
  the two apart, and the others go on reporting what they see. Not a rare
  corner on a slow link: measured at 0.5s resolution on the same pair, a
  ~670 MB store takes 102s to copy against a 244s cycle, so the copy is
  unusable **42% of the time** and losing a machine during a transfer is close
  to a coin toss.

- **Peer replication now moves a store at the speed of the wire.** Copying a
  checkpoint to another machine was running at about a third of what the same
  link carried with nothing else in the way: 529 MB/s against 1462 MB/s on a
  rented box, and the same shape on a laptop. It was not the network and not
  the disk — reading and framing the store ran at 3411 MB/s on its own. It was
  four copies of every chunk, two on each side, all of them holding the GIL.

  Two of them are gone. The sending side hands `isend` a tensor over the
  chunk's own memory instead of a `bytearray` duplicate — safe because
  `fixed_chunks` yields fresh immutable `bytes`, which is the only thing the
  copy was protecting against. The receiving side writes straight from the
  tensor's memory when nothing is half-parsed, instead of `tobytes()` into a
  `_pending` buffer, slicing a piece out, and memmoving the tail. Headers and
  file boundaries still take the buffered path, which is where the ragged
  cases always lived.

  Measured on the same machine, alternating arms: **355 MB/s → 527 MB/s, a 48%
  improvement**, which puts the transfer at the wire's own rate. A threaded
  version that also overlapped disk with network was tried and added only six
  points on top — not worth concurrency in a recovery path, so it was dropped.
  The bytes on the wire did not change: a patched peer and an unpatched one
  still understand each other.

- **`pip install ravex` now installs the autoloader.** The one-line
  `ravex_autoload.pth` ships in the wheel, so checkpointing works on a project
  with a `ravex.yaml` without anyone running `ravex enable` first. The README
  used to promise the opposite — *"installing the package changes nothing on
  its own"* — and that sentence is gone.

  What replaces it is a weaker promise that is worth more, because it is about
  what the file *does* rather than about its absence: **Ravex is inert until a
  project asks for it.** The line runs in every interpreter in the
  environment, imports only `os` and `sys`, looks for a `ravex.yaml` at or
  above the working directory, and — finding none, and no `RAVEX_ENABLED` —
  installs nothing at all and returns. It never imports torch to decide. When
  it does arm, it arms a hook that waits for `import torch` and loads the
  runtime only then, removing itself once it has fired.

  Measured with `-X importtime`: **1.8 ms**, down from 11.6 ms once `typing`
  was taken off the path (see below). Verified on a real wheel rather than on
  the configuration — the file lands in `site-packages`, not in the
  `.data/` directory that `data_files` would have put it in and that pip
  installs one level above where a `.pth` is ever executed. It is recorded in
  `RECORD`, so `pip uninstall ravex` takes it away again: an orphaned `.pth`
  importing a module that no longer exists is the classic way this scheme
  breaks, and it is not a risk here.

  `ravex enable` still exists, for putting the file back after
  `ravex disable`. `ravex disable` now says that a later
  `pip install --upgrade` will restore it.

- **`ravex/__init__.py` no longer imports `typing`.** It was 9.8 ms of the
  11.6 ms the autoloader cost — 85% — for three names used only in
  annotations, which the `from __future__ import annotations` already at the
  top of the file turns into strings that are never resolved. The annotations
  now use builtins. Visible in the public typed surface: `status()` returns
  `dict[str, object | None]` rather than `Dict[str, Optional[Any]]`, and
  `object` is stricter than `Any` for a consumer.

- **The Moonclip floor is `>=0.0.8`.** Ravex calls `restore_from_remote()`,
  which the engine grew in 0.0.8, and the backend catches the `AttributeError`
  an older one raises. That is the right thing for a method that may not be
  there, and it is also why the pin has to move: on 0.0.7 the call does not
  fail, it answers "nothing to restore" — so a rank whose disk was replaced
  starts from scratch with its own data sitting in the bucket, and takes every
  other rank with it. The code path cannot tell an empty remote from an engine
  that has no way to read one; the version requirement can. The same release
  is also the first that can put an object over 5 GiB into S3 at all, which
  `sharded_checkpoints: gather` reaches on its own.

### Added

- **The storage topology is announced at activation.** With
  `sharded_checkpoints: per_rank` and local storage, each machine writes only
  its own ranks' shards to its own disk, so no machine holds a whole
  checkpoint. It then resumes only if every machine is handed the same ranks
  again — which no launcher promises — and not at all if a machine is lost.
  The behaviour was safe and silent: the run simply started over, with no
  indication that a checkpoint existed and had not been used. Ravex now says
  which of three situations you are in, at activation rather than at the first
  failed resume.
- **Shared storage is probed, not guessed.** A local disk and an NFS mount are
  the same `type: local` pointing at a directory that exists, so the question
  is asked instead: every rank drops a uniquely named marker and looks for
  everyone else's, with `all_gather_object` as the synchronisation. A directory
  that cannot be written to answers "not shared", which is the safe reading.
- **`replicate_every`** — copies of each rank's store to a peer on another
  machine, every N checkpoints, for jobs with neither a bucket nor a shared
  filesystem. The peer is `(rank + local_world_size) % world_size`, which only
  lands on a different machine when the ranks are spread evenly, so an uneven
  layout is reported as *not* replicating rather than assumed to work. The
  exchange is point-to-point `isend`/`irecv`, never a collective — an
  all-gather would leave every rank holding `world_size` copies. All-or-nothing
  with a collective verdict: three copies of four landing is not a restore
  point and is not recorded as one. Replicas live under `replica/`, outside the
  `rank_*` names discovery scans, and count only once a completion marker is
  written last. Default 10; `0` turns it off. **The guarantee, stated exactly:
  the loss of any one machine is survivable, at a cost of at most one
  replication interval of progress.**
- **A replaced machine gets its store back.** A rank that comes up with nothing
  pulls its store from the peer that has been holding a copy, or — with a
  bucket configured — from the remote. Without this the copies existed and
  nobody read them, which protects the bytes and not the run.
- **Run identity and owner records.** Each per-rank store carries a
  `.ravex-owner` naming the run that wrote it and the machine it was written
  on. Two training histories on one disk used to be indistinguishable: on
  2026-08-19 a run restarted with a different placement wrote a second history
  beside the first, and a later restart resumed the accidental one while the
  original sat one directory away. It is also what turns "rank 2's store is not
  here" into "rank 2's store was written on node0, which is not running rank 2
  now".

- **A run is told when its cadence is expensive.** Once, and as a statement of
  what happened rather than a prediction: what fraction of wall time went into
  checkpoint handoff between the last two checkpoints. Measured on 8x RTX 5060
  Ti with a 1.5B model, `checkpoint_every=2` spends a third of wall time on
  handoff and nothing breaks — the writer keeps up, the run is simply slower
  than its author probably meant. So it does not claim the cadence is
  unsustainable and it does not quietly raise it; both would be guesses about
  a machine the process cannot see, while the ratio is a fact it can. The
  `drain` phase is left out, being the training loop's own queued GPU work
  coming due rather than a cost of checkpointing.

### Fixed

- **Replication never worked on a GPU job, and said it did.** Moving a store
  between machines means moving bytes, and bytes live on the host — but the
  transfer went out on the **default** process group, which on a multi-GPU job
  is NCCL. NCCL is a GPU collective library and refuses host tensors outright.

  Measured on 8x RTX 5060 Ti on 2026-08-21: every replication round failed
  with `No backend type associated with device type cpu`, **not one copy was
  ever made**, and the announcement at activation went on promising that
  losing a machine would cost at most one interval. The failure was reported —
  a warning per round — but the promise was louder and came first.

  Bytes now travel on a **gloo** subgroup, opened once at activation and
  reused. On a job that is already gloo the default group carries host tensors
  perfectly well and no second group is made. Where neither is possible,
  replication is declared **off at activation**, in the same message that
  would otherwise have promised it — because a run that is not protected
  should be told once, at the start, rather than a warning at a time into a
  log nobody is reading.

  Verified on the same hardware after the change: six rounds out of six, and
  both disks holding the peer's copy where before they held nothing.

  Nothing on CPU was affected, which is exactly why nothing caught it: the
  four-scenario container bench runs on gloo, where the same code is correct.
  The backend was the one variable it could not vary.

- **A bucket was a backup you could not resume from, and the documentation
  said otherwise.** Moonclip's remote support was push-only, so the step to
  resume from was read from the *local* manifest. On a six-node bench a node
  whose disk had been replaced started from scratch with its own data sitting
  in the bucket, and took every other rank with it, since a resume is agreed at
  the oldest step everyone holds. Fixed in Moonclip; Ravex now fills an empty
  store from the remote before deciding where to resume. The runtime warning
  that used to recommend S3 for a problem S3 did not solve was corrected in the
  same pass.
- **One rank disabling itself no longer hangs the other seven.** A failed
  checkpoint called `_disable`, which is per process: a full disk on one node
  of eight turned that rank off while the rest stayed on. At the next
  checkpoint the seven entered `collect_state` — a collective — and the eighth
  returned at the first line and ran ahead into the next forward. The seven
  then waited for a participant that never came, and NCCL takes half an hour to
  say so, with all eight GPUs allocated and billing throughout. Not a fast
  error, an expensive hang. The verdict is now taken once, by everyone, and
  everyone acts on it. A backend that is merely unavailable is reported rather
  than latching the runtime off.

- **After a reshuffle every copy was present and none was used.** The ring
  addresses a peer **by rank**, but a copy travels **with the disk** it was
  written to. Move the nodes round by one position and each rank comes up
  sitting on the copy of its own store — complete, and unreachable, because
  the peer that used to hold it is now elsewhere holding somebody else's. Six
  intact copies bought nothing, and the run started from scratch with the
  bytes under its feet. A rank without a store now looks for a copy of itself
  on its own disk first: a local file copy, no pairing and nothing on the
  wire. It is promoted only when the history it belongs to is unambiguous —
  named by the stores that survived, or, when none did, agreed among the
  copies themselves. A copy from an older run is refused, because resuming
  half the shards from one training history and half from another is a wrong
  model and a silent one.

### Testing

- **Two scripts for the replication transfer**, in `integration/scripts/`, both
  running two gloo processes with no GPU and no rented machine — the path was
  measured at 529 MB/s over loopback and 562 MB/s over real TCP between
  containers, so whatever binds it can be studied on a laptop.

  `measure_replication_transfer.py` times the transfer against the two bounds
  it has to be read against: the wire carrying the same volume with no file
  touched, and the disk delivering the store with nothing sent. When the
  transfer sits far below both, its `phases` arm says which step is spending
  the time. That is how the 48% came off.

  `verify_replication_transfer.py` sends a deliberately awkward store — a
  zero-length file, a one-byte file, a file exactly one chunk long, another one
  byte past the boundary, nested directories — and compares the arrival with
  the source by hash. The unit tests drive `StoreWriter` in one process; this
  is the only thing that exercises `exchange_stores` between two of them.

- **A multi-machine bench**, `integration/multinode/`. One container per rank,
  because the question is which ranks can see which directory and ranks on one
  machine all see the same one — `torchrun --nproc_per_node=6` makes six
  processes, not six machines. Four scenarios: shared storage, split storage,
  a machine replaced with an empty disk, and the nodes coming back in a
  different order. It prints what each disk holds before the resume, so a run
  can tell "the data was gone" from "the data was there and nobody looked".

  It is the first end-to-end exercise of both the shared-storage path and the
  peer replication, and it earned its keep on its first honest run by finding
  the reshuffle defect fixed above — which no unit test could have found,
  because every piece involved was correct on its own.

  Worth knowing if you extend it: `--init` is load-bearing. A round ends with
  the training script sending itself SIGKILL, and the kernel discards a
  SIGKILL aimed at PID 1 from inside its own namespace when PID 1 has no
  handler. Without an init process the script runs to completion and the bench
  reports six happy nodes having proved nothing.

### Documentation

- **The multi-machine story is written down.** It had never been: `multi-node`,
  `nnodes` and `NFS` appeared nowhere in `docs/`, the README or the sources,
  and a runtime warning is not a substitute for a page you can read before
  starting a job. See *More than one machine* in
  [docs/how-it-works.md](docs/how-it-works.md) and
  [docs/configuration.md](docs/configuration.md).
- **Keeping the checkpoints after a run ends is the user's**, stated as such,
  together with where to look: a sharded model has no checkpoint at exit, so
  the newest thing worth copying off is the last periodic one.

## 0.0.3 — 2026-08-18

**No changes to the library.** `git diff v0.0.2..v0.0.3 -- ravex/` is empty: the
package you install is byte-for-byte the 0.0.2 one. This release exists to carry
the packaging fix below and to put the release pipeline through a real
publication for the first time, which is a thing worth knowing about a version
before you wonder what it changed.

### Fixed

- **The source archive contains the changelog again.** Setuptools takes the
  readme and the licences from `pyproject.toml` but has no field for a changelog,
  so `ravex-0.0.2.tar.gz` shipped without one — the sdist being exactly the copy
  a distribution packager or an auditor reads, and the one that survives if the
  repository does not. A line of `MANIFEST.in` fixes it; this is the first
  release built with it, and the archive was checked rather than assumed.

### Infrastructure

Not shipped, but this is the release where CI started meaning something. Every
job had been failing: `--index-url` for the CPU torch wheels *replaces* PyPI
instead of adding to it, so pip could not find the build backends torch's own
dependencies need and the six interpreter jobs died in six seconds each. The
integration job hit the ten-minute runner limit to the second, and now runs in
under five with the framework tests split into a job of their own. One unit test
asserted Windows path semantics on a POSIX interpreter and had never run
anywhere it could fail.

## 0.0.2 — 2026-08-17

First release on PyPI: `pip install ravex`, or `pip install "ravex[moonclip]"` on
Linux for the delta-tracking engine. Alpha, and the limits below are the part
worth reading before you rely on it.

### What it does

Transparent checkpoint and resume for PyTorch training: no change to your script,
not one import. A killed run, restarted with the same command, continues with the
same weights, optimizer moments, LR schedule, AMP loss scale, RNG state and
position in the dataset. The test suite asserts the strong form — a run killed at
step 20 and resumed produces losses bit-identical, step by step, to the run that
was never interrupted.

Verified: plain loops, gradient accumulation, LR schedulers, AMP, `num_workers >
0`, DDP over NCCL, FSDP1 and FSDP2, and a killed `torchrun` job resuming on every
rank. The CUDA paths were last exercised on 8× RTX 5060 Ti with torch 2.12/cu130.

### Added

- **Per-rank sharded checkpoints** (`sharded_checkpoints: per_rank`). Every rank
  writes its own shard into its own store instead of gathering the whole state on
  rank 0. On a 1.48B model with Adam, collecting the state went from 15.6 s to
  1.5 s, and peak host memory from 18.1 GiB on rank 0 to 6.1 GiB on every rank
  alike. The default stays `gather`, because per-rank shards only resume at the
  world size that wrote them and losing that is not something you should acquire
  by upgrading.
- **The per-checkpoint log line reports where its time went**, not just the total:
  `handed off in <total> (collect …, flatten …, store …)`, or `copy`/`queue` on the
  `torch_save` backend. A handoff is time the training loop stands still, and the
  total alone never says which phase to look at — the last investigation of a slow
  one cost four experiments that each ruled out the wrong suspect.

### Fixed

- **The `torchrun` launcher no longer starts a runtime.** It imports torch to
  parse its own arguments, so it came through the autoloader like everything else:
  a run with eight ranks announced nine runtimes, the extra one being a process
  that never trains. Harmless in practice — it holds no model, so it never
  reached the storage backend — but harmless by construction rather than by
  design.
- **The version was declared in two places** and diverged at the first bump: the
  CLI announced 0.1.0 while the package was 0.2.0. `ravex/__init__.py` is the only
  source now, and the release refuses to publish a tag that disagrees with it.

### Known limits

These are choices, not defects, and each one costs something specific.

- **No final checkpoint at exit for a sharded model.** Shutdown is exactly where
  ranks stop being in lockstep, and a collective nobody else joins does not raise
  — it hangs. Losing the last few steps is a bounded cost; a hang on rented
  hardware is not. Set `checkpoint_every` accordingly.
- **`IterableDataset` position is not replayed.** There is no index sampler, so
  the position in the stream cannot be reproduced. Everything else is restored.
- **With HuggingFace `Trainer` or Lightning, state restoration is exact but replay
  is not.** Both iterate the dataloader on their own schedule and consume the
  global RNG around the loop, so a resumed run continues correctly from the
  checkpointed state and then sees a different shuffle. Plain loops, DDP and FSDP
  are bit-exact with randomness on; this is a framework-interaction limit.
- **Your loop's bounds are still yours.** A resumed script runs its own
  `for epoch in range(N)` from the top and has no idea 3000 steps already
  happened. Set `max_steps` and Ravex ends the run at the right step however many
  times the process restarted.
- **`auto_wrap_policy` is not optional for FSDP1** at real sizes: without it the
  model is one flat unit and every rank materialises it whole in the forward pass.
  A 1.5B model OOMs on eight 16 GiB cards.
