# Two machines, rented

The bench in `../multinode/` makes six *containers* on one host, because the
question there is which rank can see which directory. This one makes two
*machines*, because there are questions a single host cannot be wrong about.

Peer replication is the main one. On one box the copy goes over loopback: it
always succeeds, it takes milliseconds, and every number it would otherwise
expose reads zero. Three defects found on 2026-08-21 had all survived a green
container bench for exactly that reason — the bench could not vary the thing
that mattered.

Everything here is throwaway by design. It runs against boxes you rent for an
hour and destroy, and it assumes a meter is running: phases go on both machines
at once, results come home after every phase, and nothing waits on a human to
decide what to type.

## What it needs from the provider

Two machines that can open TCP connections to each other, with GPUs, reachable
over SSH with `scp` working. That is a shorter list than it sounds:

- **A private network between them, arranged at creation.** Cloud boxes are
  usually behind NAT on their own bridge, and two of them often have the *same*
  address — the peer's address is your own. Neither environment variables nor
  NCCL settings work around that. On RunPod the toggle is *Global Networking*
  and the machines then resolve each other as `<pod-id>.runpod.internal`; on
  vast.ai it is an overlay network, and only between instances in one physical
  cluster. Both must be chosen **when the machine is created** — a running box
  cannot be added to either.

  **Check it on every pod, before anything else, with one command:**

  ```sh
  hostname -I
  ```

  There must be a `10.x.x.x` in the output — on RunPod it arrives on an
  interface called `podnet1`. A pod showing only its `172.x` bridge address was
  created without the private network, cannot reach the other machine in either
  direction, and cannot be fixed: destroy it and make another. On 2026-08-30
  three pairs of boxes out of four died exactly here, and always the same way —
  the toggle set on the first pod and not the second. The failure looks like a
  plain `TimeoutError` much later, after the kit is pushed and the meter has
  been running for a while.

  The pod's own SSH string does not tell you: it is the public address, which
  every pod has whether or not it is on the private network.
- **Full SSH, not a proxy shell.** `push.sh` and `fetch.sh` move files with
  `scp`, and the convenience SSH some providers offer does not carry it.
- **A GPU whose architecture the image's torch was built for.** This is checked
  by `10-setup.sh`, which reinstalls torch when it is not: an image can ship a
  torch that answers `torch.cuda.is_available() == True` and then fails on the
  first real operation with `no kernel image is available for execution on the
  device`. Blackwell (`sm_120`) against a torch built to `sm_90` is how that was
  found.
- **Two boxes in one region are not two regions.** RunPod hands out different
  `10.x` subnets to pods in the same cluster, so the private addresses look
  reassuringly unrelated while the round trip is 0.29 ms. The mount tells the
  truth: both pods on 2026-09-10 had `/workspace` on `mfs#euro.runpod.net`,
  which is one European cluster. Check it before believing a latency number
  means anything about continents.
- **Local disk for the stores.** Some providers mount a network filesystem at
  the obvious writable path — measure a store written there and you have
  measured their storage, not this code. `KIT_ROOT` decides where everything
  lands; keep it on a real disk.

One GPU per machine is enough. These questions are about the network and the
store, not about FLOPs: one GPU each gives `WORLD_SIZE=2, LOCAL_WORLD_SIZE=1`,
which is the honest two-machine configuration, and costs a fraction of two
eight-GPU boxes that would answer nothing more.

## The disposable key

```sh
ssh-keygen -t ed25519 -N "" -C "kit-throwaway" -f ./id_throwaway
```

No passphrase, because every phase opens several connections; the price is that
whoever reads the file can enter the boxes, so it dies with them — `rm -f
id_throwaway id_throwaway.pub` belongs in the same minute as destroying the
instances. Put the public half on the provider **before** creating the machines:
a key registered on an account does not always reach a box created afterwards.

Override with `KIT_KEY=/path/to/key` if you would rather keep it elsewhere. The
scripts pass `IdentitiesOnly=yes`, so a box that rejects this key says so
immediately instead of quietly falling back to a personal one and hiding that
the throwaway was never installed.

## The sequence

From your machine, in this directory. Paste the connection strings as the
provider gives them.

```sh
KIT_ROOT=/root bash push.sh 'ssh root@IP0 -p PORT0' 'ssh root@IP1 -p PORT1'
```

Then tell each box who it is and where the other one is — an address or a name,
whatever they reach each other by, which is never the address the SSH arrived
on:

```sh
bash addrs.sh <node0-address-or-name> <node1-address-or-name>
```

```sh
bash on-both.sh 'bash $KIT/00-preflight.sh'
```

`$KIT` expands on the far side, so the same line works wherever the kit landed.
Quote it single, or your own shell eats it first.

**`00-preflight.sh` is the go/no-go, and it costs seconds.** GPUs present, a
route to the peer, torch with kernels for this GPU, and two ranks on two
machines shaking hands over **both** NCCL and the gloo subgroup — the one the
replication moves bytes on. If it does not end in `ready`, stop: destroy and
recreate rather than adjust. Nothing below it can work.

Then setup, then the phases:

```sh
bash on-both.sh 'bash $KIT/10-setup.sh'
bash on-both.sh 'bash $KIT/40-bandwidth.sh'
bash on-both.sh 'PARAMS=6e7 bash $KIT/20-correctness.sh run1'
bash fetch.sh run1
```

`40-bandwidth.sh` comes first even though it is the least interesting: it tells
you how large `PARAMS` can be before a phase takes longer than the rental. The
store each rank writes is roughly `params × 16 / world_size` bytes, and it has
to cross the wire at whatever that measurement said.

## What the preemption phase now has to show

`46-gpu92-sigterm.sh` used to answer one question — did the survivor save too.
Since 2026-09-13 it answers two more, and both are things only a link with a
real round trip in it can be wrong about.

**The announced step has to match on both machines.** The detection stopped
being a collective per step (GPU-111): a preempted rank writes one key naming
the step everybody saves at, and an ordinary step is a `check` on a key that is
not there. On loopback that key is written and read in microseconds. Here the
announcement has to cross a continent before the announced step arrives, and
the two cadence checks of `ANNOUNCE_LEAD` are what buy it the time. Two
machines printing *different* steps, or either of them printing `too late to
join`, means the lead is too short for this link — and that is the number to
change, not the protocol.

**Per-machine detection will print nothing here, and that is the pass.** The
channel is now one group per machine when the sharding stays on a machine and
one group over the job when it does not (GPU-125). One GPU per box means
`--nproc_per_node=1`, so the FSDP group spans both machines: the sharding
crosses a machine, the wide group is correct, and the new branch is not
reached. Its silence is the unchanged branch working.

Reaching the new branch needs **two GPUs per box with FSDP inside each one**,
which is a different rental. Worth one if the outer loop is ever run with
sharding underneath it, because that configuration is the one the change was
made for and nothing here has run it.

## The object collectives, which a stock image will not test

Since 2026-08-30 every agreement between ranks — the step to resume from,
which stores each machine can see, the run id, whether the storage is shared,
whether every rank succeeded — goes through `_all_gather_object`. It has two
implementations, and it picks the second **only when torch cannot reach
NumPy**:

```python
if _torch_can_reach_numpy():
    dist.all_gather_object(gathered, value)   # torch's own, as before
    return gathered
# the replacement is below here
```

Every image worth renting ships NumPy. So on two stock boxes the replacement
**never executes**, and a session that runs the phases below proves nothing
about it while the meter runs.

`RAVEX_ASSUME_NO_NUMPY=1` forces the second path. `launch` is `torchrun`, which
inherits the environment, so exporting it before a phase covers that machine's
ranks — and exporting it on **one** machine is how the mixed case is reached:

| both machines | what it exercises |
| -- | -- |
| unset | torch's collectives — the baseline, and what shipped before |
| `=1` on both | the replacement, end to end, on a real network |
| `=1` on one | a rank with NumPy and a rank without, meeting on the wire |

The third is not hypothetical: two boxes from one provider can come up from
different images. Its wire format is checked by
`TestTheObjectGatherOnTwoRanks` in `tests/test_dist_multinode.py`, which runs two
real gloo processes locally — so what is left for two machines is latency,
ordering and a peer that disappears, not the encoding.

## The outer loop without torchrun

`110-rendezvous.sh` is GPU-129 on real machines: no `torchrun`, no process
group, plain `python` on each box, and the store every node reads is a
`ravex rendezvous` in a process of its own on node 0. Three arms — `base`,
`join`, `kill0` — plus `stop` for the server, which no arm stops by itself: in
`kill0` node 0's script ends the moment its trainer dies, and the server is
exactly what has to outlive it.

**`kill0` is the arm torchrun could not pass.** Under torchrun the store lives in
node 0's agent, and the exchange reads peer addresses from it on every fetch, so
losing node 0's box stops every round. It had never been tried: the `kill` arm
of `100-outer.sh` kills node 1.

**In `join`, the number to read is the joiner's first `gather_wait_seconds`.** It
downloads the outer parameters and the momentum — twice a delta — over the real
link, inside the round it enters. On loopback that cost nothing.

**The rendezvous has no authentication.** Whoever reaches its port can take a
number and write any key; without `RAVEX_JOB_TOKEN` it can also read the job
token. A RunPod pod exposes only the ports it was created with, so 29400 is
reachable over `podnet1` and not from outside; keep it that way.

## Rules that cost money to relearn

**Interrupting the command on your machine does not stop the phase on the box.**
`Ctrl-C` kills the SSH, not the `torchrun` on the other end — a pair once ran
another 400 seconds at 100% GPU after the terminal had gone quiet. Kill it on
the box. The consolation is that an accidental interruption loses no work:
re-read `$KIT_ROOT/out`.

**`pkill -f <pattern>` kills the shell running it**, because the pattern is in
its own command line. Use the bracket trick you already use with `grep`:
`pkill -f "run_[t]rain"`.

**A phase leaves the rendezvous port held.** Two runs back to back give
`EADDRINUSE`; pass a different `MASTER_PORT` or make sure nothing survived.

**"Two ranks alive" does not say *which* run is alive.** Check what you started,
not that something is running — twice that ambiguity read as confirmation while
the processes belonged to the previous phase.

**A phase where a rank exits on purpose costs torchrun 300 seconds.** Not our
code: the surviving box's elastic agent waits on
`torchelastic/agent/terminal_state/last_member` for its own five-minute
default, and only then gives up with a `DistStoreError`. Measured on
2026-09-10, where it read exactly like a hang and took fifteen minutes and an
`ssh` to tell apart from one. Budget for it, or expect to explain it again.

**`bash watch.sh` answers "is it stuck?" without guessing.** `on-both.sh` hands
its output back only when the command ends, so a phase in the middle of a long
exchange and a phase that is wedged look identical from here. `watch.sh` prints,
for each box, the phase it announced, how many processes are running, and the
last timestamped progress lines — and `outer_run.py` beats every ten seconds
while it waits, so a stall is not silence, it is the same line with a growing
number.

**Fetch after every phase, not at the end.** A rented box can be preempted, and
anything that exists only there is a result that can be taken away.

## What is in here

| | |
| -- | -- |
| `push.sh` | copy this working tree and the kit to both boxes, print what each sees |
| `addrs.sh` | write `box.env` on each: its rank, its own address, its peer's |
| `on-both.sh` | run one command on both at once, and wait for both |
| `fetch.sh` | bring `$KIT_ROOT/out` home into `results/<tag>/` |
| `watch.sh` | what both boxes are doing right now, and whether anything is running |
| `remote/41-latency.sh` | round trip, three ways — what bandwidth does not imply |
| `remote/45-nccl-regroup.sh` | the group rebuilt without the rank that went away |
| `remote/46-gpu92-sigterm.sh` | one rank preempted, every rank saving together |
| `remote/100-outer.sh` | the outer loop over the link: base, bf16, a node killed, resume |
| `remote/110-rendezvous.sh` | the outer loop with no torchrun: base, a node joining late, node 0 killed |
| `remote/00-preflight.sh` | the go/no-go |
| `remote/10-setup.sh` | install moonclip and this ravex, fix torch if the GPU needs it |
| `remote/20-correctness.sh` | train, replicate, die like a preempted box, lose a machine, resume |
| `remote/30-measure.sh` | the replicated arm against its control |
| `remote/40-bandwidth.sh` | what the wire carries, both transports |
| `remote/50-…` to `90-…` | single-box variants: transport ceiling, size sweep, eight ranks, shared volume |

`boxes.env` and `results/` are written at run time and stay out of the
repository.
