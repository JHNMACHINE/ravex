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
- **Full SSH, not a proxy shell.** `push.sh` and `fetch.sh` move files with
  `scp`, and the convenience SSH some providers offer does not carry it.
- **A GPU whose architecture the image's torch was built for.** This is checked
  by `10-setup.sh`, which reinstalls torch when it is not: an image can ship a
  torch that answers `torch.cuda.is_available() == True` and then fails on the
  first real operation with `no kernel image is available for execution on the
  device`. Blackwell (`sm_120`) against a torch built to `sm_90` is how that was
  found.
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

**Fetch after every phase, not at the end.** A rented box can be preempted, and
anything that exists only there is a result that can be taken away.

## What is in here

| | |
| -- | -- |
| `push.sh` | copy this working tree and the kit to both boxes, print what each sees |
| `addrs.sh` | write `box.env` on each: its rank, its own address, its peer's |
| `on-both.sh` | run one command on both at once, and wait for both |
| `fetch.sh` | bring `$KIT_ROOT/out` home into `results/<tag>/` |
| `remote/00-preflight.sh` | the go/no-go |
| `remote/10-setup.sh` | install moonclip and this ravex, fix torch if the GPU needs it |
| `remote/20-correctness.sh` | train, replicate, die like a preempted box, lose a machine, resume |
| `remote/30-measure.sh` | the replicated arm against its control |
| `remote/40-bandwidth.sh` | what the wire carries, both transports |
| `remote/50-…` to `90-…` | single-box variants: transport ceiling, size sweep, eight ranks, shared volume |

`boxes.env` and `results/` are written at run time and stay out of the
repository.
