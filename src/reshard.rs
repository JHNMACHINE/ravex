//! Resuming a per-rank checkpoint onto a different number of ranks.
//!
//! Per-rank checkpointing writes each rank's own slice of every tensor and
//! nothing else, which is what keeps a checkpoint off any single rank's memory
//! budget. The price is that the slices only line up with the topology that
//! wrote them: resume eight shards onto four ranks and the first tensor raises
//! *saved shard is (384, 4096), this rank holds (512, 4096)*.
//!
//! This module is the arithmetic that makes them line up. It is deliberately
//! **pure**: no Python, no process group, no I/O. Everything here is integers,
//! so the part with no excuse for being under-tested can be tested
//! exhaustively over every `N -> M` pair in a range rather than on the one pair
//! a GPU box happens to have. [`crate::python`] is the only file that knows
//! this is reached from an interpreter.
//!
//! Two ideas carry the whole file.
//!
//! **Offsets are measured, not derived.** The obvious approach is to reproduce
//! torch's chunking rule and work out where each shard begins. Don't: the rule
//! has an uneven-tail case that `ravex._dist.collectives._rebuild_dtensor`
//! already carries a comment about, and a second implementation of it would
//! drift from the first in silence. Both sides can be *observed* instead — the
//! old lengths are the shapes of the tensors that were saved, the new lengths
//! are the shapes the live model is holding — and the plan is then a matter of
//! adding up what torch already decided.
//!
//! **A hole is worse than a failure.** If some old shard cannot be read, the
//! tempting move is to carry on with the ones that can. That produces a tensor
//! with a band of uninitialised rows: every individual shard valid, the whole
//! thing wrong, and nothing downstream able to notice. It is the same failure
//! GPU-59 and GPU-79 were about, and [`plan_reshard`] refuses rather than
//! reintroduce it through this door.

use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;

/// One DTensor placement, decoded.
///
/// `Unknown` carries what it was rather than being dropped: a reshard refuses
/// it *by name* instead of refusing it as "something", which is the difference
/// between a message a user can act on and a message they can only report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Placement {
    Shard { dim: i64 },
    Replicate,
    Partial { op: String },
    Unknown { repr: String },
}

/// Why a plan could not be made.
///
/// The three variants are three different exceptions on the Python side, and
/// the split is one the resume path already depends on.
/// [`ReshardError::Unsupported`] is `ReshardUnsupported` — "I refuse, and here
/// is why", a reason to fall back to starting from scratch with a message a
/// user can act on. [`ReshardError::Invalid`] is `ValueError` — something went
/// wrong, which is a bug. [`ReshardError::MissingHome`] is `KeyError`, kept
/// apart from the other two because a caller that has not recorded where a
/// store lives has a different thing to fix than one whose arithmetic
/// disagrees.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReshardError {
    Unsupported(String),
    Invalid(String),
    MissingHome(String),
}

pub type Result<T> = std::result::Result<T, ReshardError>;

// How torch writes a placement, for checkpoints from before placements were
// stored as data. Two forms each, and *both* are needed: what the old
// `_encode_shards` wrote was `str(placement)`, which on torch 2.12 is the
// short form — `S(0)`, `R`, `P(sum)` — while the long form is what `repr`
// gives and what anyone reading this would expect to have been written.
// Accepting only the one that looks canonical would have read every existing
// per-rank checkpoint as "unknown placement" and refused to reshard it.
//
// Compiled once. `Regex::new` per placement per tensor is the kind of cost
// that turns a port meant to be faster into one that is not, and these three
// patterns never change.
fn legacy_shard() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*(?:_?Shard\s*\(\s*(?:dim\s*=\s*)?|S\s*\(\s*)(\d+)\s*\)\s*$").unwrap()
    })
}

fn legacy_replicate() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*(?:_?Replicate\s*\(\s*\)|R)\s*$").unwrap())
}

fn legacy_partial() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"^\s*(?:_?Partial|P)\s*\(\s*(?:reduce_op\s*=\s*)?['"]?(\w*)"#).unwrap()
    })
}

/// Parse one `repr` from a checkpoint written before this was data.
///
/// A stored checkpoint is not a thing you get to reformat after the fact, so
/// these strings stay readable for as long as any of those checkpoints might be
/// resumed. Anything unrecognisable comes back as `Unknown` rather than failing
/// here: the caller that cares — [`shard_dim`] — refuses on it with a message
/// that can name the tensor, where this function could only name the string.
pub fn decode_legacy_placement(text: &str) -> Placement {
    if let Some(caps) = legacy_shard().captures(text) {
        // The capture is `\d+`, so the only way `parse` fails is an integer
        // wider than i64 — a number of mesh dimensions no machine has.
        return match caps[1].parse::<i64>() {
            Ok(dim) => Placement::Shard { dim },
            Err(_) => Placement::Unknown {
                repr: text.to_string(),
            },
        };
    }
    if legacy_replicate().is_match(text) {
        return Placement::Replicate;
    }
    if let Some(caps) = legacy_partial().captures(text) {
        let op = &caps[1];
        return Placement::Partial {
            op: if op.is_empty() {
                "sum".to_string()
            } else {
                op.to_string()
            },
        };
    }
    Placement::Unknown {
        repr: text.to_string(),
    }
}

/// The single dimension this tensor is split along, or `None` if it is not.
///
/// `None` means every rank holds the same bytes — a `Replicate` on a 1-D mesh —
/// and resharding one is copying rank 0's, which is why it is a normal answer
/// rather than an error.
///
/// Everything else in scope is a 1-D mesh with exactly one `Shard`. A 2-D mesh
/// — FSDP crossed with tensor parallel — makes the plan a cartesian problem
/// rather than an interval one, and refusing it by name costs one check and
/// saves a category of wrong answers. Widening later is additive: nothing
/// written here has to change for a 2-D planner to be added beside it.
pub fn shard_dim(placements: &[Placement], location: &str) -> Result<Option<i64>> {
    // The refusals come first and in list order, so a tensor carrying both a
    // Partial and an unreadable string is refused for the first thing wrong
    // with it rather than for whichever check happens to run first.
    for placement in placements {
        match placement {
            Placement::Partial { .. } => {
                return Err(ReshardError::Unsupported(format!(
                    "{location} is stored as a Partial value, which is a term waiting to be \
                     summed rather than a piece of the tensor. Resharding one would have to \
                     know what reduction is pending; it does not."
                )))
            }
            Placement::Unknown { repr } => {
                return Err(ReshardError::Unsupported(format!(
                    "{location} carries a placement this version does not understand ({repr})"
                )))
            }
            _ => {}
        }
    }

    let dims: Vec<i64> = placements
        .iter()
        .filter_map(|p| match p {
            Placement::Shard { dim } => Some(*dim),
            _ => None,
        })
        .collect();

    match dims.len() {
        0 => Ok(None),
        1 => Ok(Some(dims[0])),
        n => Err(ReshardError::Unsupported(format!(
            "{location} is sharded over {n} mesh dimensions. Resharding covers a 1-D mesh — \
             FSDP — where a shard is an interval; a 2-D mesh is a different problem and is \
             not attempted."
        ))),
    }
}

/// Running sum, with the leading zero: `[3, 3, 2] -> [0, 3, 6, 8]`.
///
/// Shard *q* covers global rows `[offsets[q], offsets[q + 1])`. Keeping the
/// fence-post form means both ends of every interval come out of the same list
/// and there is no `+ 1` to get wrong at a call site.
pub fn offsets_from_lengths(lengths: &[i64]) -> Vec<i64> {
    let mut running = Vec::with_capacity(lengths.len() + 1);
    running.push(0);
    for length in lengths {
        let last = *running.last().expect("seeded with a leading zero");
        running.push(last + length);
    }
    running
}

/// One piece of a new shard: which old rank holds it, the half-open interval to
/// take out of that rank's shard, and where it lands in the new one.
///
/// The destination is carried rather than implied so the caller can assert on
/// it instead of trusting the order it iterates in. At the Python boundary this
/// is the tuple `(source, (start, stop), (start, stop))`, unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub source: usize,
    pub take: (i64, i64),
    pub put: (i64, i64),
}

/// How to build each new shard out of the old ones.
///
/// Returns one list of [`Piece`] per new rank, in order, covering that rank's
/// shard from its first row to its last with no gap and no overlap.
///
/// Both arguments are *measurements*: `old_lengths[q]` is the extent of the
/// shard rank *q* actually saved, `new_lengths[r]` the extent rank *r* is
/// holding now. The two must sum to the same global extent — if they do not,
/// the checkpoint is not of this tensor, and continuing would silently truncate
/// or pad it.
///
/// The general case is not a rearrangement of whole shards. Going from four
/// ranks to three, no new shard equals any old one: each is stitched from two,
/// and one old shard feeds two new ones. Boundaries coincide only on an exact
/// halving or doubling — which is why an 8 -> 4 test passes while the general
/// mechanism stays unwritten, and why the test for this is exhaustive.
pub fn plan_reshard(old_lengths: &[i64], new_lengths: &[i64]) -> Result<Vec<Vec<Piece>>> {
    let old_total: i64 = old_lengths.iter().sum();
    let new_total: i64 = new_lengths.iter().sum();
    if old_total != new_total {
        return Err(ReshardError::Invalid(format!(
            "the saved shards add up to {old_total} along the sharded dimension and the live \
             ones to {new_total}: this checkpoint is not of this tensor"
        )));
    }

    let old_at = offsets_from_lengths(old_lengths);
    let new_at = offsets_from_lengths(new_lengths);

    let mut plan = Vec::with_capacity(new_lengths.len());
    for r in 0..new_lengths.len() {
        let (lo, hi) = (new_at[r], new_at[r + 1]);
        let mut pieces = Vec::new();
        for q in 0..old_lengths.len() {
            let (a, b) = (old_at[q], old_at[q + 1]);
            if b <= lo || a >= hi {
                continue;
            }
            let (start, stop) = (lo.max(a), hi.min(b));
            pieces.push(Piece {
                source: q,
                take: (start - a, stop - a),
                put: (start - lo, stop - lo),
            });
        }
        plan.push(pieces);
    }
    Ok(plan)
}

/// Which old ranks a plan reads from at all, in the order it first wants them.
///
/// The reachability question is asked against this rather than against every
/// old rank: on a shrink each new rank needs at most `ceil(N / M) + 1` old
/// shards, and demanding every store be readable when only some are wanted
/// would refuse resumes that are perfectly possible.
pub fn sources_needed<'a>(plan: impl IntoIterator<Item = &'a [Piece]>) -> Vec<usize> {
    let mut seen = Vec::new();
    for pieces in plan {
        for piece in pieces {
            if !seen.contains(&piece.source) {
                seen.push(piece.source);
            }
        }
    }
    seen
}

/// Assert that the pieces tile `[0, length)` exactly. Cheap, and load-bearing.
///
/// [`plan_reshard`] cannot produce a gap — it walks every old interval that
/// overlaps — so this never fires on its output. It exists because the thing it
/// guards against is unobservable downstream: a shard assembled with a hole in
/// it is the right shape, the right dtype, and wrong, and the run would train
/// on it for hours before anything looked odd.
pub fn check_covered(pieces: &[Piece], length: i64, location: &str) -> Result<()> {
    let mut covered = 0i64;
    for piece in pieces {
        let (start, stop) = piece.put;
        if start != covered {
            return Err(ReshardError::Invalid(format!(
                "{location}: the pieces of this shard do not join up — expected the next one \
                 to start at {covered}, it starts at {start}"
            )));
        }
        covered = stop;
    }
    if covered != length {
        return Err(ReshardError::Invalid(format!(
            "{location}: the pieces cover {covered} of {length} rows"
        )));
    }
    Ok(())
}

// ── where the shards are, not just what they are ────────────────────────────
//
// Everything above this line is about a tensor's shape and nothing else, which
// is what let it be tested exhaustively. The functions below add exactly one
// fact — *which machine can read which old store* — and stay pure for the same
// reason: the question "is moving these bytes affordable" has an arithmetic
// answer, and it should be available before any transport exists to answer it
// empirically.
//
// The map is deliberately "where a readable copy is" rather than "who wrote
// it". Those differ precisely in the case GPU-96 exists for: when the machine
// that wrote a store is gone, the shard survives as the copy a neighbour holds
// (GPU-76), and the neighbour is where it has to be fetched from. Feeding that
// neighbour's identity in as the home makes replica promotion the same problem
// as an ordinary cross-machine fetch, with no second code path to keep honest.
//
// A machine is a `usize` here and an opaque object at the Python boundary — a
// hostname, an owner record, a node rank. `crate::python` numbers the distinct
// objects it is handed using a Python dict, so they are compared with exactly
// the semantics the Python module used, and this half never has to know what
// they were.

/// Which machine holds a readable copy of each old rank's store, and which
/// machine each new rank runs on. A missing rank is an error, never a default.
pub type Homes = HashMap<usize, usize>;

fn home_of(homes: &Homes, rank: usize, missing: impl FnOnce() -> String) -> Result<usize> {
    homes
        .get(&rank)
        .copied()
        .ok_or_else(|| ReshardError::MissingHome(missing()))
}

/// The subset of `plan` a transport would have to move, per new rank.
///
/// Same indexing as [`plan_reshard`]'s output, so the two can be zipped: entry
/// *r* is the pieces new rank *r* cannot read from where it is standing. A rank
/// with nothing to fetch gets an empty list rather than being dropped, because
/// "this rank needs no help" and "this rank was not considered" are different
/// answers and a caller iterating the result should not have to tell them apart
/// by absence.
///
/// An old rank with no entry in `old_home` is an error rather than defaulting
/// to remote or to local. Both defaults are wrong in a way that hides: assuming
/// remote invents traffic that may not exist, and assuming local invents a
/// store that is not there, which is the same family of silent-wrong-model
/// failure the whole module is written against.
pub fn crossing_pieces(
    plan: &[Vec<Piece>],
    old_home: &Homes,
    new_home: &Homes,
) -> Result<Vec<Vec<Piece>>> {
    let mut crossing = Vec::with_capacity(plan.len());
    for (r, pieces) in plan.iter().enumerate() {
        let here = home_of(new_home, r, || {
            format!("no machine is recorded for new rank {r}")
        })?;
        let mut mine = Vec::new();
        for piece in pieces {
            let q = piece.source;
            let there = home_of(old_home, q, || {
                format!("no machine is recorded as holding old rank {q}'s store")
            })?;
            if there != here {
                mine.push(*piece);
            }
        }
        crossing.push(mine);
    }
    Ok(crossing)
}

/// A `(from machine, to machine) -> amount` tally that keeps insertion order.
///
/// Order is not part of the answer — the Python side compares these as dicts —
/// but a stable one makes two runs of the cost tool diffable, and Python's own
/// dicts have preserved insertion order since 3.7, so keeping it costs a `Vec`
/// and loses nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LinkTally {
    order: Vec<(usize, usize)>,
    total: HashMap<(usize, usize), i64>,
}

impl LinkTally {
    pub fn add(&mut self, link: (usize, usize), amount: i64) {
        match self.total.get_mut(&link) {
            Some(running) => *running += amount,
            None => {
                self.order.push(link);
                self.total.insert(link, amount);
            }
        }
    }

    /// The links in the order they were first seen, with their totals.
    pub fn entries(&self) -> impl Iterator<Item = ((usize, usize), i64)> + '_ {
        self.order.iter().map(|link| (*link, self.total[link]))
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}

/// Rows that cross, keyed by `(from machine, to machine)`. One tensor.
///
/// Rows rather than bytes because this half is shape arithmetic and a row's
/// width belongs to the caller — see [`crossing_bytes`], which is this function
/// with the multiplication done.
///
/// Directional on purpose. A shrink is not symmetric: the machines that keep
/// running pull, the ones being emptied push, and a single total would hide a
/// plan that asks one link to carry everything while another carries nothing.
/// That imbalance is the measured finding of GPU-94's pre-staging — one sender
/// serving joiners in sequence — and it is the shape of cost most likely to
/// decide this question, so it is not summed away here.
pub fn crossing_rows(plan: &[Vec<Piece>], old_home: &Homes, new_home: &Homes) -> Result<LinkTally> {
    let mut volume = LinkTally::default();
    for (r, pieces) in crossing_pieces(plan, old_home, new_home)?
        .iter()
        .enumerate()
    {
        // Both lookups were validated by `crossing_pieces`, which refuses a
        // rank it has no home for rather than reaching this line.
        let there = new_home[&r];
        for piece in pieces {
            let (start, stop) = piece.take;
            volume.add((old_home[&piece.source], there), stop - start);
        }
    }
    Ok(volume)
}

/// [`crossing_rows`] summed over many tensors, in bytes.
///
/// `tensors` is `(plan, row_bytes)` — one plan per tensor, paired with what one
/// row along *that* tensor's sharded dimension costs. The pairing is per tensor
/// because it varies per tensor: a checkpoint's rows are a hidden dimension
/// wide for one weight and a vocabulary wide for another, and a single average
/// over the model is the kind of number that looks like a measurement and is
/// not one.
///
/// This is the numerator of the only question worth asking before writing the
/// transport: at the rate the link between two machines actually carries bytes,
/// how long does a given reshard take? A whole-store rate is easy to measure
/// and easy to misread — most of a store may never need to move at all. What
/// has to move is this.
pub fn crossing_bytes(
    tensors: &[(Vec<Vec<Piece>>, i64)],
    old_home: &Homes,
    new_home: &Homes,
) -> Result<LinkTally> {
    let mut total = LinkTally::default();
    for (plan, row_bytes) in tensors {
        for (link, rows) in crossing_rows(plan, old_home, new_home)?.entries() {
            total.add(link, rows * row_bytes);
        }
    }
    Ok(total)
}

/// Ranks dealt to machines in contiguous blocks: the usual `torchrun` shape.
///
/// `torchrun` numbers ranks by node — node 0 takes the first
/// `LOCAL_WORLD_SIZE`, node 1 the next — so the old stores on one machine cover
/// one contiguous interval of the global tensor, and so do the new shards of
/// the ranks that run there.
///
/// That is worth stating as its own function because of what it implies, which
/// is not obvious and is easy to get backwards: **when both topologies are
/// dealt this way and the machine boundaries land on the same rows, nothing
/// crosses at all.** A shrink from eight ranks to four across two machines
/// moves zero bytes if each machine keeps its own half. Traffic appears when
/// the boundaries fail to line up — an odd split, machines with different GPU
/// counts — or when a machine is gone and its half has to be read from the
/// copies elsewhere. Sizing the transport off "a reshard moves the whole
/// checkpoint" would be sizing it off a case that mostly does not happen.
///
/// A remainder is spread over the first machines, one extra rank each, which is
/// the same rule `torchrun` follows and is only reached on a job whose machines
/// are not identical.
pub fn contiguous_homes(world: i64, machines: i64) -> Result<Homes> {
    if machines <= 0 {
        return Err(ReshardError::Invalid(format!(
            "a job runs on at least one machine, not {machines}"
        )));
    }
    if world < machines {
        return Err(ReshardError::Invalid(format!(
            "{world} ranks cannot be dealt over {machines} machines: some machine would hold \
             no rank, and a machine with no rank is not part of this job"
        )));
    }

    let (base, extra) = (world / machines, world % machines);
    let mut homes = Homes::with_capacity(world as usize);
    let mut rank = 0usize;
    for machine in 0..machines {
        let count = base + i64::from(machine < extra);
        for _ in 0..count {
            homes.insert(rank, machine as usize);
            rank += 1;
        }
    }
    Ok(homes)
}

/// The largest number of old shards any single new rank reads from.
///
/// The memory ceiling of a reshard, expressed as a count: a rank holds the old
/// shards it is stitching from, so this is what bounds its peak footprint. For
/// a shrink from *N* ranks to *M* it should not exceed `ceil(N / M) + 1`, and
/// that bound is the reason the global tensor never materialises anywhere.
///
/// Reported rather than asserted. The bound is a property of even splits, and a
/// checkpoint whose shards are lopsided enough could exceed it honestly; a gate
/// here would refuse a resume that is merely unusual, while a number lets the
/// caller decide whether what it is looking at is unusual or wrong.
pub fn most_sources_held(plan: &[Vec<Piece>]) -> usize {
    plan.iter().map(Vec::len).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// How torch splits `total` rows over `parts` ranks.
    ///
    /// Reproduced here and *only* here, mirroring the helper of the same name
    /// in `tests/test_dist_reshard.py`: the planner never derives an offset, it
    /// is handed measurements. This exists to manufacture plausible
    /// measurements, including the uneven tails that are the interesting case.
    fn chunked(total: i64, parts: i64) -> Vec<i64> {
        let size = total.div_euclid(parts) + i64::from(total.rem_euclid(parts) != 0);
        let mut lengths = Vec::new();
        let mut left = total;
        for _ in 0..parts {
            let take = size.min(left.max(0));
            lengths.push(take);
            left -= take;
        }
        lengths
    }

    fn even(lengths: i64, count: i64) -> Vec<i64> {
        vec![lengths; count as usize]
    }

    #[test]
    fn offsets_are_fence_posts() {
        assert_eq!(offsets_from_lengths(&[3, 3, 2]), vec![0, 3, 6, 8]);
        assert_eq!(offsets_from_lengths(&[]), vec![0]);
        // A world larger than the tensor: torch gives the tail ranks nothing,
        // and the plan has to survive a zero-length shard rather than divide by
        // it.
        assert_eq!(offsets_from_lengths(&[1, 1, 0]), vec![0, 1, 2, 2]);
    }

    #[test]
    fn every_pair_reassembles_the_tensor() {
        // The tensor is modelled as `[0, 1, ... total-1]`, so the assertion is
        // not "the shapes match" — which a plan that shuffled rows would also
        // satisfy — but that every row is in its place.
        for total in [12i64, 13, 24, 31] {
            let rows: Vec<i64> = (0..total).collect();
            for old_world in 1..=8i64 {
                for new_world in 1..=8i64 {
                    let old_lengths = chunked(total, old_world);
                    let new_lengths = chunked(total, new_world);
                    let old_at = offsets_from_lengths(&old_lengths);
                    let new_at = offsets_from_lengths(&new_lengths);

                    let plan = plan_reshard(&old_lengths, &new_lengths).expect("totals agree");
                    assert_eq!(plan.len(), new_lengths.len());

                    for (r, pieces) in plan.iter().enumerate() {
                        check_covered(pieces, new_lengths[r], "exhaustive").expect("tiled");
                        let mut built = Vec::new();
                        for piece in pieces {
                            let base = old_at[piece.source];
                            let (start, stop) = piece.take;
                            built.extend_from_slice(
                                &rows[(base + start) as usize..(base + stop) as usize],
                            );
                        }
                        let expected = &rows[new_at[r] as usize..new_at[r + 1] as usize];
                        assert_eq!(built, expected, "{old_world}->{new_world} rank {r}");
                    }
                }
            }
        }
    }

    #[test]
    fn four_to_three_stitches_every_shard_from_two() {
        // 4 -> 3 over 12 rows is the smallest case where no new shard is any
        // old shard: each is built from two, and one old shard feeds two new
        // ones. Written out rather than swept into the loop above because it is
        // the case that distinguishes a real planner from a permutation.
        let plan = plan_reshard(&[3, 3, 3, 3], &[4, 4, 4]).unwrap();
        assert_eq!(plan.iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 2, 2]);
        // New rank 1 wants rows 4-7: the tail of old rank 1, the head of old 2.
        assert_eq!(
            plan[1],
            vec![
                Piece {
                    source: 1,
                    take: (1, 3),
                    put: (0, 2)
                },
                Piece {
                    source: 2,
                    take: (0, 2),
                    put: (2, 4)
                },
            ]
        );
    }

    #[test]
    fn totals_that_disagree_are_refused() {
        assert!(matches!(
            plan_reshard(&[3, 3], &[4, 4]),
            Err(ReshardError::Invalid(ref m)) if m.contains("not of this tensor")
        ));
    }

    #[test]
    fn coverage_catches_a_gap_and_a_short_shard() {
        let gapped = [
            Piece {
                source: 0,
                take: (0, 2),
                put: (0, 2),
            },
            Piece {
                source: 1,
                take: (0, 2),
                put: (3, 5),
            },
        ];
        assert!(matches!(
            check_covered(&gapped, 5, "t"),
            Err(ReshardError::Invalid(ref m)) if m.contains("do not join up")
        ));
        let short = [Piece {
            source: 0,
            take: (0, 2),
            put: (0, 2),
        }];
        assert!(matches!(
            check_covered(&short, 5, "t"),
            Err(ReshardError::Invalid(ref m)) if m.contains("cover 2 of 5")
        ));
    }

    #[test]
    fn legacy_placements_are_still_readable() {
        // What `str(placement)` gives on torch 2.12, which is what the
        // checkpoints written before this feature actually contain...
        assert_eq!(decode_legacy_placement("S(0)"), Placement::Shard { dim: 0 });
        assert_eq!(decode_legacy_placement("S(3)"), Placement::Shard { dim: 3 });
        assert_eq!(decode_legacy_placement("R"), Placement::Replicate);
        // ...and what `repr` gives, which is what a reader would guess.
        assert_eq!(
            decode_legacy_placement("Shard(dim=1)"),
            Placement::Shard { dim: 1 }
        );
        assert_eq!(decode_legacy_placement("Replicate()"), Placement::Replicate);
        assert_eq!(
            decode_legacy_placement("P(sum)"),
            Placement::Partial { op: "sum".into() }
        );
        assert_eq!(
            decode_legacy_placement("Partial(avg)"),
            Placement::Partial { op: "avg".into() }
        );
        assert_eq!(
            decode_legacy_placement("Whatever(4)"),
            Placement::Unknown {
                repr: "Whatever(4)".into()
            }
        );
    }

    #[test]
    fn shard_dim_names_the_dimension_or_refuses_by_name() {
        assert_eq!(
            shard_dim(&[Placement::Shard { dim: 1 }], "w").unwrap(),
            Some(1)
        );
        assert_eq!(shard_dim(&[Placement::Replicate], "w").unwrap(), None);

        let two_d = [Placement::Shard { dim: 0 }, Placement::Shard { dim: 1 }];
        assert!(matches!(
            shard_dim(&two_d, "w"),
            Err(ReshardError::Unsupported(ref m)) if m.contains("2 mesh dimensions")
        ));
        assert!(matches!(
            shard_dim(&[Placement::Partial { op: "sum".into() }], "w"),
            Err(ReshardError::Unsupported(ref m)) if m.contains("Partial")
        ));
        assert!(matches!(
            shard_dim(
                &[Placement::Unknown { repr: "Whatever(4)".into() }],
                "w"
            ),
            Err(ReshardError::Unsupported(ref m)) if m.contains("Whatever")
        ));
    }

    #[test]
    fn contiguous_homes_deal_even_blocks_and_spread_a_remainder() {
        let two = contiguous_homes(8, 2).unwrap();
        assert!((0..8).all(|r| two[&r] == r / 4));
        // Five ranks over two machines is three then two, which is what
        // torchrun does and is the only shape where the boundary can land
        // off-centre.
        let odd = contiguous_homes(5, 2).unwrap();
        assert_eq!(
            (0..5).map(|r| odd[&r]).collect::<Vec<_>>(),
            vec![0, 0, 0, 1, 1]
        );
        assert!(matches!(
            contiguous_homes(2, 4),
            Err(ReshardError::Invalid(ref m)) if m.contains("cannot be dealt")
        ));
        assert!(matches!(
            contiguous_homes(4, 0),
            Err(ReshardError::Invalid(ref m)) if m.contains("at least one machine")
        ));
    }

    #[test]
    fn nothing_crosses_when_the_machine_boundaries_line_up() {
        // The headline this suite exists to pin down: traffic is what
        // misalignment costs, not what resharding costs.
        for machines in [2i64, 3, 4] {
            for old_world in [2i64, 4, 6, 8, 12] {
                for new_world in [2i64, 4, 6, 8, 12] {
                    if old_world % machines != 0 || new_world % machines != 0 {
                        continue;
                    }
                    let plan =
                        plan_reshard(&even(new_world, old_world), &even(old_world, new_world))
                            .unwrap();
                    let crossing = crossing_rows(
                        &plan,
                        &contiguous_homes(old_world, machines).unwrap(),
                        &contiguous_homes(new_world, machines).unwrap(),
                    )
                    .unwrap();
                    assert!(
                        crossing.is_empty(),
                        "{old_world} -> {new_world} over {machines} machines moved {crossing:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_misaligned_boundary_is_exactly_what_crosses() {
        // Two old shards, three new ones, over two machines. New rank 1
        // straddles the old boundary: it wants rows [2, 4) and machine 0 only
        // holds [0, 3). The single row [3, 4) is the whole cost, and it moves
        // from machine 1 to machine 0 — worked out by hand rather than by
        // re-running the implementation, so the test can disagree with it.
        let plan = plan_reshard(&[3, 3], &[2, 2, 2]).unwrap();
        let old_home = Homes::from([(0, 0), (1, 1)]);
        let new_home = Homes::from([(0, 0), (1, 0), (2, 1)]);
        let crossing = crossing_rows(&plan, &old_home, &new_home).unwrap();
        assert_eq!(crossing.entries().collect::<Vec<_>>(), vec![((1, 0), 1)]);
    }

    #[test]
    fn a_dead_machine_costs_only_what_its_neighbour_cannot_serve_locally() {
        // Six ranks over three machines, machine 1 gone, its shards readable on
        // machine 2. Two of twelve rows cross: losing a third of the cluster
        // does not mean moving a third of the checkpoint.
        let plan = plan_reshard(&[2; 6], &[3; 4]).unwrap();
        let old_home = Homes::from([(0, 0), (1, 0), (2, 2), (3, 2), (4, 2), (5, 2)]);
        let new_home = Homes::from([(0, 0), (1, 0), (2, 2), (3, 2)]);
        assert_eq!(
            crossing_rows(&plan, &old_home, &new_home)
                .unwrap()
                .entries()
                .collect::<Vec<_>>(),
            vec![((2, 0), 2)]
        );
    }

    #[test]
    fn a_missing_home_is_an_error_rather_than_a_guess() {
        let plan = plan_reshard(&[2, 2], &[4]).unwrap();
        assert!(matches!(
            crossing_pieces(&plan, &Homes::from([(0, 0)]), &Homes::from([(0, 0)])),
            Err(ReshardError::MissingHome(ref m)) if m.contains("old rank 1")
        ));
        assert!(matches!(
            crossing_pieces(&plan, &Homes::from([(0, 0), (1, 0)]), &Homes::new()),
            Err(ReshardError::MissingHome(ref m)) if m.contains("new rank 0")
        ));
    }

    #[test]
    fn crossing_and_local_partition_the_plan_exactly() {
        // A piece counted twice is a byte moved for nothing; a piece counted in
        // neither is a band of uninitialised rows, which is the failure the
        // whole module is written against.
        for old_world in 1..=8i64 {
            for new_world in 1..=8i64 {
                let plan =
                    plan_reshard(&even(new_world, old_world), &even(old_world, new_world)).unwrap();
                let old_home = contiguous_homes(old_world, old_world.min(2)).unwrap();
                let new_home = contiguous_homes(new_world, new_world.min(2)).unwrap();

                let crossing = crossing_pieces(&plan, &old_home, &new_home).unwrap();
                assert_eq!(crossing.len(), plan.len());

                for (r, (all_pieces, remote)) in plan.iter().zip(&crossing).enumerate() {
                    let expected: Vec<Piece> = all_pieces
                        .iter()
                        .copied()
                        .filter(|p| old_home[&p.source] != new_home[&r])
                        .collect();
                    let local = all_pieces.len() - expected.len();
                    assert_eq!(*remote, expected);
                    assert_eq!(remote.len() + local, all_pieces.len());
                }

                let moved: i64 = crossing_rows(&plan, &old_home, &new_home)
                    .unwrap()
                    .entries()
                    .map(|(_, rows)| rows)
                    .sum();
                assert!((0..=old_world * new_world).contains(&moved));
            }
        }
    }

    #[test]
    fn crossing_bytes_uses_each_tensor_own_row_width() {
        let plan = plan_reshard(&[3, 3], &[2, 2, 2]).unwrap(); // one row, machine 1 -> 0
        let old_home = Homes::from([(0, 0), (1, 1)]);
        let new_home = Homes::from([(0, 0), (1, 0), (2, 1)]);

        let wide = 4096 * 2; // bf16, hidden 4096
        let narrow = 2; // bf16 scalar per row

        let one = crossing_bytes(&[(plan.clone(), wide)], &old_home, &new_home).unwrap();
        assert_eq!(one.entries().collect::<Vec<_>>(), vec![((1, 0), wide)]);

        let both =
            crossing_bytes(&[(plan.clone(), wide), (plan, narrow)], &old_home, &new_home).unwrap();
        assert_eq!(
            both.entries().collect::<Vec<_>>(),
            vec![((1, 0), wide + narrow)]
        );
    }

    #[test]
    fn a_rank_never_holds_more_than_the_documented_bound() {
        // `ceil(N / M) + 1` old shards, which is what keeps the global tensor
        // unmaterialised. Exhaustive because the bound is the reason the memory
        // story holds, and the only pairs where it is obviously safe are the
        // ones a hand-picked test would have chosen.
        for old_world in 1..=16i64 {
            for new_world in 1..=16i64 {
                let plan =
                    plan_reshard(&even(new_world, old_world), &even(old_world, new_world)).unwrap();
                let bound = old_world.div_euclid(new_world)
                    + i64::from(old_world.rem_euclid(new_world) != 0)
                    + 1;
                assert!(most_sources_held(&plan) as i64 <= bound);
            }
        }
        assert_eq!(most_sources_held(&[]), 0);
    }

    #[test]
    fn sources_needed_is_narrower_than_the_old_world() {
        let plan = plan_reshard(&[2; 8], &[4; 4]).unwrap();
        assert_eq!(sources_needed(vec![plan[0].as_slice()]), vec![0, 1]);
        let all: Vec<&[Piece]> = plan.iter().map(Vec::as_slice).collect();
        let mut every = sources_needed(all);
        every.sort_unstable();
        assert_eq!(every, (0..8).collect::<Vec<usize>>());
    }
}
