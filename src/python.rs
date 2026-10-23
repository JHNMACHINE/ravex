//! The interpreter boundary, and nothing else.
//!
//! Every function here is a conversion around a call into [`crate::reshard`].
//! No arithmetic lives in this file on purpose: what can be tested by `cargo
//! test` without an interpreter should be, and a calculation that has drifted
//! down here is a calculation that only the Python suite covers.
//!
//! The signatures are the ones `ravex._dist.reshard` has always had, down to
//! the argument names and the wording of the messages. That is not politeness
//! towards the old module — it is what lets the existing test suite, which was
//! written against the Python implementation and never touched for this port,
//! stand as the oracle for the Rust one.
//!
//! Three exception types come out of here and the distinction is load-bearing.
//! `ReshardUnsupported` means "I refuse, and here is why", and `_resume.py`
//! catches it to fall back to starting from scratch. `ValueError` means
//! something went wrong, which is a bug. `KeyError` means the caller has not
//! recorded where a store lives.

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList, PySet, PyTuple};

use crate::reshard::{self, Homes, Piece, Placement, ReshardError};

create_exception!(
    reshard,
    ReshardUnsupported,
    PyException,
    "A checkpoint this planner will not attempt to reshape.\n\n\
     Separate from ``ValueError`` so the resume path can tell \"I refuse, and \
     here is why\" apart from \"something went wrong\". The first is a reason \
     to fall back to starting from scratch with a message a user can act on; \
     the second is a bug."
);

fn to_py_err(error: ReshardError) -> PyErr {
    match error {
        ReshardError::Unsupported(message) => ReshardUnsupported::new_err(message),
        ReshardError::Invalid(message) => PyValueError::new_err(message),
        ReshardError::MissingHome(message) => PyKeyError::new_err(message),
    }
}

// ── pieces and plans ────────────────────────────────────────────────────────

/// One piece, as the tuple every caller of this module already unpacks.
fn piece_to_py<'py>(py: Python<'py>, piece: &Piece) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        [
            piece.source.into_pyobject(py)?.into_any(),
            PyTuple::new(py, [piece.take.0, piece.take.1])?.into_any(),
            PyTuple::new(py, [piece.put.0, piece.put.1])?.into_any(),
        ],
    )
}

fn plan_to_py<'py>(py: Python<'py>, plan: &[Vec<Piece>]) -> PyResult<Bound<'py, PyList>> {
    let rows: Vec<Bound<'py, PyList>> = plan
        .iter()
        .map(|pieces| {
            let converted: Vec<Bound<'py, PyTuple>> = pieces
                .iter()
                .map(|piece| piece_to_py(py, piece))
                .collect::<PyResult<_>>()?;
            PyList::new(py, converted)
        })
        .collect::<PyResult<_>>()?;
    PyList::new(py, rows)
}

/// A plan handed back in — from `plan_reshard`, or hand-written by a test.
///
/// Anything iterable of iterables of three-element tuples is accepted, which is
/// what the Python implementation accepted by virtue of only ever indexing
/// them.
fn plan_from_py(plan: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Piece>>> {
    let mut out = Vec::new();
    for pieces in plan.try_iter()? {
        let mut row = Vec::new();
        for piece in pieces?.try_iter()? {
            let (source, take, put): (usize, (i64, i64), (i64, i64)) = piece?.extract()?;
            row.push(Piece { source, take, put });
        }
        out.push(row);
    }
    Ok(out)
}

// ── machines, which are whatever the caller says they are ───────────────────

/// Numbers the distinct machine identities in a pair of home maps.
///
/// The identities are opaque — a hostname, an owner record, a node rank — and
/// the Python module only ever compared them for equality and used them as dict
/// keys. Both of those are done here by a real `dict`, so the comparison is
/// Python's own: `__eq__` and `__hash__` of whatever was passed in, with no
/// second notion of equality invented on the Rust side. The planner downstream
/// then works with the small integers this hands out and never has to know.
struct Machines<'py> {
    numbers: Bound<'py, PyDict>,
    objects: Vec<Py<PyAny>>,
}

impl<'py> Machines<'py> {
    fn new(py: Python<'py>) -> PyResult<Self> {
        Ok(Self {
            numbers: PyDict::new(py),
            objects: Vec::new(),
        })
    }

    fn number(&mut self, machine: &Bound<'py, PyAny>) -> PyResult<usize> {
        if let Some(known) = self.numbers.get_item(machine)? {
            return known.extract();
        }
        let assigned = self.objects.len();
        self.numbers.set_item(machine, assigned)?;
        self.objects.push(machine.clone().unbind());
        Ok(assigned)
    }

    fn object(&self, number: usize) -> &Py<PyAny> {
        &self.objects[number]
    }

    /// One `{rank: machine}` mapping, numbered.
    ///
    /// Entries whose key is not a rank — not an integer, or negative — are
    /// skipped rather than refused. That is not leniency for its own sake: the
    /// Python implementation indexed these maps by rank and never enumerated
    /// them, so an entry no rank can name was already invisible, and raising on
    /// one here would refuse a call that used to work.
    fn read(&mut self, homes: &Bound<'py, PyAny>) -> PyResult<Homes> {
        let mut read = Homes::new();
        let items = homes.call_method0("items")?;
        for item in items.try_iter()? {
            let (rank, machine): (Bound<'py, PyAny>, Bound<'py, PyAny>) = item?.extract()?;
            let Ok(rank) = rank.extract::<usize>() else {
                continue;
            };
            let machine = self.number(&machine)?;
            read.insert(rank, machine);
        }
        Ok(read)
    }

    /// A `{(from, to): amount}` tally, with the identities put back.
    fn tally_to_py(&self, py: Python<'py>, tally: &reshard::LinkTally) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for ((from, to), amount) in tally.entries() {
            let link = PyTuple::new(py, [self.object(from), self.object(to)])?;
            out.set_item(link, amount)?;
        }
        Ok(out)
    }
}

// ── placements ──────────────────────────────────────────────────────────────

fn placement_to_py<'py>(py: Python<'py>, placement: &Placement) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    match placement {
        Placement::Shard { dim } => {
            out.set_item("kind", "shard")?;
            out.set_item("dim", dim)?;
        }
        Placement::Replicate => out.set_item("kind", "replicate")?,
        Placement::Partial { op } => {
            out.set_item("kind", "partial")?;
            out.set_item("op", op)?;
        }
        Placement::Unknown { repr } => {
            out.set_item("kind", "unknown")?;
            out.set_item("repr", repr)?;
        }
    }
    Ok(out)
}

/// ``placement.is_replicate()`` and friends, tolerating their absence.
fn asks_yes(placement: &Bound<'_, PyAny>, question: &str) -> bool {
    let Ok(method) = placement.getattr(question) else {
        return false;
    };
    // A placement type with a hostile API is not worth a traceback: the caller
    // gets `unknown` and refuses by name, which is the better message anyway.
    method
        .call0()
        .and_then(|answer| answer.is_truthy())
        .unwrap_or(false)
}

/// One DTensor placement as plain data.
///
/// What used to be written here was ``str(placement)`` — ``"Shard(dim=0)"`` —
/// which reads well in a dump and badly everywhere else. The only way back from
/// it is a parser, and a parser for torch's ``repr`` is a dependency on a string
/// nobody promised to keep stable. Resharding has to *ask* which dimension a
/// tensor was split along, so the answer is written down as an answer.
///
/// Duck-typed on purpose: this takes a torch object and knows nothing about
/// torch, so the encoding lives beside the planner that consumes it rather than
/// inside the module that happens to hold a DTensor.
///
/// ``Partial`` is named rather than folded in with ``Replicate``. A partial
/// value is a term waiting to be summed, not a copy of the whole — treating one
/// as the other would produce a tensor that is quietly a fraction of what it
/// should be, which is exactly the class of error this module exists to avoid.
#[pyfunction]
fn encode_placement<'py>(placement: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let py = placement.py();

    // `getattr(placement, "dim", None)`: an absent attribute is the question
    // answered "no", which is what every non-shard placement says.
    if let Ok(dim) = placement.getattr("dim") {
        if !dim.is_none() {
            let dim: i64 = py.get_type::<PyInt>().call1((dim,))?.extract()?;
            return placement_to_py(py, &Placement::Shard { dim });
        }
    }

    if asks_yes(placement, "is_replicate") {
        return placement_to_py(py, &Placement::Replicate);
    }

    if asks_yes(placement, "is_partial") {
        let op = match placement.getattr("reduce_op") {
            Ok(op) if !op.is_none() => op.str()?.extract()?,
            _ => "sum".to_string(),
        };
        return placement_to_py(py, &Placement::Partial { op });
    }

    // Neither a shard nor anything this torch will admit to. Carrying the repr
    // keeps the checkpoint self-describing: a reshard refuses it by name
    // instead of refusing it as "something".
    placement_to_py(
        py,
        &Placement::Unknown {
            repr: placement.str()?.extract()?,
        },
    )
}

/// The placement list of a saved shard, whichever way it was written.
///
/// Two shapes reach this. Checkpoints written from here on hold the dicts
/// [`encode_placement`] produces. Checkpoints written before that hold torch's
/// ``repr`` strings, and they are still resumable — a stored checkpoint is not
/// a thing you get to reformat after the fact, and the strings are parseable
/// well enough to recover the one field that matters.
///
/// Anything unrecognisable comes back as ``unknown`` rather than raising. The
/// caller that cares — [`shard_dim`] — refuses on it with a message that can
/// name the tensor; raising here would only be able to name the string.
#[pyfunction]
fn decode_placements<'py>(saved: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let py = saved.py();
    if !(saved.is_instance_of::<PyList>() || saved.is_instance_of::<PyTuple>()) {
        return Ok(PyList::empty(py));
    }

    let mut decoded: Vec<Bound<'py, PyDict>> = Vec::new();
    for entry in saved.try_iter()? {
        let entry = entry?;

        if let Ok(mapping) = entry.cast::<PyDict>() {
            let kind = mapping
                .get_item("kind")?
                .and_then(|k| k.extract::<String>().ok());

            match kind.as_deref() {
                Some("shard") => {
                    // `int(entry["dim"])`, and the three ways it can fail — no
                    // such key, a value that is not a number, a string that is
                    // not one — all mean the same thing: this entry says it is
                    // a shard and does not say of what.
                    let dim = mapping
                        .get_item("dim")?
                        .ok_or(())
                        .and_then(|value| {
                            py.get_type::<PyInt>()
                                .call1((value,))
                                .map_err(|_| ())
                        })
                        .and_then(|value| value.extract::<i64>().map_err(|_| ()));
                    decoded.push(match dim {
                        Ok(dim) => placement_to_py(py, &Placement::Shard { dim })?,
                        Err(()) => placement_to_py(
                            py,
                            &Placement::Unknown {
                                repr: mapping.repr()?.extract()?,
                            },
                        )?,
                    });
                }
                // Copied rather than re-encoded, so a key this version has not
                // heard of survives a decode instead of being quietly dropped.
                Some("replicate") | Some("partial") | Some("unknown") => {
                    decoded.push(mapping.copy()?)
                }
                _ => decoded.push(placement_to_py(
                    py,
                    &Placement::Unknown {
                        repr: mapping.repr()?.extract()?,
                    },
                )?),
            }
            continue;
        }

        let text: String = entry.str()?.extract()?;
        decoded.push(placement_to_py(py, &reshard::decode_legacy_placement(&text))?);
    }
    PyList::new(py, decoded)
}

/// The single dimension this tensor is split along, or None if it is not.
///
/// ``None`` means every rank holds the same bytes — a ``Replicate`` on a 1-D
/// mesh — and resharding one is copying rank 0's, which is why it is a normal
/// answer rather than an error.
///
/// Everything else in scope is a 1-D mesh with exactly one ``Shard``. A 2-D
/// mesh — FSDP crossed with tensor parallel — makes the plan a cartesian
/// problem rather than an interval one, and refusing it by name costs one check
/// and saves a category of wrong answers.
#[pyfunction]
#[pyo3(signature = (placements, r#where = "a tensor"))]
fn shard_dim(placements: &Bound<'_, PyAny>, r#where: &str) -> PyResult<Option<i64>> {
    let py = placements.py();

    // The dimension itself is deliberately *not* read here. A list with two
    // shards is refused for having two, and a shard entry that names no
    // dimension must raise the same `KeyError` it always did — but only if it
    // is the one shard whose dimension is actually wanted. So this pass reads
    // kinds, and the dimension is fetched afterwards from the entry the
    // planner settled on.
    let mut kinds: Vec<Placement> = Vec::new();
    let mut shards: Vec<Bound<'_, PyAny>> = Vec::new();
    for placement in placements.try_iter()? {
        let placement = placement?;
        let kind = placement
            .call_method1("get", ("kind",))?
            .extract::<String>()
            .unwrap_or_default();
        kinds.push(match kind.as_str() {
            "shard" => {
                shards.push(placement);
                Placement::Shard { dim: 0 }
            }
            "partial" => Placement::Partial {
                op: String::new(),
            },
            "unknown" => Placement::Unknown {
                repr: placement
                    .call_method1("get", ("repr", "?"))?
                    .str()?
                    .extract()?,
            },
            // Everything else — "replicate", and any kind this version has not
            // heard of — is neither sharded nor a refusal, so it contributes
            // nothing to the answer. That was true of the Python version by
            // omission; it is true here by construction.
            _ => Placement::Replicate,
        });
    }

    if reshard::shard_dim(&kinds, r#where)
        .map_err(to_py_err)?
        .is_none()
    {
        return Ok(None);
    }

    let dim = shards[0].get_item("dim")?;
    Ok(Some(
        py.get_type::<PyInt>()
            .call1((dim,))?
            .extract()?,
    ))
}

// ── the planner ─────────────────────────────────────────────────────────────

/// Running sum, with the leading zero: ``[3, 3, 2] -> [0, 3, 6, 8]``.
#[pyfunction]
fn offsets_from_lengths(lengths: Vec<i64>) -> Vec<i64> {
    reshard::offsets_from_lengths(&lengths)
}

/// How to build each new shard out of the old ones.
#[pyfunction]
fn plan_reshard<'py>(
    py: Python<'py>,
    old_lengths: Vec<i64>,
    new_lengths: Vec<i64>,
) -> PyResult<Bound<'py, PyList>> {
    let plan = reshard::plan_reshard(&old_lengths, &new_lengths).map_err(to_py_err)?;
    plan_to_py(py, &plan)
}

/// Which old ranks a plan reads from at all.
#[pyfunction]
fn sources_needed<'py>(py: Python<'py>, plan: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PySet>> {
    let plan = plan_from_py(plan)?;
    let borrowed: Vec<&[Piece]> = plan.iter().map(Vec::as_slice).collect();
    PySet::new(py, reshard::sources_needed(borrowed))
}

/// Assert that the pieces tile ``[0, length)`` exactly.
#[pyfunction]
fn check_covered(pieces: &Bound<'_, PyAny>, length: i64, r#where: &str) -> PyResult<()> {
    let mut row = Vec::new();
    for piece in pieces.try_iter()? {
        let (source, take, put): (usize, (i64, i64), (i64, i64)) = piece?.extract()?;
        row.push(Piece { source, take, put });
    }
    reshard::check_covered(&row, length, r#where).map_err(to_py_err)
}

/// The subset of ``plan`` a transport would have to move, per new rank.
#[pyfunction]
fn crossing_pieces<'py>(
    py: Python<'py>,
    plan: &Bound<'py, PyAny>,
    old_home: &Bound<'py, PyAny>,
    new_home: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let mut machines = Machines::new(py)?;
    let plan = plan_from_py(plan)?;
    let old_home = machines.read(old_home)?;
    let new_home = machines.read(new_home)?;
    let crossing = reshard::crossing_pieces(&plan, &old_home, &new_home).map_err(to_py_err)?;
    plan_to_py(py, &crossing)
}

/// Rows that cross, keyed by ``(from machine, to machine)``. One tensor.
#[pyfunction]
fn crossing_rows<'py>(
    py: Python<'py>,
    plan: &Bound<'py, PyAny>,
    old_home: &Bound<'py, PyAny>,
    new_home: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut machines = Machines::new(py)?;
    let plan = plan_from_py(plan)?;
    let old_home = machines.read(old_home)?;
    let new_home = machines.read(new_home)?;
    let tally = reshard::crossing_rows(&plan, &old_home, &new_home).map_err(to_py_err)?;
    machines.tally_to_py(py, &tally)
}

/// [`crossing_rows`] summed over many tensors, in bytes.
#[pyfunction]
fn crossing_bytes<'py>(
    py: Python<'py>,
    tensors: &Bound<'py, PyAny>,
    old_home: &Bound<'py, PyAny>,
    new_home: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut machines = Machines::new(py)?;
    let old_home = machines.read(old_home)?;
    let new_home = machines.read(new_home)?;

    let mut plans: Vec<(Vec<Vec<Piece>>, i64)> = Vec::new();
    for tensor in tensors.try_iter()? {
        let (plan, row_bytes): (Bound<'py, PyAny>, Bound<'py, PyAny>) = tensor?.extract()?;
        let row_bytes: i64 = py
            .get_type::<pyo3::types::PyInt>()
            .call1((row_bytes,))?
            .extract()?;
        plans.push((plan_from_py(&plan)?, row_bytes));
    }

    let tally = reshard::crossing_bytes(&plans, &old_home, &new_home).map_err(to_py_err)?;
    machines.tally_to_py(py, &tally)
}

/// Ranks dealt to machines in contiguous blocks: the usual ``torchrun`` shape.
#[pyfunction]
fn contiguous_homes(py: Python<'_>, world: i64, machines: i64) -> PyResult<Bound<'_, PyDict>> {
    let homes = reshard::contiguous_homes(world, machines).map_err(to_py_err)?;
    let out = PyDict::new(py);
    // Rank order, not hash order: this is read by a person as often as by a
    // caller, and `{0: 0, 1: 0, 2: 1}` is the shape the docstring describes.
    let mut ranks: Vec<&usize> = homes.keys().collect();
    ranks.sort_unstable();
    for rank in ranks {
        out.set_item(rank, homes[rank])?;
    }
    Ok(out)
}

/// The largest number of old shards any single new rank reads from.
#[pyfunction]
fn most_sources_held(plan: &Bound<'_, PyAny>) -> PyResult<usize> {
    Ok(reshard::most_sources_held(&plan_from_py(plan)?))
}

/// Everything `ravex._dist.reshard` re-exports.
pub fn register_reshard(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();

    let unsupported = py.get_type::<ReshardUnsupported>();
    // So a traceback names the module a reader can open. `create_exception!`
    // can only be given a bare identifier, and `ravex._dist.reshard` is where
    // this class is imported from and documented.
    unsupported.setattr("__module__", "ravex._dist.reshard")?;
    module.add("ReshardUnsupported", unsupported)?;

    module.add_function(wrap_pyfunction!(encode_placement, module)?)?;
    module.add_function(wrap_pyfunction!(decode_placements, module)?)?;
    module.add_function(wrap_pyfunction!(shard_dim, module)?)?;
    module.add_function(wrap_pyfunction!(offsets_from_lengths, module)?)?;
    module.add_function(wrap_pyfunction!(plan_reshard, module)?)?;
    module.add_function(wrap_pyfunction!(sources_needed, module)?)?;
    module.add_function(wrap_pyfunction!(check_covered, module)?)?;
    module.add_function(wrap_pyfunction!(crossing_pieces, module)?)?;
    module.add_function(wrap_pyfunction!(crossing_rows, module)?)?;
    module.add_function(wrap_pyfunction!(crossing_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(contiguous_homes, module)?)?;
    module.add_function(wrap_pyfunction!(most_sources_held, module)?)?;
    Ok(())
}

/// The compiled half of Ravex.
///
/// Underscored because nothing outside the package should import it:
/// `ravex._dist.reshard` is the name this has always had from the outside, and
/// it stays the name after the implementation moved.
///
/// Flat, rather than one submodule per area. A submodule of an extension has no
/// place to put a type stub — `ravex/_core.pyi` can describe a module and not a
/// package, and a `ravex/_core/` directory beside `_core.pyd` would shadow the
/// extension it is meant to describe. The package ships `py.typed`, so the
/// choice is between a flat module that can be typed and a tidy one that cannot.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_reshard(module)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
