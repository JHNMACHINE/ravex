//! The arithmetic core of Ravex.
//!
//! Ravex is a Python package that attaches itself to a running PyTorch program,
//! and most of it has to stay that way: the patches, the registry and the
//! bootstrap live on introspection of live Python objects, and moving them here
//! would only mean crossing this boundary on every `optimizer.step()`.
//!
//! What is here instead is the part of the engine that never touches torch —
//! integers, offsets and byte formats. [`reshard`] is the first of it: the
//! planner that lines a per-rank checkpoint up with a different number of
//! ranks.
//!
//! Two layers, and the split is the point. Every module beside this one is pure
//! Rust with no `pyo3` in sight, so it can be tested exhaustively by `cargo
//! test` and linked from another Rust program. [`python`] is the only file that
//! knows an interpreter exists; it converts, it names exceptions, and it holds
//! no arithmetic of its own.
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod reshard;
pub mod transport;

#[cfg(feature = "extension-module")]
mod python;
