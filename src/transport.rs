//! Moving a store between machines: the framing, and the socket under it.
//!
//! This is the second thing to leave Python, after the reshard planner, and it
//! left for a different reason. The planner moved because it was pure
//! arithmetic; this moved because of what it is *made of*. A replication round
//! reads files, frames them, pushes bytes and writes files — no torch, no
//! process group, no interpreter object anywhere in it — and the Python it
//! replaced paid for that in three places at once: a copy per chunk, a memmove
//! of the pending buffer at every file boundary, and a loop in which reading
//! the disk and writing the socket took turns instead of overlapping.
//!
//! **The wire format is not new and must not drift.** It is the one
//! `ravex/_dist/replication.py` has always written, byte for byte, because a
//! replica written by one version is read back by another and because
//! `exchange_stores` — which still moves its bytes over torch collectives —
//! shares this framing. Little-endian throughout:
//!
//! ```text
//! header:  u32 count
//!          per entry: u16 name_len, u64 size, name bytes, u8 already_there
//! bodies:  the bytes of every entry whose flag is 0, in header order
//! ```
//!
//! A separate, flagless form of the same entry list — [`encode_manifest`] — is
//! what a receiver sends *first* to say what it already holds, so the sender
//! can leave those files out. That one is never itself skippable.
//!
//! **What is deliberately not here.** The transport for `exchange_stores`
//! itself: that one is addressed by rank inside a process group, and giving it
//! a socket means answering where peers find each other, which is a design
//! question and not a port. See GPU-109. What this module does replace is the
//! pre-staging path, which already had a socket and an address and was only
//! ever a socket for want of a registered process group.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::sync_channel;

/// Bytes per chunk on the wire, matching `replication.CHUNK`.
///
/// Bounded on purpose: a real shard is over a gigabyte, and a whole-store
/// buffer would be that allocation on the training process at checkpoint time.
pub const CHUNK: usize = 4 * 1024 * 1024;

/// Written last, deleted first. A replica is only trustworthy while it exists.
pub const COMPLETE_MARKER: &str = ".ravex-replica-ok";

/// How many chunks the reader thread may run ahead of the socket.
///
/// The whole point of the thread is that the disk and the wire stop taking
/// turns, and one chunk of slack is enough for that. Three is the memory this
/// costs — `3 * chunk`, 12 MiB at the default — and it is bounded here rather
/// than left to grow because the process holding it is the one training.
const AHEAD: usize = 3;

#[derive(Debug)]
pub enum TransportError {
    Io(io::Error),
    /// The stream did not say what the format requires. Carries what was
    /// expected, because a desynchronised stream is otherwise unreadable from
    /// the outside.
    Protocol(String),
    /// The peer hung up with bytes still owed. Its own variant rather than a
    /// `Protocol` with a particular wording, because it is the one failure a
    /// caller acts on differently: a machine that went away mid-round is what
    /// replication exists to survive, and the Python this replaces raised
    /// `ConnectionError` for exactly this and nothing else.
    PeerClosed(String),
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportError::Io(error) => write!(f, "{}", error),
            TransportError::Protocol(what) => write!(f, "{}", what),
            TransportError::PeerClosed(what) => write!(f, "{}", what),
        }
    }
}

impl From<io::Error> for TransportError {
    fn from(error: io::Error) -> Self {
        TransportError::Io(error)
    }
}

pub type Result<T> = std::result::Result<T, TransportError>;

fn protocol<T>(what: impl Into<String>) -> Result<T> {
    Err(TransportError::Protocol(what.into()))
}

// ─── the file list ───────────────────────────────────────────────────────────

/// Every file under a store, relative to it, with its size. Sorted.
///
/// Sorted so both ends agree on order without exchanging it, and so a failure
/// part-way through leaves a prefix rather than a scatter. Names are joined
/// with `/` on every platform, because they travel.
///
/// A file whose size cannot be read is left out rather than reported, which is
/// what the Python did: the list describes what is readable here, and a
/// directory being walked while a peer prunes it is ordinary.
pub fn store_files(root: &Path) -> io::Result<Vec<(String, u64)>> {
    let mut found = Vec::new();
    let mut stack = vec![(root.to_path_buf(), String::new())];

    while let Some((directory, prefix)) = stack.pop() {
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            // Same tolerance as the size check below, for the same reason: a
            // store is a live directory and this is a description of it, not
            // an assertion about it.
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let name = match entry.file_name().into_string() {
                Ok(name) => name,
                // A name that is not UTF-8 cannot cross the wire, where the
                // format says the bytes are UTF-8. Skipping it here means the
                // receiver never hears of it; sending it would mean a stream
                // neither end could parse.
                Err(_) => continue,
            };
            let relative = if prefix.is_empty() {
                name
            } else {
                format!("{}/{}", prefix, name)
            };
            let path = entry.path();
            let is_dir = match entry.file_type() {
                Ok(kind) => kind.is_dir(),
                Err(_) => continue,
            };
            if is_dir {
                stack.push((path, relative));
            } else if let Ok(metadata) = fs::metadata(&path) {
                found.push((relative, metadata.len()));
            }
        }
    }

    found.sort();
    Ok(found)
}

// ─── the two encodings of an entry list ──────────────────────────────────────

const COUNT_SIZE: usize = 4;
const ENTRY_SIZE: usize = 10; // u16 name_len + u64 size
const FLAG_SIZE: usize = 1;

fn put_entry(blob: &mut Vec<u8>, name: &str, size: u64) {
    blob.extend_from_slice(&(name.len() as u16).to_le_bytes());
    blob.extend_from_slice(&size.to_le_bytes());
    blob.extend_from_slice(name.as_bytes());
}

/// `(name, size)` pairs, no flag: what one side already holds.
pub fn encode_manifest(entries: &[(String, u64)]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(COUNT_SIZE + entries.len() * (ENTRY_SIZE + 16));
    blob.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (name, size) in entries {
        put_entry(&mut blob, name, *size);
    }
    blob
}

/// The inverse of [`encode_manifest`].
pub fn parse_manifest(blob: &[u8]) -> Result<Vec<(String, u64)>> {
    if blob.len() < COUNT_SIZE {
        return protocol("manifest shorter than its own count");
    }
    let count = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
    let mut offset = COUNT_SIZE;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        if blob.len() < offset + ENTRY_SIZE {
            return protocol("manifest ended inside an entry");
        }
        let name_len =
            u16::from_le_bytes(blob[offset..offset + 2].try_into().unwrap()) as usize;
        let size = u64::from_le_bytes(blob[offset + 2..offset + 10].try_into().unwrap());
        offset += ENTRY_SIZE;
        if blob.len() < offset + name_len {
            return protocol("manifest ended inside a name");
        }
        let name = match std::str::from_utf8(&blob[offset..offset + name_len]) {
            Ok(name) => name.to_string(),
            Err(_) => return protocol("manifest carried a name that is not UTF-8"),
        };
        offset += name_len;
        entries.push((name, size));
    }
    Ok(entries)
}

/// The transfer header: the same entries, each with a flag saying whether its
/// bytes follow.
pub fn encode_header(entries: &[(String, u64)], skip: &HashSet<String>) -> Vec<u8> {
    let mut blob = Vec::with_capacity(COUNT_SIZE + entries.len() * (ENTRY_SIZE + 17));
    blob.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (name, size) in entries {
        put_entry(&mut blob, name, *size);
        blob.push(u8::from(skip.contains(name)));
    }
    blob
}

/// Bytes a transfer of `root` will produce, without producing them.
///
/// Must be called with the same `skip` the transfer gets, or the two ends
/// disagree about the stream's length.
pub fn encoded_size(root: &Path, skip: &HashSet<String>) -> io::Result<u64> {
    let entries = store_files(root)?;
    let mut total = COUNT_SIZE as u64;
    for (name, size) in &entries {
        total += (ENTRY_SIZE + FLAG_SIZE + name.len()) as u64;
        if !skip.contains(name) {
            total += size;
        }
    }
    Ok(total)
}

fn on_disk(root: &Path, relative: &str) -> PathBuf {
    let mut path = root.to_path_buf();
    for piece in relative.split('/') {
        path.push(piece);
    }
    path
}

// ─── the sending half ────────────────────────────────────────────────────────

/// Read `root` off disk and frame it, one block at a time.
///
/// The blocks are ragged on purpose — a header, then whatever came off each
/// file — because that is what the Python generator yielded and what
/// `exchange_stores` re-cuts with `fixed_chunks`. Matching it keeps one wire
/// format and one set of tests rather than two of each.
pub struct StoreEncoder {
    root: PathBuf,
    entries: Vec<(String, u64)>,
    skip: HashSet<String>,
    index: usize,
    handle: Option<File>,
    left: u64,
    chunk: usize,
    header: Option<Vec<u8>>,
}

impl StoreEncoder {
    pub fn new(root: &Path, chunk: usize, skip: HashSet<String>) -> io::Result<Self> {
        let entries = store_files(root)?;
        let header = encode_header(&entries, &skip);
        Ok(StoreEncoder {
            root: root.to_path_buf(),
            entries,
            skip,
            index: 0,
            handle: None,
            left: 0,
            chunk: chunk.max(1),
            header: Some(header),
        })
    }

    /// The next block, or `None` when the store is spent.
    pub fn next_block(&mut self) -> io::Result<Option<Vec<u8>>> {
        if let Some(header) = self.header.take() {
            return Ok(Some(header));
        }

        loop {
            if self.handle.is_none() {
                let (name, size) = match self.entries.get(self.index) {
                    Some(entry) => entry.clone(),
                    None => return Ok(None),
                };
                self.index += 1;
                if self.skip.contains(&name) {
                    continue;
                }
                self.handle = Some(File::open(on_disk(&self.root, &name))?);
                self.left = size;
                if size == 0 {
                    self.handle = None;
                    continue;
                }
            }

            let want = self.left.min(self.chunk as u64) as usize;
            let mut block = vec![0u8; want];
            let handle = self.handle.as_mut().expect("open by construction");
            let read = read_some(handle, &mut block)?;
            if read == 0 {
                // Truncated under us. The zeros already in `block` are the
                // padding the Python did explicitly: the framing stays aligned
                // and the file is condemned later, which is a better failure
                // than a desynchronised stream.
                self.left -= want as u64;
            } else {
                block.truncate(read);
                self.left -= read as u64;
            }
            if self.left == 0 {
                self.handle = None;
            }
            return Ok(Some(block));
        }
    }
}

/// One `read`, retried past `Interrupted` only. A short read is not an error
/// here: the caller sizes the next block from what came back.
fn read_some(handle: &mut File, buffer: &mut [u8]) -> io::Result<usize> {
    loop {
        match handle.read(buffer) {
            Ok(count) => return Ok(count),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
}

/// Push a store into `out`, reading the disk on another thread.
///
/// The overlap is the reason this function exists rather than a loop at the
/// call site. Measured on 2026-09-05 before any of this was written: the wire
/// ran at 4.5 GB/s, reading and framing the store at 1.2 GB/s, and the two of
/// them alternating at 0.99 GB/s — so the transfer was paying for the disk and
/// the socket in series when it could have paid for the slower of the two.
/// One bounded channel is the whole fix; `AHEAD` says what it costs in memory.
/// Returns the bytes written, which is what a caller reports and a test
/// asserts on: the point of the skip list is that a second round is smaller
/// than the first, and this is the number that says so.
pub fn send_store<W: Write>(
    out: &mut W,
    root: &Path,
    chunk: usize,
    skip: HashSet<String>,
) -> Result<u64> {
    let mut encoder = StoreEncoder::new(root, chunk, skip)?;
    let (blocks, arriving) = sync_channel::<io::Result<Vec<u8>>>(AHEAD);

    let reader = std::thread::spawn(move || loop {
        match encoder.next_block() {
            Ok(None) => return,
            Ok(Some(block)) => {
                if blocks.send(Ok(block)).is_err() {
                    return; // the socket gave up; stop reading for nobody
                }
            }
            Err(error) => {
                let _ = blocks.send(Err(error));
                return;
            }
        }
    });

    let mut written = 0u64;
    let mut failure = None;
    for block in arriving.iter() {
        match block {
            Ok(block) => {
                if let Err(error) = out.write_all(&block) {
                    failure = Some(TransportError::Io(error));
                    break;
                }
                written += block.len() as u64;
            }
            Err(error) => {
                failure = Some(TransportError::Io(error));
                break;
            }
        }
    }

    // Dropping the receiver is what unblocks a reader thread that is still
    // holding a chunk, so this join cannot hang on a failed send.
    drop(arriving);
    let _ = reader.join();

    match failure {
        Some(error) => Err(error),
        None => {
            out.flush()?;
            Ok(written)
        }
    }
}

// ─── the receiving half ──────────────────────────────────────────────────────

#[derive(Clone)]
struct Entry {
    name: String,
    size: u64,
    already_there: bool,
}

/// Turns the stream back into files, writing as the bytes arrive.
///
/// Streaming rather than buffering the whole store: the buffer would be a
/// second copy of the store on the receiving disk, which then also has to hold
/// the store itself.
///
/// Written in place rather than staged and swapped, because disk is the binding
/// constraint on these machines. Safety comes instead from [`COMPLETE_MARKER`],
/// removed before the first byte lands and written only once the last one has.
///
/// **The pending buffer has a cursor.** The Python it replaces did
/// `del self._pending[:take]`, which memmoves the tail, and at a 64 MiB chunk
/// that memmove was most of what the receiving side did — the measured collapse
/// from 529 MB/s to 113 MB/s as the chunk grew (GPU-84). Here the head is a
/// cursor and the buffer is compacted only when the dead prefix is worth
/// reclaiming.
pub struct StoreWriter {
    path: PathBuf,
    pending: Vec<u8>,
    at: usize,
    entries: Option<Vec<Entry>>,
    index: usize,
    left: u64,
    handle: Option<BufWriter<File>>,
    /// Set when a file flagged "already here" turns out not to be, on disk,
    /// what the sender was told it was. `complete` must not go true over that:
    /// a wrong local file is exactly as unusable as a missing one.
    broken: bool,
}

impl StoreWriter {
    pub fn new(path: &Path) -> Self {
        // Removed first: from here until `commit`, this directory is not a
        // replica of anything and must not be read as one.
        let _ = fs::remove_file(path.join(COMPLETE_MARKER));
        StoreWriter {
            path: path.to_path_buf(),
            pending: Vec::new(),
            at: 0,
            entries: None,
            index: 0,
            left: 0,
            handle: None,
            broken: false,
        }
    }

    fn buffered(&self) -> usize {
        self.pending.len() - self.at
    }

    fn take_front(&mut self, count: usize) {
        self.at += count;
        if self.at == self.pending.len() {
            self.pending.clear();
            self.at = 0;
        } else if self.at > (1 << 20) && self.at * 2 > self.pending.len() {
            self.pending.drain(..self.at);
            self.at = 0;
        }
    }

    /// Take one chunk off the wire.
    ///
    /// The fast path writes straight from the caller's memory, and it is the
    /// normal case because the files in a store are large and the chunks are
    /// not. Headers and file boundaries fall through to the buffered path,
    /// which is where the ragged cases have always been handled.
    pub fn feed(&mut self, block: &[u8]) -> Result<()> {
        let mut view = block;

        while !view.is_empty()
            && self.buffered() == 0
            && self.entries.is_some()
            && self.handle.is_some()
        {
            let take = (self.left as usize).min(view.len());
            let handle = self.handle.as_mut().expect("checked just above");
            handle.write_all(&view[..take])?;
            self.left -= take as u64;
            view = &view[take..];
            if self.left == 0 {
                self.finish_file()?;
            }
        }

        // Falls through even with nothing left over, and that is not a
        // formality: a zero-length file is created by *reaching* it, not by
        // writing to it, so a chunk that ends exactly on a file boundary must
        // still hand control back to `write_body` or a trailing empty file is
        // never made.
        if !view.is_empty() {
            self.pending.extend_from_slice(view);
        }

        loop {
            if self.entries.is_none() && !self.read_header()? {
                return Ok(());
            }
            if !self.write_body()? {
                return Ok(());
            }
        }
    }

    fn read_header(&mut self) -> Result<bool> {
        let blob = &self.pending[self.at..];
        if blob.len() < COUNT_SIZE {
            return Ok(false);
        }
        let count = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
        let mut offset = COUNT_SIZE;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            if blob.len() < offset + ENTRY_SIZE + FLAG_SIZE {
                return Ok(false);
            }
            let name_len =
                u16::from_le_bytes(blob[offset..offset + 2].try_into().unwrap()) as usize;
            let size = u64::from_le_bytes(blob[offset + 2..offset + 10].try_into().unwrap());
            offset += ENTRY_SIZE;
            if blob.len() < offset + name_len + FLAG_SIZE {
                return Ok(false);
            }
            let name = match std::str::from_utf8(&blob[offset..offset + name_len]) {
                Ok(name) => name.to_string(),
                Err(_) => return protocol("a file name on the wire is not UTF-8"),
            };
            offset += name_len;
            let already_there = blob[offset] != 0;
            offset += FLAG_SIZE;
            entries.push(Entry {
                name,
                size,
                already_there,
            });
        }
        self.take_front(offset);
        self.entries = Some(entries);
        Ok(true)
    }

    /// A file the sender did not send because we said we already had it.
    ///
    /// Said moments earlier, by us — but checked again rather than trusted,
    /// because a wrong file marked complete is the one failure this whole
    /// module exists to rule out.
    fn verify_skip(&mut self, name: &str, size: u64) {
        let ok = fs::metadata(on_disk(&self.path, name))
            .map(|found| found.len() == size)
            .unwrap_or(false);
        if !ok {
            self.broken = true;
        }
    }

    fn write_body(&mut self) -> Result<bool> {
        loop {
            if self.handle.is_none() {
                let entry = {
                    let entries = self.entries.as_ref().expect("header parsed first");
                    match entries.get(self.index) {
                        Some(entry) => entry.clone(),
                        None => return Ok(false),
                    }
                };
                if entry.already_there {
                    self.verify_skip(&entry.name, entry.size);
                    self.index += 1;
                    continue;
                }
                let target = on_disk(&self.path, &entry.name);
                if let Some(parent) = target.parent() {
                    fs::create_dir_all(parent)?;
                }
                self.handle = Some(BufWriter::new(File::create(target)?));
                self.left = entry.size;
                if entry.size == 0 {
                    self.finish_file()?;
                    continue;
                }
            }

            if self.buffered() == 0 {
                return Ok(false);
            }
            let take = (self.left as usize).min(self.buffered());
            let from = self.at;
            // Split the borrow: the handle and the buffer are different fields
            // and the compiler cannot see that through two method calls.
            let (pending, handle) = (&self.pending, self.handle.as_mut().unwrap());
            handle.write_all(&pending[from..from + take])?;
            self.take_front(take);
            self.left -= take as u64;
            if self.left == 0 {
                self.finish_file()?;
            }
        }
    }

    fn finish_file(&mut self) -> Result<()> {
        if let Some(mut handle) = self.handle.take() {
            handle.flush()?;
        }
        self.index += 1;
        Ok(())
    }

    /// The header, once it has been parsed: name, size, and whether the sender
    /// left the bytes out. `None` until then.
    ///
    /// Read-only, and here because a test asserts on the order the entries
    /// arrive in — the promise that a skip flag does not reshuffle the
    /// transfer. That test was written against the Python implementation and
    /// is not being rewritten for this one.
    pub fn entries(&self) -> Option<Vec<(String, u64, bool)>> {
        self.entries.as_ref().map(|entries| {
            entries
                .iter()
                .map(|entry| (entry.name.clone(), entry.size, entry.already_there))
                .collect()
        })
    }

    /// Whether every file the header promised is present here, whole.
    pub fn complete(&self) -> bool {
        match &self.entries {
            Some(entries) => {
                self.index >= entries.len() && self.handle.is_none() && !self.broken
            }
            None => false,
        }
    }

    pub fn close(&mut self) -> Result<()> {
        if let Some(mut handle) = self.handle.take() {
            handle.flush()?;
        }
        Ok(())
    }

    /// Drop what the source no longer has, then mark the copy good.
    ///
    /// Without the pruning the replica only grows: the source is pruned by
    /// retention and the copy is not, so a long run leaves a copy holding every
    /// step ever sent while the original holds three.
    pub fn commit(&mut self) -> Result<bool> {
        if !self.complete() {
            return Ok(false);
        }
        let wanted: HashSet<&str> = self
            .entries
            .as_ref()
            .expect("complete implies entries")
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();

        for (name, _) in store_files(&self.path)? {
            if name == COMPLETE_MARKER || wanted.contains(name.as_str()) {
                continue;
            }
            let _ = fs::remove_file(on_disk(&self.path, &name));
        }

        match fs::write(self.path.join(COMPLETE_MARKER), b"ok") {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

/// Read a store off `input` into `destination`, until the header says it is
/// whole or the peer stops talking.
///
/// The loop ends on `complete` alone: the embedded header already says how much
/// to expect, so no length prefix or connection close is needed, and the
/// connection is left open for whatever round comes next. An empty read still
/// ends the loop rather than spinning, but that is the belt, not the buckle.
///
/// Single-threaded, unlike [`send_store`], and the asymmetry is deliberate. To
/// overlap the socket with the disk here, the reading side would have to know
/// when to stop reading *before* the writing side has parsed the header — and a
/// reader that guesses wrong blocks forever on a chunk that is never sent. The
/// send side has no such problem because the encoder knows the length up front.
pub fn receive_store<R: Read>(input: &mut R, destination: &Path) -> Result<bool> {
    let mut writer = StoreWriter::new(destination);
    let mut buffer = vec![0u8; 1 << 16];

    while !writer.complete() {
        let read = match input.read(&mut buffer) {
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => {
                let _ = writer.close();
                return Err(TransportError::Io(error));
            }
        };
        if read == 0 {
            break;
        }
        if let Err(error) = writer.feed(&buffer[..read]) {
            let _ = writer.close();
            return Err(error);
        }
    }

    writer.close()?;
    writer.commit()
}

// ─── the pre-staging pair, over a connection someone else owns ───────────────

/// Big-endian, unlike every other integer here, because that is what
/// `_LENGTH_PREFIX` in `ravex/_dist/elastic.py` has always packed and this end
/// of the conversation is not the one that gets to choose.
const PREFIX_SIZE: usize = 8;

fn read_exactly<R: Read>(input: &mut R, want: usize) -> Result<Vec<u8>> {
    let mut buffer = vec![0u8; want];
    let mut filled = 0;
    while filled < want {
        match input.read(&mut buffer[filled..]) {
            Ok(0) => {
                return Err(TransportError::PeerClosed(format!(
                    "peer closed while {} of {} bytes were still expected",
                    want - filled,
                    want
                )))
            }
            Ok(count) => filled += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(TransportError::Io(error)),
        }
    }
    Ok(buffer)
}

/// Push `source` to whatever [`prestage_receive`] holds at the other end,
/// leaving out the files the peer says it already has.
///
/// Does not shut the connection down afterwards, on purpose: the header already
/// tells the receiver how many files of what size to expect, which is what lets
/// it end its read loop without a message boundary from the transport. Leaving
/// the connection open is what makes several rounds over one socket possible —
/// a coarse pass while the old ranks are still far from the cutover, then
/// smaller and smaller deltas, without a new handshake for each.
pub fn prestage_send<S: Read + Write>(
    stream: &mut S,
    source: &Path,
    chunk: usize,
) -> Result<u64> {
    let prefix = read_exactly(stream, PREFIX_SIZE)?;
    let manifest_len = u64::from_be_bytes(prefix[..].try_into().unwrap()) as usize;
    let peer: std::collections::HashMap<String, u64> = if manifest_len > 0 {
        parse_manifest(&read_exactly(stream, manifest_len)?)?
            .into_iter()
            .collect()
    } else {
        std::collections::HashMap::new()
    };

    // Names alone, no hash, and that is safe for the reason GPU-97 gives: the
    // files a store is made of are immutable once written.
    let skip: HashSet<String> = store_files(source)?
        .into_iter()
        .filter(|(name, size)| peer.get(name) == Some(size))
        .map(|(name, _)| name)
        .collect();

    send_store(stream, source, chunk, skip)
}

/// The other end of [`prestage_send`]: say what is already here, then take what
/// comes back.
pub fn prestage_receive<S: Read + Write>(stream: &mut S, destination: &Path) -> Result<bool> {
    let blob = encode_manifest(&store_files(destination)?);
    let mut greeting = Vec::with_capacity(PREFIX_SIZE + blob.len());
    greeting.extend_from_slice(&(blob.len() as u64).to_be_bytes());
    greeting.extend_from_slice(&blob);
    stream.write_all(&greeting)?;
    stream.flush()?;

    receive_store(stream, destination)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn scratch(name: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "ravex-transport-{}-{}-{}",
            name,
            std::process::id(),
            unique
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write(root: &Path, relative: &str, bytes: &[u8]) {
        let target = on_disk(root, relative);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(target, bytes).unwrap();
    }

    fn round_trip(source: &Path, destination: &Path, skip: HashSet<String>) -> bool {
        let mut wire = Vec::new();
        send_store(&mut wire, source, 64, skip).unwrap();
        receive_store(&mut wire.as_slice(), destination).unwrap()
    }

    #[test]
    fn a_store_arrives_whole() {
        let source = scratch("source");
        let destination = scratch("destination");
        write(&source, "a.bin", &[7u8; 300]);
        write(&source, "nested/b.bin", &[9u8; 5]);

        assert!(round_trip(&source, &destination, HashSet::new()));
        assert_eq!(fs::read(destination.join("a.bin")).unwrap(), vec![7u8; 300]);
        assert_eq!(
            fs::read(destination.join("nested").join("b.bin")).unwrap(),
            vec![9u8; 5]
        );
        assert!(destination.join(COMPLETE_MARKER).exists());
    }

    #[test]
    fn an_empty_file_still_arrives() {
        // The case the buffered path exists for: a zero-length file is created
        // by being reached, not by being written to.
        let source = scratch("empty-source");
        let destination = scratch("empty-destination");
        write(&source, "a.bin", &[1u8; 128]);
        write(&source, "z-empty.bin", &[]);

        assert!(round_trip(&source, &destination, HashSet::new()));
        assert!(destination.join("z-empty.bin").exists());
        assert_eq!(fs::metadata(destination.join("z-empty.bin")).unwrap().len(), 0);
    }

    #[test]
    fn a_skipped_file_is_not_sent_but_is_still_promised() {
        let source = scratch("skip-source");
        let destination = scratch("skip-destination");
        write(&source, "kept.bin", &[3u8; 64]);
        write(&source, "already.bin", &[4u8; 64]);
        // The receiver has it, byte for byte, which is what lets it be skipped.
        write(&destination, "already.bin", &[4u8; 64]);

        let skip: HashSet<String> = ["already.bin".to_string()].into_iter().collect();
        let mut wire = Vec::new();
        send_store(&mut wire, &source, 16, skip.clone()).unwrap();

        assert_eq!(wire.len() as u64, encoded_size(&source, &skip).unwrap());
        assert!(receive_store(&mut wire.as_slice(), &destination).unwrap());
        assert_eq!(fs::read(destination.join("already.bin")).unwrap(), vec![4u8; 64]);
    }

    #[test]
    fn a_skip_the_receiver_cannot_honour_is_not_marked_complete() {
        // The sender was told the file was here and left it out; it is not
        // here. Anything but a refusal turns a recovery into a wrong model.
        let source = scratch("lie-source");
        let destination = scratch("lie-destination");
        write(&source, "missing.bin", &[5u8; 32]);

        let skip: HashSet<String> = ["missing.bin".to_string()].into_iter().collect();
        let mut wire = Vec::new();
        send_store(&mut wire, &source, 16, skip).unwrap();

        assert!(!receive_store(&mut wire.as_slice(), &destination).unwrap());
        assert!(!destination.join(COMPLETE_MARKER).exists());
    }

    #[test]
    fn what_the_source_dropped_is_pruned_from_the_copy() {
        let source = scratch("prune-source");
        let destination = scratch("prune-destination");
        write(&source, "current.bin", &[1u8; 16]);
        write(&destination, "step_00000001.bin", &[2u8; 16]);

        assert!(round_trip(&source, &destination, HashSet::new()));
        assert!(!destination.join("step_00000001.bin").exists());
        assert!(destination.join("current.bin").exists());
    }

    #[test]
    fn the_marker_is_gone_while_the_copy_is_half_written() {
        let destination = scratch("torn");
        fs::write(destination.join(COMPLETE_MARKER), b"ok").unwrap();

        let mut writer = StoreWriter::new(&destination);
        assert!(!destination.join(COMPLETE_MARKER).exists());
        assert!(!writer.complete());
        assert!(!writer.commit().unwrap());
    }

    #[test]
    fn a_stream_split_at_every_byte_parses_the_same() {
        // The framing has to survive arriving one byte at a time, because a
        // socket is entitled to deliver it that way.
        let source = scratch("dribble-source");
        let destination = scratch("dribble-destination");
        write(&source, "a.bin", &[6u8; 70]);
        write(&source, "b/c.bin", &[8u8; 3]);

        let mut wire = Vec::new();
        send_store(&mut wire, &source, 1 << 20, HashSet::new()).unwrap();

        let mut writer = StoreWriter::new(&destination);
        for byte in &wire {
            writer.feed(std::slice::from_ref(byte)).unwrap();
        }
        writer.close().unwrap();
        assert!(writer.commit().unwrap());
        assert_eq!(fs::read(destination.join("a.bin")).unwrap(), vec![6u8; 70]);
    }

    #[test]
    fn the_manifest_round_trips() {
        let entries = vec![
            ("a.bin".to_string(), 12u64),
            ("nested/b.bin".to_string(), 0u64),
        ];
        assert_eq!(parse_manifest(&encode_manifest(&entries)).unwrap(), entries);
    }

    #[test]
    fn a_truncated_manifest_is_an_error_not_a_guess() {
        let blob = encode_manifest(&[("a.bin".to_string(), 12)]);
        assert!(parse_manifest(&blob[..blob.len() - 2]).is_err());
    }

    #[test]
    fn the_prestage_pair_moves_a_store_over_a_socket() {
        // A real socket rather than two buffers: the pair exists because of
        // what a socket does to a stream, and a test that never splits a write
        // would not be testing the part that has ever gone wrong.
        let source = scratch("prestage-source");
        let destination = scratch("prestage-destination");
        write(&source, "a.bin", &[2u8; 5000]);
        write(&source, "b/c.bin", &[3u8; 17]);
        // Already there, byte for byte, so this round must not send it again.
        write(&destination, "a.bin", &[2u8; 5000]);

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let sending = source.clone();
        let sender = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            prestage_send(&mut stream, &sending, CHUNK).unwrap();
        });

        let mut stream = std::net::TcpStream::connect(address).unwrap();
        let ok = prestage_receive(&mut stream, &destination).unwrap();
        sender.join().unwrap();

        assert!(ok);
        assert_eq!(
            fs::read(destination.join("b").join("c.bin")).unwrap(),
            vec![3u8; 17]
        );
        assert!(destination.join(COMPLETE_MARKER).exists());
    }

    #[test]
    fn a_peer_that_hangs_up_mid_greeting_is_named_not_guessed_at() {
        let source = scratch("hangup-source");
        write(&source, "a.bin", &[1u8; 8]);
        let mut half = &[0u8, 0, 0][..];
        let mut sink = Vec::new();
        let mut both = Duplex {
            input: &mut half,
            output: &mut sink,
        };
        match prestage_send(&mut both, &source, CHUNK) {
            Err(TransportError::PeerClosed(said)) => assert!(said.contains("peer closed")),
            other => panic!("expected the peer-closed error, got {:?}", other),
        }
    }

    /// A reader and a writer bolted together, for the cases where a real socket
    /// would only add a thread.
    struct Duplex<'a> {
        input: &'a mut &'a [u8],
        output: &'a mut Vec<u8>,
    }

    impl Read for Duplex<'_> {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            self.input.read(buffer)
        }
    }

    impl Write for Duplex<'_> {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            self.output.write(buffer)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn encoded_size_matches_what_is_produced() {
        let source = scratch("size");
        write(&source, "a.bin", &[1u8; 1000]);
        write(&source, "deep/nested/b.bin", &[2u8; 7]);

        let mut wire = Vec::new();
        send_store(&mut wire, &source, 128, HashSet::new()).unwrap();
        assert_eq!(
            wire.len() as u64,
            encoded_size(&source, &HashSet::new()).unwrap()
        );
    }
}
