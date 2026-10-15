//! S3-compatible object storage for the cache of compiled dependencies
//! (GPU-181), signed by `rusty-s3` (GPU-203).
//!
//! The signing is `rusty-s3`'s, as in Moonclip since 0.2: through 0.7.2 this
//! module carried a copy of Moonclip's hand-written SigV4, which is exactly
//! the kind of cryptography that is easy to get subtly wrong at the edges,
//! and a fix to it belonged in both copies. `rusty-s3` signs and does no
//! I/O, so it brings neither the AWS SDK's dependency tree nor its checksum
//! headers, which GCS and R2 reject. What stays ours is what is ours: the
//! requests (`ureq`), the retries, the multipart upload and its abort, and
//! the little XML the answers need.
//!
//! Still a copy of Moonclip's client rather than a call into it, and that is
//! a decision: Moonclip's Python binding exposes its checkpoint manager and
//! nothing else, so using its client from here would mean a new public
//! surface in Moonclip, released before every Ravex release that needs it.
//! What is copied is now only the use of `rusty-s3` - addressing, retries,
//! multipart - not a signature; what stayed behind is what only a checkpoint
//! needs: ranged reads, deletes, the listing of abandoned uploads, and the
//! header-signed listing R2 wants with a session token.
//!
//! Every request is a **presigned URL**: the signature is in its query
//! string, made the moment the request is sent, and the body is not hashed
//! into it (`UNSIGNED-PAYLOAD`). Whether a cached file is the right one is
//! decided a layer up, by the key stored beside it.
//!
//! # What is not covered
//!
//! * **Temporary credentials.** Only a long-lived access key and secret: no
//!   session token is signed or sent. Nodes do not need one - their cache goes
//!   through URLs the backend signs (GPU-200) - so this client serves only
//!   whoever runs `ravex cache` with keys of their own.
//! * **Clock skew.** The signature is stamped with the local clock; a host
//!   more than fifteen minutes off signs requests that are never accepted.
//! * **Checksum headers.** None are sent, which is also why this works
//!   unmodified against R2 and GCS.

use std::fmt::Write as FmtWrite;
use std::io::Read;
use std::time::Duration;

use rusty_s3::{Bucket, Credentials, Method, UrlStyle};
use url::Url;

/// A failed request, split the one way callers act on.
#[derive(Debug)]
pub enum S3Error {
    /// The object is not there: a 404, which a cache treats as a miss.
    NotFound(String),
    /// Anything else, with the service's own words when it gave some.
    Storage(String),
}

impl std::fmt::Display for S3Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            S3Error::NotFound(key) => write!(f, "no object at {key}"),
            S3Error::Storage(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for S3Error {}

pub type Result<T> = std::result::Result<T, S3Error>;

// ─── Configuration ──────────────────────────────────────────────────

/// Where the bucket is and how to sign for it.
#[derive(Debug, Clone)]
pub struct S3Config {
    pub bucket: String,
    /// Key prefix, like a folder. A trailing slash is added when missing.
    pub prefix: String,
    /// `auto` for R2, which accepts any region in the signature.
    pub region: String,
    /// The service's address; AWS when None. For R2,
    /// `https://<account>.r2.cloudflarestorage.com`.
    pub endpoint: Option<String>,
    pub access_key: String,
    pub secret_key: String,
    /// `http://endpoint/bucket/key` when true, `http://bucket.endpoint/key`
    /// when false. A custom endpoint wants path style: virtual-hosted needs a
    /// DNS name per bucket, which MinIO does not have.
    pub path_style: bool,
    pub timeout_secs: u64,
    /// Above this many bytes, a write becomes a multipart upload. Movable so a
    /// test can watch that path for the price of a few megabytes.
    pub single_put_limit: usize,
}

impl S3Config {
    /// Path style whenever the endpoint is not AWS's own.
    pub fn with_auto_path_style(mut self) -> Self {
        if self.endpoint.is_some() {
            self.path_style = true;
        }
        self
    }

    fn normalized_prefix(&self) -> String {
        let mut p = self.prefix.trim_matches('/').to_string();
        if !p.is_empty() {
            p.push('/');
        }
        p
    }

    fn object_key(&self, rel_path: &str) -> String {
        format!("{}{}", self.normalized_prefix(), rel_path)
    }
}

// ─── Addressing and signing ─────────────────────────────────────────

/// How long a signed request stays valid. It is signed the moment it is
/// sent, so this only has to cover the request starting: an hour leaves room
/// for a slow connection to open, and none for a URL found in a log later.
const SIGNED_FOR_SECS: u64 = 3600;

/// The bucket as `rusty-s3` addresses it: AWS's endpoint for the region when
/// none is given, path-style or virtual-hosted as configured.
fn bucket_of(config: &S3Config) -> Result<Bucket> {
    let mut endpoint = config
        .endpoint
        .clone()
        .unwrap_or_else(|| format!("https://s3.{}.amazonaws.com", config.region));
    // The bucket's address is joined onto this one, and a join replaces the
    // last segment of a path that has no trailing slash: an endpoint with a
    // path of its own would lose it.
    if !endpoint.ends_with('/') {
        endpoint.push('/');
    }
    let url = Url::parse(&endpoint).map_err(|e| {
        S3Error::Storage(format!("S3 endpoint {endpoint:?} is not an address: {e}"))
    })?;
    let style = if config.path_style {
        UrlStyle::Path
    } else {
        UrlStyle::VirtualHost
    };
    Bucket::new(url, style, config.bucket.clone(), config.region.clone())
        .map_err(|e| S3Error::Storage(format!("S3 bucket {:?}: {e}", config.bucket)))
}

/// Above this, S3 refuses one `PUT` (its limit is 5 GiB) and a write is split.
pub const SINGLE_PUT_LIMIT: usize = 4 * 1024 * 1024 * 1024;

/// Larger than S3's 5 MiB floor: parts cost a request each, and 10000 is the
/// ceiling.
const MIN_PART_SIZE: usize = 16 * 1024 * 1024;

const MAX_PARTS: usize = 10_000;

/// Part size for an object of `total` bytes, growing so the count stays under
/// [`MAX_PARTS`], with some headroom left for rounding.
fn part_size_for(total: usize) -> usize {
    let spread = total.div_ceil(MAX_PARTS - 16);
    MIN_PART_SIZE.max(spread)
}

// ─── The client ─────────────────────────────────────────────────────

pub struct S3Storage {
    config: S3Config,
    bucket: Bucket,
    credentials: Credentials,
    agent: ureq::Agent,
}

/// Three attempts in all: enough to ride out a rolling restart on the far
/// side, short enough that a broken endpoint is reported while it matters.
const MAX_ATTEMPTS: u32 = 3;

impl S3Storage {
    /// Fails only on an endpoint or bucket name that cannot be an address,
    /// which is better said here than on the first request.
    pub fn new(config: S3Config) -> Result<Self> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(config.timeout_secs))
            .timeout_read(Duration::from_secs(config.timeout_secs * 10))
            .timeout_write(Duration::from_secs(config.timeout_secs * 10))
            .build();
        let bucket = bucket_of(&config)?;
        let credentials = Credentials::new(&config.access_key, &config.secret_key);
        Ok(S3Storage {
            config,
            bucket,
            credentials,
            agent,
        })
    }

    /// The address of one request to `key`, signed now: `rusty-s3`'s SigV4,
    /// in the query string.
    ///
    /// `query` is any of S3's own parameters (`uploads`, `partNumber`...).
    /// `rusty-s3` merges them with its `X-Amz-*` ones as two runs already in
    /// order, so they are sorted here first: out of order, the URL would carry
    /// them in an order other than the one the signature covers.
    fn signed_url(&self, method: &str, key: &str, query: &[(&str, &str)]) -> Result<Url> {
        let method = match method {
            "GET" => Method::Get,
            "PUT" => Method::Put,
            "DELETE" => Method::Delete,
            "HEAD" => Method::Head,
            "POST" => Method::Post,
            _ => return Err(S3Error::Storage(format!("unknown method {method}"))),
        };
        let url = self
            .bucket
            .object_url(key)
            .map_err(|e| S3Error::Storage(format!("S3 key {key:?} is not a path: {e}")))?;
        let mut query = query.to_vec();
        query.sort_unstable();
        Ok(rusty_s3::signing::sign(
            &jiff::Timestamp::now(),
            method,
            url,
            self.credentials.key(),
            self.credentials.secret(),
            self.credentials.token(),
            self.bucket.region(),
            SIGNED_FOR_SECS,
            query.iter().copied(),
            std::iter::empty(),
        ))
    }

    /// One request, retried on what is worth retrying: transport failures,
    /// 5xx and 429, which object stores produce as a matter of course. A 4xx
    /// otherwise - a wrong key, a missing bucket, a bad signature - fails the
    /// same way twice. Every request here is idempotent, so a retry cannot
    /// compound.
    fn do_request(
        &self,
        method: &str,
        key: &str,
        body: Option<&[u8]>,
        query_params: &[(&str, &str)],
    ) -> Result<ureq::Response> {
        let mut backoff = Duration::from_millis(200);
        let mut attempt = 0;
        loop {
            attempt += 1;
            match self.attempt_request(method, key, body, query_params) {
                Ok(resp) => return Ok(resp),
                Err(e) if attempt < MAX_ATTEMPTS && is_retryable(&e) => {
                    std::thread::sleep(backoff);
                    backoff *= 2;
                }
                Err(e) => return Err(e),
            }
        }
    }

    fn attempt_request(
        &self,
        method: &str,
        key: &str,
        body: Option<&[u8]>,
        query_params: &[(&str, &str)],
    ) -> Result<ureq::Response> {
        let url = self.signed_url(method, key, query_params)?;
        // Nothing to set beyond the URL: the signature covers only the host,
        // and ureq sends that and the body's length on its own.
        let req = match method {
            "GET" => self.agent.get(url.as_str()),
            "PUT" => self.agent.put(url.as_str()),
            "DELETE" => self.agent.delete(url.as_str()),
            "HEAD" => self.agent.head(url.as_str()),
            "POST" => self.agent.post(url.as_str()),
            _ => return Err(S3Error::Storage(format!("unknown method {method}"))),
        };
        let response = match body {
            Some(data) => req.send_bytes(data),
            None => req.call(),
        };
        response.map_err(|e| status_error(key, e))
    }

    /// Write the whole object, in parts past [`S3Config::single_put_limit`].
    pub fn put(&self, rel_path: &str, data: &[u8]) -> Result<()> {
        if data.len() > self.config.single_put_limit {
            return self.put_multipart(rel_path, data);
        }
        self.do_request("PUT", &self.config.object_key(rel_path), Some(data), &[])?;
        Ok(())
    }

    /// The object's bytes; [`S3Error::NotFound`] under the caller's path when
    /// there is none.
    pub fn get(&self, rel_path: &str) -> Result<Vec<u8>> {
        let key = self.config.object_key(rel_path);
        let resp = self
            .do_request("GET", &key, None, &[])
            .map_err(|e| match e {
                S3Error::NotFound(_) => S3Error::NotFound(rel_path.to_string()),
                other => other,
            })?;
        let mut buf = Vec::new();
        resp.into_reader()
            .read_to_end(&mut buf)
            .map_err(|e| S3Error::Storage(format!("S3 read error: {e}")))?;
        Ok(buf)
    }

    pub fn exists(&self, rel_path: &str) -> Result<bool> {
        match self.do_request("HEAD", &self.config.object_key(rel_path), None, &[]) {
            Ok(_) => Ok(true),
            Err(S3Error::NotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Every path under `prefix`, relative to the configured prefix, across
    /// as many `ListObjectsV2` pages as there are.
    pub fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let full_prefix = self.config.object_key(prefix);
        let base = self.config.normalized_prefix();
        let mut all_keys = Vec::new();
        let mut continuation: Option<String> = None;
        loop {
            let mut params: Vec<(&str, &str)> = vec![("list-type", "2"), ("prefix", &full_prefix)];
            let held;
            if let Some(ref token) = continuation {
                held = token.clone();
                params.push(("continuation-token", &held));
            }
            let body = self
                .do_request("GET", "", None, &params)?
                .into_string()
                .map_err(|e| S3Error::Storage(format!("S3 list parse error: {e}")))?;
            for key in extract_xml_values(&body, "Key") {
                match key.strip_prefix(&base) {
                    Some(rel) => all_keys.push(rel.to_string()),
                    None => all_keys.push(key),
                }
            }
            if !body.contains("<IsTruncated>true</IsTruncated>") {
                break;
            }
            match extract_xml_value(&body, "NextContinuationToken") {
                Some(token) => continuation = Some(token),
                None => break,
            }
        }
        Ok(all_keys)
    }

    /// Upload in parts: create, one call per part, complete - and on every
    /// other way out, abort. An upload neither completed nor aborted keeps its
    /// parts in the bucket, invisible to a listing and billed.
    pub fn put_multipart(&self, rel_path: &str, data: &[u8]) -> Result<()> {
        let key = &self.config.object_key(rel_path);
        let body = self
            .do_request("POST", key, None, &[("uploads", "")])?
            .into_string()
            .map_err(|e| S3Error::Storage(format!("S3 multipart start: {e}")))?;
        let upload_id = extract_xml_value(&body, "UploadId")
            .ok_or_else(|| S3Error::Storage("S3 multipart start returned no UploadId".into()))?;
        match self.upload_parts(key, data, &upload_id) {
            Ok(()) => Ok(()),
            Err(e) => {
                // The upload's own failure is what the caller needs; a failed
                // cleanup is a second line, never a replacement.
                if let Err(cleanup) =
                    self.do_request("DELETE", key, None, &[("uploadId", upload_id.as_str())])
                {
                    eprintln!(
                        "[ravex] could not abort the multipart upload of {key}: {cleanup}. \
                         Its parts are still in the bucket, invisible to a listing and billed."
                    );
                }
                Err(e)
            }
        }
    }

    fn upload_parts(&self, key: &str, data: &[u8], upload_id: &str) -> Result<()> {
        let size = part_size_for(data.len());
        let mut completed = String::from("<CompleteMultipartUpload>");
        // `chunks` gives equal parts and a smaller last one, which R2 demands:
        // every part but the last the same size, or the whole upload fails.
        for (index, chunk) in data.chunks(size).enumerate() {
            let number = (index + 1).to_string();
            let response = self.do_request(
                "PUT",
                key,
                Some(chunk),
                &[("partNumber", number.as_str()), ("uploadId", upload_id)],
            )?;
            let etag = response.header("ETag").ok_or_else(|| {
                S3Error::Storage(format!("S3 part {number} came back without an ETag"))
            })?;
            let _ = write!(
                completed,
                "<Part><PartNumber>{number}</PartNumber><ETag>{etag}</ETag></Part>"
            );
        }
        completed.push_str("</CompleteMultipartUpload>");
        let body = self
            .do_request(
                "POST",
                key,
                Some(completed.as_bytes()),
                &[("uploadId", upload_id)],
            )?
            .into_string()
            .map_err(|e| S3Error::Storage(format!("S3 multipart complete: {e}")))?;
        // S3 can answer 200 and report the failure in the body.
        if body.contains("<Error>") {
            let code = extract_xml_value(&body, "Code").unwrap_or_else(|| "unknown".into());
            return Err(S3Error::Storage(format!(
                "S3 refused to complete the multipart upload of {key}: {code}"
            )));
        }
        Ok(())
    }
}

/// 404 becomes [`S3Error::NotFound`] here, by status, never by looking for
/// "404" in a message that may hold a key with those digits.
fn status_error(key: &str, e: ureq::Error) -> S3Error {
    match e {
        ureq::Error::Status(404, _) => S3Error::NotFound(key.to_string()),
        ureq::Error::Status(code, resp) => {
            let body = resp.into_string().unwrap_or_default();
            S3Error::Storage(format!("S3 HTTP {code}: {body}"))
        }
        ureq::Error::Transport(t) => S3Error::Storage(format!("S3 transport error: {t}")),
    }
}

fn is_retryable(e: &S3Error) -> bool {
    match e {
        S3Error::Storage(msg) if msg.starts_with("S3 transport error") => true,
        S3Error::Storage(msg) => {
            let code = msg
                .strip_prefix("S3 HTTP ")
                .and_then(|rest| rest.split(':').next())
                .and_then(|c| c.trim().parse::<u16>().ok());
            matches!(code, Some(429) | Some(500..=599))
        }
        S3Error::NotFound(_) => false,
    }
}

// ─── Minimal XML, enough for ListObjectsV2 ──────────────────────────

/// S3 escapes key names in its XML; `&amp;` goes last so `&amp;lt;` comes
/// back as the text `&lt;`.
fn unescape_xml(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn extract_xml_values(xml: &str, tag: &str) -> Vec<String> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let mut results = Vec::new();
    let mut search_from = 0;
    while let Some(start) = xml[search_from..].find(&open) {
        let abs_start = search_from + start + open.len();
        match xml[abs_start..].find(&close) {
            Some(end) => {
                results.push(unescape_xml(&xml[abs_start..abs_start + end]));
                search_from = abs_start + end + close.len();
            }
            None => break,
        }
    }
    results
}

fn extract_xml_value(xml: &str, tag: &str) -> Option<String> {
    extract_xml_values(xml, tag).into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> S3Config {
        S3Config {
            bucket: "examplebucket".into(),
            prefix: "".into(),
            region: "us-east-1".into(),
            endpoint: None,
            access_key: "AKIAIOSFODNN7EXAMPLE".into(),
            secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".into(),
            path_style: false,
            timeout_secs: 30,
            single_put_limit: SINGLE_PUT_LIMIT,
        }
    }

    /// Where requests go, both ways of naming a bucket, with and without an
    /// endpoint of one's own.
    #[test]
    fn the_bucket_is_addressed_as_configured() {
        let address = |endpoint: Option<&str>, path_style| {
            let mut c = config();
            c.endpoint = endpoint.map(Into::into);
            c.path_style = path_style;
            bucket_of(&c).unwrap().base_url().to_string()
        };
        // Virtual-hosted, AWS's default.
        assert_eq!(address(None, false), "https://examplebucket.s3.us-east-1.amazonaws.com/");
        assert_eq!(address(None, true), "https://s3.us-east-1.amazonaws.com/examplebucket/");
        // A gateway, path-style as they want.
        assert_eq!(address(Some("http://127.0.0.1:9000"), true), "http://127.0.0.1:9000/examplebucket/");
        assert_eq!(address(Some("http://127.0.0.1:9000/"), true), "http://127.0.0.1:9000/examplebucket/");
        // An endpoint with a path of its own keeps it.
        assert_eq!(address(Some("https://gw.example/s3"), true), "https://gw.example/s3/examplebucket/");
        let mut broken = config();
        broken.endpoint = Some("not an address".into());
        assert!(S3Storage::new(broken).is_err());
    }

    /// A key is encoded as SigV4 wants it in the path: everything but the
    /// unreserved characters and `/`. A wheel's local version is a `+`.
    #[test]
    fn a_key_is_encoded_into_the_path() {
        let store = S3Storage::new(config()).unwrap();
        let url = store
            .signed_url("GET", "wheels/flash_attn-2.7.4+cu128 a&b.whl", &[])
            .unwrap();
        assert_eq!(url.path(), "/wheels/flash_attn-2.7.4%2Bcu128%20a%26b.whl");
    }

    /// The signature is in the query, with the long-lived key's credential
    /// and nothing that asks for a checksum: R2 and GCS refuse those.
    #[test]
    fn a_request_is_signed_in_its_query_and_asks_for_no_checksum() {
        let store = S3Storage::new(config()).unwrap();
        let url = store.signed_url("PUT", "wheels/x.whl", &[]).unwrap();
        let query = url.query().unwrap();
        assert!(query.contains("X-Amz-Algorithm=AWS4-HMAC-SHA256"));
        assert!(query.contains("X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F"));
        assert!(query.contains("X-Amz-SignedHeaders=host"));
        assert!(query.contains("X-Amz-Signature="));
        assert!(!query.contains("X-Amz-Security-Token"));
        assert!(!query.to_ascii_lowercase().contains("checksum"));
    }

    /// S3's own parameters reach the URL, whatever order they are given in:
    /// `rusty-s3` merges them as an ordered run, so they are sorted first.
    #[test]
    fn the_parameters_of_a_request_are_signed_in_order() {
        let mut c = config();
        c.endpoint = Some("http://127.0.0.1:9000".into());
        c.path_style = true;
        let store = S3Storage::new(c).unwrap();
        let url = store
            .signed_url("PUT", "k", &[("uploadId", "u-1"), ("partNumber", "3")])
            .unwrap();
        let query = url.query().unwrap();
        let names: Vec<&str> = query.split('&').map(|pair| pair.split('=').next().unwrap()).collect();
        // The signature last, and everything it covers in order before it.
        let (last, signed) = names.split_last().unwrap();
        assert_eq!(*last, "X-Amz-Signature");
        assert!(signed.windows(2).all(|pair| pair[0] <= pair[1]), "{names:?}");
        assert!(query.contains("partNumber=3") && query.contains("uploadId=u-1"));
        // A listing's prefix, encoded, and an empty value, as `?uploads` is.
        let url = store
            .signed_url("GET", "", &[("prefix", "x/"), ("list-type", "2")])
            .unwrap();
        assert_eq!(url.path(), "/examplebucket/");
        assert!(url.query().unwrap().contains("prefix=x%2F"));
        let url = store.signed_url("POST", "k", &[("uploads", "")]).unwrap();
        assert!(url.query().unwrap().contains("uploads="));
    }

    #[test]
    fn prefixes_are_normalized() {
        let mut c = config();
        c.prefix = "/foo/bar/".into();
        assert_eq!(c.object_key("wheels/x"), "foo/bar/wheels/x");
        c.prefix = "".into();
        assert_eq!(c.object_key("wheels/x"), "wheels/x");
    }

    #[test]
    fn no_object_size_produces_more_parts_than_s3_accepts() {
        let gib = 1024 * 1024 * 1024usize;
        for total in [
            SINGLE_PUT_LIMIT + 1,
            11 * gib,
            160 * gib,
            1024 * gib,
            5 * 1024 * gib,
        ] {
            let size = part_size_for(total);
            assert!(total.div_ceil(size) <= MAX_PARTS);
            assert!((5 * 1024 * 1024..=5 * gib).contains(&size));
        }
        assert_eq!(part_size_for(0), MIN_PART_SIZE);
    }

    #[test]
    fn listed_keys_come_back_unescaped() {
        let xml = "<Contents><Key>wheels/a&amp;b.whl</Key></Contents><Contents><Key>x&amp;lt;y</Key></Contents>";
        assert_eq!(
            extract_xml_values(xml, "Key"),
            vec!["wheels/a&b.whl", "x&lt;y"]
        );
    }

    #[test]
    fn retries_cover_the_far_side_failing_and_nothing_else() {
        assert!(is_retryable(&S3Error::Storage(
            "S3 transport error: reset".into()
        )));
        assert!(is_retryable(&S3Error::Storage(
            "S3 HTTP 429: SlowDown".into()
        )));
        assert!(is_retryable(&S3Error::Storage("S3 HTTP 503: retry".into())));
        assert!(!is_retryable(&S3Error::Storage(
            "S3 HTTP 403: SignatureDoesNotMatch".into()
        )));
        assert!(!is_retryable(&S3Error::NotFound("wheels/x".into())));
        // A key holding 404 is a server error here, not a missing object.
        assert!(is_retryable(&S3Error::Storage(
            "S3 HTTP 500: step_404/x".into()
        )));
    }
}
