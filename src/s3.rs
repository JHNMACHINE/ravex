//! S3-compatible object storage over a hand-written SigV4, for the cache of
//! compiled dependencies (GPU-181).
//!
//! Copied from Moonclip's `src/s3.rs` rather than reached through it, and that
//! is a decision: Moonclip's Python binding exposes its checkpoint manager and
//! nothing else, so using its client from here would have meant a new public
//! surface in Moonclip, released before every Ravex release that needs it.
//! What came across is the signing, the addressing, the retries and the
//! multipart upload, with their reasons and their tests; what stayed behind is
//! what only a checkpoint needs - ranged reads, deletes, the listing of
//! abandoned uploads. A fix to the signing in one copy belongs in the other.
//!
//! Two departures from Moonclip's copy, both to keep this crate's dependency
//! list short: the timestamp is formatted from [`std::time::SystemTime`]
//! instead of `chrono`, and the error is this module's own [`S3Error`].
//!
//! # What is not covered
//!
//! The same as in Moonclip, and for the same reasons:
//!
//! * **Temporary credentials.** Only a long-lived access key and secret: no
//!   `x-amz-security-token` is signed or sent.
//! * **Clock skew.** The signature is stamped with the local clock; a host
//!   more than fifteen minutes off signs requests that are never accepted.
//! * **Checksum headers.** None are sent, which is also why this works
//!   unmodified against R2 and GCS. Whether a cached file is the right one is
//!   decided a layer up, by the key stored beside it.

use std::fmt::Write as FmtWrite;
use std::io::Read;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

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

// ─── AWS Signature V4 ───────────────────────────────────────────────

fn sha256_hex(data: &[u8]) -> String {
    to_hex(&Sha256::digest(data))
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(s, "{:02x}", byte).unwrap();
    }
    s
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC takes a key of any length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn signing_key(secret: &str, date: &str, region: &str, service: &str) -> Vec<u8> {
    let k_date = hmac_sha256(format!("AWS4{}", secret).as_bytes(), date.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

/// Percent-encode per AWS: everything but the unreserved characters.
fn uri_encode(s: &str, encode_slash: bool) -> String {
    let mut result = String::with_capacity(s.len() * 2);
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                result.push(byte as char);
            }
            b'/' if !encode_slash => result.push('/'),
            _ => write!(result, "%{:02X}", byte).unwrap(),
        }
    }
    result
}

/// `(YYYYMMDD, YYYYMMDDTHHMMSSZ)` for a moment given in seconds since 1970.
///
/// The civil date from a day count is Howard Hinnant's `civil_from_days`,
/// exact for every day after 1970 - which is the whole range a request can be
/// signed in. It replaces Moonclip's `chrono` for two strings.
fn amz_stamps(unix_secs: u64) -> (String, String) {
    let days = (unix_secs / 86_400) as i64;
    let rest = unix_secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    let date = format!("{year:04}{month:02}{day:02}");
    let time = format!(
        "{date}T{:02}{:02}{:02}Z",
        rest / 3_600,
        rest % 3_600 / 60,
        rest % 60
    );
    (date, time)
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

struct SignedRequest {
    url: String,
    headers: Vec<(String, String)>,
}

fn sign_request(
    config: &S3Config,
    method: &str,
    key: &str,
    payload_hash: &str,
    query_params: &[(&str, &str)],
    content_length: Option<usize>,
    unix_secs: u64,
) -> SignedRequest {
    let (date_stamp, amz_date) = amz_stamps(unix_secs);

    let encoded_key = uri_encode(key, false);
    let canonical_uri = if config.path_style {
        format!("/{}/{}", config.bucket, encoded_key)
    } else {
        format!("/{}", encoded_key)
    };

    let mut sorted_params: Vec<(&str, &str)> = query_params.to_vec();
    sorted_params.sort_by_key(|(k, _)| *k);
    let canonical_qs: String = sorted_params
        .iter()
        .map(|(k, v)| format!("{}={}", uri_encode(k, true), uri_encode(v, true)))
        .collect::<Vec<_>>()
        .join("&");

    let default_endpoint = format!("https://s3.{}.amazonaws.com", config.region);
    let endpoint = config.endpoint.as_deref().unwrap_or(&default_endpoint);
    let endpoint = endpoint.trim_end_matches('/');
    let query = if canonical_qs.is_empty() {
        String::new()
    } else {
        format!("?{canonical_qs}")
    };
    let url = if config.path_style {
        format!("{}/{}/{}{}", endpoint, config.bucket, encoded_key, query)
    } else {
        let scheme_end = endpoint.find("://").map(|i| i + 3).unwrap_or(0);
        let (scheme, rest) = endpoint.split_at(scheme_end);
        format!(
            "{}{}.{}/{}{}",
            scheme, config.bucket, rest, encoded_key, query
        )
    };

    // The host exactly as ureq will send it, lowercased.
    let host = url
        .split("://")
        .nth(1)
        .unwrap_or("")
        .split('/')
        .next()
        .unwrap_or("")
        .to_lowercase();

    // Canonical headers, in alphabetical order as SigV4 requires.
    let mut headers_to_sign: Vec<(String, String)> = vec![
        ("host".into(), host),
        ("x-amz-content-sha256".into(), payload_hash.to_string()),
        ("x-amz-date".into(), amz_date.clone()),
    ];
    if let Some(len) = content_length {
        headers_to_sign.push(("content-length".into(), len.to_string()));
    }
    headers_to_sign.sort_by(|a, b| a.0.cmp(&b.0));
    let canonical_headers: String = headers_to_sign
        .iter()
        .map(|(name, value)| format!("{}:{}\n", name, value))
        .collect();
    let signed_headers: String = headers_to_sign
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>()
        .join(";");

    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        method, canonical_uri, canonical_qs, canonical_headers, signed_headers, payload_hash
    );
    let credential_scope = format!("{}/{}/s3/aws4_request", date_stamp, config.region);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        amz_date,
        credential_scope,
        sha256_hex(canonical_request.as_bytes())
    );
    let sig_key = signing_key(&config.secret_key, &date_stamp, &config.region, "s3");
    let signature = to_hex(&hmac_sha256(&sig_key, string_to_sign.as_bytes()));
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        config.access_key, credential_scope, signed_headers, signature
    );

    // Host and Content-Length are not set here: ureq sets them from the URL
    // and the body, to the values signed above.
    SignedRequest {
        url,
        headers: vec![
            ("Authorization".into(), authorization),
            ("x-amz-content-sha256".into(), payload_hash.into()),
            ("x-amz-date".into(), amz_date),
        ],
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ─── The client ─────────────────────────────────────────────────────

pub struct S3Storage {
    config: S3Config,
    agent: ureq::Agent,
}

/// Three attempts in all: enough to ride out a rolling restart on the far
/// side, short enough that a broken endpoint is reported while it matters.
const MAX_ATTEMPTS: u32 = 3;

impl S3Storage {
    pub fn new(config: S3Config) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(config.timeout_secs))
            .timeout_read(Duration::from_secs(config.timeout_secs * 10))
            .timeout_write(Duration::from_secs(config.timeout_secs * 10))
            .build();
        S3Storage { config, agent }
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
        let payload_hash = sha256_hex(body.unwrap_or(b""));
        let signed = sign_request(
            &self.config,
            method,
            key,
            &payload_hash,
            query_params,
            body.map(|d| d.len()),
            now_secs(),
        );
        let mut req = match method {
            "GET" => self.agent.get(&signed.url),
            "PUT" => self.agent.put(&signed.url),
            "DELETE" => self.agent.delete(&signed.url),
            "HEAD" => self.agent.head(&signed.url),
            "POST" => self.agent.post(&signed.url),
            _ => return Err(S3Error::Storage(format!("unknown method {method}"))),
        };
        for (k, v) in &signed.headers {
            req = req.set(k, v);
        }
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

    #[test]
    fn stamps_match_the_calendar() {
        assert_eq!(
            amz_stamps(0),
            ("19700101".into(), "19700101T000000Z".into())
        );
        // 2013-05-24T00:00:00Z, the moment of AWS's published S3 examples.
        assert_eq!(amz_stamps(1_369_353_600).1, "20130524T000000Z");
        // A leap day, and the second before a new year.
        assert_eq!(amz_stamps(1_709_164_800).0, "20240229");
        assert_eq!(amz_stamps(1_798_761_599).1, "20261231T235959Z");
    }

    /// The key is a function of the secret and the day and nothing else. That
    /// the signature it makes is one a real service accepts is the MinIO
    /// test's to show (`tests/test_cache_s3.py`), not this one's.
    #[test]
    fn the_signing_key_depends_on_the_day() {
        let key = signing_key(
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "20130524",
            "us-east-1",
            "s3",
        );
        let again = signing_key(
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "20130524",
            "us-east-1",
            "s3",
        );
        assert_eq!(key, again);
        assert_ne!(
            key,
            signing_key(
                "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "20130525",
                "us-east-1",
                "s3"
            )
        );
    }

    #[test]
    fn a_request_names_the_bucket_where_its_style_says() {
        let mut c = config();
        let hash = sha256_hex(b"");
        let hosted = sign_request(&c, "GET", "a b/c.whl", &hash, &[], None, 1_369_353_600);
        assert_eq!(
            hosted.url,
            "https://examplebucket.s3.us-east-1.amazonaws.com/a%20b/c.whl"
        );
        c.endpoint = Some("http://127.0.0.1:9000/".into());
        c.path_style = true;
        let path = sign_request(
            &c,
            "GET",
            "",
            &hash,
            &[("prefix", "x/"), ("list-type", "2")],
            None,
            1_369_353_600,
        );
        assert_eq!(
            path.url,
            "http://127.0.0.1:9000/examplebucket/?list-type=2&prefix=x%2F"
        );
        let auth = &path.headers[0].1;
        assert!(auth.starts_with(
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request"
        ));
        assert!(auth.contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date,"));
    }

    #[test]
    fn test_uri_encode() {
        assert_eq!(uri_encode("hello world", true), "hello%20world");
        assert_eq!(uri_encode("path/to/file", false), "path/to/file");
        assert_eq!(uri_encode("path/to/file", true), "path%2Fto%2Ffile");
        assert_eq!(
            uri_encode("flash_attn-2.7.4+cu128-cp312.whl", false),
            "flash_attn-2.7.4%2Bcu128-cp312.whl"
        );
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
