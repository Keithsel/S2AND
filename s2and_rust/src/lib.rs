use arrow::array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, Int32Array, Int64Array, LargeListArray,
    LargeStringArray, ListArray, StringArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::DataType;
use arrow::ipc::reader::FileReader as ArrowFileReader;
use arrow::record_batch::RecordBatch;
use cld2::{detect_language_ext as cld2_detect_language_ext, Format as Cld2Format};
use fasttext::FastText;
use memmap2::Mmap;
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods, ToPyArray,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyModule, PyTuple};
use pyo3::Bound;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{hash_map::Entry, BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

fn py_len(s: &str) -> usize {
    // Python len() semantics for str: count of Unicode scalar values
    s.chars().count()
}

const DROPPED_AFFIXES: [&str; 48] = [
    "ab", "am", "ap", "abu", "al", "auf", "aus", "bar", "bath", "bat", "bet", "bint", "dall",
    "dalla", "das", "de", "degli", "del", "dell", "della", "dem", "den", "der", "di", "do", "dos",
    "ds", "du", "el", "ibn", "im", "jr", "la", "las", "le", "los", "mac", "mc", "mhic", "mic",
    "ter", "und", "van", "vom", "von", "zu", "zum", "zur",
];

const FNV_OFFSET: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;
const ARROW_BATCH_LOOKUP_INDEX_MAGIC: &[u8; 8] = b"S2ABI001";
const ARROW_BATCH_LOOKUP_INDEX_HEADER_LEN: usize = 48;
const ARROW_BATCH_LOOKUP_INDEX_RECORD_LEN: usize = 16;
const ARROW_BATCH_LOOKUP_INDEX_SOURCE_HASH_DOMAIN: &[u8] =
    b"s2and-arrow-batch-lookup-index-source\0";
const ARROW_BATCH_LOOKUP_INDEX_SOURCE_SAMPLE_BYTES: u64 = 65_536;

fn is_dropped_affix(token: &str) -> bool {
    DROPPED_AFFIXES.contains(&token)
}

#[derive(Clone, Serialize, Deserialize)]
struct CounterData {
    // FNV-1a 64-bit hashes of original string keys, sorted ascending.
    // Values are f32 counts. Binary search replaces HashMap lookup.
    // Trade-off: collisions are possible but rare at expected key counts.
    // If they occur, colliding keys merge silently in this representation.
    entries: Vec<(u64, f32)>,
    sum: f32,
}

#[inline(always)]
fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[inline(always)]
fn fnv64_update(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[derive(Clone, Serialize, Deserialize)]
struct NameCountsData {
    first: f64,
    first_last: f64,
    last: f64,
    last_first_initial: f64,
}

#[derive(Default)]
struct RawNameCountMaps {
    first: HashMap<String, f64>,
    last: HashMap<String, f64>,
    first_last: HashMap<String, f64>,
    last_first_initial: HashMap<String, f64>,
    index: Option<RawNameCountIndex>,
}

#[derive(Clone, Copy)]
enum RawNameCountKind {
    First,
    Last,
    FirstLast,
    LastFirstInitial,
}

impl RawNameCountKind {
    fn key(self) -> &'static str {
        match self {
            RawNameCountKind::First => "first",
            RawNameCountKind::Last => "last",
            RawNameCountKind::FirstLast => "first_last",
            RawNameCountKind::LastFirstInitial => "last_first_initial",
        }
    }
}

const NAME_COUNTS_INDEX_MAGIC: &[u8; 8] = b"S2NCI001";
const NAME_COUNTS_INDEX_HASH_DOMAIN: &[u8] = b"s2and-name-counts-index-v1\0";
const NAME_COUNTS_INDEX_HEADER_LEN: usize = 32;
const NAME_COUNTS_INDEX_RECORD_LEN: usize = 40;

struct RawNameCountIndex {
    first: RawNameCountIndexFile,
    last: RawNameCountIndexFile,
    first_last: RawNameCountIndexFile,
    last_first_initial: RawNameCountIndexFile,
}

struct RawNameCountIndexPaths {
    first: PathBuf,
    last: PathBuf,
    first_last: PathBuf,
    last_first_initial: PathBuf,
}

impl RawNameCountIndex {
    fn open(path: &str) -> PyResult<Self> {
        let paths = resolve_name_counts_index_paths(path)?;
        Ok(Self {
            first: RawNameCountIndexFile::open(&paths.first, RawNameCountKind::First)?,
            last: RawNameCountIndexFile::open(&paths.last, RawNameCountKind::Last)?,
            first_last: RawNameCountIndexFile::open(
                &paths.first_last,
                RawNameCountKind::FirstLast,
            )?,
            last_first_initial: RawNameCountIndexFile::open(
                &paths.last_first_initial,
                RawNameCountKind::LastFirstInitial,
            )?,
        })
    }

    fn get(&self, kind: RawNameCountKind, name: &str) -> Option<f64> {
        match kind {
            RawNameCountKind::First => self.first.get(kind, name),
            RawNameCountKind::Last => self.last.get(kind, name),
            RawNameCountKind::FirstLast => self.first_last.get(kind, name),
            RawNameCountKind::LastFirstInitial => self.last_first_initial.get(kind, name),
        }
    }
}

struct RawNameCountIndexFile {
    mmap: Mmap,
    record_count: usize,
    blob_offset: usize,
    blob_len: usize,
}

impl RawNameCountIndexFile {
    fn open(path: &Path, kind: RawNameCountKind) -> PyResult<Self> {
        let file = File::open(path).map_err(|err| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "failed to open name-count index file {}: {}",
                path.display(),
                err
            ))
        })?;
        // The writer produces immutable binary sidecars. Mapping avoids reading
        // the multi-GB global name-count artifact into Rust heap memory.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|err| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "failed to mmap name-count index file {}: {}",
                path.display(),
                err
            ))
        })?;
        if mmap.len() < NAME_COUNTS_INDEX_HEADER_LEN {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "name-count index file {} is shorter than the header",
                path.display()
            )));
        }
        if &mmap[0..8] != NAME_COUNTS_INDEX_MAGIC {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "name-count index file {} has unsupported magic for kind {}",
                path.display(),
                kind.key(),
            )));
        }
        let record_count = read_u64_le(&mmap, 8)? as usize;
        let blob_offset = read_u64_le(&mmap, 16)? as usize;
        let blob_len = read_u64_le(&mmap, 24)? as usize;
        let records_end = NAME_COUNTS_INDEX_HEADER_LEN
            .checked_add(
                record_count
                    .checked_mul(NAME_COUNTS_INDEX_RECORD_LEN)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyOverflowError::new_err(format!(
                            "name-count index file {} has too many records",
                            path.display()
                        ))
                    })?,
            )
            .ok_or_else(|| {
                pyo3::exceptions::PyOverflowError::new_err(format!(
                    "name-count index file {} record section overflows",
                    path.display()
                ))
            })?;
        let blob_end = blob_offset.checked_add(blob_len).ok_or_else(|| {
            pyo3::exceptions::PyOverflowError::new_err(format!(
                "name-count index file {} blob section overflows",
                path.display()
            ))
        })?;
        if blob_offset < records_end || blob_end > mmap.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "name-count index file {} has invalid record/blob offsets",
                path.display()
            )));
        }
        if record_count > 1 {
            let read_pair = |index: usize| {
                let offset = NAME_COUNTS_INDEX_HEADER_LEN + index * NAME_COUNTS_INDEX_RECORD_LEN;
                (
                    read_u64_le_unchecked(&mmap, offset),
                    read_u64_le_unchecked(&mmap, offset + 8),
                )
            };
            let mut previous_index = 0usize;
            let mut previous_pair = read_pair(0);
            for index in 1..record_count {
                let pair = read_pair(index);
                if pair < previous_pair {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "name-count index file {} is not sorted for kind {}: record {} {:?} follows record {} {:?}",
                        path.display(),
                        kind.key(),
                        index,
                        pair,
                        previous_index,
                        previous_pair
                    )));
                }
                previous_index = index;
                previous_pair = pair;
            }
        }
        Ok(Self {
            mmap,
            record_count,
            blob_offset,
            blob_len,
        })
    }

    fn record_offset(&self, index: usize) -> usize {
        NAME_COUNTS_INDEX_HEADER_LEN + index * NAME_COUNTS_INDEX_RECORD_LEN
    }

    fn record_hash_pair(&self, index: usize) -> (u64, u64) {
        let offset = self.record_offset(index);
        (
            read_u64_le_unchecked(&self.mmap, offset),
            read_u64_le_unchecked(&self.mmap, offset + 8),
        )
    }

    fn get(&self, kind: RawNameCountKind, name: &str) -> Option<f64> {
        let name_bytes = name.as_bytes();
        let (hash_1, hash_2) = name_counts_index_hashes(kind, name_bytes);
        let mut lower = 0usize;
        let mut upper = self.record_count;
        while lower < upper {
            let middle = lower + (upper - lower) / 2;
            let (middle_hash_1, middle_hash_2) = self.record_hash_pair(middle);
            if middle_hash_1 < hash_1 || (middle_hash_1 == hash_1 && middle_hash_2 < hash_2) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }

        let mut index = lower;
        while index < self.record_count {
            let (record_hash_1, record_hash_2) = self.record_hash_pair(index);
            if record_hash_1 != hash_1 || record_hash_2 != hash_2 {
                break;
            }
            let offset = self.record_offset(index);
            let name_offset = read_u64_le_unchecked(&self.mmap, offset + 16) as usize;
            let name_len = read_u32_le_unchecked(&self.mmap, offset + 24) as usize;
            if name_offset
                .checked_add(name_len)
                .map_or(false, |end| end <= self.blob_len)
            {
                let start = self.blob_offset + name_offset;
                let end = start + name_len;
                if &self.mmap[start..end] == name_bytes {
                    return Some(read_f64_le_unchecked(&self.mmap, offset + 32));
                }
            }
            index += 1;
        }
        None
    }
}

impl RawNameCountMaps {
    fn from_index(index: RawNameCountIndex) -> Self {
        Self {
            first: HashMap::new(),
            last: HashMap::new(),
            first_last: HashMap::new(),
            last_first_initial: HashMap::new(),
            index: Some(index),
        }
    }

    fn has_data(&self) -> bool {
        self.index.is_some()
            || !self.first.is_empty()
            || !self.last.is_empty()
            || !self.first_last.is_empty()
            || !self.last_first_initial.is_empty()
    }

    fn get(&self, kind: RawNameCountKind, name: &str) -> Option<f64> {
        if let Some(index) = self.index.as_ref() {
            return index.get(kind, name);
        }
        match kind {
            RawNameCountKind::First => self.first.get(name),
            RawNameCountKind::Last => self.last.get(name),
            RawNameCountKind::FirstLast => self.first_last.get(name),
            RawNameCountKind::LastFirstInitial => self.last_first_initial.get(name),
        }
        .copied()
    }
}

fn name_counts_index_manifest_path(
    index_dir: &Path,
    files: &serde_json::Map<String, serde_json::Value>,
    kind: &str,
) -> PyResult<PathBuf> {
    let path_value = files
        .get(kind)
        .and_then(|entry| entry.get("path"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "name-count index manifest {} is missing files.{}.path",
                index_dir.join("manifest.json").display(),
                kind
            ))
        })?;
    let raw_path = PathBuf::from(path_value);
    let resolved = if raw_path.is_absolute() {
        raw_path
    } else {
        index_dir.join(raw_path)
    };
    if !resolved.exists() {
        return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "name-count index manifest {} points to missing file {}",
            index_dir.join("manifest.json").display(),
            resolved.display()
        )));
    }
    Ok(resolved)
}

fn read_name_counts_index_manifest(index_dir: &Path) -> PyResult<RawNameCountIndexPaths> {
    let manifest_path = index_dir.join("manifest.json");
    let manifest_text = fs::read_to_string(&manifest_path).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "failed to read name-count index manifest {}: {}",
            manifest_path.display(),
            err
        ))
    })?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_text).map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "failed to parse name-count index manifest {}: {}",
            manifest_path.display(),
            err
        ))
    })?;
    let files = manifest
        .get("files")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "name-count index manifest {} is missing files",
                manifest_path.display()
            ))
        })?;
    Ok(RawNameCountIndexPaths {
        first: name_counts_index_manifest_path(index_dir, files, "first")?,
        last: name_counts_index_manifest_path(index_dir, files, "last")?,
        first_last: name_counts_index_manifest_path(index_dir, files, "first_last")?,
        last_first_initial: name_counts_index_manifest_path(
            index_dir,
            files,
            "last_first_initial",
        )?,
    })
}

fn resolve_name_counts_index_paths(path: &str) -> PyResult<RawNameCountIndexPaths> {
    let direct = PathBuf::from(path);
    let nested = direct.join("name_counts_index");
    for index_dir in [&direct, &nested] {
        if index_dir.join("manifest.json").exists() {
            return read_name_counts_index_manifest(index_dir);
        }
    }
    Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
        "name-count index path {} does not contain manifest.json",
        path
    )))
}

fn name_counts_index_hashes(kind: RawNameCountKind, name_bytes: &[u8]) -> (u64, u64) {
    let first = fnv64(name_bytes);
    let mut second = FNV_OFFSET;
    second = fnv64_update(second, NAME_COUNTS_INDEX_HASH_DOMAIN);
    second = fnv64_update(second, kind.key().as_bytes());
    second = fnv64_update(second, b"\0");
    second = fnv64_update(second, name_bytes);
    (first, second)
}

fn read_u64_le(bytes: &[u8], offset: usize) -> PyResult<u64> {
    let end = offset.checked_add(8).ok_or_else(|| {
        pyo3::exceptions::PyOverflowError::new_err("u64 offset overflows while reading index")
    })?;
    let slice = bytes.get(offset..end).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("u64 offset is outside name-count index")
    })?;
    Ok(read_u64_le_unchecked(slice, 0))
}

#[inline(always)]
fn read_u64_le_unchecked(bytes: &[u8], offset: usize) -> u64 {
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(raw)
}

#[inline(always)]
fn read_u32_le_unchecked(bytes: &[u8], offset: usize) -> u32 {
    let mut raw = [0u8; 4];
    raw.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(raw)
}

#[inline(always)]
fn read_f64_le_unchecked(bytes: &[u8], offset: usize) -> f64 {
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[offset..offset + 8]);
    f64::from_le_bytes(raw)
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
enum ClusterId {
    Int(i64),
    Str(String),
}

type PaperId = String;

#[derive(Clone, Serialize, Deserialize)]
struct SignatureData {
    // Python author_info_first_normalized_without_apostrophe.
    first: Option<String>,
    middle: Option<String>,
    last_normalized: Option<String>,
    orcid: Option<String>,
    email: Option<String>,
    affiliations: Option<CounterData>,
    coauthor_blocks: Option<HashSet<String>>,
    coauthor_ngrams: Option<CounterData>,
    coauthors: Option<HashSet<String>>,
    position: i64,
    paper_id: PaperId,
    name_counts: Option<NameCountsData>,
    // Same canonical first-name field used by Python name_text_features.
    adv_name: Option<String>,
}

impl SignatureData {
    fn first_without_apostrophe(&self) -> Option<&str> {
        self.first.as_deref()
    }

    fn adv_name_for_features(&self) -> Option<&str> {
        self.adv_name.as_deref()
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct PaperData {
    venue_ngrams: Option<CounterData>,
    title_words: Option<CounterData>,
    title_chars: Option<CounterData>,
    ref_authors: Option<CounterData>,
    ref_titles: Option<CounterData>,
    ref_venues: Option<CounterData>,
    ref_blocks: Option<CounterData>,
    ref_details_present: bool,
    references: HashSet<PaperId>,
    year: Option<i64>,
    has_abstract: bool,
    predicted_language: Option<String>,
    is_reliable: bool,
    journal_ngrams: Option<CounterData>,
    specter: Option<Vec<f32>>,
    #[serde(default)]
    specter_norm: Option<f64>,
}

#[derive(Clone)]
struct StageSignatureInput {
    sig_id: String,
    paper_id: PaperId,
    raw_first: String,
    raw_middle: String,
    raw_last: String,
    email: Option<String>,
    position: i64,
    affiliation_values: Vec<String>,
    orcid: Option<String>,
}

#[derive(Clone)]
struct StagePaperInput {
    paper_id: PaperId,
    raw_title: String,
    raw_venue: String,
    raw_journal: String,
    raw_authors: Vec<(i64, String)>,
    year: Option<i64>,
    has_abstract: bool,
    predicted_language: Option<String>,
    is_reliable: bool,
}

#[derive(Clone)]
struct StagePaperPreprocessed {
    authors: Vec<(i64, String)>,
    year: Option<i64>,
    has_abstract: bool,
    predicted_language: Option<String>,
    is_reliable: bool,
    title_words: Option<CounterData>,
    title_chars: Option<CounterData>,
    venue_ngrams: Option<CounterData>,
    journal_ngrams: Option<CounterData>,
}

#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
struct RustFeaturizer {
    signatures: HashMap<String, SignatureData>,
    #[serde(default)]
    signature_ids: Vec<String>,
    papers: HashMap<PaperId, PaperData>,
    name_tuples: HashMap<String, HashSet<String>>,
    cluster_seeds_disallow: HashSet<(String, String)>,
    cluster_seeds_require: HashMap<String, ClusterId>,
    compute_reference_features: bool,
    cluster_seed_require_value: f64,
    cluster_seed_disallow_value: f64,
    #[serde(skip)]
    json_ingest_telemetry: Option<JsonIngestTelemetry>,
    #[serde(skip)]
    cached_signature_id_order: OnceLock<Vec<String>>,
    #[serde(skip)]
    cluster_seeds_disallow_index: OnceLock<HashMap<String, HashSet<String>>>,
}

#[derive(Clone)]
struct RetrievalSummaryData {
    component_key: String,
    size: usize,
    first_name_counts: Vec<(String, f32)>,
    middle_initial_counts: Option<CounterData>,
    coauthor_counts: Option<CounterData>,
    non_mega_coauthor_counts: Option<CounterData>,
    affiliation_counts: Option<CounterData>,
    venue_counts: Option<CounterData>,
    title_counts: Option<CounterData>,
    max_paper_author_count: usize,
    year_min: Option<i64>,
    year_max: Option<i64>,
    year_mean: Option<f64>,
    orcid_hashes: Vec<u64>,
    specter_centroid: Option<Vec<f32>>,
    specter_centroid_norm: Option<f64>,
    exemplar_vectors: Vec<Vec<f32>>,
    exemplar_norms: Vec<f64>,
}

#[derive(Clone, Copy)]
struct RetrievalQueryTerm {
    hash: u64,
    token_count: u8,
}

#[derive(Clone)]
struct RetrievalQueryData {
    first: String,
    has_full_first: bool,
    middle_initial_hashes: Vec<u64>,
    coauthor_hashes: Vec<u64>,
    coauthor_terms: Vec<RetrievalQueryTerm>,
    affiliation_hashes: Vec<u64>,
    affiliation_terms: Vec<RetrievalQueryTerm>,
    venue_hashes: Vec<u64>,
    title_hashes: Vec<u64>,
    year: Option<i64>,
    orcid_hash: Option<u64>,
    specter: Option<Arc<Vec<f32>>>,
    specter_norm: Option<f64>,
}

#[derive(Clone, Copy)]
struct RetrievalHybridWeights {
    centroid: f64,
    coauthor: f64,
    affiliation: f64,
    middle: f64,
    first_name: f64,
}

const RETRIEVAL_FEATURE_ORDER: [&str; 5] = [
    "centroid",
    "coauthor",
    "affiliation",
    "middle",
    "first_name",
];
const DEFAULT_HYBRID_CENTROID_POLICY_NAME: &str = "h_wang_any_input_v2";
const DEFAULT_HYBRID_CENTROID_WEIGHTS: [f64; 5] =
    [0.527232, 0.223412, 0.146909, 0.009439, 0.093007];
const DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS: [f64; 5] =
    [0.520012, 0.220264, 0.109278, 0.150447, 0.0];
const DEFAULT_HYBRID_EXEMPLAR_4_WEIGHTS: [f64; 5] = [0.40, 0.23, 0.12, 0.05, 0.07];
const INCREMENTAL_LINKING_PAIR_PLAN_ROW_SIGNALS: [&str; 1] = ["row_orcid_match"];
const RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE: f64 = -0.25;
const RETRIEVAL_YEAR_SCORE_DECAY_YEARS: f64 = 15.0;
const RETRIEVAL_YEAR_SCORE_RANGE_GAP: i64 = 10;
const RETRIEVAL_YEAR_SCORE_RANGE_PENALTY: f64 = 0.15;
const RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP: i64 = 35;
const RETRIEVAL_MEGA_AUTHOR_THRESHOLD: usize = 50;

impl RetrievalHybridWeights {
    fn from_array(weights: [f64; 5]) -> Self {
        Self {
            centroid: weights[0],
            coauthor: weights[1],
            affiliation: weights[2],
            middle: weights[3],
            first_name: weights[4],
        }
    }
}

fn filter_excluded_candidate_indices(
    indices: Vec<usize>,
    excluded_candidate_indices: Option<&HashSet<usize>>,
) -> Vec<usize> {
    let Some(excluded) = excluded_candidate_indices else {
        return indices;
    };
    if excluded.is_empty() {
        return indices;
    }
    indices
        .into_iter()
        .filter(|index| !excluded.contains(index))
        .collect()
}

fn default_candidate_indices(
    summary_count: usize,
    base_candidate_indices: Option<&[usize]>,
    excluded_candidate_indices: Option<&HashSet<usize>>,
) -> Vec<usize> {
    let values = base_candidate_indices.map_or_else(
        || (0..summary_count).collect::<Vec<_>>(),
        |base_values| base_values.to_vec(),
    );
    filter_excluded_candidate_indices(values, excluded_candidate_indices)
}

#[derive(Clone, Copy)]
enum RetrievalFirstNameMode {
    Prefix,
    ExactOnly,
    ExactThenPrefixHalf,
    PrefixLengthRatio,
    ExactThenPrefixLengthRatio,
}

#[derive(Clone, Copy)]
enum RetrievalSpecterMode {
    Centroid,
    ExemplarMax,
    CentroidExemplar50_50,
    CentroidExemplar25_75,
    CentroidExemplar75_25,
    MaxOfCentroidExemplar,
}

#[derive(Clone, Copy)]
struct RetrievalOverlapConfig {
    use_idf: bool,
    per_term_cap: Option<f64>,
    total_cap: Option<f64>,
    min_token_count: u8,
    unigram_weight: f64,
    multi_token_weight: f64,
}

#[derive(Clone, Copy)]
struct RetrievalExperimentalConfig {
    first_name_mode: RetrievalFirstNameMode,
    specter_mode: RetrievalSpecterMode,
    coauthor: RetrievalOverlapConfig,
    drop_candidate_mega_coauthors: bool,
    mega_coauthor_rescue_query_coverage: Option<f64>,
    mega_coauthor_rescue_min_shared_blocks: usize,
    affiliation: RetrievalOverlapConfig,
}

#[pyclass]
struct RustHybridCentroidRetriever {
    summaries: Vec<RetrievalSummaryData>,
    component_index_by_key: HashMap<String, usize>,
    coauthor_cluster_df: HashMap<u64, usize>,
    non_mega_coauthor_cluster_df: HashMap<u64, usize>,
    affiliation_cluster_df: HashMap<u64, usize>,
}

#[pyclass]
struct RustNameCompatibleSubblockSelector {
    signature_to_subblock: HashMap<String, String>,
    subblock_to_components: HashMap<String, Vec<String>>,
    subblock_tokens_by_subblock: HashMap<String, Vec<String>>,
    name_tuples: HashMap<String, HashSet<String>>,
}

#[derive(Default)]
struct RetrievalPairPlanQueryResult {
    row_query_signature_indices: Vec<u32>,
    row_component_keys: Vec<String>,
    row_retrieval_scores: Vec<f32>,
    row_retrieval_ranks: Vec<u16>,
    row_component_sizes: Vec<u32>,
    row_named_signature_counts: Vec<u32>,
    row_dominant_first_names: Vec<String>,
    row_candidate_year_min: Vec<i32>,
    row_candidate_year_max: Vec<i32>,
    row_candidate_year_range_missing: Vec<u8>,
    row_query_first_tokens: Vec<String>,
    row_query_years: Vec<i32>,
    row_query_year_missing: Vec<u8>,
    row_query_has_affiliations: Vec<u8>,
    row_query_has_coauthors: Vec<u8>,
    row_orcid_match: Vec<u8>,
    row_middle_initial_compatibility: Vec<f32>,
    row_affiliation_overlap: Vec<f32>,
    row_coauthor_overlap: Vec<f32>,
    row_venue_overlap: Vec<f32>,
    row_year_compatibility: Vec<f32>,
    row_title_overlap: Vec<f32>,
    row_specter_centroid_similarity: Vec<f32>,
    row_specter_exemplar_similarity: Vec<f32>,
    right_signature_indices_by_row: Vec<Vec<u32>>,
}

fn year_signal_value(year: Option<i64>, field_name: &str) -> Result<(i32, u8), String> {
    let Some(value) = year else {
        return Ok((i32::MIN, 1));
    };
    let converted = i32::try_from(value)
        .map_err(|_| format!("{field_name} is outside the supported i32 range: {value}"))?;
    if converted == i32::MIN {
        return Err(format!(
            "{field_name} uses reserved missing-year sentinel value: {value}"
        ));
    }
    Ok((converted, 0))
}

struct RetrievalCandidateSelection {
    indices: Vec<usize>,
    return_all: bool,
}

#[derive(Clone, Copy)]
struct PairAggregateRowRange {
    row_offset: usize,
    start: usize,
    stop: usize,
}

struct PairAggregateBuffers {
    counts: Vec<u32>,
    valid_counts: Vec<u64>,
    sums: Vec<f64>,
    mins: Vec<f64>,
    maxs: Vec<f64>,
}

fn checked_retrieved_row_index(base_row_index: u32, local_row_index: usize) -> PyResult<u32> {
    let local_row_index = u32::try_from(local_row_index).map_err(|_| {
        pyo3::exceptions::PyOverflowError::new_err("retrieved candidate row count exceeds u32")
    })?;
    base_row_index.checked_add(local_row_index).ok_or_else(|| {
        pyo3::exceptions::PyOverflowError::new_err("retrieved candidate row count exceeds u32")
    })
}

fn validate_positive_top_k(top_k: usize) -> PyResult<()> {
    if top_k == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "top_k must be positive",
        ));
    }
    Ok(())
}

fn validate_retrieval_rank_top_k(top_k: usize) -> PyResult<()> {
    validate_positive_top_k(top_k)?;
    if top_k > u16::MAX as usize {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "top_k must be <= {} because retrieval_ranks are stored as uint16",
            u16::MAX
        )));
    }
    Ok(())
}

struct LinkerPairDistanceAccumulator {
    counts: Vec<u32>,
    sums: Vec<f64>,
    mins: Vec<f64>,
    top_distances: Vec<f64>,
    hard_disallow_pair_count: u64,
}

impl LinkerPairDistanceAccumulator {
    fn new(row_count: usize) -> Self {
        Self {
            counts: vec![0_u32; row_count],
            sums: vec![0.0_f64; row_count],
            mins: vec![f64::INFINITY; row_count],
            top_distances: vec![f64::INFINITY; row_count * 5],
            hard_disallow_pair_count: 0,
        }
    }

    fn merge_from(&mut self, other: Self) {
        for row in 0..self.counts.len() {
            self.counts[row] = self.counts[row].saturating_add(other.counts[row]);
            self.sums[row] += other.sums[row];
            if other.mins[row] < self.mins[row] {
                self.mins[row] = other.mins[row];
            }
            let top_start = row * 5;
            for value in other.top_distances[top_start..top_start + 5].iter() {
                RustFeaturizer::update_top5_distance(
                    &mut self.top_distances[top_start..top_start + 5],
                    *value,
                );
            }
        }
        self.hard_disallow_pair_count = self
            .hard_disallow_pair_count
            .saturating_add(other.hard_disallow_pair_count);
    }
}

impl RustHybridCentroidRetriever {
    fn default_hybrid_weights_for_query(query_data: &RetrievalQueryData) -> RetrievalHybridWeights {
        if query_data.has_full_first {
            RetrievalHybridWeights::from_array(DEFAULT_HYBRID_CENTROID_WEIGHTS)
        } else {
            RetrievalHybridWeights::from_array(DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS)
        }
    }

    fn default_experimental_config_for_query(
        query_data: &RetrievalQueryData,
    ) -> RetrievalExperimentalConfig {
        RetrievalExperimentalConfig {
            first_name_mode: if query_data.has_full_first {
                RetrievalFirstNameMode::ExactThenPrefixHalf
            } else {
                RetrievalFirstNameMode::Prefix
            },
            specter_mode: RetrievalSpecterMode::MaxOfCentroidExemplar,
            coauthor: RetrievalOverlapConfig {
                use_idf: true,
                per_term_cap: Some(0.35),
                ..default_overlap_config()
            },
            drop_candidate_mega_coauthors: true,
            mega_coauthor_rescue_query_coverage: Some(0.995),
            mega_coauthor_rescue_min_shared_blocks: 3,
            affiliation: RetrievalOverlapConfig {
                use_idf: true,
                ..default_overlap_config()
            },
        }
    }

    fn summary_for_candidate_index<'a>(
        &'a self,
        idx: usize,
        override_index: Option<usize>,
        override_summary: Option<&'a RetrievalSummaryData>,
    ) -> &'a RetrievalSummaryData {
        match (override_index, override_summary) {
            (Some(replaced_idx), Some(replaced_summary)) if idx == replaced_idx => replaced_summary,
            _ => &self.summaries[idx],
        }
    }

    fn compare_scored_candidates(
        &self,
        left: &(usize, f32),
        right: &(usize, f32),
        override_index: Option<usize>,
        override_summary: Option<&RetrievalSummaryData>,
    ) -> Ordering {
        let left_component_key = self
            .summary_for_candidate_index(left.0, override_index, override_summary)
            .component_key
            .as_str();
        let right_component_key = self
            .summary_for_candidate_index(right.0, override_index, override_summary)
            .component_key
            .as_str();
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_component_key.cmp(right_component_key))
    }

    fn keep_sorted_top_k_scored_candidates(
        &self,
        scored: &mut Vec<(usize, f32)>,
        top_k: usize,
        override_index: Option<usize>,
        override_summary: Option<&RetrievalSummaryData>,
    ) {
        if top_k == 0 {
            scored.clear();
            return;
        }
        let limit = top_k.min(scored.len());
        if limit < scored.len() {
            scored.select_nth_unstable_by(limit - 1, |left, right| {
                self.compare_scored_candidates(left, right, override_index, override_summary)
            });
            scored.truncate(limit);
        }
        scored.sort_unstable_by(|left, right| {
            self.compare_scored_candidates(left, right, override_index, override_summary)
        });
    }

    fn score_top_k_candidate_indices_default_inner(
        &self,
        query_data: &RetrievalQueryData,
        candidate_indices: &[usize],
        top_k: usize,
        override_index: Option<usize>,
        override_summary: Option<&RetrievalSummaryData>,
    ) -> Vec<(usize, f32)> {
        let weights = Self::default_hybrid_weights_for_query(query_data);
        let config = Self::default_experimental_config_for_query(query_data);
        let mut scored = candidate_indices
            .iter()
            .map(|idx| {
                let summary =
                    self.summary_for_candidate_index(*idx, override_index, override_summary);
                (
                    *idx,
                    score_experimental_hybrid_centroid_query(
                        query_data,
                        summary,
                        weights,
                        config,
                        &self.coauthor_cluster_df,
                        &self.non_mega_coauthor_cluster_df,
                        &self.affiliation_cluster_df,
                        self.summaries.len(),
                    ),
                )
            })
            .collect::<Vec<_>>();
        self.keep_sorted_top_k_scored_candidates(
            &mut scored,
            top_k,
            override_index,
            override_summary,
        );
        scored
    }

    fn scored_candidates_to_keys_scores(
        &self,
        scored: Vec<(usize, f32)>,
        override_index: Option<usize>,
        override_summary: Option<&RetrievalSummaryData>,
    ) -> (Vec<String>, Vec<f32>) {
        let component_keys = scored
            .iter()
            .map(|(idx, _)| {
                self.summary_for_candidate_index(*idx, override_index, override_summary)
                    .component_key
                    .clone()
            })
            .collect();
        let scores = scored.iter().map(|(_, score)| *score).collect();
        (component_keys, scores)
    }

    fn hard_filtered_candidate_indices_for_query(
        &self,
        query_data: &RetrievalQueryData,
        mut candidate_indices: Vec<usize>,
    ) -> RetrievalCandidateSelection {
        if let Some(selection) =
            self.orcid_candidate_selection_for_query(query_data, candidate_indices.iter().copied())
        {
            return selection;
        }

        let middle_filtered: Vec<usize> = candidate_indices
            .iter()
            .copied()
            .filter(|idx| {
                !has_middle_initial_conflict(
                    &query_data.middle_initial_hashes,
                    &self.summaries[*idx].middle_initial_counts,
                )
            })
            .collect();
        if !middle_filtered.is_empty() {
            candidate_indices = middle_filtered;
        }

        let year_filtered: Vec<usize> = candidate_indices
            .iter()
            .copied()
            .filter(|idx| {
                !has_impossible_year_conflict(
                    query_data.year,
                    &self.summaries[*idx],
                    RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP,
                )
            })
            .collect();
        if !year_filtered.is_empty() {
            candidate_indices = year_filtered;
        }

        RetrievalCandidateSelection {
            indices: candidate_indices,
            return_all: false,
        }
    }

    fn orcid_candidate_selection_for_query<I>(
        &self,
        query_data: &RetrievalQueryData,
        candidate_indices: I,
    ) -> Option<RetrievalCandidateSelection>
    where
        I: IntoIterator<Item = usize>,
    {
        let orcid_hash = query_data.orcid_hash?;
        let orcid_matches: Vec<usize> = candidate_indices
            .into_iter()
            .filter(|idx| contains_hashed_value(&self.summaries[*idx].orcid_hashes, orcid_hash))
            .collect();
        if orcid_matches.is_empty() {
            None
        } else {
            Some(RetrievalCandidateSelection {
                indices: orcid_matches,
                return_all: true,
            })
        }
    }

    fn candidate_indices_for_pair_plan_query(
        &self,
        query_data: &RetrievalQueryData,
        base_candidate_indices: Option<&[usize]>,
        excluded_candidate_indices: Option<&HashSet<usize>>,
        query_signature_id: Option<&str>,
        selector: Option<&RustNameCompatibleSubblockSelector>,
        global_backfill_count: usize,
        allow_global_orcid_override: bool,
    ) -> RetrievalCandidateSelection {
        if allow_global_orcid_override {
            if let Some(selection) =
                self.orcid_candidate_selection_for_query(query_data, 0..self.summaries.len())
            {
                return selection;
            }
        }
        let selected = if query_data.has_full_first {
            match (query_signature_id, selector) {
                (Some(signature_id), Some(selector)) => selector
                    .select_candidate_indices_for_summaries(
                        signature_id,
                        &query_data.first,
                        &self.summaries,
                        base_candidate_indices,
                        global_backfill_count,
                    ),
                _ => None,
            }
        } else {
            None
        };
        let candidate_indices = match selected {
            Some(indices) => {
                let filtered =
                    filter_excluded_candidate_indices(indices, excluded_candidate_indices);
                if filtered.is_empty() {
                    default_candidate_indices(
                        self.summaries.len(),
                        base_candidate_indices,
                        excluded_candidate_indices,
                    )
                } else {
                    filtered
                }
            }
            None => default_candidate_indices(
                self.summaries.len(),
                base_candidate_indices,
                excluded_candidate_indices,
            ),
        };
        self.hard_filtered_candidate_indices_for_query(query_data, candidate_indices)
    }

    fn build_pair_plan_query_result(
        &self,
        current_query: &RetrievalQueryData,
        row_query_first_token: &str,
        query_signature_index: u32,
        base_candidate_indices: Option<&[usize]>,
        excluded_candidate_indices: Option<&HashSet<usize>>,
        query_signature_id: Option<&str>,
        component_member_indices: &HashMap<String, Vec<u32>>,
        top_k: usize,
        selector: Option<&RustNameCompatibleSubblockSelector>,
        global_backfill_count: usize,
        allow_global_orcid_override: bool,
    ) -> Result<RetrievalPairPlanQueryResult, String> {
        let selection = self.candidate_indices_for_pair_plan_query(
            current_query,
            base_candidate_indices,
            excluded_candidate_indices,
            query_signature_id,
            selector,
            global_backfill_count,
            allow_global_orcid_override,
        );
        if selection.indices.is_empty() {
            return Ok(RetrievalPairPlanQueryResult::default());
        }
        let effective_top_k = if selection.return_all {
            selection.indices.len()
        } else {
            top_k
        };
        let scored = self.score_top_k_candidate_indices_default_inner(
            current_query,
            &selection.indices,
            effective_top_k,
            None,
            None,
        );

        let mut result = RetrievalPairPlanQueryResult::default();
        let (query_year, query_year_missing) = year_signal_value(current_query.year, "query year")?;
        let query_has_affiliations = u8::from(!current_query.affiliation_hashes.is_empty());
        let query_has_coauthors = u8::from(!current_query.coauthor_hashes.is_empty());
        for (rank_offset, (summary_index, score)) in scored.iter().enumerate() {
            let summary = &self.summaries[*summary_index];
            let Some(member_indices) = component_member_indices.get(&summary.component_key) else {
                return Err(format!(
                    "Missing component members for retrieved component_key: {}",
                    summary.component_key
                ));
            };
            let chooser_features = chooser_summary_features(current_query, summary);
            let mut dominant_first_name = "";
            let mut dominant_first_count = 0.0f32;
            let mut named_signature_count = 0.0f32;
            for (first_name, count) in summary.first_name_counts.iter() {
                named_signature_count += *count;
                if *count > dominant_first_count
                    || (*count == dominant_first_count && first_name.as_str() > dominant_first_name)
                {
                    dominant_first_name = first_name.as_str();
                    dominant_first_count = *count;
                }
            }

            result
                .row_query_signature_indices
                .push(query_signature_index);
            result
                .row_component_keys
                .push(summary.component_key.clone());
            result.row_retrieval_scores.push(*score);
            result
                .row_retrieval_ranks
                .push((rank_offset + 1).min(u16::MAX as usize) as u16);
            result
                .row_component_sizes
                .push(summary.size.min(u32::MAX as usize) as u32);
            result
                .row_named_signature_counts
                .push(named_signature_count.round().max(0.0) as u32);
            result
                .row_dominant_first_names
                .push(dominant_first_name.to_string());
            let (candidate_year_min, candidate_year_min_missing) =
                year_signal_value(summary.year_min, "candidate year_min")?;
            let (candidate_year_max, candidate_year_max_missing) =
                year_signal_value(summary.year_max, "candidate year_max")?;
            result.row_candidate_year_min.push(candidate_year_min);
            result.row_candidate_year_max.push(candidate_year_max);
            result.row_candidate_year_range_missing.push(u8::from(
                candidate_year_min_missing != 0 || candidate_year_max_missing != 0,
            ));
            result
                .row_query_first_tokens
                .push(row_query_first_token.to_string());
            result.row_query_years.push(query_year);
            result.row_query_year_missing.push(query_year_missing);
            result
                .row_query_has_affiliations
                .push(query_has_affiliations);
            result.row_query_has_coauthors.push(query_has_coauthors);
            result
                .row_orcid_match
                .push(u8::from(current_query.orcid_hash.is_some_and(
                    |orcid_hash| contains_hashed_value(&summary.orcid_hashes, orcid_hash),
                )));
            result
                .row_middle_initial_compatibility
                .push(chooser_features[0]);
            result.row_affiliation_overlap.push(chooser_features[1]);
            result.row_coauthor_overlap.push(chooser_features[2]);
            result.row_venue_overlap.push(chooser_features[3]);
            result.row_year_compatibility.push(chooser_features[4]);
            result.row_title_overlap.push(chooser_features[5]);
            result
                .row_specter_centroid_similarity
                .push(chooser_features[6]);
            result
                .row_specter_exemplar_similarity
                .push(chooser_features[7]);
            result
                .right_signature_indices_by_row
                .push(member_indices.clone());
        }
        Ok(result)
    }

    fn extract_candidate_indices_by_query_signature_id(
        &self,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<HashMap<String, Vec<usize>>> {
        let candidate_keys_by_query = extract_string_vec_map(obj)?;
        let mut out = HashMap::with_capacity(candidate_keys_by_query.len());
        for (query_signature_id, component_keys) in candidate_keys_by_query {
            let mut indices = Vec::with_capacity(component_keys.len());
            for component_key in component_keys {
                let Some(candidate_index) = self.component_index_by_key.get(&component_key) else {
                    return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                        "Unknown component_key for RustHybridCentroidRetriever query window: {component_key}"
                    )));
                };
                indices.push(*candidate_index);
            }
            out.insert(query_signature_id, indices);
        }
        Ok(out)
    }

    fn score_top_k_candidate_indices_experimental(
        &self,
        py: Python<'_>,
        query_data: &RetrievalQueryData,
        candidate_indices: &[usize],
        top_k: usize,
        num_threads: Option<usize>,
        override_index: Option<usize>,
        override_summary: Option<&RetrievalSummaryData>,
        weights: RetrievalHybridWeights,
        config: RetrievalExperimentalConfig,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        if candidate_indices.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut scored: Vec<(usize, f32)> = py.allow_threads(|| {
            let compute = || {
                candidate_indices
                    .par_iter()
                    .map(|idx| {
                        let summary = self.summary_for_candidate_index(
                            *idx,
                            override_index,
                            override_summary,
                        );
                        (
                            *idx,
                            score_experimental_hybrid_centroid_query(
                                query_data,
                                summary,
                                weights,
                                config,
                                &self.coauthor_cluster_df,
                                &self.non_mega_coauthor_cluster_df,
                                &self.affiliation_cluster_df,
                                self.summaries.len(),
                            ),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        self.keep_sorted_top_k_scored_candidates(
            &mut scored,
            top_k,
            override_index,
            override_summary,
        );
        Ok(self.scored_candidates_to_keys_scores(scored, override_index, override_summary))
    }
}

#[derive(Clone, Default)]
struct JsonIngestTelemetry {
    json_parse_seconds: f64,
    paper_preprocess_seconds: f64,
    reference_counter_seconds: f64,
    signature_preprocess_seconds: f64,
    cluster_seed_seconds: f64,
    missing_specter_paper_count: usize,
    defaulted_name_count_signature_count: usize,
    defaulted_name_count_first_count: usize,
    defaulted_name_count_first_last_count: usize,
    defaulted_name_count_last_count: usize,
    defaulted_name_count_last_first_initial_count: usize,
    defaulted_signature_author_position_count: usize,
    defaulted_paper_author_position_count: usize,
}

const RAYON_POOL_CACHE_MAX_ENTRIES: usize = 8;

static RAYON_POOL_CACHE: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();

fn rayon_pool_cache() -> &'static Mutex<HashMap<usize, Arc<rayon::ThreadPool>>> {
    RAYON_POOL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cached_rayon_pool(thread_count: usize) -> Option<Arc<rayon::ThreadPool>> {
    if thread_count == 0 {
        return None;
    }
    if let Ok(cache) = rayon_pool_cache().lock() {
        if let Some(pool) = cache.get(&thread_count) {
            return Some(Arc::clone(pool));
        }
    }

    let built_pool = ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .ok()?;
    let built_pool = Arc::new(built_pool);
    if let Ok(mut cache) = rayon_pool_cache().lock() {
        if cache.len() >= RAYON_POOL_CACHE_MAX_ENTRIES && !cache.contains_key(&thread_count) {
            if let Some(remove_key) = cache.keys().copied().find(|key| *key != thread_count) {
                cache.remove(&remove_key);
            }
        }
        let pooled = cache
            .entry(thread_count)
            .or_insert_with(|| Arc::clone(&built_pool));
        return Some(Arc::clone(pooled));
    }
    Some(built_pool)
}

fn install_with_optional_rayon_pool<T, F>(num_threads: Option<usize>, compute: F) -> T
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    if let Some(thread_count) = num_threads {
        let threads = thread_count.max(1);
        if let Some(pool) = cached_rayon_pool(threads) {
            return pool.install(compute);
        }
    }
    compute()
}

fn upper_triangle_total_pairs(block_size: usize) -> usize {
    block_size.saturating_mul(block_size.saturating_sub(1)) / 2
}

fn upper_triangle_pairs_for_range(
    block_size: usize,
    start_offset: usize,
    max_pairs: Option<usize>,
) -> PyResult<Vec<(usize, usize)>> {
    let total_pairs = upper_triangle_total_pairs(block_size);
    if start_offset > total_pairs {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "start_offset out of range: start_offset={} total_pairs={}",
            start_offset, total_pairs
        )));
    }
    let remaining = total_pairs.saturating_sub(start_offset);
    let pair_count = max_pairs.unwrap_or(remaining).min(remaining);
    if pair_count == 0 {
        return Ok(Vec::new());
    }

    let mut local_i = 0usize;
    let mut offset_in_row = start_offset;
    while local_i + 1 < block_size {
        let row_pairs = block_size - local_i - 1;
        if offset_in_row < row_pairs {
            break;
        }
        offset_in_row -= row_pairs;
        local_i += 1;
    }
    if local_i + 1 >= block_size {
        return Ok(Vec::new());
    }
    let mut local_j = local_i + 1 + offset_in_row;
    let mut pairs = Vec::with_capacity(pair_count);
    for _ in 0..pair_count {
        pairs.push((local_i, local_j));
        local_j += 1;
        if local_j >= block_size {
            local_i += 1;
            local_j = local_i + 1;
        }
    }
    Ok(pairs)
}

fn json_ingest_telemetry_to_py(
    py: Python<'_>,
    telemetry: &JsonIngestTelemetry,
) -> PyResult<Py<PyDict>> {
    let stage_seconds = PyDict::new(py);
    stage_seconds.set_item("json_parse_seconds", telemetry.json_parse_seconds)?;
    stage_seconds.set_item(
        "paper_preprocess_seconds",
        telemetry.paper_preprocess_seconds,
    )?;
    stage_seconds.set_item(
        "reference_counter_seconds",
        telemetry.reference_counter_seconds,
    )?;
    stage_seconds.set_item(
        "signature_preprocess_seconds",
        telemetry.signature_preprocess_seconds,
    )?;
    stage_seconds.set_item("cluster_seed_seconds", telemetry.cluster_seed_seconds)?;

    let telemetry_dict = PyDict::new(py);
    telemetry_dict.set_item("stage_seconds", stage_seconds)?;
    let counts = PyDict::new(py);
    counts.set_item(
        "missing_specter_paper_count",
        telemetry.missing_specter_paper_count,
    )?;
    counts.set_item(
        "defaulted_name_count_signature_count",
        telemetry.defaulted_name_count_signature_count,
    )?;
    counts.set_item(
        "defaulted_name_count_first_count",
        telemetry.defaulted_name_count_first_count,
    )?;
    counts.set_item(
        "defaulted_name_count_first_last_count",
        telemetry.defaulted_name_count_first_last_count,
    )?;
    counts.set_item(
        "defaulted_name_count_last_count",
        telemetry.defaulted_name_count_last_count,
    )?;
    counts.set_item(
        "defaulted_name_count_last_first_initial_count",
        telemetry.defaulted_name_count_last_first_initial_count,
    )?;
    counts.set_item(
        "defaulted_signature_author_position_count",
        telemetry.defaulted_signature_author_position_count,
    )?;
    counts.set_item(
        "defaulted_paper_author_position_count",
        telemetry.defaulted_paper_author_position_count,
    )?;
    telemetry_dict.set_item("counts", counts)?;
    Ok(telemetry_dict.unbind())
}

fn extract_counter(obj: &Bound<'_, PyAny>) -> PyResult<Option<CounterData>> {
    if obj.is_none() {
        return Ok(None);
    }
    let dict = obj.downcast::<PyDict>()?;
    if dict.len() == 0 {
        return Ok(None);
    }
    let mut entries: Vec<(u64, f32)> = Vec::with_capacity(dict.len());
    let mut sum = 0.0f32;
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        let val: f64 = v.extract()?;
        let val32 = val as f32;
        sum += val32;
        entries.push((fnv64(key.as_bytes()), val32));
    }
    entries.sort_unstable_by_key(|e| e.0);
    Ok(Some(CounterData { entries, sum }))
}

fn extract_reference_details_counters(
    py: Python<'_>,
    ref_details_obj: &Bound<'_, PyAny>,
) -> PyResult<(
    Option<CounterData>,
    Option<CounterData>,
    Option<CounterData>,
    Option<CounterData>,
)> {
    let tuple = ref_details_obj.extract::<(PyObject, PyObject, PyObject, PyObject)>()?;
    Ok((
        extract_counter(&tuple.0.bind(py))?,
        extract_counter(&tuple.1.bind(py))?,
        extract_counter(&tuple.2.bind(py))?,
        extract_counter(&tuple.3.bind(py))?,
    ))
}

fn extract_optional_string_set(obj: &Bound<'_, PyAny>) -> PyResult<Option<HashSet<String>>> {
    if obj.is_none() {
        return Ok(None);
    }
    let mut out = HashSet::new();
    for item in PyIterator::from_object(obj)? {
        let v: String = item?.extract()?;
        out.insert(v);
    }
    if out.is_empty() {
        Ok(None)
    } else {
        Ok(Some(out))
    }
}

fn canonical_signature_pair_ref<'a>(a: &'a str, b: &'a str) -> (&'a str, &'a str) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn canonical_signature_pair_owned(a: String, b: String) -> (String, String) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn canonical_signature_pair_cloned(a: &str, b: &str) -> (String, String) {
    let (left, right) = canonical_signature_pair_ref(a, b);
    (left.to_string(), right.to_string())
}

fn extract_pair_set(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<(String, String)>> {
    if obj.is_none() {
        return Ok(HashSet::new());
    }
    let mut out = HashSet::new();
    for item in PyIterator::from_object(obj)? {
        let tuple = item?;
        let (a, b): (String, String) = tuple.extract()?;
        out.insert(canonical_signature_pair_owned(a, b));
    }
    Ok(out)
}

fn insert_name_tuple_alias(map: &mut HashMap<String, HashSet<String>>, a: String, b: String) {
    map.entry(a.clone())
        .or_insert_with(HashSet::new)
        .insert(b.clone());
    map.entry(b).or_insert_with(HashSet::new).insert(a);
}

fn extract_name_tuples_map(obj: &Bound<'_, PyAny>) -> PyResult<HashMap<String, HashSet<String>>> {
    if obj.is_none() {
        return Ok(HashMap::new());
    }
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    for item in PyIterator::from_object(obj)? {
        let tuple = item?;
        let (a, b): (String, String) = tuple.extract()?;
        insert_name_tuple_alias(&mut out, a, b);
    }
    Ok(out)
}

fn extract_cluster_seeds_require(obj: &Bound<'_, PyAny>) -> PyResult<HashMap<String, ClusterId>> {
    if obj.is_none() {
        return Ok(HashMap::new());
    }
    let dict = obj.downcast::<PyDict>()?;
    let mut out = HashMap::with_capacity(dict.len());
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        let val: ClusterId = if let Ok(i) = v.extract::<i64>() {
            ClusterId::Int(i)
        } else if let Ok(s) = v.extract::<String>() {
            ClusterId::Str(s)
        } else if let Ok(u) = v.extract::<u64>() {
            ClusterId::Int(u as i64)
        } else {
            ClusterId::Str(v.str()?.to_string())
        };
        out.insert(key, val);
    }
    Ok(out)
}

fn cluster_id_to_string(cluster_id: &ClusterId) -> String {
    match cluster_id {
        ClusterId::Int(value) => value.to_string(),
        ClusterId::Str(value) => value.clone(),
    }
}

fn extract_id_string(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(s);
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(i.to_string());
    }
    if let Ok(u) = obj.extract::<u64>() {
        return Ok(u.to_string());
    }
    Ok(obj.str()?.to_string())
}

fn extract_set_id_string(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<PaperId>> {
    if obj.is_none() {
        return Ok(HashSet::new());
    }
    let mut out = HashSet::new();
    for item in PyIterator::from_object(obj)? {
        let v = extract_id_string(&item?)?;
        out.insert(v);
    }
    Ok(out)
}

fn extract_string_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for item in PyIterator::from_object(obj)? {
        out.push(item?.extract()?);
    }
    Ok(out)
}

fn get_namedtuple_item_or_attr<'py>(
    obj: &Bound<'py, PyAny>,
    allow_tuple_fastpath: bool,
    tuple_index: usize,
    attr_name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if allow_tuple_fastpath {
        return obj.get_item(tuple_index).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "NamedTuple fast-path index out of range or object not indexable: index={} attr={}",
                tuple_index, attr_name
            ))
        });
    }
    obj.getattr(attr_name)
}

fn validate_namedtuple_fastpath_contract(
    sample_obj: &Bound<'_, PyAny>,
    required_fields: &[(usize, &str)],
    tuple_label: &str,
) -> PyResult<bool> {
    let fields_obj = match sample_obj.getattr("_fields") {
        Ok(fields_obj) => fields_obj,
        Err(_) => return Ok(false),
    };

    let field_names: Vec<String> = fields_obj.extract().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{} fast-path expected _fields to be a tuple/list of field names",
            tuple_label
        ))
    })?;

    let max_required_index = required_fields
        .iter()
        .map(|(index, _)| *index)
        .max()
        .unwrap_or(0);
    if sample_obj.get_item(max_required_index).is_err() || field_names.len() <= max_required_index {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} fast-path contract mismatch: object is not indexable to required max index {} or _fields len={}",
            tuple_label,
            max_required_index,
            field_names.len()
        )));
    }

    for (field_index, expected_name) in required_fields {
        let actual_name = field_names
            .get(*field_index)
            .map(|s| s.as_str())
            .unwrap_or("<missing>");
        if actual_name != *expected_name {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{} fast-path contract mismatch at index {}: expected '{}' got '{}'",
                tuple_label, field_index, expected_name, actual_name
            )));
        }
    }

    Ok(true)
}

fn validate_dict_namedtuple_fastpath_contract(
    rows: &Bound<'_, PyDict>,
    required_fields: &[(usize, &str)],
    tuple_label: &str,
) -> PyResult<bool> {
    if let Some((_, sample_obj)) = rows.iter().next() {
        return validate_namedtuple_fastpath_contract(&sample_obj, required_fields, tuple_label);
    }
    Ok(false)
}

fn extract_paper_authors_with_positions(obj: &Bound<'_, PyAny>) -> PyResult<Vec<(i64, String)>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for item in PyIterator::from_object(obj)? {
        let author_obj = item?;
        let (position, author_name) = if let Ok(author_tuple) = author_obj.downcast::<PyTuple>() {
            if author_tuple.len() >= 2 {
                let author_name: String = author_tuple.get_item(0)?.extract()?;
                let position: i64 = author_tuple.get_item(1)?.extract()?;
                (position, author_name)
            } else {
                let position: i64 = author_obj.getattr("position")?.extract()?;
                let author_name: String = author_obj.getattr("author_name")?.extract()?;
                (position, author_name)
            }
        } else {
            let position: i64 = author_obj.getattr("position")?.extract()?;
            let author_name: String = author_obj.getattr("author_name")?.extract()?;
            (position, author_name)
        };
        out.push((position, author_name));
    }
    Ok(out)
}

fn extract_required_string_set(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<String>> {
    let mut out = HashSet::new();
    for item in PyIterator::from_object(obj)? {
        out.insert(item?.extract()?);
    }
    Ok(out)
}

fn extract_affiliation_stopwords(py: Python<'_>) -> PyResult<HashSet<String>> {
    let text_module = py.import("s2and.text")?;
    let stopwords_obj = text_module.getattr("AFFILIATIONS_STOP_WORDS")?;
    extract_required_string_set(&stopwords_obj)
}

fn prefilter_affiliation_text(affiliations: &[String], stopwords: &HashSet<String>) -> String {
    if affiliations.is_empty() {
        return String::new();
    }
    let mut tokens: Vec<&str> = Vec::new();
    for word in affiliations
        .iter()
        .flat_map(|affiliation| affiliation.split_whitespace())
    {
        if !stopwords.contains(word) && py_len(word) > 1 {
            tokens.push(word);
        }
    }
    tokens.join(" ")
}

fn ensure_unidecode_for_text(
    unidecode_fn: &Bound<'_, PyAny>,
    text: &str,
    unidecode_char_map: &mut HashMap<char, String>,
) -> PyResult<()> {
    if text.is_ascii() {
        return Ok(());
    }
    for ch in text.chars() {
        if ch.is_ascii() || unidecode_char_map.contains_key(&ch) {
            continue;
        }
        let mapped: String = unidecode_fn.call1((ch.to_string(),))?.extract()?;
        unidecode_char_map.insert(ch, mapped);
    }
    Ok(())
}

fn normalize_ascii_text_compat(text: &str, special_case_apostrophes: bool) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut prev_space = true;
    for byte in text.bytes() {
        let lowered = byte.to_ascii_lowercase();
        if lowered.is_ascii_alphabetic() {
            normalized.push(lowered as char);
            prev_space = false;
        } else if special_case_apostrophes && lowered == b'\'' {
            continue;
        } else if !prev_space {
            normalized.push(' ');
            prev_space = true;
        }
    }
    while normalized.ends_with(' ') {
        normalized.pop();
    }
    normalized
}

fn normalize_text_compat_from_map(
    text: &str,
    special_case_apostrophes: bool,
    unidecode_char_map: &HashMap<char, String>,
) -> String {
    if text.is_empty() {
        return String::new();
    }
    if text.is_ascii() {
        return normalize_ascii_text_compat(text, special_case_apostrophes);
    }

    let mut transliterated = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii() {
            transliterated.push(ch.to_ascii_lowercase());
            continue;
        }
        let mapped = unidecode_char_map
            .get(&ch)
            .unwrap_or_else(|| panic!("missing unidecode mapping for non-ASCII character {ch:?}"));
        for mapped_ch in mapped.chars() {
            for lowered in mapped_ch.to_lowercase() {
                transliterated.push(lowered);
            }
        }
    }

    let source = if special_case_apostrophes {
        transliterated.replace('\'', "")
    } else {
        transliterated
    };
    let mut normalized = String::with_capacity(source.len());
    let mut prev_space = true;
    for ch in source.chars() {
        if ch.is_ascii_alphabetic() {
            normalized.push(ch);
            prev_space = false;
        } else if !prev_space {
            normalized.push(' ');
            prev_space = true;
        }
    }
    while normalized.ends_with(' ') {
        normalized.pop();
    }
    normalized
}

fn first_normalized_token_python_compat(
    first_normalized: &str,
    middle_normalized: &str,
    name_prefixes: &HashSet<String>,
) -> String {
    let joined = format!("{} {}", first_normalized, middle_normalized);
    let mut parts: Vec<String> = joined.split(' ').map(|token| token.to_string()).collect();
    if let Some(prefix) = parts.first() {
        if name_prefixes.contains(prefix) {
            parts.remove(0);
        }
    }
    parts.get(0).cloned().unwrap_or_default()
}

fn is_name_dash(ch: char) -> bool {
    matches!(
        ch,
        '-' | '\u{2010}'
            | '\u{2011}'
            | '\u{2012}'
            | '\u{2013}'
            | '\u{2014}'
            | '\u{2212}'
            | '\u{FE58}'
            | '\u{FE63}'
            | '\u{FF0D}'
    )
}

fn contains_name_dash(value: &str) -> bool {
    value.chars().any(is_name_dash)
}

fn split_first_middle_hyphen_aware_compat(
    first_raw: &str,
    middle_raw: &str,
    name_prefixes: &HashSet<String>,
    unidecode_char_map: &HashMap<char, String>,
) -> (String, String) {
    let has_dash_in_first = contains_name_dash(first_raw);
    let first_noapos = normalize_text_compat_from_map(first_raw, true, unidecode_char_map);
    let middle_norm = normalize_text_compat_from_map(middle_raw, false, unidecode_char_map);

    let mut f_parts: Vec<String> = first_noapos
        .split_whitespace()
        .map(|token| token.to_string())
        .collect();
    let m_parts: Vec<String> = middle_norm
        .split_whitespace()
        .map(|token| token.to_string())
        .collect();
    if let Some(prefix) = f_parts.first() {
        if name_prefixes.contains(prefix) {
            f_parts.remove(0);
        }
    }

    if f_parts.is_empty() {
        return (String::new(), m_parts.join(" "));
    }
    if has_dash_in_first {
        return (f_parts.join(" "), m_parts.join(" "));
    }
    let first = f_parts[0].clone();
    let middle = f_parts[1..]
        .iter()
        .chain(m_parts.iter())
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    (first, middle)
}

fn compute_block_compat(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }
    let name_parts: Vec<&str> = name.split(' ').collect();
    if name_parts.len() == 1 {
        return name_parts[0].to_string();
    }
    let Some(first_initial) = name_parts[0].chars().next() else {
        return String::new();
    };
    format!("{} {}", first_initial, name_parts[name_parts.len() - 1])
}

fn parse_fasttext_label(label: &str) -> String {
    label.rsplit("__").next().unwrap_or(label).to_string()
}

fn resolve_fasttext_model_path(py: Python<'_>) -> Option<String> {
    let consts = py.import("s2and.consts").ok()?;
    let fasttext_path: String = consts.getattr("FASTTEXT_PATH").ok()?.extract().ok()?;
    let file_cache = py.import("s2and.file_cache").ok()?;
    let cached_path = file_cache.getattr("cached_path").ok()?;
    cached_path.call1((fasttext_path,)).ok()?.extract().ok()
}

fn python_fasttext_loading_enabled(py: Python<'_>) -> bool {
    py.import("s2and.text")
        .and_then(|text_module| text_module.getattr("fasttext_loading_enabled"))
        .and_then(|enabled_fn| enabled_fn.call0())
        .and_then(|enabled| enabled.extract::<bool>())
        .unwrap_or(true)
}

fn emit_runtime_warning(py: Python<'_>, message: &str) {
    if let Ok(warnings) = py.import("warnings") {
        let _ = warnings.call_method1("warn", (message.to_string(),));
    }
}

struct LanguageDetectorCompat {
    fasttext: Option<FastText>,
}

impl LanguageDetectorCompat {
    fn new(py: Python<'_>) -> Self {
        if !python_fasttext_loading_enabled(py) {
            return Self { fasttext: None };
        }
        let fasttext = resolve_fasttext_model_path(py).and_then(|model_path| {
            let mut model = FastText::new();
            match model.load_model(&model_path) {
                Ok(()) => Some(model),
                Err(err) => {
                    let warning = format!(
                        "s2and_rust: failed to load fastText model at '{}' ({}); falling back to CLD2-only language detection.",
                        model_path, err
                    );
                    emit_runtime_warning(py, &warning);
                    eprintln!("{warning}");
                    None
                }
            }
        });
        Self { fasttext }
    }

    fn detect(&self, text: &str) -> (bool, bool, String) {
        if text.split_whitespace().count() <= 1 {
            return (false, false, "un".to_string());
        }

        let mut alpha_count = 0usize;
        let mut uppercase_count = 0usize;
        for ch in text.chars() {
            if ch.is_alphabetic() {
                alpha_count += 1;
                if ch.is_uppercase() {
                    uppercase_count += 1;
                }
            }
        }
        if alpha_count == 0 {
            return (false, false, "un".to_string());
        }

        let predicted_language_ft = if let Some(fasttext_model) = &self.fasttext {
            let uppercase_ratio = uppercase_count as f64 / alpha_count as f64;
            let mut fasttext_input = text.replace('\n', " ");
            if uppercase_ratio > 0.9 {
                fasttext_input = fasttext_input.to_lowercase();
            }
            match fasttext_model.predict(&fasttext_input, 1, 0.0) {
                Ok(predictions) => predictions
                    .first()
                    .map(|prediction| parse_fasttext_label(&prediction.label))
                    .unwrap_or_else(|| "un_ft".to_string()),
                Err(_) => "un_ft".to_string(),
            }
        } else {
            "un_ft".to_string()
        };

        let cld2_result = cld2_detect_language_ext(text, Cld2Format::Text, &Default::default());
        let mut predicted_language_2 = match cld2_result.scores[0].language {
            Some(lang) => lang.0.to_string(),
            None => "un_2".to_string(),
        };
        if predicted_language_2 == "un" {
            predicted_language_2 = "un_2".to_string();
        }

        let (predicted_language, is_reliable) =
            if predicted_language_ft == "un_ft" && predicted_language_2 == "un_2" {
                ("un".to_string(), false)
            } else if predicted_language_ft == "un_ft" {
                (predicted_language_2, true)
            } else if predicted_language_2 == "un_2" {
                (predicted_language_ft, true)
            } else if predicted_language_2 != predicted_language_ft {
                ("un".to_string(), false)
            } else {
                (predicted_language_2, true)
            };

        let is_english = predicted_language == "en";
        (is_reliable, is_english, predicted_language)
    }
}

fn counter_data_from_usize_map(counter_map: HashMap<String, usize>) -> Option<CounterData> {
    if counter_map.is_empty() {
        return None;
    }
    let mut entries: Vec<(u64, f32)> = counter_map
        .iter()
        .map(|(k, v)| (fnv64(k.as_bytes()), *v as f32))
        .collect();
    entries.sort_unstable_by_key(|e| e.0);
    let sum: f32 = entries.iter().map(|e| e.1).sum();
    Some(CounterData { entries, sum })
}

fn counter_data_from_hash_count_map(counter_map: HashMap<u64, usize>) -> Option<CounterData> {
    if counter_map.is_empty() {
        return None;
    }
    let mut entries: Vec<(u64, f32)> = counter_map
        .into_iter()
        .map(|(hash, count)| (hash, count as f32))
        .collect();
    entries.sort_unstable_by_key(|entry| entry.0);
    let sum: f32 = entries.iter().map(|entry| entry.1).sum();
    Some(CounterData { entries, sum })
}

fn increment_df_from_counter(counter: &Option<CounterData>, df_map: &mut HashMap<u64, usize>) {
    if let Some(counter_data) = counter.as_ref() {
        for (hash, _count) in counter_data.entries.iter() {
            *df_map.entry(*hash).or_insert(0) += 1;
        }
    }
}

fn hash_string_values(values: &HashSet<String>) -> Vec<u64> {
    let mut hashes: Vec<u64> = values.iter().map(|value| fnv64(value.as_bytes())).collect();
    hashes.sort_unstable();
    hashes.dedup();
    hashes
}

fn query_terms_from_values(values: &HashSet<String>) -> Vec<RetrievalQueryTerm> {
    let mut terms: Vec<RetrievalQueryTerm> = values
        .iter()
        .map(|value| RetrievalQueryTerm {
            hash: fnv64(value.as_bytes()),
            token_count: term_token_count(value),
        })
        .collect();
    terms.sort_unstable_by_key(|term| term.hash);
    terms.dedup_by_key(|term| term.hash);
    terms
}

fn term_set_from_normalized_text(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .filter(|token| !token.is_empty())
        .map(|token| token.to_string())
        .collect()
}

fn is_orcid_dash(ch: char) -> bool {
    matches!(
        ch,
        '-' | '\u{2010}'
            | '\u{2011}'
            | '\u{2012}'
            | '\u{2013}'
            | '\u{2014}'
            | '\u{2212}'
            | '\u{FE58}'
            | '\u{FE63}'
            | '\u{FF0D}'
    )
}

fn normalize_orcid_owned(value: &str) -> Option<String> {
    let chars: Vec<char> = value.trim().chars().collect();
    for start in 0..chars.len() {
        if !chars[start].is_ascii_digit() {
            continue;
        }
        if start > 0
            && (chars[start - 1].is_ascii_digit()
                || chars[start - 1] == 'X'
                || chars[start - 1] == 'x')
        {
            continue;
        }
        let mut compact = String::with_capacity(16);
        let mut idx = start;
        let mut valid = true;
        for (group_index, group_len) in [4usize, 4, 4, 3].iter().enumerate() {
            for _ in 0..*group_len {
                if idx >= chars.len() || !chars[idx].is_ascii_digit() {
                    valid = false;
                    break;
                }
                compact.push(chars[idx]);
                idx += 1;
            }
            if !valid {
                break;
            }
            if group_index < 3 && idx < chars.len() && is_orcid_dash(chars[idx]) {
                idx += 1;
            }
        }
        if !valid || idx >= chars.len() {
            continue;
        }
        let check_digit = chars[idx];
        if !(check_digit.is_ascii_digit() || check_digit == 'X' || check_digit == 'x') {
            continue;
        }
        compact.push(check_digit.to_ascii_uppercase());
        idx += 1;
        let bytes = compact.as_bytes();
        if bytes.len() == 16
            && bytes[..15].iter().all(|byte| byte.is_ascii_digit())
            && (bytes[15].is_ascii_digit() || bytes[15] == b'X')
            && (idx == chars.len()
                || !(chars[idx].is_ascii_digit() || chars[idx] == 'X' || chars[idx] == 'x'))
        {
            return Some(format!(
                "{}-{}-{}-{}",
                &compact[0..4],
                &compact[4..8],
                &compact[8..12],
                &compact[12..16]
            ));
        }
    }
    None
}

fn normalize_orcid_compact_owned(value: &str) -> Option<String> {
    normalize_orcid_owned(value).map(|orcid| orcid.replace('-', ""))
}

#[derive(Clone)]
struct RawArrowSignature {
    paper_id: String,
    author_first: String,
    author_middle: String,
    author_last: String,
    author_suffix: String,
    author_block: Option<String>,
    affiliations: Vec<String>,
    email: Option<String>,
    orcid: Option<String>,
    position: Option<i64>,
}

#[derive(Clone)]
struct RawArrowPaper {
    title: String,
    abstract_text: String,
    venue: String,
    journal_name: String,
    year: Option<i64>,
    predicted_language: Option<String>,
    is_reliable: Option<bool>,
}

#[derive(Clone)]
struct RawArrowFeature {
    query: RetrievalQueryData,
    name_counts: Option<NameCountsData>,
    paper_author_count: usize,
    query_author: String,
}

struct RawArrowAuthorSignalData {
    paper_author_names: HashSet<String>,
    local10_author_names: HashSet<String>,
}

struct RawArrowSummarySignalData {
    name_counts_values: Vec<NameCountsData>,
    member_paper_author_names: Vec<HashSet<String>>,
    member_paper_author_counts: Vec<usize>,
    member_local10_author_names: Vec<HashSet<String>>,
    member_signature_ids: Vec<String>,
}

struct RawArrowNameCountRarityRow {
    last_name_count_min_rarity: f32,
    candidate_last_name_count_min_rarity: f32,
    candidate_last_first_name_count_min_rarity: f32,
    last_first_name_count_min_rarity: f32,
    first_prefix_x_last_first_name_count_min_rarity: f32,
}

struct RawArrowPaperEvidenceRow {
    paper_author_list_max_jaccard: f32,
    paper_author_list_max_containment: f32,
    paper_author_list_max_overlap_count: f32,
    local_author_window10_jaccard_max: f32,
    local_author_window10_overlap_count_max: f32,
    best_author_count_log_absdiff: f32,
}

fn io_error_to_py(context: &str, path: &str, err: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyIOError::new_err(format!("{context} '{}': {err}", path))
}

fn arrow_error_to_py(context: &str, path: &str, err: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("{context} '{}': {err}", path))
}

fn source_file_sample_fingerprint(path: &str, source_size: u64) -> PyResult<u64> {
    let mut file = File::open(path).map_err(|err| {
        io_error_to_py(
            "failed to open Arrow IPC file for fingerprinting",
            path,
            err,
        )
    })?;
    let mut digest = fnv64(ARROW_BATCH_LOOKUP_INDEX_SOURCE_HASH_DOMAIN);
    digest = fnv64_update(digest, &source_size.to_le_bytes());
    let first_len =
        std::cmp::min(ARROW_BATCH_LOOKUP_INDEX_SOURCE_SAMPLE_BYTES, source_size) as usize;
    let mut buffer = vec![0u8; first_len];
    if first_len > 0 {
        file.read_exact(&mut buffer).map_err(|err| {
            io_error_to_py(
                "failed to read Arrow IPC file fingerprint prefix",
                path,
                err,
            )
        })?;
        digest = fnv64_update(digest, &buffer);
    }
    if source_size > ARROW_BATCH_LOOKUP_INDEX_SOURCE_SAMPLE_BYTES {
        let suffix_start = std::cmp::max(
            ARROW_BATCH_LOOKUP_INDEX_SOURCE_SAMPLE_BYTES,
            source_size - ARROW_BATCH_LOOKUP_INDEX_SOURCE_SAMPLE_BYTES,
        );
        file.seek(SeekFrom::Start(suffix_start)).map_err(|err| {
            io_error_to_py(
                "failed to seek Arrow IPC file fingerprint suffix",
                path,
                err,
            )
        })?;
        let mut suffix = vec![0u8; (source_size - suffix_start) as usize];
        file.read_exact(&mut suffix).map_err(|err| {
            io_error_to_py(
                "failed to read Arrow IPC file fingerprint suffix",
                path,
                err,
            )
        })?;
        digest = fnv64_update(digest, &suffix);
    }
    Ok(digest)
}

fn read_arrow_batches(path: &str) -> PyResult<Vec<RecordBatch>> {
    let file = File::open(path)
        .map_err(|err| io_error_to_py("failed to open Arrow IPC file", path, err))?;
    let reader = ArrowFileReader::try_new(file, None)
        .map_err(|err| arrow_error_to_py("failed to read Arrow IPC schema from", path, err))?;
    reader
        .map(|batch| {
            batch.map_err(|err| {
                arrow_error_to_py("failed to read Arrow IPC record batch from", path, err)
            })
        })
        .collect()
}

struct ArrowBatchLookupIndex {
    mmap: Mmap,
    record_count: usize,
}

impl ArrowBatchLookupIndex {
    fn open(path: &str, source_arrow_path: &str, key_column: &str) -> PyResult<Self> {
        let file = File::open(path)
            .map_err(|err| io_error_to_py("failed to open Arrow batch lookup index", path, err))?;
        let mmap = unsafe {
            Mmap::map(&file).map_err(|err| {
                io_error_to_py("failed to memory-map Arrow batch lookup index", path, err)
            })?
        };
        if mmap.len() < ARROW_BATCH_LOOKUP_INDEX_HEADER_LEN {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow batch lookup index '{path}' is shorter than its header"
            )));
        }
        let magic = &mmap[0..8];
        if magic != ARROW_BATCH_LOOKUP_INDEX_MAGIC {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow batch lookup index '{path}' has invalid magic bytes"
            )));
        }
        let source_metadata = fs::metadata(source_arrow_path).map_err(|err| {
            io_error_to_py(
                "failed to stat Arrow IPC file for batch lookup index validation",
                source_arrow_path,
                err,
            )
        })?;
        let source_size = source_metadata.len();
        let record_count = u64::from_le_bytes(
            mmap[8..16]
                .try_into()
                .expect("slice length is checked by fixed header length"),
        ) as usize;
        let indexed_source_size = u64::from_le_bytes(
            mmap[16..24]
                .try_into()
                .expect("indexed source-size slice has fixed length"),
        );
        let indexed_key_column_hash = u64::from_le_bytes(
            mmap[32..40]
                .try_into()
                .expect("indexed key-column hash slice has fixed length"),
        );
        let expected_key_column_hash = fnv64(key_column.as_bytes());
        if indexed_key_column_hash != expected_key_column_hash {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow batch lookup index '{path}' was built for a different key column: \
                 indexed hash={indexed_key_column_hash} expected hash={expected_key_column_hash} \
                 key_column='{key_column}'"
            )));
        }
        let indexed_source_fingerprint = u64::from_le_bytes(
            mmap[40..48]
                .try_into()
                .expect("indexed source fingerprint slice has fixed length"),
        );
        let source_fingerprint = source_file_sample_fingerprint(source_arrow_path, source_size)?;
        if indexed_source_size != source_size || indexed_source_fingerprint != source_fingerprint {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow batch lookup index '{path}' is stale for '{source_arrow_path}': \
                 indexed size/fingerprint=({indexed_source_size}, {indexed_source_fingerprint}) \
                 current size/fingerprint=({source_size}, {source_fingerprint})"
            )));
        }
        let expected_len = ARROW_BATCH_LOOKUP_INDEX_HEADER_LEN
            .checked_add(
                record_count
                    .checked_mul(ARROW_BATCH_LOOKUP_INDEX_RECORD_LEN)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyOverflowError::new_err(
                            "Arrow batch lookup index record count overflows usize",
                        )
                    })?,
            )
            .ok_or_else(|| {
                pyo3::exceptions::PyOverflowError::new_err(
                    "Arrow batch lookup index length overflows usize",
                )
            })?;
        if mmap.len() != expected_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Arrow batch lookup index '{path}' length {} does not match expected length {expected_len} \
                 (record_count={record_count}, header_len={ARROW_BATCH_LOOKUP_INDEX_HEADER_LEN}, \
                 record_len={ARROW_BATCH_LOOKUP_INDEX_RECORD_LEN})",
                mmap.len()
            )));
        }
        Ok(Self { mmap, record_count })
    }

    fn record_offset(&self, index: usize) -> usize {
        ARROW_BATCH_LOOKUP_INDEX_HEADER_LEN + index * ARROW_BATCH_LOOKUP_INDEX_RECORD_LEN
    }

    fn record_hash(&self, index: usize) -> u64 {
        let offset = self.record_offset(index);
        u64::from_le_bytes(
            self.mmap[offset..offset + 8]
                .try_into()
                .expect("record hash slice has fixed length"),
        )
    }

    fn record_batch_index(&self, index: usize) -> u32 {
        let offset = self.record_offset(index) + 8;
        u32::from_le_bytes(
            self.mmap[offset..offset + 4]
                .try_into()
                .expect("record batch-index slice has fixed length"),
        )
    }

    fn lower_bound(&self, hash: u64) -> usize {
        let mut lo = 0usize;
        let mut hi = self.record_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.record_hash(mid) < hash {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    fn batch_indices_for_keys(&self, keys: &HashSet<String>) -> HashSet<usize> {
        let mut out = HashSet::new();
        for key in keys {
            let hash = fnv64(key.as_bytes());
            let mut index = self.lower_bound(hash);
            while index < self.record_count && self.record_hash(index) == hash {
                out.insert(self.record_batch_index(index) as usize);
                index += 1;
            }
        }
        out
    }
}

#[derive(Clone, Copy, Default)]
struct IndexedArrowReadStats {
    batches_read: usize,
    rows_scanned: usize,
}

fn read_indexed_arrow_batches(
    path: &str,
    index_path: &str,
    key_column: &str,
    keep_ids: &HashSet<String>,
) -> PyResult<(Vec<RecordBatch>, IndexedArrowReadStats)> {
    if keep_ids.is_empty() {
        return Ok((Vec::new(), IndexedArrowReadStats::default()));
    }
    let index = ArrowBatchLookupIndex::open(index_path, path, key_column)?;
    let mut batch_indices: Vec<usize> =
        index.batch_indices_for_keys(keep_ids).into_iter().collect();
    batch_indices.sort_unstable();
    let file = File::open(path)
        .map_err(|err| io_error_to_py("failed to open Arrow IPC file", path, err))?;
    let mut reader = ArrowFileReader::try_new(file, None)
        .map_err(|err| arrow_error_to_py("failed to read Arrow IPC schema from", path, err))?;
    let mut batches = Vec::with_capacity(batch_indices.len());
    let mut rows_scanned = 0usize;
    for batch_index in batch_indices {
        reader.set_index(batch_index).map_err(|err| {
            arrow_error_to_py("failed to seek Arrow IPC record batch in", path, err)
        })?;
        let batch = reader
            .next()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Arrow IPC file '{path}' is missing indexed record batch {batch_index}"
                ))
            })?
            .map_err(|err| {
                arrow_error_to_py("failed to read Arrow IPC record batch from", path, err)
            })?;
        rows_scanned += batch.num_rows();
        batches.push(batch);
    }
    let batches_read = batches.len();
    Ok((
        batches,
        IndexedArrowReadStats {
            batches_read,
            rows_scanned,
        },
    ))
}

fn arrow_column_index(batch: &RecordBatch, name: &str, path: &str) -> PyResult<usize> {
    batch.schema().index_of(name).map_err(|err| {
        pyo3::exceptions::PyKeyError::new_err(format!(
            "missing Arrow column '{name}' in '{path}': {err}"
        ))
    })
}

fn arrow_optional_column_index(batch: &RecordBatch, name: &str) -> Option<usize> {
    batch.schema().index_of(name).ok()
}

fn arrow_first_existing_column_index(
    batch: &RecordBatch,
    path: &str,
    names: &[&str],
) -> PyResult<usize> {
    for name in names {
        if let Ok(index) = batch.schema().index_of(name) {
            return Ok(index);
        }
    }
    Err(pyo3::exceptions::PyKeyError::new_err(format!(
        "missing Arrow column in '{path}'; expected one of {names:?}"
    )))
}

fn arrow_required_string(array: &dyn Array, row: usize, context: &str) -> PyResult<String> {
    if array.is_null(row) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{context} is null at row {row}"
        )));
    }
    arrow_optional_string(array, row, context).and_then(|value| {
        value.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("{context} is null at row {row}"))
        })
    })
}

enum ArrowStringColumn<'a> {
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
    Int64(&'a Int64Array),
    Int32(&'a Int32Array),
    UInt64(&'a UInt64Array),
    UInt32(&'a UInt32Array),
    Null,
}

impl<'a> ArrowStringColumn<'a> {
    fn from_array(array: &'a dyn Array, context: &str) -> PyResult<Self> {
        match array.data_type() {
            DataType::Utf8 => Ok(Self::Utf8(
                array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "{context} is not a Utf8 array"
                        ))
                    })?,
            )),
            DataType::LargeUtf8 => Ok(Self::LargeUtf8(
                array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .ok_or_else(|| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "{context} is not a LargeUtf8 array"
                        ))
                    })?,
            )),
            DataType::Int64 => Ok(Self::Int64(
                array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} is not an Int64 array"
                    ))
                })?,
            )),
            DataType::Int32 => Ok(Self::Int32(
                array.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} is not an Int32 array"
                    ))
                })?,
            )),
            DataType::UInt64 => Ok(Self::UInt64(
                array
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "{context} is not a UInt64 array"
                        ))
                    })?,
            )),
            DataType::UInt32 => Ok(Self::UInt32(
                array
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "{context} is not a UInt32 array"
                        ))
                    })?,
            )),
            DataType::Null => Ok(Self::Null),
            other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "{context} must be a string or integer column, got {other:?}"
            ))),
        }
    }

    fn optional_value(&self, row: usize) -> Option<Cow<'a, str>> {
        match self {
            Self::Utf8(values) => (!values.is_null(row)).then(|| Cow::Borrowed(values.value(row))),
            Self::LargeUtf8(values) => {
                (!values.is_null(row)).then(|| Cow::Borrowed(values.value(row)))
            }
            Self::Int64(values) => {
                (!values.is_null(row)).then(|| Cow::Owned(values.value(row).to_string()))
            }
            Self::Int32(values) => {
                (!values.is_null(row)).then(|| Cow::Owned(values.value(row).to_string()))
            }
            Self::UInt64(values) => {
                (!values.is_null(row)).then(|| Cow::Owned(values.value(row).to_string()))
            }
            Self::UInt32(values) => {
                (!values.is_null(row)).then(|| Cow::Owned(values.value(row).to_string()))
            }
            Self::Null => None,
        }
    }

    fn optional_owned(&self, row: usize) -> Option<String> {
        self.optional_value(row).map(Cow::into_owned)
    }

    fn required_value(&self, row: usize, context: &str) -> PyResult<Cow<'a, str>> {
        self.optional_value(row).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("{context} is null at row {row}"))
        })
    }
}

enum ArrowI64Column<'a> {
    Int64(&'a Int64Array),
    Int32(&'a Int32Array),
    UInt64(&'a UInt64Array),
    UInt32(&'a UInt32Array),
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
    Null,
}

impl<'a> ArrowI64Column<'a> {
    fn from_array(array: &'a dyn Array, context: &str) -> PyResult<Self> {
        match array.data_type() {
            DataType::Int64 => Ok(Self::Int64(
                array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} is not an Int64 array"
                    ))
                })?,
            )),
            DataType::Int32 => Ok(Self::Int32(
                array.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} is not an Int32 array"
                    ))
                })?,
            )),
            DataType::UInt64 => Ok(Self::UInt64(
                array
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "{context} is not a UInt64 array"
                        ))
                    })?,
            )),
            DataType::UInt32 => Ok(Self::UInt32(
                array
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "{context} is not a UInt32 array"
                        ))
                    })?,
            )),
            DataType::Utf8 => Ok(Self::Utf8(
                array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "{context} is not a Utf8 array"
                        ))
                    })?,
            )),
            DataType::LargeUtf8 => Ok(Self::LargeUtf8(
                array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .ok_or_else(|| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "{context} is not a LargeUtf8 array"
                        ))
                    })?,
            )),
            DataType::Null => Ok(Self::Null),
            other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "{context} must be an integer column, got {other:?}"
            ))),
        }
    }

    fn optional_value(&self, row: usize, context: &str) -> PyResult<Option<i64>> {
        match self {
            Self::Int64(values) => Ok((!values.is_null(row)).then(|| values.value(row))),
            Self::Int32(values) => Ok((!values.is_null(row)).then(|| values.value(row) as i64)),
            Self::UInt64(values) => {
                if values.is_null(row) {
                    return Ok(None);
                }
                i64::try_from(values.value(row)).map(Some).map_err(|_| {
                    pyo3::exceptions::PyOverflowError::new_err(format!(
                        "{context} value at row {row} exceeds i64"
                    ))
                })
            }
            Self::UInt32(values) => Ok((!values.is_null(row)).then(|| values.value(row) as i64)),
            Self::Utf8(values) => {
                if values.is_null(row) {
                    return Ok(None);
                }
                values.value(row).parse::<i64>().map(Some).map_err(|err| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "{context} string value at row {row} is not an integer: {err}"
                    ))
                })
            }
            Self::LargeUtf8(values) => {
                if values.is_null(row) {
                    return Ok(None);
                }
                values.value(row).parse::<i64>().map(Some).map_err(|err| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "{context} string value at row {row} is not an integer: {err}"
                    ))
                })
            }
            Self::Null => Ok(None),
        }
    }

    fn required_value(&self, row: usize, context: &str) -> PyResult<i64> {
        self.optional_value(row, context)?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("{context} is null at row {row}"))
        })
    }
}

fn arrow_optional_string(array: &dyn Array, row: usize, context: &str) -> PyResult<Option<String>> {
    if array.is_null(row) {
        return Ok(None);
    }
    match array.data_type() {
        DataType::Utf8 => {
            let values = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!("{context} is not a Utf8 array"))
                })?;
            Ok(Some(values.value(row).to_string()))
        }
        DataType::LargeUtf8 => {
            let values = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} is not a LargeUtf8 array"
                    ))
                })?;
            Ok(Some(values.value(row).to_string()))
        }
        DataType::Int64 => {
            let values = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                pyo3::exceptions::PyTypeError::new_err(format!("{context} is not an Int64 array"))
            })?;
            Ok(Some(values.value(row).to_string()))
        }
        DataType::Int32 => {
            let values = array.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                pyo3::exceptions::PyTypeError::new_err(format!("{context} is not an Int32 array"))
            })?;
            Ok(Some(values.value(row).to_string()))
        }
        DataType::UInt64 => {
            let values = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} is not a UInt64 array"
                    ))
                })?;
            Ok(Some(values.value(row).to_string()))
        }
        DataType::UInt32 => {
            let values = array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} is not a UInt32 array"
                    ))
                })?;
            Ok(Some(values.value(row).to_string()))
        }
        DataType::Null => Ok(None),
        other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{context} must be a string or integer column, got {other:?}"
        ))),
    }
}

fn arrow_optional_i64(array: &dyn Array, row: usize, context: &str) -> PyResult<Option<i64>> {
    ArrowI64Column::from_array(array, context)?.optional_value(row, context)
}

fn arrow_optional_bool(array: &dyn Array, row: usize, context: &str) -> PyResult<Option<bool>> {
    if array.is_null(row) {
        return Ok(None);
    }
    match array.data_type() {
        DataType::Boolean => {
            let values = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} is not a Boolean array"
                    ))
                })?;
            Ok(Some(values.value(row)))
        }
        DataType::Int64 | DataType::Int32 | DataType::UInt64 | DataType::UInt32 => {
            arrow_optional_i64(array, row, context).map(|value| value.map(|integer| integer != 0))
        }
        DataType::Null => Ok(None),
        other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{context} must be a boolean column, got {other:?}"
        ))),
    }
}

fn arrow_string_array_values(array: &dyn Array, context: &str) -> PyResult<Vec<String>> {
    match array.data_type() {
        DataType::Utf8 => {
            let values = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!("{context} is not a Utf8 array"))
                })?;
            let mut out = Vec::with_capacity(values.len());
            for idx in 0..values.len() {
                if values.is_null(idx) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "{context} cannot contain null list elements"
                    )));
                }
                out.push(values.value(idx).to_string());
            }
            Ok(out)
        }
        DataType::LargeUtf8 => {
            let values = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} is not a LargeUtf8 array"
                    ))
                })?;
            let mut out = Vec::with_capacity(values.len());
            for idx in 0..values.len() {
                if values.is_null(idx) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "{context} cannot contain null list elements"
                    )));
                }
                out.push(values.value(idx).to_string());
            }
            Ok(out)
        }
        DataType::Null => {
            if array.len() == 0 {
                Ok(Vec::new())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{context} cannot contain null list elements"
                )))
            }
        }
        other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{context} list values must be strings, got {other:?}"
        ))),
    }
}

fn arrow_optional_string_list(
    array: &dyn Array,
    row: usize,
    context: &str,
) -> PyResult<Vec<String>> {
    if array.is_null(row) {
        return Ok(Vec::new());
    }
    match array.data_type() {
        DataType::List(_) => {
            let values = array.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                pyo3::exceptions::PyTypeError::new_err(format!("{context} is not a List array"))
            })?;
            let item_values = values.value(row);
            arrow_string_array_values(item_values.as_ref(), context)
        }
        DataType::LargeList(_) => {
            let values = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} is not a LargeList array"
                    ))
                })?;
            let item_values = values.value(row);
            arrow_string_array_values(item_values.as_ref(), context)
        }
        other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{context} must be a list<string> column, got {other:?}"
        ))),
    }
}

fn arrow_optional_f32_vector(
    array: &dyn Array,
    row: usize,
    context: &str,
) -> PyResult<Option<Vec<f32>>> {
    if array.is_null(row) {
        return Ok(None);
    }
    match array.data_type() {
        DataType::FixedSizeList(_, _) => {
            let values = array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} is not a FixedSizeList array"
                    ))
                })?;
            let item_values = values.value(row);
            let floats = item_values
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{context} FixedSizeList values must be float32"
                    ))
                })?;
            let mut out = Vec::with_capacity(floats.len());
            for idx in 0..floats.len() {
                if floats.is_null(idx) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "{context} has null float value at row {row}, offset {idx}"
                    )));
                }
                out.push(floats.value(idx));
            }
            Ok(Some(out))
        }
        other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{context} must be a fixed_size_list<float32> column, got {other:?}"
        ))),
    }
}

fn read_raw_arrow_signatures_from_batches(
    path: &str,
    batches: Vec<RecordBatch>,
    keep_signature_ids: Option<&HashSet<String>>,
) -> PyResult<HashMap<String, RawArrowSignature>> {
    let mut out = HashMap::new();
    for batch in batches {
        let signature_id_col = batch.column(arrow_column_index(&batch, "signature_id", path)?);
        let signature_id_values =
            ArrowStringColumn::from_array(signature_id_col.as_ref(), "signature_id")?;
        let paper_id_col = batch.column(arrow_column_index(&batch, "paper_id", path)?);
        let paper_id_values = ArrowStringColumn::from_array(paper_id_col.as_ref(), "paper_id")?;
        let first_col = batch.column(arrow_column_index(&batch, "author_first", path)?);
        let first_values = ArrowStringColumn::from_array(first_col.as_ref(), "author_first")?;
        let middle_col = batch.column(arrow_column_index(&batch, "author_middle", path)?);
        let middle_values = ArrowStringColumn::from_array(middle_col.as_ref(), "author_middle")?;
        let last_col = batch.column(arrow_column_index(&batch, "author_last", path)?);
        let last_values = ArrowStringColumn::from_array(last_col.as_ref(), "author_last")?;
        let suffix_col = batch.column(arrow_column_index(&batch, "author_suffix", path)?);
        let suffix_values = ArrowStringColumn::from_array(suffix_col.as_ref(), "author_suffix")?;
        let affiliations_col =
            batch.column(arrow_column_index(&batch, "author_affiliations", path)?);
        let orcid_col = batch.column(arrow_column_index(&batch, "author_orcid", path)?);
        let orcid_values = ArrowStringColumn::from_array(orcid_col.as_ref(), "author_orcid")?;
        let position_col = batch.column(arrow_column_index(&batch, "author_position", path)?);
        let position_values = ArrowI64Column::from_array(position_col.as_ref(), "author_position")?;
        let author_block_col =
            arrow_optional_column_index(&batch, "author_block").map(|index| batch.column(index));
        let author_block_values = match author_block_col.as_ref() {
            Some(col) => Some(ArrowStringColumn::from_array(col.as_ref(), "author_block")?),
            None => None,
        };
        let email_col =
            arrow_optional_column_index(&batch, "author_email").map(|index| batch.column(index));
        let email_values = match email_col.as_ref() {
            Some(col) => Some(ArrowStringColumn::from_array(col.as_ref(), "author_email")?),
            None => None,
        };
        for row in 0..batch.num_rows() {
            let signature_id_value = signature_id_values.required_value(row, "signature_id")?;
            if keep_signature_ids.map_or(false, |keep| !keep.contains(signature_id_value.as_ref()))
            {
                continue;
            }
            if signature_id_value.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "signatures Arrow cannot contain empty signature_id values",
                ));
            }
            let signature_id = signature_id_value.into_owned();
            match out.entry(signature_id) {
                Entry::Occupied(entry) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "signatures Arrow contains duplicate signature_id: {:?}",
                        entry.key()
                    )));
                }
                Entry::Vacant(entry) => {
                    let paper_id = paper_id_values
                        .required_value(row, "paper_id")?
                        .into_owned();
                    entry.insert(RawArrowSignature {
                        paper_id,
                        author_first: first_values.optional_owned(row).unwrap_or_default(),
                        author_middle: middle_values.optional_owned(row).unwrap_or_default(),
                        author_last: last_values.optional_owned(row).unwrap_or_default(),
                        author_suffix: suffix_values.optional_owned(row).unwrap_or_default(),
                        author_block: author_block_values
                            .as_ref()
                            .and_then(|col| col.optional_owned(row)),
                        affiliations: arrow_optional_string_list(
                            affiliations_col.as_ref(),
                            row,
                            "author_affiliations",
                        )?,
                        email: email_values
                            .as_ref()
                            .and_then(|col| col.optional_owned(row)),
                        orcid: orcid_values
                            .optional_owned(row)
                            .and_then(|value| normalize_orcid_owned(&value)),
                        position: position_values.optional_value(row, "author_position")?,
                    });
                }
            }
        }
    }
    Ok(out)
}

fn read_raw_arrow_papers_from_batches(
    path: &str,
    batches: Vec<RecordBatch>,
    keep_paper_ids: Option<&HashSet<String>>,
) -> PyResult<HashMap<String, RawArrowPaper>> {
    let mut out = HashMap::new();
    for batch in batches {
        let paper_id_col = batch.column(arrow_column_index(&batch, "paper_id", path)?);
        let paper_id_values = ArrowStringColumn::from_array(paper_id_col.as_ref(), "paper_id")?;
        let title_col = batch.column(arrow_column_index(&batch, "title", path)?);
        let title_values = ArrowStringColumn::from_array(title_col.as_ref(), "title")?;
        let abstract_col =
            arrow_optional_column_index(&batch, "abstract").map(|index| batch.column(index));
        let abstract_values = match abstract_col.as_ref() {
            Some(col) => Some(ArrowStringColumn::from_array(col.as_ref(), "abstract")?),
            None => None,
        };
        let venue_col = batch.column(arrow_column_index(&batch, "venue", path)?);
        let venue_values = ArrowStringColumn::from_array(venue_col.as_ref(), "venue")?;
        let journal_col = batch.column(arrow_column_index(&batch, "journal_name", path)?);
        let journal_values = ArrowStringColumn::from_array(journal_col.as_ref(), "journal_name")?;
        let year_col = batch.column(arrow_column_index(&batch, "year", path)?);
        let year_values = ArrowI64Column::from_array(year_col.as_ref(), "year")?;
        let predicted_language_col = arrow_optional_column_index(&batch, "predicted_language")
            .map(|index| batch.column(index));
        let predicted_language_values = match predicted_language_col.as_ref() {
            Some(col) => Some(ArrowStringColumn::from_array(
                col.as_ref(),
                "predicted_language",
            )?),
            None => None,
        };
        let is_reliable_col = arrow_optional_column_index(&batch, "is_reliable")
            .map(|index| batch.column(index).as_ref());
        for row in 0..batch.num_rows() {
            let paper_id_value = paper_id_values.required_value(row, "paper_id")?;
            if keep_paper_ids.map_or(false, |keep| !keep.contains(paper_id_value.as_ref())) {
                continue;
            }
            if paper_id_value.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "papers Arrow cannot contain empty paper_id values",
                ));
            }
            let paper_id = paper_id_value.into_owned();
            match out.entry(paper_id) {
                Entry::Occupied(entry) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "papers Arrow contains duplicate paper_id: {:?}",
                        entry.key()
                    )));
                }
                Entry::Vacant(entry) => {
                    entry.insert(RawArrowPaper {
                        title: title_values.optional_owned(row).unwrap_or_default(),
                        abstract_text: abstract_values
                            .as_ref()
                            .and_then(|col| col.optional_owned(row))
                            .unwrap_or_default(),
                        venue: venue_values.optional_owned(row).unwrap_or_default(),
                        journal_name: journal_values.optional_owned(row).unwrap_or_default(),
                        year: year_values.optional_value(row, "year")?,
                        predicted_language: predicted_language_values
                            .as_ref()
                            .and_then(|col| col.optional_owned(row)),
                        is_reliable: match is_reliable_col {
                            Some(col) => arrow_optional_bool(col, row, "is_reliable")?,
                            None => None,
                        },
                    });
                }
            }
        }
    }
    Ok(out)
}

fn read_raw_arrow_paper_authors_from_batches(
    path: &str,
    batches: Vec<RecordBatch>,
    keep_paper_ids: Option<&HashSet<String>>,
) -> PyResult<HashMap<String, Vec<(i64, String)>>> {
    let mut out: HashMap<String, Vec<(i64, String)>> = HashMap::new();
    for batch in batches {
        let paper_id_col = batch.column(arrow_column_index(&batch, "paper_id", path)?);
        let paper_id_values = ArrowStringColumn::from_array(paper_id_col.as_ref(), "paper_id")?;
        let position_col = batch.column(arrow_column_index(&batch, "position", path)?);
        let position_values = ArrowI64Column::from_array(position_col.as_ref(), "position")?;
        let author_name_col = batch.column(arrow_column_index(&batch, "author_name", path)?);
        let author_name_values =
            ArrowStringColumn::from_array(author_name_col.as_ref(), "author_name")?;
        for row in 0..batch.num_rows() {
            let paper_id_value = paper_id_values.required_value(row, "paper_id")?;
            if keep_paper_ids.map_or(false, |keep| !keep.contains(paper_id_value.as_ref())) {
                continue;
            }
            if paper_id_value.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "paper_authors Arrow cannot contain empty paper_id values",
                ));
            }
            let paper_id = paper_id_value.into_owned();
            let position = position_values.required_value(row, "position")?;
            let author_name = author_name_values.optional_owned(row).unwrap_or_default();
            out.entry(paper_id)
                .or_default()
                .push((position, author_name));
        }
    }
    for (paper_id, authors) in out.iter_mut() {
        authors.sort_by_key(|(position, _name)| *position);
        for window in authors.windows(2) {
            if window[0].0 == window[1].0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "paper_authors Arrow contains duplicate (paper_id, position): ({:?}, {})",
                    paper_id, window[0].0
                )));
            }
        }
    }
    Ok(out)
}

fn read_raw_arrow_cluster_seeds(
    path: &str,
) -> PyResult<(Vec<String>, HashMap<String, Vec<String>>)> {
    let mut component_order = Vec::new();
    let mut members_by_component: HashMap<String, Vec<String>> = HashMap::new();
    let mut component_by_signature_id = HashMap::<String, String>::new();
    for batch in read_arrow_batches(path)? {
        let signature_id_col = batch.column(arrow_column_index(&batch, "signature_id", path)?);
        let cluster_id_col = batch.column(arrow_column_index(&batch, "cluster_id", path)?);
        for row in 0..batch.num_rows() {
            let signature_id =
                arrow_required_string(signature_id_col.as_ref(), row, "signature_id")?;
            let component_key = arrow_required_string(cluster_id_col.as_ref(), row, "cluster_id")?;
            if signature_id.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "cluster_seeds Arrow cannot contain empty signature_id values",
                ));
            }
            if component_key.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "cluster_seeds Arrow cannot contain empty cluster_id values: {signature_id:?}"
                )));
            }
            if let Some(existing_component_key) = component_by_signature_id.get(&signature_id) {
                if existing_component_key == &component_key {
                    continue;
                }
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "cluster_seeds Arrow assigns signature_id {signature_id:?} to multiple clusters: \
                     {existing_component_key:?} and {component_key:?}"
                )));
            }
            component_by_signature_id.insert(signature_id.clone(), component_key.clone());
            if !members_by_component.contains_key(&component_key) {
                component_order.push(component_key.clone());
            }
            members_by_component
                .entry(component_key)
                .or_default()
                .push(signature_id);
        }
    }
    Ok((component_order, members_by_component))
}

fn read_raw_arrow_cluster_seed_disallows(path: &str) -> PyResult<HashSet<(String, String)>> {
    let mut out = HashSet::new();
    for batch in read_arrow_batches(path)? {
        let left_col = batch.column(arrow_column_index(&batch, "signature_id_1", path)?);
        let right_col = batch.column(arrow_column_index(&batch, "signature_id_2", path)?);
        for row in 0..batch.num_rows() {
            let left = arrow_required_string(left_col.as_ref(), row, "signature_id_1")?;
            let right = arrow_required_string(right_col.as_ref(), row, "signature_id_2")?;
            if left == right {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "cluster_seed_disallows contains a self-pair for signature_id={left:?}"
                )));
            }
            let pair = canonical_signature_pair_owned(left, right);
            out.insert(pair);
        }
    }
    Ok(out)
}

fn read_raw_arrow_specter_from_batches(
    path: &str,
    batches: Vec<RecordBatch>,
    keep_paper_ids: Option<&HashSet<String>>,
) -> PyResult<HashMap<String, Vec<f32>>> {
    let mut out = HashMap::new();
    let mut seen_paper_ids = HashSet::<String>::new();
    for batch in batches {
        let paper_id_col = batch.column(arrow_column_index(&batch, "paper_id", path)?);
        let paper_id_values = ArrowStringColumn::from_array(paper_id_col.as_ref(), "paper_id")?;
        let embedding_col = batch.column(arrow_column_index(&batch, "embedding", path)?);
        for row in 0..batch.num_rows() {
            let paper_id_value = paper_id_values.required_value(row, "paper_id")?;
            if keep_paper_ids.map_or(false, |keep| !keep.contains(paper_id_value.as_ref())) {
                continue;
            }
            if paper_id_value.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "specter Arrow cannot contain empty paper_id values",
                ));
            }
            let paper_id = paper_id_value.into_owned();
            if !seen_paper_ids.insert(paper_id.clone()) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "specter Arrow contains duplicate paper_id: {paper_id:?}"
                )));
            }
            if let Some(vector) =
                arrow_optional_f32_vector(embedding_col.as_ref(), row, "embedding")?
            {
                if !vector.is_empty() {
                    insert_nonzero_specter_row(&mut out, &paper_id, &vector);
                }
            }
        }
    }
    Ok(out)
}

fn read_raw_arrow_with_optional_index<T, F>(
    path: &str,
    index_path: Option<&str>,
    key_column: &str,
    keep_ids: Option<&HashSet<String>>,
    full_scan_without_index: bool,
    read_from_batches: F,
) -> PyResult<(T, IndexedArrowReadStats)>
where
    F: Fn(&str, Vec<RecordBatch>, Option<&HashSet<String>>) -> PyResult<T>,
{
    if let (Some(index_path), Some(keep_ids)) = (index_path, keep_ids) {
        let (batches, stats) = read_indexed_arrow_batches(path, index_path, key_column, keep_ids)?;
        return Ok((read_from_batches(path, batches, Some(keep_ids))?, stats));
    }
    if keep_ids.is_some() && index_path.is_none() && !full_scan_without_index {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Refusing filtered full scan of Arrow IPC file '{path}' without a batch lookup index for key column \
             '{key_column}'. Provide the matching *_batch_index path or set full_scan_without_index=true for an \
             explicit small/full-scan path."
        )));
    }
    let batches = read_arrow_batches(path)?;
    let stats = IndexedArrowReadStats {
        batches_read: batches.len(),
        rows_scanned: batches.iter().map(|batch| batch.num_rows()).sum(),
    };
    let loaded = read_from_batches(path, batches, keep_ids)?;
    Ok((loaded, stats))
}

fn read_raw_arrow_signatures_with_optional_index(
    path: &str,
    index_path: Option<&str>,
    keep_signature_ids: Option<&HashSet<String>>,
    full_scan_without_index: bool,
) -> PyResult<(HashMap<String, RawArrowSignature>, IndexedArrowReadStats)> {
    read_raw_arrow_with_optional_index(
        path,
        index_path,
        "signature_id",
        keep_signature_ids,
        full_scan_without_index,
        read_raw_arrow_signatures_from_batches,
    )
}

fn read_raw_arrow_papers_with_optional_index(
    path: &str,
    index_path: Option<&str>,
    keep_paper_ids: &HashSet<String>,
    full_scan_without_index: bool,
) -> PyResult<(HashMap<String, RawArrowPaper>, IndexedArrowReadStats)> {
    read_raw_arrow_with_optional_index(
        path,
        index_path,
        "paper_id",
        Some(keep_paper_ids),
        full_scan_without_index,
        read_raw_arrow_papers_from_batches,
    )
}

fn read_raw_arrow_paper_authors_with_optional_index(
    path: &str,
    index_path: Option<&str>,
    keep_paper_ids: &HashSet<String>,
    full_scan_without_index: bool,
) -> PyResult<(HashMap<String, Vec<(i64, String)>>, IndexedArrowReadStats)> {
    read_raw_arrow_with_optional_index(
        path,
        index_path,
        "paper_id",
        Some(keep_paper_ids),
        full_scan_without_index,
        read_raw_arrow_paper_authors_from_batches,
    )
}

fn read_raw_arrow_specter_with_optional_index(
    path: &str,
    index_path: Option<&str>,
    keep_paper_ids: &HashSet<String>,
    full_scan_without_index: bool,
) -> PyResult<(HashMap<String, Vec<f32>>, IndexedArrowReadStats)> {
    read_raw_arrow_with_optional_index(
        path,
        index_path,
        "paper_id",
        Some(keep_paper_ids),
        full_scan_without_index,
        read_raw_arrow_specter_from_batches,
    )
}

fn read_raw_name_counts_index(path: &str) -> PyResult<RawNameCountMaps> {
    Ok(RawNameCountMaps::from_index(RawNameCountIndex::open(path)?))
}

fn read_raw_arrow_name_tuples(path: &str) -> PyResult<HashMap<String, HashSet<String>>> {
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    for batch in read_arrow_batches(path)? {
        let left_col = batch.column(arrow_first_existing_column_index(
            &batch,
            path,
            &["name_1", "left", "name_a", "first_name"],
        )?);
        let right_col = batch.column(arrow_first_existing_column_index(
            &batch,
            path,
            &["name_2", "right", "name_b", "second_name"],
        )?);
        for row in 0..batch.num_rows() {
            let left = arrow_required_string(left_col.as_ref(), row, "name_pairs.name_1")?;
            let right = arrow_required_string(right_col.as_ref(), row, "name_pairs.name_2")?;
            insert_name_tuple_alias(&mut out, left, right);
        }
    }
    Ok(out)
}

fn extract_path_mapping_string(
    paths: &Bound<'_, PyAny>,
    key: &str,
    required: bool,
) -> PyResult<Option<String>> {
    let dict = paths.downcast::<PyDict>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("Arrow path bundle must be a dict-like mapping")
    })?;
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => Ok(Some(value.extract::<String>()?)),
        _ if required => Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "Arrow path bundle is missing required key: {key}"
        ))),
        _ => Ok(None),
    }
}

fn extract_name_counts_index_path(paths: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    extract_path_mapping_string(paths, "name_counts_index", false)
}

struct SignatureTextFields<'a> {
    author_first: &'a str,
    author_middle: &'a str,
    author_last: &'a str,
    author_suffix: &'a str,
    affiliations: &'a [String],
}

struct PaperTextFields<'a> {
    title: &'a str,
    venue: &'a str,
    journal_name: &'a str,
}

fn ensure_unidecode_for_signature_texts<'a>(
    unidecode_fn: &Bound<'_, PyAny>,
    signatures: impl IntoIterator<Item = SignatureTextFields<'a>>,
    unidecode_char_map: &mut HashMap<char, String>,
) -> PyResult<()> {
    for signature in signatures {
        ensure_unidecode_for_text(unidecode_fn, signature.author_first, unidecode_char_map)?;
        ensure_unidecode_for_text(unidecode_fn, signature.author_middle, unidecode_char_map)?;
        ensure_unidecode_for_text(unidecode_fn, signature.author_last, unidecode_char_map)?;
        ensure_unidecode_for_text(unidecode_fn, signature.author_suffix, unidecode_char_map)?;
        for affiliation in signature.affiliations.iter() {
            ensure_unidecode_for_text(unidecode_fn, affiliation, unidecode_char_map)?;
        }
    }
    Ok(())
}

fn ensure_unidecode_for_paper_texts<'a>(
    unidecode_fn: &Bound<'_, PyAny>,
    papers: impl IntoIterator<Item = PaperTextFields<'a>>,
    unidecode_char_map: &mut HashMap<char, String>,
) -> PyResult<()> {
    for paper in papers {
        ensure_unidecode_for_text(unidecode_fn, paper.title, unidecode_char_map)?;
        ensure_unidecode_for_text(unidecode_fn, paper.venue, unidecode_char_map)?;
        ensure_unidecode_for_text(unidecode_fn, paper.journal_name, unidecode_char_map)?;
    }
    Ok(())
}

fn ensure_unidecode_for_paper_author_texts<'a>(
    unidecode_fn: &Bound<'_, PyAny>,
    paper_authors: impl IntoIterator<Item = &'a [(i64, String)]>,
    unidecode_char_map: &mut HashMap<char, String>,
) -> PyResult<()> {
    for authors in paper_authors {
        for (_position, author_name) in authors.iter() {
            ensure_unidecode_for_text(unidecode_fn, author_name, unidecode_char_map)?;
        }
    }
    Ok(())
}

fn ensure_unidecode_for_raw_arrow_inputs(
    unidecode_fn: &Bound<'_, PyAny>,
    signatures: &HashMap<String, RawArrowSignature>,
    papers: &HashMap<String, RawArrowPaper>,
    paper_authors: &HashMap<String, Vec<(i64, String)>>,
    unidecode_char_map: &mut HashMap<char, String>,
) -> PyResult<()> {
    ensure_unidecode_for_signature_texts(
        unidecode_fn,
        signatures.values().map(|signature| SignatureTextFields {
            author_first: &signature.author_first,
            author_middle: &signature.author_middle,
            author_last: &signature.author_last,
            author_suffix: &signature.author_suffix,
            affiliations: &signature.affiliations,
        }),
        unidecode_char_map,
    )?;
    ensure_unidecode_for_paper_texts(
        unidecode_fn,
        papers.values().map(|paper| PaperTextFields {
            title: &paper.title,
            venue: &paper.venue,
            journal_name: &paper.journal_name,
        }),
        unidecode_char_map,
    )?;
    ensure_unidecode_for_paper_author_texts(
        unidecode_fn,
        paper_authors.values().map(Vec::as_slice),
        unidecode_char_map,
    )?;
    Ok(())
}

fn preprocess_stage_papers(
    paper_inputs: &[StagePaperInput],
    preprocess: bool,
    unidecode_char_map: &HashMap<char, String>,
    stop_words: &HashSet<String>,
    venue_stop_words: &HashSet<String>,
) -> Vec<(PaperId, StagePaperPreprocessed)> {
    paper_inputs
        .par_iter()
        .map(|paper_input| {
            let title =
                normalize_text_compat_from_map(&paper_input.raw_title, false, unidecode_char_map);
            let venue = if preprocess {
                normalize_text_compat_from_map(&paper_input.raw_venue, false, unidecode_char_map)
            } else {
                paper_input.raw_venue.clone()
            };
            let journal_name = if preprocess {
                normalize_text_compat_from_map(&paper_input.raw_journal, false, unidecode_char_map)
            } else {
                paper_input.raw_journal.clone()
            };
            let authors = paper_input
                .raw_authors
                .iter()
                .map(|(position, raw_name)| {
                    (
                        *position,
                        normalize_text_compat_from_map(raw_name, false, unidecode_char_map),
                    )
                })
                .collect::<Vec<_>>();
            let title_words =
                counter_data_from_usize_map(word_ngrams_counter_python_compat(&title, stop_words));
            let title_chars = if preprocess {
                counter_data_from_usize_map(char_ngrams_counter_python_compat(
                    &title,
                    false,
                    true,
                    Some(stop_words),
                ))
            } else {
                None
            };
            let venue_ngrams = if preprocess {
                counter_data_from_usize_map(char_ngrams_counter_python_compat(
                    &venue,
                    false,
                    true,
                    Some(venue_stop_words),
                ))
            } else {
                None
            };
            let journal_ngrams = if preprocess {
                counter_data_from_usize_map(char_ngrams_counter_python_compat(
                    &journal_name,
                    false,
                    true,
                    Some(venue_stop_words),
                ))
            } else {
                None
            };
            (
                paper_input.paper_id.clone(),
                StagePaperPreprocessed {
                    authors,
                    year: paper_input.year,
                    has_abstract: paper_input.has_abstract,
                    predicted_language: paper_input.predicted_language.clone(),
                    is_reliable: paper_input.is_reliable,
                    title_words,
                    title_chars,
                    venue_ngrams,
                    journal_ngrams,
                },
            )
        })
        .collect::<Vec<_>>()
}

fn preprocess_stage_signatures(
    signature_inputs: &[StageSignatureInput],
    preprocessed_papers: &HashMap<PaperId, StagePaperPreprocessed>,
    raw_name_counts: &RawNameCountMaps,
    name_prefixes: &HashSet<String>,
    affiliation_stopwords: &HashSet<String>,
    unidecode_char_map: &HashMap<char, String>,
    preprocess: bool,
) -> Vec<(String, SignatureData)> {
    signature_inputs
        .par_iter()
        .map(|entry| {
            let middle_normalized =
                normalize_text_compat_from_map(&entry.raw_middle, false, unidecode_char_map);
            let first_normalized =
                normalize_text_compat_from_map(&entry.raw_first, false, unidecode_char_map);
            let first_normalized_token = first_normalized_token_python_compat(
                &first_normalized,
                &middle_normalized,
                name_prefixes,
            );
            let (first_without_apostrophe, middle_without_apostrophe) =
                split_first_middle_hyphen_aware_compat(
                    &entry.raw_first,
                    &entry.raw_middle,
                    name_prefixes,
                    unidecode_char_map,
                );
            let last_normalized =
                normalize_text_compat_from_map(&entry.raw_last, false, unidecode_char_map);
            let mut coauthor_list: Vec<String> = Vec::new();
            if let Some(preprocessed_paper) = preprocessed_papers.get(&entry.paper_id) {
                for (author_position, author_name) in preprocessed_paper.authors.iter() {
                    if *author_position != entry.position {
                        coauthor_list.push(author_name.clone());
                    }
                }
            }
            let coauthors = if coauthor_list.is_empty() {
                None
            } else {
                Some(coauthor_list.iter().cloned().collect::<HashSet<String>>())
            };
            let mut coauthor_blocks_set: HashSet<String> = HashSet::new();
            for coauthor in coauthor_list.iter() {
                coauthor_blocks_set.insert(compute_block_compat(coauthor));
            }
            let coauthor_blocks = if coauthor_blocks_set.is_empty() {
                None
            } else {
                Some(coauthor_blocks_set)
            };

            let normalized_affiliations: Vec<String> = if preprocess {
                entry
                    .affiliation_values
                    .iter()
                    .filter_map(|affiliation| {
                        let normalized =
                            normalize_text_compat_from_map(affiliation, false, unidecode_char_map);
                        if normalized.is_empty() {
                            None
                        } else {
                            Some(normalized)
                        }
                    })
                    .collect()
            } else {
                entry.affiliation_values.clone()
            };
            let affiliation_text = if preprocess {
                prefilter_affiliation_text(&normalized_affiliations, affiliation_stopwords)
            } else {
                String::new()
            };
            let coauthor_text = if preprocess {
                coauthor_list.join(" ")
            } else {
                String::new()
            };
            let affiliations = if preprocess && !affiliation_text.is_empty() {
                counter_data_from_usize_map(word_ngrams_counter(&affiliation_text))
            } else {
                None
            };
            let coauthor_ngrams = if preprocess && !coauthor_text.is_empty() {
                counter_data_from_usize_map(char_ngrams_counter(&coauthor_text))
            } else {
                None
            };
            let normalized_orcid = entry
                .orcid
                .as_ref()
                .and_then(|value| normalize_orcid_compact_owned(value));
            let name_counts = build_name_counts_data_from_artifact(
                raw_name_counts,
                &entry.raw_first,
                &first_normalized_token,
                &first_without_apostrophe,
                &entry.raw_last,
                &last_normalized,
            )
            .data;
            (
                entry.sig_id.clone(),
                SignatureData {
                    first: Some(first_without_apostrophe.clone()),
                    middle: Some(middle_without_apostrophe),
                    last_normalized: Some(last_normalized),
                    orcid: normalized_orcid,
                    email: entry.email.clone(),
                    affiliations,
                    coauthor_blocks,
                    coauthor_ngrams,
                    coauthors,
                    position: entry.position,
                    paper_id: entry.paper_id.clone(),
                    name_counts,
                    adv_name: Some(first_without_apostrophe),
                },
            )
        })
        .collect::<Vec<_>>()
}

fn build_raw_arrow_feature(
    signature: &RawArrowSignature,
    paper: Option<&RawArrowPaper>,
    paper_authors: Option<&Vec<(i64, String)>>,
    specter_by_paper_id: Option<&HashMap<String, Arc<Vec<f32>>>>,
    raw_name_counts: &RawNameCountMaps,
    name_prefixes: &HashSet<String>,
    affiliation_stopwords: &HashSet<String>,
    unidecode_char_map: &HashMap<char, String>,
    orcid_enabled: bool,
) -> RawArrowFeature {
    let (first, middle) = split_first_middle_hyphen_aware_compat(
        &signature.author_first,
        &signature.author_middle,
        name_prefixes,
        unidecode_char_map,
    );
    let last_normalized =
        normalize_text_compat_from_map(&signature.author_last, false, unidecode_char_map);
    let name_counts = build_name_counts_data_from_artifact(
        raw_name_counts,
        &signature.author_first,
        &first,
        &first,
        &signature.author_last,
        &last_normalized,
    )
    .data;
    let middle_initials: HashSet<String> = middle
        .split_whitespace()
        .filter_map(|token| token.chars().next().map(|ch| ch.to_string()))
        .collect();

    let mut coauthor_blocks = HashSet::new();
    let mut paper_author_count = 0usize;
    if let Some(authors) = paper_authors {
        paper_author_count = authors.len();
        if let Some(signature_position) = signature.position {
            for (position, author_name) in authors.iter() {
                let normalized =
                    normalize_text_compat_from_map(author_name, false, unidecode_char_map);
                if *position == signature_position {
                    continue;
                }
                if normalized.is_empty() {
                    continue;
                }
                let block = compute_block_compat(&normalized);
                if !block.is_empty() {
                    coauthor_blocks.insert(block);
                }
            }
        }
    }

    let mut normalized_affiliations = Vec::with_capacity(signature.affiliations.len());
    for affiliation in signature.affiliations.iter() {
        let normalized = normalize_text_compat_from_map(affiliation, false, unidecode_char_map);
        if !normalized.is_empty() {
            normalized_affiliations.push(normalized);
        }
    }
    let affiliation_text =
        prefilter_affiliation_text(&normalized_affiliations, affiliation_stopwords);
    let affiliation_terms: HashSet<String> =
        word_ngrams_counter(&affiliation_text).into_keys().collect();

    let (venue_terms, title_terms, year) = if let Some(paper_data) = paper {
        let venue_text = [paper_data.venue.as_str(), paper_data.journal_name.as_str()]
            .iter()
            .filter(|part| !part.trim().is_empty())
            .copied()
            .collect::<Vec<_>>()
            .join(" ");
        let normalized_venue =
            normalize_text_compat_from_map(&venue_text, false, unidecode_char_map);
        let normalized_title =
            normalize_text_compat_from_map(&paper_data.title, false, unidecode_char_map);
        (
            term_set_from_normalized_text(&normalized_venue),
            term_set_from_normalized_text(&normalized_title),
            paper_data.year,
        )
    } else {
        (HashSet::new(), HashSet::new(), None)
    };

    let coauthor_terms = query_terms_from_values(&coauthor_blocks);
    let coauthor_hashes = coauthor_terms.iter().map(|term| term.hash).collect();
    let affiliation_query_terms = query_terms_from_values(&affiliation_terms);
    let affiliation_hashes = affiliation_query_terms
        .iter()
        .map(|term| term.hash)
        .collect();
    let venue_hashes = hash_string_values(&venue_terms);
    let title_hashes = hash_string_values(&title_terms);
    let middle_initial_hashes = hash_string_values(&middle_initials);
    let orcid_hash = if orcid_enabled {
        signature
            .orcid
            .as_ref()
            .and_then(|value| normalize_orcid_str(value).map(|orcid| fnv64(orcid.as_bytes())))
    } else {
        None
    };
    let specter = specter_by_paper_id
        .and_then(|values| values.get(&signature.paper_id))
        .map(Arc::clone);
    let specter_norm = specter.as_ref().map(|values| {
        values
            .iter()
            .map(|value| {
                let val = *value as f64;
                val * val
            })
            .sum::<f64>()
            .sqrt()
    });
    let query_author = [
        signature.author_first.as_str(),
        signature.author_middle.as_str(),
        signature.author_last.as_str(),
        signature.author_suffix.as_str(),
    ]
    .iter()
    .filter(|value| !value.trim().is_empty())
    .copied()
    .collect::<Vec<_>>()
    .join(" ");

    RawArrowFeature {
        query: RetrievalQueryData {
            has_full_first: py_len(&first) > 1,
            first,
            middle_initial_hashes,
            coauthor_hashes,
            coauthor_terms,
            affiliation_hashes,
            affiliation_terms: affiliation_query_terms,
            venue_hashes,
            title_hashes,
            year,
            orcid_hash,
            specter,
            specter_norm,
        },
        name_counts,
        paper_author_count,
        query_author,
    }
}

fn build_raw_arrow_author_signal_data(
    signature: &RawArrowSignature,
    paper_authors: Option<&Vec<(i64, String)>>,
    unidecode_char_map: &HashMap<char, String>,
) -> RawArrowAuthorSignalData {
    let mut paper_author_names = HashSet::new();
    let mut local10_author_names = HashSet::new();
    if let Some(authors) = paper_authors {
        for (position, author_name) in authors.iter() {
            let normalized = normalize_text_compat_from_map(author_name, false, unidecode_char_map);
            if normalized.is_empty() {
                continue;
            }
            paper_author_names.insert(normalized.clone());
            if let Some(author_position) = signature.position {
                if *position != author_position && (*position).abs_diff(author_position) <= 10 {
                    local10_author_names.insert(normalized);
                }
            }
        }
    }
    RawArrowAuthorSignalData {
        paper_author_names,
        local10_author_names,
    }
}

fn mask_raw_arrow_query(
    base: &RetrievalQueryData,
    requested_view: &str,
) -> Result<(RetrievalQueryData, String), String> {
    let resolved_view = if requested_view == "auto" {
        if base.has_full_first {
            "full"
        } else {
            "initial_only"
        }
    } else if requested_view == "full" || requested_view == "initial_only" {
        requested_view
    } else {
        return Err(format!("Unknown query view: {requested_view}"));
    };
    if resolved_view == "full" {
        return Ok((base.clone(), resolved_view.to_string()));
    }

    let masked = RetrievalQueryData {
        first: base
            .first
            .chars()
            .next()
            .map(|ch| ch.to_string())
            .unwrap_or_default(),
        has_full_first: false,
        middle_initial_hashes: Vec::new(),
        coauthor_hashes: base.coauthor_hashes.clone(),
        coauthor_terms: base.coauthor_terms.clone(),
        affiliation_hashes: base.affiliation_hashes.clone(),
        affiliation_terms: base.affiliation_terms.clone(),
        venue_hashes: base.venue_hashes.clone(),
        title_hashes: base.title_hashes.clone(),
        year: base.year,
        orcid_hash: base.orcid_hash,
        specter: base.specter.clone(),
        specter_norm: base.specter_norm,
    };
    Ok((masked, resolved_view.to_string()))
}

fn vector_norm_f32(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| {
            let val = *value as f64;
            val * val
        })
        .sum::<f64>()
        .sqrt()
}

fn euclidean_distance_f32(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| {
            let diff = (*left_value as f64) - (*right_value as f64);
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}

fn validate_raw_arrow_specter_dimensions(
    component_key: &str,
    vectors: &[&[f32]],
) -> Result<(), String> {
    let Some(first) = vectors.first() else {
        return Ok(());
    };
    let expected_dim = first.len();
    for vector in vectors.iter().skip(1) {
        if vector.len() != expected_dim {
            return Err(format!(
                "component_key '{}' has mixed SPECTER dimensions: expected {}, got {}",
                component_key,
                expected_dim,
                vector.len()
            ));
        }
    }
    Ok(())
}

fn select_raw_arrow_exemplars(vectors: &[&[f32]], max_exemplars: usize) -> Vec<Vec<f32>> {
    if max_exemplars == 0 || vectors.is_empty() {
        return Vec::new();
    }
    if vectors.len() <= max_exemplars {
        return vectors.iter().map(|vector| vector.to_vec()).collect();
    }
    let dim = vectors[0].len();
    let mut centroid = vec![0.0f32; dim];
    for vector in vectors.iter() {
        for (idx, value) in vector.iter().enumerate() {
            centroid[idx] += *value;
        }
    }
    for value in centroid.iter_mut() {
        *value /= vectors.len() as f32;
    }

    let mut selected = Vec::<usize>::new();
    let mut best_index = 0usize;
    let mut best_distance = f64::NEG_INFINITY;
    for (idx, vector) in vectors.iter().enumerate() {
        let distance = euclidean_distance_f32(vector, &centroid);
        if distance > best_distance {
            best_distance = distance;
            best_index = idx;
        }
    }
    selected.push(best_index);
    let mut selected_flags = vec![false; vectors.len()];
    selected_flags[best_index] = true;
    let mut min_distances = vec![f64::INFINITY; vectors.len()];
    for (idx, vector) in vectors.iter().enumerate() {
        if idx != best_index {
            min_distances[idx] = euclidean_distance_f32(vector, vectors[best_index]);
        }
    }

    while selected.len() < max_exemplars {
        let mut next_index = None;
        let mut next_distance = f64::NEG_INFINITY;
        for (idx, distance) in min_distances.iter().enumerate() {
            if selected_flags[idx] {
                continue;
            }
            if *distance > next_distance {
                next_distance = *distance;
                next_index = Some(idx);
            }
        }
        let Some(idx) = next_index else {
            break;
        };
        selected.push(idx);
        selected_flags[idx] = true;
        for (candidate_idx, vector) in vectors.iter().enumerate() {
            if selected_flags[candidate_idx] {
                continue;
            }
            let distance = euclidean_distance_f32(vector, vectors[idx]);
            if distance < min_distances[candidate_idx] {
                min_distances[candidate_idx] = distance;
            }
        }
    }
    selected
        .into_iter()
        .map(|idx| vectors[idx].to_vec())
        .collect()
}

fn build_raw_arrow_summary(
    component_key: &str,
    signature_ids: &[String],
    features_by_signature_id: &HashMap<String, RawArrowFeature>,
    max_exemplars: usize,
) -> Result<RetrievalSummaryData, String> {
    let mut first_name_counts: HashMap<String, usize> = HashMap::new();
    let mut middle_initial_counts: HashMap<u64, usize> = HashMap::new();
    let mut coauthor_counts: HashMap<u64, usize> = HashMap::new();
    let mut non_mega_coauthor_counts: HashMap<u64, usize> = HashMap::new();
    let mut affiliation_counts: HashMap<u64, usize> = HashMap::new();
    let mut venue_counts: HashMap<u64, usize> = HashMap::new();
    let mut title_counts: HashMap<u64, usize> = HashMap::new();
    let mut years = Vec::<i64>::new();
    let mut orcid_hashes = Vec::<u64>::new();
    let mut specter_vectors = Vec::<&[f32]>::new();
    let mut max_paper_author_count = 0usize;

    for signature_id in signature_ids {
        let feature = features_by_signature_id.get(signature_id).ok_or_else(|| {
            format!(
                "cluster seed signature_id '{}' is missing from computed raw Arrow features",
                signature_id
            )
        })?;
        if py_len(&feature.query.first) > 1 {
            *first_name_counts
                .entry(feature.query.first.clone())
                .or_insert(0) += 1;
        }
        for hash in feature.query.middle_initial_hashes.iter() {
            *middle_initial_counts.entry(*hash).or_insert(0) += 1;
        }
        for hash in feature.query.coauthor_hashes.iter() {
            *coauthor_counts.entry(*hash).or_insert(0) += 1;
            if feature.paper_author_count < RETRIEVAL_MEGA_AUTHOR_THRESHOLD {
                *non_mega_coauthor_counts.entry(*hash).or_insert(0) += 1;
            }
        }
        for hash in feature.query.affiliation_hashes.iter() {
            *affiliation_counts.entry(*hash).or_insert(0) += 1;
        }
        for hash in feature.query.venue_hashes.iter() {
            *venue_counts.entry(*hash).or_insert(0) += 1;
        }
        for hash in feature.query.title_hashes.iter() {
            *title_counts.entry(*hash).or_insert(0) += 1;
        }
        if let Some(year) = feature.query.year {
            years.push(year);
        }
        if let Some(orcid_hash) = feature.query.orcid_hash {
            orcid_hashes.push(orcid_hash);
        }
        if let Some(specter) = feature.query.specter.as_ref() {
            specter_vectors.push(specter.as_slice());
        }
        max_paper_author_count = max_paper_author_count.max(feature.paper_author_count);
    }
    validate_raw_arrow_specter_dimensions(component_key, &specter_vectors)?;

    let mut first_name_pairs: Vec<(String, f32)> = first_name_counts
        .into_iter()
        .map(|(name, count)| (name, count as f32))
        .collect();
    first_name_pairs.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    years.sort_unstable();
    orcid_hashes.sort_unstable();
    orcid_hashes.dedup();
    let year_min = years.first().copied();
    let year_max = years.last().copied();
    let year_mean = if years.is_empty() {
        None
    } else {
        Some(years.iter().sum::<i64>() as f64 / years.len() as f64)
    };

    let specter_centroid = if specter_vectors.is_empty() {
        None
    } else {
        let dim = specter_vectors[0].len();
        let mut centroid = vec![0.0f32; dim];
        for vector in specter_vectors.iter() {
            for (idx, value) in vector.iter().enumerate() {
                centroid[idx] += *value;
            }
        }
        for value in centroid.iter_mut() {
            *value /= specter_vectors.len() as f32;
        }
        Some(centroid)
    };
    let specter_centroid_norm = specter_centroid
        .as_ref()
        .map(|values| vector_norm_f32(values));
    let exemplar_vectors = select_raw_arrow_exemplars(&specter_vectors, max_exemplars);
    let exemplar_norms = exemplar_vectors
        .iter()
        .map(|values| vector_norm_f32(values))
        .collect();

    Ok(RetrievalSummaryData {
        component_key: component_key.to_string(),
        size: signature_ids.len(),
        first_name_counts: first_name_pairs,
        middle_initial_counts: counter_data_from_hash_count_map(middle_initial_counts),
        coauthor_counts: counter_data_from_hash_count_map(coauthor_counts),
        non_mega_coauthor_counts: counter_data_from_hash_count_map(non_mega_coauthor_counts),
        affiliation_counts: counter_data_from_hash_count_map(affiliation_counts),
        venue_counts: counter_data_from_hash_count_map(venue_counts),
        title_counts: counter_data_from_hash_count_map(title_counts),
        max_paper_author_count,
        year_min,
        year_max,
        year_mean,
        orcid_hashes,
        specter_centroid,
        specter_centroid_norm,
        exemplar_vectors,
        exemplar_norms,
    })
}

fn build_raw_arrow_summary_signals(
    signature_ids: &[String],
    features_by_signature_id: &HashMap<String, RawArrowFeature>,
    signatures: &HashMap<String, RawArrowSignature>,
    paper_authors: &HashMap<String, Vec<(i64, String)>>,
    unidecode_char_map: &HashMap<char, String>,
) -> Result<RawArrowSummarySignalData, String> {
    let mut name_counts_values = Vec::<NameCountsData>::new();
    let mut member_paper_author_names = Vec::<HashSet<String>>::with_capacity(signature_ids.len());
    let mut member_paper_author_counts = Vec::<usize>::with_capacity(signature_ids.len());
    let mut member_local10_author_names =
        Vec::<HashSet<String>>::with_capacity(signature_ids.len());
    let mut member_signature_ids = Vec::<String>::with_capacity(signature_ids.len());

    for signature_id in signature_ids {
        let feature = features_by_signature_id.get(signature_id).ok_or_else(|| {
            format!(
                "cluster seed signature_id '{}' is missing from computed raw Arrow features",
                signature_id
            )
        })?;
        if let Some(name_counts) = feature.name_counts.as_ref() {
            name_counts_values.push(name_counts.clone());
        }
        let signature = signatures.get(signature_id).ok_or_else(|| {
            format!(
                "cluster seed signature_id '{}' is missing from signatures",
                signature_id
            )
        })?;
        let author_signals = build_raw_arrow_author_signal_data(
            signature,
            paper_authors.get(&signature.paper_id),
            unidecode_char_map,
        );
        member_paper_author_names.push(author_signals.paper_author_names);
        member_paper_author_counts.push(feature.paper_author_count);
        member_local10_author_names.push(author_signals.local10_author_names);
        member_signature_ids.push(signature_id.clone());
    }

    Ok(RawArrowSummarySignalData {
        name_counts_values,
        member_paper_author_names,
        member_paper_author_counts,
        member_local10_author_names,
        member_signature_ids,
    })
}

fn round_six(value: f64) -> f32 {
    ((value * 1_000_000.0).round() / 1_000_000.0) as f32
}

fn valid_positive_finite(value: f64) -> Option<f64> {
    if value.is_finite() && value > 0.0 {
        Some(value)
    } else {
        None
    }
}

fn update_minimum(target: &mut Option<f64>, value: f64) {
    let Some(valid_value) = valid_positive_finite(value) else {
        return;
    };
    *target = Some(match target {
        Some(current) => current.min(valid_value),
        None => valid_value,
    });
}

fn name_count_rarity(value: Option<f64>) -> f32 {
    match value {
        Some(count) if count.is_finite() && count > 0.0 => round_six(1.0 / count.sqrt()),
        _ => 0.0,
    }
}

fn raw_arrow_name_count_rarity_row(
    query: &RetrievalQueryData,
    query_name_counts: &Option<NameCountsData>,
    summary: &RetrievalSummaryData,
    summary_signals: &RawArrowSummarySignalData,
) -> RawArrowNameCountRarityRow {
    let mut candidate_first_last_min = None;
    let mut candidate_last_min = None;
    for candidate_counts in summary_signals.name_counts_values.iter() {
        update_minimum(&mut candidate_first_last_min, candidate_counts.first_last);
        update_minimum(&mut candidate_last_min, candidate_counts.last);
    }
    let candidate_last_name_count_min_rarity = name_count_rarity(candidate_last_min);
    let candidate_last_first_name_count_min_rarity = name_count_rarity(candidate_first_last_min);

    let mut last_name_count_min_rarity = 0.0f32;
    let mut last_first_name_count_min_rarity = 0.0f32;
    if let Some(query_counts) = query_name_counts.as_ref() {
        let mut observed_minima: [Option<f64>; 6] = [None, None, None, None, None, None];
        for candidate_counts in summary_signals.name_counts_values.iter() {
            let values = compute_name_counts_data(Some(query_counts), Some(candidate_counts));
            for (index, value) in values.iter().enumerate() {
                update_minimum(&mut observed_minima[index], *value);
            }
        }
        last_name_count_min_rarity = name_count_rarity(observed_minima[2]);
        if query.has_full_first {
            last_first_name_count_min_rarity = name_count_rarity(observed_minima[1]);
        }
    }

    let mut first_prefix_match = 0.0f64;
    if py_len(&query.first) > 1 && summary.size > 0 {
        for (candidate_first, count) in summary.first_name_counts.iter() {
            if py_len(candidate_first) > 1
                && same_prefix_tokens_compat(&query.first, candidate_first)
            {
                first_prefix_match =
                    first_prefix_match.max((*count as f64) / (summary.size as f64));
            }
        }
    }

    RawArrowNameCountRarityRow {
        last_name_count_min_rarity,
        candidate_last_name_count_min_rarity,
        candidate_last_first_name_count_min_rarity,
        last_first_name_count_min_rarity,
        first_prefix_x_last_first_name_count_min_rarity: round_six(
            first_prefix_match * (last_first_name_count_min_rarity as f64),
        ),
    }
}

fn set_intersection_count(left: &HashSet<String>, right: &HashSet<String>) -> usize {
    if left.len() <= right.len() {
        left.iter().filter(|value| right.contains(*value)).count()
    } else {
        right.iter().filter(|value| left.contains(*value)).count()
    }
}

fn raw_arrow_paper_evidence_row(
    query_signature_id: &str,
    query_paper_author_count: usize,
    query_author_signals: &RawArrowAuthorSignalData,
    summary_signals: &RawArrowSummarySignalData,
) -> RawArrowPaperEvidenceRow {
    let mut best_author_jaccard = 0.0f64;
    let mut best_author_containment = 0.0f64;
    let mut best_author_overlap = 0.0f64;
    let mut best_local10_jaccard = 0.0f64;
    let mut best_local10_overlap_count = 0.0f64;
    let mut best_author_count_log_absdiff: Option<f64> = None;

    for (((candidate_names, candidate_count), candidate_local10_names), candidate_signature_id) in
        summary_signals
            .member_paper_author_names
            .iter()
            .zip(summary_signals.member_paper_author_counts.iter())
            .zip(summary_signals.member_local10_author_names.iter())
            .zip(summary_signals.member_signature_ids.iter())
    {
        if query_signature_id == candidate_signature_id {
            continue;
        }
        let intersection =
            set_intersection_count(&query_author_signals.paper_author_names, candidate_names);
        let union =
            query_author_signals.paper_author_names.len() + candidate_names.len() - intersection;
        if union > 0 {
            best_author_jaccard = best_author_jaccard.max((intersection as f64) / (union as f64));
        }
        let denominator = query_author_signals
            .paper_author_names
            .len()
            .min(candidate_names.len());
        if denominator > 0 {
            best_author_containment =
                best_author_containment.max((intersection as f64) / (denominator as f64));
        }
        best_author_overlap = best_author_overlap.max(intersection as f64);

        let local10_intersection = set_intersection_count(
            &query_author_signals.local10_author_names,
            candidate_local10_names,
        );
        let local10_union = query_author_signals.local10_author_names.len()
            + candidate_local10_names.len()
            - local10_intersection;
        if local10_union > 0 {
            best_local10_jaccard =
                best_local10_jaccard.max((local10_intersection as f64) / (local10_union as f64));
        }
        best_local10_overlap_count = best_local10_overlap_count.max(local10_intersection as f64);

        let count_delta =
            ((query_paper_author_count as f64).ln_1p() - (*candidate_count as f64).ln_1p()).abs();
        best_author_count_log_absdiff = Some(match best_author_count_log_absdiff {
            Some(current) => current.min(count_delta),
            None => count_delta,
        });
    }

    RawArrowPaperEvidenceRow {
        paper_author_list_max_jaccard: round_six(best_author_jaccard),
        paper_author_list_max_containment: round_six(best_author_containment),
        paper_author_list_max_overlap_count: round_six(best_author_overlap),
        local_author_window10_jaccard_max: round_six(best_local10_jaccard),
        local_author_window10_overlap_count_max: round_six(best_local10_overlap_count),
        best_author_count_log_absdiff: round_six(best_author_count_log_absdiff.unwrap_or(0.0)),
    }
}

fn extract_specter_for_paper_id(
    spec_dict: &Bound<'_, PyDict>,
    paper_id: &str,
) -> PyResult<Option<Vec<f32>>> {
    if let Ok(Some(val)) = spec_dict.get_item(paper_id) {
        return extract_specter_vec(&val);
    }
    if let Ok(i) = paper_id.parse::<i64>() {
        if let Ok(Some(val)) = spec_dict.get_item(i) {
            return extract_specter_vec(&val);
        }
    }
    if let Ok(u) = paper_id.parse::<u64>() {
        if let Ok(Some(val)) = spec_dict.get_item(u) {
            return extract_specter_vec(&val);
        }
    }
    Ok(None)
}

fn extract_string_opt(obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if obj.is_none() {
        Ok(None)
    } else {
        Ok(Some(obj.extract()?))
    }
}

fn extract_name_counts_data(obj: &Bound<'_, PyAny>) -> PyResult<Option<NameCountsData>> {
    if obj.is_none() {
        return Ok(None);
    }
    let first: Option<f64> = obj.getattr("first")?.extract()?;
    let first_last: Option<f64> = obj.getattr("first_last")?.extract()?;
    let last: Option<f64> = obj.getattr("last")?.extract()?;
    let last_first_initial: Option<f64> = obj.getattr("last_first_initial")?.extract()?;
    Ok(Some(NameCountsData {
        first: first.unwrap_or(f64::NAN),
        first_last: first_last.unwrap_or(f64::NAN),
        last: last.unwrap_or(f64::NAN),
        last_first_initial: last_first_initial.unwrap_or(f64::NAN),
    }))
}

fn extract_specter_vec(obj: &Bound<'_, PyAny>) -> PyResult<Option<Vec<f32>>> {
    if obj.is_none() {
        return Ok(None);
    }
    if let Ok(arr) = obj.downcast::<PyArray1<f32>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        let all_zero = slice.iter().all(|v| *v == 0.0);
        if all_zero {
            return Ok(None);
        }
        return Ok(Some(slice.to_vec()));
    }
    if let Ok(arr) = obj.downcast::<PyArray1<f64>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        let all_zero = slice.iter().all(|v| *v == 0.0);
        if all_zero {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(slice.len());
        for v in slice {
            out.push(*v as f32);
        }
        return Ok(Some(out));
    }
    // Fallback: try to extract as Vec<f64>
    let vec_f64: Vec<f64> = obj.extract()?;
    let all_zero = vec_f64.iter().all(|v| *v == 0.0);
    if all_zero {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(vec_f64.len());
    for v in vec_f64 {
        out.push(v as f32);
    }
    Ok(Some(out))
}

fn insert_nonzero_specter_row(out: &mut HashMap<String, Vec<f32>>, paper_id: &str, values: &[f32]) {
    if values.iter().all(|value| *value == 0.0) {
        return;
    }
    out.insert(paper_id.to_string(), values.to_vec());
}

fn extract_feature_block_specter_by_paper(
    feature_block: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, Vec<f32>>> {
    let paper_ids_obj = feature_block.getattr("specter_paper_ids")?;
    let paper_ids: Vec<String> = PyIterator::from_object(&paper_ids_obj)?
        .map(|item| item.and_then(|value| value.extract::<String>()))
        .collect::<PyResult<Vec<_>>>()?;
    let embeddings_obj = feature_block.getattr("specter_embeddings")?;
    if embeddings_obj.is_none() {
        if paper_ids.is_empty() {
            return Ok(HashMap::new());
        }
        return Err(pyo3::exceptions::PyValueError::new_err(
            "FeatureBlock specter_paper_ids requires specter_embeddings",
        ));
    }
    if paper_ids.is_empty() {
        return Ok(HashMap::new());
    }

    let mut out = HashMap::with_capacity(paper_ids.len());
    if let Ok(arr) = embeddings_obj.downcast::<PyArray2<f32>>() {
        let shape = arr.shape();
        if shape.len() != 2 || shape[0] != paper_ids.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "FeatureBlock specter_embeddings shape must be ({}, d), got {:?}",
                paper_ids.len(),
                shape
            )));
        }
        let cols = shape[1];
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        for (row_index, paper_id) in paper_ids.iter().enumerate() {
            let start = row_index * cols;
            insert_nonzero_specter_row(&mut out, paper_id, &slice[start..start + cols]);
        }
        return Ok(out);
    }
    if let Ok(arr) = embeddings_obj.downcast::<PyArray2<f64>>() {
        let shape = arr.shape();
        if shape.len() != 2 || shape[0] != paper_ids.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "FeatureBlock specter_embeddings shape must be ({}, d), got {:?}",
                paper_ids.len(),
                shape
            )));
        }
        let cols = shape[1];
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        for (row_index, paper_id) in paper_ids.iter().enumerate() {
            let start = row_index * cols;
            let values = slice[start..start + cols]
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            insert_nonzero_specter_row(&mut out, paper_id, &values);
        }
        return Ok(out);
    }

    let rows: Vec<Vec<f64>> = embeddings_obj.extract()?;
    if rows.len() != paper_ids.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "FeatureBlock specter_embeddings row count must match specter_paper_ids: {} != {}",
            rows.len(),
            paper_ids.len()
        )));
    }
    for (paper_id, row) in paper_ids.iter().zip(rows.iter()) {
        let values = row.iter().map(|value| *value as f32).collect::<Vec<_>>();
        insert_nonzero_specter_row(&mut out, paper_id, &values);
    }
    Ok(out)
}

fn extract_feature_block_name_tuples(
    py: Python<'_>,
    name_tuples: Option<&Bound<'_, PyAny>>,
) -> PyResult<HashMap<String, HashSet<String>>> {
    let Some(obj) = name_tuples else {
        return load_name_tuples_from_text_path(py, None);
    };
    if obj.is_none() {
        return Ok(HashMap::new());
    }
    if let Ok(value) = obj.extract::<String>() {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized.is_empty() || normalized == "none" {
            return Ok(HashMap::new());
        }
        if normalized == "filtered" {
            return load_name_tuples_from_text_path(py, None);
        }
        return load_name_tuples_from_text_path(py, Some(value.as_str()));
    }
    extract_name_tuples_map(obj)
}

fn extract_feature_block_cluster_seeds_require(
    feature_block: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, ClusterId>> {
    let obj = feature_block.getattr("cluster_seeds_require")?;
    let mut out = HashMap::new();
    for item in PyIterator::from_object(&obj)? {
        let (signature_id, component_id): (String, String) = item?.extract()?;
        out.insert(signature_id, ClusterId::Str(component_id));
    }
    Ok(out)
}

fn extract_feature_block_cluster_seeds_disallow(
    feature_block: &Bound<'_, PyAny>,
) -> PyResult<HashSet<(String, String)>> {
    let obj = feature_block.getattr("cluster_seeds_disallow")?;
    extract_pair_set(&obj)
}

fn extract_u32_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    if let Ok(arr) = obj.downcast::<PyArray1<u32>>() {
        let readonly = arr.readonly();
        return Ok(readonly.as_slice()?.to_vec());
    }
    if let Ok(arr) = obj.downcast::<PyArray1<u64>>() {
        let readonly = arr.readonly();
        return readonly
            .as_slice()?
            .iter()
            .map(|value| {
                u32::try_from(*value).map_err(|_| {
                    pyo3::exceptions::PyOverflowError::new_err(format!(
                        "component member signature index exceeds u32: {value}"
                    ))
                })
            })
            .collect();
    }
    let values: Vec<u64> = obj.extract()?;
    values
        .into_iter()
        .map(|value| {
            u32::try_from(value).map_err(|_| {
                pyo3::exceptions::PyOverflowError::new_err(format!(
                    "component member signature index exceeds u32: {value}"
                ))
            })
        })
        .collect()
}

fn extract_component_member_indices(obj: &Bound<'_, PyAny>) -> PyResult<HashMap<String, Vec<u32>>> {
    let mut out = HashMap::new();
    let items = obj.call_method0("items")?;
    for item in PyIterator::from_object(&items)? {
        let tuple = item?.downcast_into::<PyTuple>()?;
        if tuple.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "component_member_indices_by_key.items() yielded a non-pair",
            ));
        }
        let component_key: String = tuple.get_item(0)?.extract()?;
        let members = extract_u32_vec(&tuple.get_item(1)?)?;
        out.insert(component_key, members);
    }
    Ok(out)
}

fn extract_specter_vec_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut vectors = Vec::new();
    for item in PyIterator::from_object(obj)? {
        if let Some(vector) = extract_specter_vec(&item?)? {
            vectors.push(vector);
        }
    }
    Ok(vectors)
}

fn extract_string_count_pairs(obj: &Bound<'_, PyAny>) -> PyResult<Vec<(String, f32)>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let dict = obj.downcast::<PyDict>()?;
    if dict.len() == 0 {
        return Ok(Vec::new());
    }
    let mut entries = Vec::with_capacity(dict.len());
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        let val: f64 = v.extract()?;
        entries.push((key, val as f32));
    }
    Ok(entries)
}

fn term_token_count(value: &str) -> u8 {
    value
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .count()
        .min(u8::MAX as usize) as u8
}

fn extract_string_hashes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut hashes = Vec::new();
    for item in PyIterator::from_object(obj)? {
        let value: String = item?.extract()?;
        hashes.push(fnv64(value.as_bytes()));
    }
    hashes.sort_unstable();
    hashes.dedup();
    Ok(hashes)
}

fn normalize_orcid_str(value: &str) -> Option<String> {
    normalize_orcid_owned(value)
}

fn extract_orcid_hashes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut hashes = Vec::new();
    for item in PyIterator::from_object(obj)? {
        let value: String = item?.extract()?;
        if let Some(orcid) = normalize_orcid_str(&value) {
            hashes.push(fnv64(orcid.as_bytes()));
        }
    }
    hashes.sort_unstable();
    hashes.dedup();
    Ok(hashes)
}

fn extract_query_terms(obj: &Bound<'_, PyAny>) -> PyResult<Vec<RetrievalQueryTerm>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut terms = Vec::new();
    for item in PyIterator::from_object(obj)? {
        let value: String = item?.extract()?;
        terms.push(RetrievalQueryTerm {
            hash: fnv64(value.as_bytes()),
            token_count: term_token_count(&value),
        });
    }
    terms.sort_unstable_by_key(|term| term.hash);
    terms.dedup_by_key(|term| term.hash);
    Ok(terms)
}

fn extract_optional_orcid_hash(obj: &Bound<'_, PyAny>) -> PyResult<Option<u64>> {
    if obj.is_none() {
        return Ok(None);
    }
    let value: String = obj.extract()?;
    Ok(normalize_orcid_str(&value).map(|orcid| fnv64(orcid.as_bytes())))
}

fn same_prefix_tokens_compat(a: &str, b: &str) -> bool {
    let ta: Vec<&str> = a.split_whitespace().collect();
    let tb: Vec<&str> = b.split_whitespace().collect();
    for (x, y) in ta.iter().zip(tb.iter()) {
        if !(x.starts_with(y) || y.starts_with(x)) {
            return false;
        }
    }
    true
}

fn exact_name_match_compat(a: &str, b: &str) -> bool {
    !a.is_empty() && a == b
}

fn counter_query_overlap_hashes(
    query_hashes: &[u64],
    counter: &Option<CounterData>,
    size: usize,
) -> f64 {
    let Some(counter_data) = counter.as_ref() else {
        return 0.0;
    };
    if size == 0 || query_hashes.is_empty() || counter_data.entries.is_empty() {
        return 0.0;
    }
    let mut overlap = 0.0f64;
    for query_hash in query_hashes {
        if let Ok(index) = counter_data
            .entries
            .binary_search_by_key(query_hash, |entry| entry.0)
        {
            overlap += (counter_data.entries[index].1 as f64) / (size as f64);
        }
    }
    overlap / (query_hashes.len() as f64)
}

fn overlap_idf_weight(df_map: &HashMap<u64, usize>, hash: u64, total_summary_count: usize) -> f64 {
    let df = df_map.get(&hash).copied().unwrap_or(0) as f64;
    (((total_summary_count as f64) + 1.0) / (df + 1.0)).ln() + 1.0
}

fn overlap_query_term_weight(
    term: &RetrievalQueryTerm,
    df_map: &HashMap<u64, usize>,
    total_summary_count: usize,
    config: RetrievalOverlapConfig,
) -> f64 {
    if term.token_count < config.min_token_count {
        return 0.0;
    }
    let mut weight = if term.token_count <= 1 {
        config.unigram_weight
    } else {
        config.multi_token_weight
    };
    if config.use_idf {
        weight *= overlap_idf_weight(df_map, term.hash, total_summary_count);
    }
    weight.max(0.0)
}

fn weighted_counter_query_overlap(
    query_terms: &[RetrievalQueryTerm],
    counter: &Option<CounterData>,
    size: usize,
    df_map: &HashMap<u64, usize>,
    total_summary_count: usize,
    config: RetrievalOverlapConfig,
) -> f64 {
    let Some(counter_data) = counter.as_ref() else {
        return 0.0;
    };
    if size == 0 || query_terms.is_empty() || counter_data.entries.is_empty() {
        return 0.0;
    }
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for term in query_terms {
        let query_weight = overlap_query_term_weight(term, df_map, total_summary_count, config);
        if query_weight <= 0.0 {
            continue;
        }
        denominator += query_weight;
        if let Ok(index) = counter_data
            .entries
            .binary_search_by_key(&term.hash, |entry| entry.0)
        {
            let mut contribution = (counter_data.entries[index].1 as f64) / (size as f64);
            if let Some(cap) = config.per_term_cap {
                contribution = contribution.min(cap);
            }
            numerator += query_weight * contribution;
        }
    }
    if denominator <= 0.0 {
        return 0.0;
    }
    let mut score = numerator / denominator;
    if let Some(cap) = config.total_cap {
        score = score.min(cap);
    }
    score
}

fn middle_initial_score_hashes(
    query_hashes: &[u64],
    counter: &Option<CounterData>,
    size: usize,
) -> f64 {
    let Some(counter_data) = counter.as_ref() else {
        return 0.0;
    };
    if size == 0 || query_hashes.is_empty() || counter_data.entries.is_empty() {
        return 0.0;
    }
    let mut overlap = 0.0f64;
    let mut overlap_found = false;
    for query_hash in query_hashes {
        if let Ok(index) = counter_data
            .entries
            .binary_search_by_key(query_hash, |entry| entry.0)
        {
            overlap += (counter_data.entries[index].1 as f64) / (size as f64);
            overlap_found = true;
        }
    }
    if overlap_found {
        overlap / (query_hashes.len() as f64)
    } else {
        RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE
    }
}

fn first_name_score_mode(
    query_first: &str,
    counts: &[(String, f32)],
    size: usize,
    mode: RetrievalFirstNameMode,
) -> f64 {
    if size == 0 || py_len(query_first) <= 1 || counts.is_empty() {
        return 0.0;
    }
    let mut best = 0.0f64;
    for (first_name, count) in counts.iter() {
        if py_len(first_name) <= 1 {
            continue;
        }
        let share = (*count as f64) / (size as f64);
        let candidate = match mode {
            RetrievalFirstNameMode::Prefix => {
                if same_prefix_tokens_compat(query_first, first_name) {
                    share
                } else {
                    0.0
                }
            }
            RetrievalFirstNameMode::ExactOnly => {
                if exact_name_match_compat(query_first, first_name) {
                    share
                } else {
                    0.0
                }
            }
            RetrievalFirstNameMode::ExactThenPrefixHalf => {
                if exact_name_match_compat(query_first, first_name) {
                    share
                } else if same_prefix_tokens_compat(query_first, first_name) {
                    share * 0.5
                } else {
                    0.0
                }
            }
            RetrievalFirstNameMode::PrefixLengthRatio => {
                if same_prefix_tokens_compat(query_first, first_name) {
                    let query_len = py_len(query_first) as f64;
                    let candidate_len = py_len(first_name) as f64;
                    share * (query_len.min(candidate_len) / query_len.max(candidate_len))
                } else {
                    0.0
                }
            }
            RetrievalFirstNameMode::ExactThenPrefixLengthRatio => {
                if exact_name_match_compat(query_first, first_name) {
                    share
                } else if same_prefix_tokens_compat(query_first, first_name) {
                    let query_len = py_len(query_first) as f64;
                    let candidate_len = py_len(first_name) as f64;
                    share * (query_len.min(candidate_len) / query_len.max(candidate_len)) * 0.75
                } else {
                    0.0
                }
            }
        };
        best = best.max(candidate);
    }
    best
}

fn year_score(query_year: Option<i64>, summary: &RetrievalSummaryData) -> f64 {
    let Some(query_year_value) = query_year else {
        return 0.0;
    };
    let Some(year_mean) = summary.year_mean else {
        return 0.0;
    };
    let distance = ((query_year_value as f64) - year_mean).abs();
    let mut score = (1.0 - (distance / RETRIEVAL_YEAR_SCORE_DECAY_YEARS)).max(0.0);
    if let (Some(year_min), Some(year_max)) = (summary.year_min, summary.year_max) {
        if query_year_value < year_min - RETRIEVAL_YEAR_SCORE_RANGE_GAP
            || query_year_value > year_max + RETRIEVAL_YEAR_SCORE_RANGE_GAP
        {
            score -= RETRIEVAL_YEAR_SCORE_RANGE_PENALTY;
        }
    }
    score
}

fn contains_hashed_value(sorted_hashes: &[u64], target: u64) -> bool {
    sorted_hashes.binary_search(&target).is_ok()
}

fn has_middle_initial_conflict(query_hashes: &[u64], counter: &Option<CounterData>) -> bool {
    let Some(counter_data) = counter.as_ref() else {
        return false;
    };
    if query_hashes.is_empty() || counter_data.entries.is_empty() {
        return false;
    }
    !query_hashes.iter().any(|query_hash| {
        counter_data
            .entries
            .binary_search_by_key(query_hash, |entry| entry.0)
            .is_ok()
    })
}

fn has_impossible_year_conflict(
    query_year: Option<i64>,
    summary: &RetrievalSummaryData,
    max_year_gap: i64,
) -> bool {
    let Some(query_year_value) = query_year else {
        return false;
    };
    let (Some(year_min), Some(year_max)) = (summary.year_min, summary.year_max) else {
        return false;
    };
    query_year_value < year_min - max_year_gap || query_year_value > year_max + max_year_gap
}

fn extract_retrieval_summary(
    obj: &Bound<'_, PyAny>,
    include_exemplars: bool,
) -> PyResult<RetrievalSummaryData> {
    let component_key: String = obj.getattr("component_key")?.extract()?;
    let size: usize = obj.getattr("size")?.extract()?;
    let first_name_counts = extract_string_count_pairs(&obj.getattr("first_name_counts")?)?;
    let middle_initial_counts = extract_counter(&obj.getattr("middle_initial_counts")?)?;
    let coauthor_counts = extract_counter(&obj.getattr("coauthor_counts")?)?;
    let non_mega_coauthor_counts = extract_counter(&obj.getattr("non_mega_coauthor_counts")?)?;
    let affiliation_counts = extract_counter(&obj.getattr("affiliation_counts")?)?;
    let venue_counts = extract_counter(&obj.getattr("venue_counts")?)?;
    let title_counts = extract_counter(&obj.getattr("title_counts")?)?;
    let max_paper_author_count: usize = obj.getattr("max_paper_author_count")?.extract()?;
    let year_min: Option<i64> = obj.getattr("year_min")?.extract()?;
    let year_max: Option<i64> = obj.getattr("year_max")?.extract()?;
    let year_mean: Option<f64> = obj.getattr("year_mean")?.extract()?;
    let orcid_hashes = extract_orcid_hashes(&obj.getattr("orcid_values")?)?;
    let specter_centroid = extract_specter_vec(&obj.getattr("specter_centroid")?)?;
    let specter_centroid_norm = specter_centroid.as_ref().map(|values| {
        values
            .iter()
            .map(|value| {
                let val = *value as f64;
                val * val
            })
            .sum::<f64>()
            .sqrt()
    });
    let exemplar_vectors = if include_exemplars {
        extract_specter_vec_list(&obj.getattr("exemplar_vectors")?)?
    } else {
        Vec::new()
    };
    let exemplar_norms = exemplar_vectors
        .iter()
        .map(|values| {
            values
                .iter()
                .map(|value| {
                    let val = *value as f64;
                    val * val
                })
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    Ok(RetrievalSummaryData {
        component_key,
        size,
        first_name_counts,
        middle_initial_counts,
        coauthor_counts,
        non_mega_coauthor_counts,
        affiliation_counts,
        venue_counts,
        title_counts,
        max_paper_author_count,
        year_min,
        year_max,
        year_mean,
        orcid_hashes,
        specter_centroid,
        specter_centroid_norm,
        exemplar_vectors,
        exemplar_norms,
    })
}

fn extract_retrieval_query(obj: &Bound<'_, PyAny>) -> PyResult<RetrievalQueryData> {
    let first: String = obj.getattr("first")?.extract()?;
    let has_full_first = match obj.getattr("has_full_first") {
        Ok(value) => value.extract()?,
        Err(_) => first.chars().count() > 1,
    };
    let middle_initial_hashes = extract_string_hashes(&obj.getattr("middle_initials")?)?;
    let coauthor_terms = extract_query_terms(&obj.getattr("coauthor_blocks")?)?;
    let coauthor_hashes = coauthor_terms.iter().map(|term| term.hash).collect();
    let affiliation_terms = extract_query_terms(&obj.getattr("affiliation_terms")?)?;
    let affiliation_hashes = affiliation_terms.iter().map(|term| term.hash).collect();
    let venue_hashes = extract_string_hashes(&obj.getattr("venue_terms")?)?;
    let title_hashes = extract_string_hashes(&obj.getattr("title_terms")?)?;
    let year: Option<i64> = obj.getattr("year")?.extract()?;
    let orcid_hash = extract_optional_orcid_hash(&obj.getattr("orcid")?)?;
    let specter = extract_specter_vec(&obj.getattr("specter")?)?;
    let specter_norm = specter.as_ref().map(|values| {
        values
            .iter()
            .map(|value| {
                let val = *value as f64;
                val * val
            })
            .sum::<f64>()
            .sqrt()
    });
    let specter = specter.map(Arc::new);

    Ok(RetrievalQueryData {
        first,
        has_full_first,
        middle_initial_hashes,
        coauthor_hashes,
        coauthor_terms,
        affiliation_hashes,
        affiliation_terms,
        venue_hashes,
        title_hashes,
        year,
        orcid_hash,
        specter,
        specter_norm,
    })
}

fn extract_retrieval_weights(weights: Vec<f64>) -> PyResult<RetrievalHybridWeights> {
    if weights.len() != 5 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected 5 retrieval weights, got {}",
            weights.len()
        )));
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Retrieval weights must all be finite",
        ));
    }
    Ok(RetrievalHybridWeights::from_array([
        weights[0], weights[1], weights[2], weights[3], weights[4],
    ]))
}

fn default_overlap_config() -> RetrievalOverlapConfig {
    RetrievalOverlapConfig {
        use_idf: false,
        per_term_cap: None,
        total_cap: None,
        min_token_count: 1,
        unigram_weight: 1.0,
        multi_token_weight: 1.0,
    }
}

fn parse_first_name_mode(mode: &str) -> PyResult<RetrievalFirstNameMode> {
    match mode {
        "prefix" => Ok(RetrievalFirstNameMode::Prefix),
        "exact_only" => Ok(RetrievalFirstNameMode::ExactOnly),
        "exact_then_prefix_half" => Ok(RetrievalFirstNameMode::ExactThenPrefixHalf),
        "prefix_length_ratio" => Ok(RetrievalFirstNameMode::PrefixLengthRatio),
        "exact_then_prefix_length_ratio" => Ok(RetrievalFirstNameMode::ExactThenPrefixLengthRatio),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown first_name_mode: {mode}"
        ))),
    }
}

fn parse_specter_mode(mode: &str) -> PyResult<RetrievalSpecterMode> {
    match mode {
        "centroid" => Ok(RetrievalSpecterMode::Centroid),
        "exemplar_max" => Ok(RetrievalSpecterMode::ExemplarMax),
        "centroid_exemplar_50_50" => Ok(RetrievalSpecterMode::CentroidExemplar50_50),
        "centroid_exemplar_25_75" => Ok(RetrievalSpecterMode::CentroidExemplar25_75),
        "centroid_exemplar_75_25" => Ok(RetrievalSpecterMode::CentroidExemplar75_25),
        "max_centroid_exemplar" => Ok(RetrievalSpecterMode::MaxOfCentroidExemplar),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown specter_mode: {mode}"
        ))),
    }
}

fn build_experimental_config(
    first_name_mode: &str,
    specter_mode: &str,
    coauthor_use_idf: bool,
    coauthor_per_term_cap: Option<f64>,
    coauthor_total_cap: Option<f64>,
    drop_candidate_mega_coauthors: bool,
    mega_coauthor_rescue_query_coverage: Option<f64>,
    mega_coauthor_rescue_min_shared_blocks: usize,
    affiliation_use_idf: bool,
    affiliation_per_term_cap: Option<f64>,
    affiliation_total_cap: Option<f64>,
    affiliation_min_token_count: usize,
    affiliation_unigram_weight: f64,
    affiliation_multi_token_weight: f64,
) -> PyResult<RetrievalExperimentalConfig> {
    if affiliation_min_token_count == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "affiliation_min_token_count must be positive",
        ));
    }
    if !affiliation_unigram_weight.is_finite() || !affiliation_multi_token_weight.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Affiliation structure weights must be finite",
        ));
    }
    if let Some(coverage) = mega_coauthor_rescue_query_coverage {
        if !coverage.is_finite() || coverage <= 0.0 || coverage > 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mega_coauthor_rescue_query_coverage must be in (0, 1]",
            ));
        }
    }
    if mega_coauthor_rescue_min_shared_blocks == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mega_coauthor_rescue_min_shared_blocks must be positive",
        ));
    }
    Ok(RetrievalExperimentalConfig {
        first_name_mode: parse_first_name_mode(first_name_mode)?,
        specter_mode: parse_specter_mode(specter_mode)?,
        coauthor: RetrievalOverlapConfig {
            use_idf: coauthor_use_idf,
            per_term_cap: coauthor_per_term_cap,
            total_cap: coauthor_total_cap,
            ..default_overlap_config()
        },
        drop_candidate_mega_coauthors,
        mega_coauthor_rescue_query_coverage,
        mega_coauthor_rescue_min_shared_blocks,
        affiliation: RetrievalOverlapConfig {
            use_idf: affiliation_use_idf,
            per_term_cap: affiliation_per_term_cap,
            total_cap: affiliation_total_cap,
            min_token_count: affiliation_min_token_count.min(u8::MAX as usize) as u8,
            unigram_weight: affiliation_unigram_weight,
            multi_token_weight: affiliation_multi_token_weight,
        },
    })
}

fn specter_exemplar_score(query: &RetrievalQueryData, summary: &RetrievalSummaryData) -> f64 {
    let (Some(query_specter), Some(query_norm)) = (query.specter.as_ref(), query.specter_norm)
    else {
        return 0.0;
    };
    summary
        .exemplar_vectors
        .iter()
        .zip(summary.exemplar_norms.iter())
        .map(|(vector, norm)| cosine_sim_with_norms(query_specter, query_norm, vector, *norm))
        .fold(0.0f64, f64::max)
}

fn query_counter_overlap_count(
    query_terms: &[RetrievalQueryTerm],
    counter: &Option<CounterData>,
) -> usize {
    let Some(counter_data) = counter.as_ref() else {
        return 0;
    };
    query_terms
        .iter()
        .filter(|term| {
            counter_data
                .entries
                .binary_search_by_key(&term.hash, |entry| entry.0)
                .is_ok()
        })
        .count()
}

fn should_rescue_candidate_mega_coauthors(
    query: &RetrievalQueryData,
    summary: &RetrievalSummaryData,
    config: RetrievalExperimentalConfig,
) -> bool {
    if !config.drop_candidate_mega_coauthors {
        return false;
    }
    let Some(min_query_coverage) = config.mega_coauthor_rescue_query_coverage else {
        return false;
    };
    if summary.max_paper_author_count < RETRIEVAL_MEGA_AUTHOR_THRESHOLD
        || query.coauthor_terms.is_empty()
    {
        return false;
    }

    let full_overlap = query_counter_overlap_count(&query.coauthor_terms, &summary.coauthor_counts);
    if full_overlap < config.mega_coauthor_rescue_min_shared_blocks {
        return false;
    }
    let filtered_overlap =
        query_counter_overlap_count(&query.coauthor_terms, &summary.non_mega_coauthor_counts);
    if full_overlap <= filtered_overlap {
        return false;
    }

    (full_overlap as f64) / (query.coauthor_terms.len() as f64) >= min_query_coverage
}

fn score_experimental_hybrid_centroid_query(
    query: &RetrievalQueryData,
    summary: &RetrievalSummaryData,
    weights: RetrievalHybridWeights,
    config: RetrievalExperimentalConfig,
    coauthor_cluster_df: &HashMap<u64, usize>,
    non_mega_coauthor_cluster_df: &HashMap<u64, usize>,
    affiliation_cluster_df: &HashMap<u64, usize>,
    total_summary_count: usize,
) -> f32 {
    let use_non_mega_coauthor_counter = config.drop_candidate_mega_coauthors
        && summary.max_paper_author_count >= RETRIEVAL_MEGA_AUTHOR_THRESHOLD
        && !should_rescue_candidate_mega_coauthors(query, summary, config);
    let (coauthor_counts, coauthor_df) = if use_non_mega_coauthor_counter {
        (
            &summary.non_mega_coauthor_counts,
            non_mega_coauthor_cluster_df,
        )
    } else {
        (&summary.coauthor_counts, coauthor_cluster_df)
    };
    let coauthor_score = weighted_counter_query_overlap(
        &query.coauthor_terms,
        coauthor_counts,
        summary.size,
        coauthor_df,
        total_summary_count,
        config.coauthor,
    );
    let affiliation_score = weighted_counter_query_overlap(
        &query.affiliation_terms,
        &summary.affiliation_counts,
        summary.size,
        affiliation_cluster_df,
        total_summary_count,
        config.affiliation,
    );
    let middle_score = middle_initial_score_hashes(
        &query.middle_initial_hashes,
        &summary.middle_initial_counts,
        summary.size,
    );
    let first_name_score = first_name_score_mode(
        &query.first,
        &summary.first_name_counts,
        summary.size,
        config.first_name_mode,
    );
    let centroid_score = match (
        query.specter.as_ref(),
        query.specter_norm,
        summary.specter_centroid.as_ref(),
        summary.specter_centroid_norm,
    ) {
        (Some(query_specter), Some(query_norm), Some(summary_specter), Some(summary_norm)) => {
            cosine_sim_with_norms(query_specter, query_norm, summary_specter, summary_norm)
        }
        _ => 0.0,
    };
    let exemplar_score = specter_exemplar_score(query, summary);
    let specter_score = match config.specter_mode {
        RetrievalSpecterMode::Centroid => centroid_score,
        RetrievalSpecterMode::ExemplarMax => exemplar_score,
        RetrievalSpecterMode::CentroidExemplar50_50 => 0.5 * centroid_score + 0.5 * exemplar_score,
        RetrievalSpecterMode::CentroidExemplar25_75 => {
            0.25 * centroid_score + 0.75 * exemplar_score
        }
        RetrievalSpecterMode::CentroidExemplar75_25 => {
            0.75 * centroid_score + 0.25 * exemplar_score
        }
        RetrievalSpecterMode::MaxOfCentroidExemplar => centroid_score.max(exemplar_score),
    };
    (weights.centroid * specter_score
        + weights.coauthor * coauthor_score
        + weights.affiliation * affiliation_score
        + weights.middle * middle_score
        + weights.first_name * first_name_score) as f32
}

fn chooser_summary_features(
    query: &RetrievalQueryData,
    summary: &RetrievalSummaryData,
) -> [f32; 8] {
    let middle_score = middle_initial_score_hashes(
        &query.middle_initial_hashes,
        &summary.middle_initial_counts,
        summary.size,
    ) as f32;
    let affiliation_score = counter_query_overlap_hashes(
        &query.affiliation_hashes,
        &summary.affiliation_counts,
        summary.size,
    ) as f32;
    let coauthor_score = counter_query_overlap_hashes(
        &query.coauthor_hashes,
        &summary.coauthor_counts,
        summary.size,
    ) as f32;
    let venue_score =
        counter_query_overlap_hashes(&query.venue_hashes, &summary.venue_counts, summary.size)
            as f32;
    let year_score_value = year_score(query.year, summary) as f32;
    let title_score =
        counter_query_overlap_hashes(&query.title_hashes, &summary.title_counts, summary.size)
            as f32;
    let specter_centroid_score = match (
        query.specter.as_ref(),
        query.specter_norm,
        summary.specter_centroid.as_ref(),
        summary.specter_centroid_norm,
    ) {
        (Some(query_specter), Some(query_norm), Some(summary_specter), Some(summary_norm)) => {
            cosine_sim_with_norms(query_specter, query_norm, summary_specter, summary_norm) as f32
        }
        _ => 0.0,
    };
    let specter_exemplar_score = specter_exemplar_score(query, summary) as f32;
    [
        middle_score,
        affiliation_score,
        coauthor_score,
        venue_score,
        year_score_value,
        title_score,
        specter_centroid_score,
        specter_exemplar_score,
    ]
}

fn update_cluster_df_from_counter(
    obj: &Bound<'_, PyAny>,
    df_map: &mut HashMap<u64, usize>,
) -> PyResult<()> {
    if obj.is_none() {
        return Ok(());
    }
    let dict = obj.downcast::<PyDict>()?;
    for (key_obj, _value_obj) in dict.iter() {
        let key: String = key_obj.extract()?;
        let hash = fnv64(key.as_bytes());
        *df_map.entry(hash).or_insert(0) += 1;
    }
    Ok(())
}

fn specter_payload_to_dict<'py>(
    py: Python<'py>,
    payload: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    if let Ok(dict) = payload.downcast::<PyDict>() {
        return Ok(dict.clone());
    }

    if let Ok(tuple_payload) = payload.downcast::<PyTuple>() {
        if tuple_payload.len() != 2 {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported specter pickle tuple payload; expected (X, keys), got tuple length {}",
                tuple_payload.len()
            )));
        }
        let matrix = tuple_payload.get_item(0)?;
        let keys = tuple_payload.get_item(1)?;
        let out = PyDict::new(py);
        for (idx, key_item) in PyIterator::from_object(&keys)?.enumerate() {
            let key = key_item?;
            let row = matrix.get_item(idx)?;
            out.set_item(key, row)?;
        }
        return Ok(out);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Unsupported specter pickle payload; expected dict or (X, keys) tuple",
    ))
}

fn load_pickle_dict<'py>(py: Python<'py>, path: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
    let builtins = py.import("builtins")?;
    let pickle = py.import("pickle")?;
    let file_obj = builtins.call_method1("open", (path, "rb"))?;
    let loaded = pickle.call_method1("load", (&file_obj,));
    let _ = file_obj.call_method0("close");
    match loaded {
        Ok(value) => {
            if value.is_none() {
                Ok(None)
            } else {
                Ok(Some(specter_payload_to_dict(py, &value)?))
            }
        }
        Err(err) => Err(err),
    }
}

type JsonValue = serde_json::Value;
type JsonObject = serde_json::Map<String, JsonValue>;

fn load_json_value(path: &str) -> PyResult<JsonValue> {
    let file = File::open(path).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!("failed to open JSON path {}: {}", path, err))
    })?;
    let reader = BufReader::new(file);
    serde_json::from_reader::<_, JsonValue>(reader).map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "failed to parse JSON path {}: {}",
            path, err
        ))
    })
}

fn json_as_object<'a>(value: &'a JsonValue, context: &str) -> PyResult<&'a JsonObject> {
    value.as_object().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("{} must be a JSON object", context))
    })
}

fn json_value_to_string(value: &JsonValue) -> Option<String> {
    match value {
        JsonValue::String(v) => Some(v.clone()),
        JsonValue::Number(v) => Some(v.to_string()),
        JsonValue::Bool(v) => Some(v.to_string()),
        _ => None,
    }
}

fn json_value_to_id(value: &JsonValue) -> Option<String> {
    match value {
        JsonValue::String(v) => Some(v.clone()),
        JsonValue::Number(v) => Some(v.to_string()),
        _ => None,
    }
}

fn json_value_to_i64(value: &JsonValue, context: &str) -> PyResult<Option<i64>> {
    match value {
        JsonValue::Number(v) => {
            if let Some(i) = v.as_i64() {
                Ok(Some(i))
            } else if let Some(u) = v.as_u64() {
                i64::try_from(u).map(Some).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "{context} is outside i64 range: {u}"
                    ))
                })
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{context} must be an integer, got floating-point value {v}"
                )))
            }
        }
        JsonValue::Null => Ok(None),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{context} must be an integer or null"
        ))),
    }
}

fn json_get_required<'a>(obj: &'a JsonObject, key: &str, context: &str) -> PyResult<&'a JsonValue> {
    obj.get(key).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!(
            "missing required key '{}' in {}",
            key, context
        ))
    })
}

fn json_get_string(obj: &JsonObject, key: &str, default: &str) -> String {
    match obj.get(key) {
        None | Some(JsonValue::Null) => default.to_string(),
        Some(value) => json_value_to_string(value).unwrap_or_else(|| default.to_string()),
    }
}

fn json_get_optional_string(obj: &JsonObject, key: &str) -> Option<String> {
    obj.get(key).and_then(|value| match value {
        JsonValue::Null => None,
        _ => json_value_to_string(value),
    })
}

fn json_get_i64_optional(obj: &JsonObject, key: &str, context: &str) -> PyResult<Option<i64>> {
    let field_context = format!("{context}.{key}");
    match obj.get(key) {
        Some(value) => json_value_to_i64(value, &field_context),
        None => Ok(None),
    }
}

fn json_get_string_list(value: Option<&JsonValue>) -> Vec<String> {
    let Some(array) = value.and_then(JsonValue::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(array.len());
    for item in array {
        if let Some(text) = json_value_to_string(item) {
            out.push(text);
        }
    }
    out
}

fn json_get_id_set(value: Option<&JsonValue>) -> HashSet<PaperId> {
    let Some(array) = value.and_then(JsonValue::as_array) else {
        return HashSet::new();
    };
    let mut out = HashSet::with_capacity(array.len());
    for item in array {
        if let Some(id) = json_value_to_id(item) {
            out.insert(id);
        }
    }
    out
}

fn json_extract_string_f64_map(value: Option<&JsonValue>) -> HashMap<String, f64> {
    let Some(object) = value.and_then(JsonValue::as_object) else {
        return HashMap::new();
    };
    let mut out = HashMap::with_capacity(object.len());
    for (key, val) in object {
        if let Some(f) = val.as_f64() {
            out.insert(key.clone(), f);
        } else if let Some(i) = val.as_i64() {
            out.insert(key.clone(), i as f64);
        } else if let Some(u) = val.as_u64() {
            out.insert(key.clone(), u as f64);
        }
    }
    out
}

fn load_raw_name_counts_from_json_path(
    path: Option<&str>,
    expected_normalization_version: Option<&str>,
    allow_normalization_version_mismatch: bool,
) -> PyResult<RawNameCountMaps> {
    let Some(path_value) = path else {
        return Ok(RawNameCountMaps::default());
    };
    let counts_json = load_json_value(path_value)?;
    let counts_obj = json_as_object(&counts_json, "name counts payload")?;

    // Validate normalization_version if an expected version was provided.
    if let Some(expected) = expected_normalization_version {
        match counts_obj.get("normalization_version") {
            None => {
                let msg = format!(
                    "Missing normalization_version in name counts artifact; fail-fast by default. \
                     path={} expected={} set allow_normalization_version_mismatch=true explicitly to override",
                    path_value, expected,
                );
                if !allow_normalization_version_mismatch {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(msg));
                }
                eprintln!("WARNING: {}", msg);
            }
            Some(artifact_val) => {
                let artifact_version = artifact_val.as_str().unwrap_or("");
                if artifact_version != expected {
                    let msg = format!(
                        "Normalization version mismatch between runtime and name-count artifact; \
                         fail-fast by default. path={} expected={} artifact={} \
                         set allow_normalization_version_mismatch=true explicitly to override",
                        path_value, expected, artifact_version,
                    );
                    if !allow_normalization_version_mismatch {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(msg));
                    }
                    eprintln!("WARNING: {}", msg);
                }
            }
        }
    }

    let first = json_extract_string_f64_map(counts_obj.get("first_dict"));
    let last = json_extract_string_f64_map(counts_obj.get("last_dict"));
    let first_last = json_extract_string_f64_map(counts_obj.get("first_last_dict"));
    let last_first_initial = json_extract_string_f64_map(counts_obj.get("last_first_initial_dict"));
    Ok(RawNameCountMaps {
        first,
        last,
        first_last,
        last_first_initial,
        index: None,
    })
}

fn default_name_tuples_path(py: Python<'_>) -> PyResult<String> {
    let consts = py.import("s2and.consts")?;
    let package_data_dir: String = consts.getattr("_PACKAGE_DATA_DIR")?.extract()?;
    let pathlib = py.import("pathlib")?;
    let path_obj = pathlib
        .getattr("Path")?
        .call1((package_data_dir,))?
        .call_method1("joinpath", ("s2and_name_tuples_filtered.txt",))?;
    path_obj.call_method0("as_posix")?.extract()
}

fn load_name_tuples_from_text_path(
    py: Python<'_>,
    path: Option<&str>,
) -> PyResult<HashMap<String, HashSet<String>>> {
    let effective_path = match path {
        Some(value) => value.to_string(),
        None => default_name_tuples_path(py)?,
    };
    if !Path::new(&effective_path).exists() {
        return Ok(HashMap::new());
    }
    let text = fs::read_to_string(&effective_path).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "failed to read name tuples path {}: {}",
            effective_path, err
        ))
    })?;
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((a, b)) = trimmed.split_once(',') {
            insert_name_tuple_alias(&mut out, a.to_string(), b.to_string());
        }
    }
    Ok(out)
}

fn has_name_counts_artifact(raw_name_counts: &RawNameCountMaps) -> bool {
    raw_name_counts.has_data()
}

fn canonical_last_for_counts(raw_last: &str, normalized_last: &str) -> String {
    if contains_name_dash(raw_last) || normalized_last.contains(' ') {
        normalized_last.replace(' ', "")
    } else {
        normalized_last.to_string()
    }
}

#[derive(Clone, Copy, Default)]
struct NameCountsDefaultTelemetry {
    first: bool,
    first_last: bool,
    last: bool,
    last_first_initial: bool,
}

impl NameCountsDefaultTelemetry {
    fn any(self) -> bool {
        self.first || self.first_last || self.last || self.last_first_initial
    }
}

struct NameCountsBuildResult {
    data: Option<NameCountsData>,
    telemetry: NameCountsDefaultTelemetry,
}

fn build_name_counts_data_from_artifact(
    raw_name_counts: &RawNameCountMaps,
    raw_first: &str,
    _first_normalized_token: &str,
    first_without_apostrophe: &str,
    raw_last: &str,
    last_normalized: &str,
) -> NameCountsBuildResult {
    if !has_name_counts_artifact(raw_name_counts) {
        return NameCountsBuildResult {
            data: None,
            telemetry: NameCountsDefaultTelemetry::default(),
        };
    }

    let mut telemetry = NameCountsDefaultTelemetry::default();
    let mut first_for_counts = first_without_apostrophe
        .split(' ')
        .next()
        .unwrap_or("")
        .to_string();
    if contains_name_dash(raw_first) {
        let joined = first_without_apostrophe.replace(' ', "");
        if !joined.is_empty() {
            first_for_counts = joined;
        }
    }

    let last_for_counts = canonical_last_for_counts(raw_last, last_normalized);
    let first_last_key = format!("{} {}", first_for_counts, last_for_counts)
        .trim()
        .to_string();
    let first_initial = first_for_counts
        .chars()
        .next()
        .map(|ch| ch.to_string())
        .unwrap_or_default();
    let last_first_initial_key = format!("{} {}", last_for_counts, first_initial)
        .trim()
        .to_string();

    let first = if py_len(&first_for_counts) > 1 {
        match raw_name_counts.get(RawNameCountKind::First, &first_for_counts) {
            Some(value) => value,
            None => {
                telemetry.first = true;
                1.0
            }
        }
    } else {
        f64::NAN
    };
    let first_last = if py_len(&first_for_counts) > 1 {
        match raw_name_counts.get(RawNameCountKind::FirstLast, &first_last_key) {
            Some(value) => value,
            None => {
                telemetry.first_last = true;
                1.0
            }
        }
    } else {
        f64::NAN
    };
    let last = match raw_name_counts.get(RawNameCountKind::Last, &last_for_counts) {
        Some(value) => value,
        None => {
            telemetry.last = true;
            1.0
        }
    };
    let last_first_initial =
        match raw_name_counts.get(RawNameCountKind::LastFirstInitial, &last_first_initial_key) {
            Some(value) => value,
            None => {
                telemetry.last_first_initial = true;
                1.0
            }
        };

    NameCountsBuildResult {
        data: Some(NameCountsData {
            first,
            first_last,
            last,
            last_first_initial,
        }),
        telemetry,
    }
}

fn count_initials(s: &str) -> HashMap<char, usize> {
    let mut counts = HashMap::new();
    for part in s.split(' ') {
        if !part.is_empty() {
            if let Some(ch) = part.chars().next() {
                *counts.entry(ch).or_insert(0) += 1;
            }
        }
    }
    counts
}

fn lasts_equivalent_for_constraint(l1: &str, l2: &str) -> bool {
    if l1 == l2 {
        return true;
    }
    l1.replace(' ', "") == l2.replace(' ', "")
}

fn same_prefix_tokens(a: &str, b: &str) -> bool {
    let mut ita = a.split_whitespace();
    let mut itb = b.split_whitespace();
    loop {
        match (ita.next(), itb.next()) {
            (Some(x), Some(y)) => {
                if !(x.starts_with(y) || y.starts_with(x)) {
                    return false;
                }
            }
            _ => return true,
        }
    }
}

fn name_tuple_contains(map: &HashMap<String, HashSet<String>>, a: &str, b: &str) -> bool {
    map.get(a).map_or(false, |vals| vals.contains(b))
        || map.get(b).map_or(false, |vals| vals.contains(a))
}

fn first_name_forms(value: &str) -> (String, String, String) {
    let normalized = value.to_string();
    let parts: Vec<&str> = normalized.split_whitespace().collect();
    let joined = parts.join("");
    let token = parts
        .first()
        .map_or_else(|| normalized.clone(), |part| (*part).to_string());
    (normalized, joined, token)
}

fn first_names_name_compatible(
    first_1: &str,
    first_2: &str,
    name_tuples: &HashMap<String, HashSet<String>>,
) -> bool {
    if same_prefix_tokens(first_1, first_2) {
        return true;
    }
    let forms_1 = first_name_forms(first_1);
    let forms_2 = first_name_forms(first_2);
    name_tuple_contains(name_tuples, &forms_1.0, &forms_2.0)
        || name_tuple_contains(name_tuples, &forms_1.1, &forms_2.1)
        || name_tuple_contains(name_tuples, &forms_1.2, &forms_2.2)
}

fn subblock_tokens_from_key(subblock_key: &str) -> Vec<String> {
    let mut values = HashSet::new();
    for raw_token in subblock_key.split(',') {
        let token = raw_token
            .trim()
            .split_once('|')
            .map_or(raw_token.trim(), |(token, _rest)| token.trim());
        if py_len(&token) > 1 {
            values.insert(token.to_string());
        }
    }
    let mut out: Vec<String> = values.into_iter().collect();
    out.sort_unstable();
    out
}

#[derive(Clone)]
struct SubblockingSignatureRow {
    signature_id: String,
    first: String,
    middle: String,
    orcid: Option<String>,
}

fn normalize_subblocking_signature_rows(
    rows: &mut [SubblockingSignatureRow],
    name_prefixes: &HashSet<String>,
    unidecode_char_map: &HashMap<char, String>,
) {
    for row in rows.iter_mut() {
        let (first, middle) = split_first_middle_hyphen_aware_compat(
            &row.first,
            &row.middle,
            name_prefixes,
            unidecode_char_map,
        );
        row.first = first;
        row.middle = middle;
    }
}

fn read_subblocking_signature_rows_from_batches(
    path: &str,
    batches: Vec<RecordBatch>,
    keep_signature_ids: Option<&HashSet<String>>,
) -> PyResult<HashMap<String, SubblockingSignatureRow>> {
    let mut out = HashMap::new();
    for batch in batches {
        let signature_id_col = batch.column(arrow_column_index(&batch, "signature_id", path)?);
        let signature_id_values =
            ArrowStringColumn::from_array(signature_id_col.as_ref(), "signature_id")?;
        let first_col = batch.column(arrow_column_index(&batch, "author_first", path)?);
        let first_values = ArrowStringColumn::from_array(first_col.as_ref(), "author_first")?;
        let middle_col = batch.column(arrow_column_index(&batch, "author_middle", path)?);
        let middle_values = ArrowStringColumn::from_array(middle_col.as_ref(), "author_middle")?;
        let orcid_col = batch.column(arrow_column_index(&batch, "author_orcid", path)?);
        let orcid_values = ArrowStringColumn::from_array(orcid_col.as_ref(), "author_orcid")?;
        for row in 0..batch.num_rows() {
            let signature_id_value = signature_id_values.required_value(row, "signature_id")?;
            if keep_signature_ids.map_or(false, |keep| !keep.contains(signature_id_value.as_ref()))
            {
                continue;
            }
            if signature_id_value.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "signatures Arrow cannot contain empty signature_id values",
                ));
            }
            let signature_id = signature_id_value.into_owned();
            match out.entry(signature_id.clone()) {
                Entry::Occupied(entry) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "signatures Arrow contains duplicate signature_id: {:?}",
                        entry.key()
                    )));
                }
                Entry::Vacant(entry) => {
                    entry.insert(SubblockingSignatureRow {
                        signature_id,
                        first: first_values.optional_owned(row).unwrap_or_default(),
                        middle: middle_values.optional_owned(row).unwrap_or_default(),
                        orcid: orcid_values
                            .optional_owned(row)
                            .and_then(|value| normalize_orcid_owned(&value)),
                    });
                }
            }
        }
    }
    Ok(out)
}

fn read_subblocking_signature_rows_with_optional_index(
    path: &str,
    index_path: Option<&str>,
    keep_signature_ids: Option<&HashSet<String>>,
    full_scan_without_index: bool,
) -> PyResult<(
    HashMap<String, SubblockingSignatureRow>,
    IndexedArrowReadStats,
)> {
    read_raw_arrow_with_optional_index(
        path,
        index_path,
        "signature_id",
        keep_signature_ids,
        full_scan_without_index,
        read_subblocking_signature_rows_from_batches,
    )
}

#[derive(Default)]
struct OrderedSubblocks {
    entries: Vec<(String, Vec<String>)>,
}

impl OrderedSubblocks {
    fn insert(&mut self, key: String, signature_ids: Vec<String>) {
        if let Some((_existing_key, existing_values)) = self
            .entries
            .iter_mut()
            .find(|(existing_key, _)| *existing_key == key)
        {
            *existing_values = signature_ids;
        } else {
            self.entries.push((key, signature_ids));
        }
    }

    fn remove(&mut self, key: &str) -> Option<Vec<String>> {
        let position = self
            .entries
            .iter()
            .position(|(existing_key, _)| existing_key == key)?;
        Some(self.entries.remove(position).1)
    }

    fn get(&self, key: &str) -> Option<&Vec<String>> {
        self.entries
            .iter()
            .find(|(existing_key, _)| existing_key == key)
            .map(|(_key, values)| values)
    }

    fn get_mut(&mut self, key: &str) -> Option<&mut Vec<String>> {
        self.entries
            .iter_mut()
            .find(|(existing_key, _)| existing_key == key)
            .map(|(_key, values)| values)
    }

    fn iter(&self) -> impl Iterator<Item = (&String, &Vec<String>)> {
        self.entries.iter().map(|(key, values)| (key, values))
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn to_hashmap(&self) -> HashMap<String, Vec<String>> {
        self.entries
            .iter()
            .map(|(key, values)| (key.clone(), values.clone()))
            .collect()
    }
}

#[derive(Default)]
struct SubblockingTelemetry {
    maximum_size: usize,
    input_signature_count: usize,
    single_letter_first_name_signature_count: usize,
    multi_letter_first_name_signature_count: usize,
    first_name_dead_end_block_count: usize,
    first_name_dead_end_signature_count: usize,
    specter_fallback_candidate_block_count: usize,
    specter_fallback_candidate_signature_count: usize,
    specter_non_invoked_candidate_block_count: usize,
    specter_non_invoked_candidate_signature_count: usize,
    specter_invocation_count: usize,
    specter_input_signature_count: usize,
    pre_merge_subblock_count: usize,
    pre_merge_specter_labeled_subblock_count: usize,
    pre_merge_specter_labeled_signature_count: usize,
    orcid_subblocking_enabled: bool,
    orcid_merge_skipped_due_to_capacity_count: usize,
    orcid_merge_skipped_due_to_capacity_signature_count: usize,
    final_subblock_count: usize,
    final_specter_labeled_subblock_count: usize,
    final_specter_labeled_signature_count: usize,
}

impl SubblockingTelemetry {
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let telemetry = PyDict::new(py);
        telemetry.set_item("maximum_size", self.maximum_size)?;
        telemetry.set_item("input_signature_count", self.input_signature_count)?;
        telemetry.set_item(
            "single_letter_first_name_signature_count",
            self.single_letter_first_name_signature_count,
        )?;
        telemetry.set_item(
            "multi_letter_first_name_signature_count",
            self.multi_letter_first_name_signature_count,
        )?;
        telemetry.set_item(
            "first_name_dead_end_block_count",
            self.first_name_dead_end_block_count,
        )?;
        telemetry.set_item(
            "first_name_dead_end_signature_count",
            self.first_name_dead_end_signature_count,
        )?;
        telemetry.set_item(
            "specter_fallback_candidate_block_count",
            self.specter_fallback_candidate_block_count,
        )?;
        telemetry.set_item(
            "specter_fallback_candidate_signature_count",
            self.specter_fallback_candidate_signature_count,
        )?;
        telemetry.set_item(
            "specter_non_invoked_candidate_block_count",
            self.specter_non_invoked_candidate_block_count,
        )?;
        telemetry.set_item(
            "specter_non_invoked_candidate_signature_count",
            self.specter_non_invoked_candidate_signature_count,
        )?;
        telemetry.set_item("specter_invocation_count", self.specter_invocation_count)?;
        telemetry.set_item(
            "specter_input_signature_count",
            self.specter_input_signature_count,
        )?;
        telemetry.set_item("pre_merge_subblock_count", self.pre_merge_subblock_count)?;
        telemetry.set_item(
            "pre_merge_specter_labeled_subblock_count",
            self.pre_merge_specter_labeled_subblock_count,
        )?;
        telemetry.set_item(
            "pre_merge_specter_labeled_signature_count",
            self.pre_merge_specter_labeled_signature_count,
        )?;
        telemetry.set_item("orcid_subblocking_enabled", self.orcid_subblocking_enabled)?;
        telemetry.set_item(
            "orcid_merge_skipped_due_to_capacity_count",
            self.orcid_merge_skipped_due_to_capacity_count,
        )?;
        telemetry.set_item(
            "orcid_merge_skipped_due_to_capacity_signature_count",
            self.orcid_merge_skipped_due_to_capacity_signature_count,
        )?;
        telemetry.set_item("final_subblock_count", self.final_subblock_count)?;
        telemetry.set_item(
            "final_specter_labeled_subblock_count",
            self.final_specter_labeled_subblock_count,
        )?;
        telemetry.set_item(
            "final_specter_labeled_signature_count",
            self.final_specter_labeled_signature_count,
        )?;
        Ok(telemetry.unbind())
    }
}

#[derive(Clone)]
struct PrefixCount {
    name: String,
    count: usize,
    first_index: usize,
    signature_ids: Vec<String>,
}

fn py_prefix(value: &str, width: usize) -> String {
    value.chars().take(width).collect()
}

fn prefix_counts(names: &[String], signature_ids: &[String], width: usize) -> Vec<PrefixCount> {
    let mut counts: HashMap<String, PrefixCount> = HashMap::new();
    for (index, (name, signature_id)) in names.iter().zip(signature_ids.iter()).enumerate() {
        let prefix = py_prefix(name, width);
        let entry = counts.entry(prefix.clone()).or_insert_with(|| PrefixCount {
            name: prefix,
            count: 0,
            first_index: index,
            signature_ids: Vec::new(),
        });
        entry.count += 1;
        entry.signature_ids.push(signature_id.clone());
    }
    let mut out: Vec<PrefixCount> = counts.into_values().collect();
    out.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.first_index.cmp(&right.first_index))
    });
    out
}

fn subdivide_helper_rust(
    mut names: Vec<String>,
    mut signature_ids: Vec<String>,
    maximum_size: usize,
    starting_k: usize,
) -> (OrderedSubblocks, OrderedSubblocks) {
    let n_signature_ids = signature_ids.len();
    if n_signature_ids == 0 {
        return (OrderedSubblocks::default(), OrderedSubblocks::default());
    }
    let mut output = OrderedSubblocks::default();
    let mut output_cant_subdivide = OrderedSubblocks::default();
    let max_len = names.iter().map(|name| py_len(name)).max().unwrap_or(0);
    let mut clean_break = false;

    for width in starting_k..=max_len {
        let counts = prefix_counts(&names, &signature_ids, width);
        let good_size_counts: Vec<&PrefixCount> = counts
            .iter()
            .filter(|count| count.count <= maximum_size)
            .collect();
        if good_size_counts.is_empty() {
            for count in counts.iter() {
                output_cant_subdivide.insert(count.name.clone(), count.signature_ids.clone());
            }
            clean_break = true;
            break;
        }

        for count in good_size_counts {
            output.insert(count.name.clone(), count.signature_ids.clone());
        }

        let bad_names: HashSet<String> = counts
            .iter()
            .filter(|count| count.count > maximum_size)
            .map(|count| count.name.clone())
            .collect();
        let mut next_names = Vec::new();
        let mut next_signature_ids = Vec::new();
        for (name, signature_id) in names.into_iter().zip(signature_ids.into_iter()) {
            if bad_names.contains(&py_prefix(&name, width)) {
                next_names.push(name);
                next_signature_ids.push(signature_id);
            }
        }
        names = next_names;
        signature_ids = next_signature_ids;
    }

    if !names.is_empty() && !clean_break {
        output_cant_subdivide.insert("final".to_string(), signature_ids);
    }
    (output, output_cant_subdivide)
}

fn specter_labeled_subblock_stats(subblocks: &OrderedSubblocks) -> (usize, usize) {
    let mut subblock_count = 0usize;
    let mut signature_count = 0usize;
    for (key, values) in subblocks.iter() {
        if key.contains("|specter=") {
            subblock_count += 1;
            signature_count += values.len();
        }
    }
    (subblock_count, signature_count)
}

struct SubblockMergeMetadata {
    size: usize,
    first_name: String,
    middle_name: Option<String>,
    name_for_splits: Option<String>,
    lookup: Option<String>,
}

fn subblock_merge_candidate_metadata(key: &str, size: usize) -> SubblockMergeMetadata {
    let key_parts: Vec<&str> = key.split('|').collect();
    let first_name = key_parts.first().copied().unwrap_or_default().to_string();
    let middle_name = if key_parts.len() > 1 {
        Some(
            key_parts[1]
                .split_once('=')
                .map_or("", |(_left, right)| right)
                .to_string(),
        )
    } else {
        None
    };
    let name_for_splits = if py_len(&first_name) > 1 {
        Some(first_name.clone())
    } else if py_len(&first_name) == 1 && middle_name.is_some() {
        middle_name.clone()
    } else {
        None
    };
    let lookup = name_for_splits
        .as_ref()
        .map(|name| name.split(' ').next().unwrap_or_default().to_string());
    SubblockMergeMetadata {
        size,
        first_name,
        middle_name,
        name_for_splits,
        lookup,
    }
}

fn common_prefix_char_count(left: &str, right: &str) -> usize {
    left.chars()
        .zip(right.chars())
        .take_while(|(left_char, right_char)| left_char == right_char)
        .count()
}

fn sorted_subblock_merge_candidates(
    output: &OrderedSubblocks,
    maximum_size: usize,
    first_k_letter_counts_sorted: &HashMap<String, HashMap<String, f64>>,
) -> PyResult<Vec<((String, String), f64)>> {
    let mut metadata: HashMap<String, SubblockMergeMetadata> = HashMap::new();
    let mut mergeable_keys = Vec::<String>::new();
    for (key, values) in output.iter() {
        if values.len() >= maximum_size {
            continue;
        }
        let row = subblock_merge_candidate_metadata(key, values.len());
        if row.name_for_splits.is_none() {
            continue;
        }
        metadata.insert(key.clone(), row);
        mergeable_keys.push(key.clone());
    }

    let mut candidates = Vec::<((String, String), f64)>::new();
    for left_index in 0..mergeable_keys.len() {
        for right_index in (left_index + 1)..mergeable_keys.len() {
            let left_key = &mergeable_keys[left_index];
            let right_key = &mergeable_keys[right_index];
            let left = metadata.get(left_key).expect("merge metadata exists");
            let right = metadata.get(right_key).expect("merge metadata exists");
            if left.size + right.size >= maximum_size {
                continue;
            }
            let both_multi_letter = py_len(&left.first_name) > 1 && py_len(&right.first_name) > 1;
            let both_single_letter_with_middle = py_len(&left.first_name) == 1
                && py_len(&right.first_name) == 1
                && left.middle_name.is_some()
                && right.middle_name.is_some();
            if !both_multi_letter && !both_single_letter_with_middle {
                continue;
            }

            let left_name = left.name_for_splits.as_ref().expect("merge name exists");
            let right_name = right.name_for_splits.as_ref().expect("merge name exists");
            let pair = (left_key.clone(), right_key.clone());
            if left_name == right_name {
                let score = match (&left.middle_name, &right.middle_name) {
                    (Some(left_middle), Some(right_middle)) => {
                        common_prefix_char_count(left_middle, right_middle)
                    }
                    _ => 0,
                };
                candidates.push((pair, 1e10 + score as f64));
            } else if same_prefix_tokens(left_name, right_name) {
                let score = py_len(left_name).min(py_len(right_name));
                candidates.push((pair, 1e5 + score as f64));
            } else if let (Some(left_lookup), Some(right_lookup)) = (&left.lookup, &right.lookup) {
                if let Some(right_counts) = first_k_letter_counts_sorted.get(left_lookup) {
                    if let Some(score) = right_counts.get(right_lookup) {
                        candidates.push((pair, *score));
                    }
                }
            }
        }
    }
    candidates.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| right.0 .0.cmp(&left.0 .0))
            .then_with(|| right.0 .1.cmp(&left.0 .1))
    });
    Ok(candidates)
}

fn merge_small_subblocks(
    output: &mut OrderedSubblocks,
    maximum_size: usize,
    first_k_letter_counts_sorted: &HashMap<String, HashMap<String, f64>>,
) -> PyResult<()> {
    let candidates =
        sorted_subblock_merge_candidates(output, maximum_size, first_k_letter_counts_sorted)?;
    let mut merging_log: BTreeMap<usize, HashSet<String>> = BTreeMap::new();
    let mut inverse_merging_log: HashMap<String, usize> = HashMap::new();
    let mut cluster_id = 0usize;

    for (pair, _score) in candidates {
        let pair_1_cluster_id = inverse_merging_log.get(&pair.0).copied();
        let pair_2_cluster_id = inverse_merging_log.get(&pair.1).copied();
        if pair_1_cluster_id.is_none() && pair_2_cluster_id.is_none() {
            let mut keys = HashSet::new();
            keys.insert(pair.0.clone());
            keys.insert(pair.1.clone());
            merging_log.insert(cluster_id, keys);
            inverse_merging_log.insert(pair.0.clone(), cluster_id);
            inverse_merging_log.insert(pair.1.clone(), cluster_id);
            cluster_id += 1;
        } else if pair_1_cluster_id.is_some()
            && pair_2_cluster_id.is_some()
            && pair_1_cluster_id == pair_2_cluster_id
        {
            continue;
        } else {
            let proposed_cluster = match (pair_1_cluster_id, pair_2_cluster_id) {
                (Some(left_id), Some(right_id)) if left_id != right_id => merging_log
                    .get(&left_id)
                    .expect("left merge cluster exists")
                    .union(
                        merging_log
                            .get(&right_id)
                            .expect("right merge cluster exists"),
                    )
                    .cloned()
                    .collect::<HashSet<_>>(),
                (Some(left_id), None) => {
                    let mut cluster = merging_log
                        .get(&left_id)
                        .expect("left merge cluster exists")
                        .clone();
                    cluster.insert(pair.0.clone());
                    cluster.insert(pair.1.clone());
                    cluster
                }
                (None, Some(right_id)) => {
                    let mut cluster = merging_log
                        .get(&right_id)
                        .expect("right merge cluster exists")
                        .clone();
                    cluster.insert(pair.0.clone());
                    cluster.insert(pair.1.clone());
                    cluster
                }
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "This should never happen",
                    ))
                }
            };
            let size_of_proposed: usize = proposed_cluster
                .iter()
                .map(|key| output.get(key).map_or(0, Vec::len))
                .sum();
            if size_of_proposed <= maximum_size {
                if let Some(left_id) = pair_1_cluster_id {
                    merging_log.insert(left_id, proposed_cluster.clone());
                    if let Some(right_id) = pair_2_cluster_id {
                        merging_log.remove(&right_id);
                    }
                    for key in proposed_cluster {
                        inverse_merging_log.insert(key, left_id);
                    }
                } else if let Some(right_id) = pair_2_cluster_id {
                    merging_log.insert(right_id, proposed_cluster.clone());
                    if let Some(left_id) = pair_1_cluster_id {
                        merging_log.remove(&left_id);
                    }
                    for key in proposed_cluster {
                        inverse_merging_log.insert(key, right_id);
                    }
                }
            }
        }
    }

    let mut counter_of_keys: HashMap<String, usize> = HashMap::new();
    for keys_to_merge in merging_log.values() {
        for key in keys_to_merge {
            *counter_of_keys.entry(key.clone()).or_insert(0) += 1;
        }
    }
    if counter_of_keys.values().any(|count| *count != 1) {
        return Err(pyo3::exceptions::PyAssertionError::new_err(
            "A subblock key appears in more than one merge cluster",
        ));
    }

    let merge_cluster_ids: Vec<usize> = merging_log.keys().copied().collect();
    for merge_cluster_id in merge_cluster_ids {
        let mut keys_to_merge: Vec<String> = merging_log
            .get(&merge_cluster_id)
            .expect("merge cluster exists")
            .iter()
            .cloned()
            .collect();
        keys_to_merge.sort_unstable();
        let key_of_keys = keys_to_merge.join(", ");
        let mut signature_ids_stacked = Vec::<String>::new();
        for key in keys_to_merge.iter() {
            if let Some(values) = output.get(key) {
                signature_ids_stacked.extend(values.iter().cloned());
            }
        }
        output.insert(key_of_keys, signature_ids_stacked);
        for key in keys_to_merge {
            output.remove(&key);
        }
    }
    Ok(())
}

fn apply_orcid_subblocking(
    output: &mut OrderedSubblocks,
    row_by_signature_id: &HashMap<String, SubblockingSignatureRow>,
    maximum_size: usize,
    telemetry: &mut SubblockingTelemetry,
) {
    let mut sig_id_to_subblock_id: HashMap<String, String> = HashMap::new();
    let mut sig_id_order = Vec::<String>::new();
    for (subblock_id, sig_ids) in output.iter() {
        for sig_id in sig_ids {
            if !sig_id_to_subblock_id.contains_key(sig_id) {
                sig_id_order.push(sig_id.clone());
            }
            sig_id_to_subblock_id.insert(sig_id.clone(), subblock_id.clone());
        }
    }

    let mut orcid_to_sig_ids: HashMap<String, Vec<String>> = HashMap::new();
    let mut orcid_order = Vec::<String>::new();
    for sig_id in sig_id_order.iter() {
        let Some(row) = row_by_signature_id.get(sig_id) else {
            continue;
        };
        let Some(orcid_raw) = row.orcid.as_ref() else {
            continue;
        };
        let Some(orcid) = normalize_orcid_owned(orcid_raw) else {
            continue;
        };
        if !orcid_to_sig_ids.contains_key(&orcid) {
            orcid_order.push(orcid.clone());
        }
        orcid_to_sig_ids
            .entry(orcid)
            .or_default()
            .push(sig_id.clone());
    }

    for orcid in orcid_order {
        let Some(orcid_sig_ids) = orcid_to_sig_ids.get(&orcid) else {
            continue;
        };
        let mut current_subblock_counts: HashMap<String, usize> = HashMap::new();
        for sig_id in orcid_sig_ids {
            if let Some(subblock_id) = sig_id_to_subblock_id.get(sig_id) {
                *current_subblock_counts
                    .entry(subblock_id.clone())
                    .or_insert(0) += 1;
            }
        }
        let mut unique_subblock_ids: Vec<String> =
            current_subblock_counts.keys().cloned().collect();
        if unique_subblock_ids.len() <= 1 {
            continue;
        }
        unique_subblock_ids.sort_by(|left, right| {
            let left_score = left.matches("specter").count() * 10 + left.matches('|').count();
            let right_score = right.matches("specter").count() * 10 + right.matches('|').count();
            left_score.cmp(&right_score).then_with(|| left.cmp(right))
        });

        let total_orcid_sig_count = orcid_sig_ids.len();
        let feasible_subblock_ids: Vec<String> = unique_subblock_ids
            .iter()
            .filter(|subblock_id| {
                let current_count = current_subblock_counts
                    .get(*subblock_id)
                    .copied()
                    .unwrap_or(0);
                output.get(subblock_id).map_or(0, Vec::len)
                    + (total_orcid_sig_count - current_count)
                    <= maximum_size
            })
            .cloned()
            .collect();
        if feasible_subblock_ids.is_empty() {
            telemetry.orcid_merge_skipped_due_to_capacity_count += 1;
            telemetry.orcid_merge_skipped_due_to_capacity_signature_count += total_orcid_sig_count;
            continue;
        }

        let subblock_id_to_move_to = feasible_subblock_ids[0].clone();
        let mut sig_ids_to_move = Vec::<String>::new();
        let mut moved_sig_ids_by_source: HashMap<String, HashSet<String>> = HashMap::new();
        for sig_id in orcid_sig_ids {
            let Some(original_subblock_id) = sig_id_to_subblock_id.get(sig_id).cloned() else {
                continue;
            };
            if original_subblock_id != subblock_id_to_move_to {
                sig_ids_to_move.push(sig_id.clone());
                moved_sig_ids_by_source
                    .entry(original_subblock_id)
                    .or_default()
                    .insert(sig_id.clone());
            }
        }

        if let Some(target_values) = output.get_mut(&subblock_id_to_move_to) {
            target_values.extend(sig_ids_to_move.iter().cloned());
        }
        for sig_id in sig_ids_to_move {
            sig_id_to_subblock_id.insert(sig_id, subblock_id_to_move_to.clone());
        }
        for (original_subblock_id, moved_sig_ids) in moved_sig_ids_by_source {
            let Some(source_values) = output.get_mut(&original_subblock_id) else {
                continue;
            };
            source_values.retain(|sig_id| !moved_sig_ids.contains(sig_id));
            if source_values.is_empty() {
                output.remove(&original_subblock_id);
            }
        }
    }
}

fn extract_string_vec_entries(obj: &Bound<'_, PyAny>) -> PyResult<Vec<(String, Vec<String>)>> {
    let dict = obj.downcast::<PyDict>()?;
    let mut out = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        out.push((key.extract()?, value.extract()?));
    }
    Ok(out)
}

fn make_subblocks_with_telemetry_from_rows(
    py: Python<'_>,
    rows: Vec<SubblockingSignatureRow>,
    maximum_size: usize,
    first_k_letter_counts_sorted: HashMap<String, HashMap<String, f64>>,
    fallback_cluster_fn: &Bound<'_, PyAny>,
    anddata: &Bound<'_, PyAny>,
    compute_block_fn: Option<&Bound<'_, PyAny>>,
    use_orcid_subblocking: bool,
) -> PyResult<(HashMap<String, Vec<String>>, Py<PyDict>)> {
    if maximum_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "maximum_size must be positive",
        ));
    }

    let signature_ids: Vec<String> = rows.iter().map(|row| row.signature_id.clone()).collect();
    let row_by_signature_id: HashMap<String, SubblockingSignatureRow> = rows
        .iter()
        .map(|row| (row.signature_id.clone(), row.clone()))
        .collect();
    let single_letter_flags: Vec<bool> = rows.iter().map(|row| py_len(&row.first) <= 1).collect();
    let single_letter_count = single_letter_flags
        .iter()
        .filter(|is_single_letter| **is_single_letter)
        .count();
    let mut telemetry = SubblockingTelemetry {
        maximum_size,
        input_signature_count: rows.len(),
        single_letter_first_name_signature_count: single_letter_count,
        multi_letter_first_name_signature_count: rows.len().saturating_sub(single_letter_count),
        orcid_subblocking_enabled: use_orcid_subblocking,
        ..SubblockingTelemetry::default()
    };

    let first_letter = rows
        .iter()
        .find_map(|row| row.first.chars().next().map(|ch| ch.to_string()))
        .unwrap_or_else(|| "?".to_string());

    let mut multi_first_names = Vec::<String>::new();
    let mut multi_signature_ids = Vec::<String>::new();
    let mut single_middle_names = Vec::<String>::new();
    let mut single_signature_ids = Vec::<String>::new();
    for (row, is_single_letter) in rows.iter().zip(single_letter_flags.iter()) {
        if *is_single_letter {
            single_middle_names.push(row.middle.clone());
            single_signature_ids.push(row.signature_id.clone());
        } else {
            multi_first_names.push(row.first.clone());
            multi_signature_ids.push(row.signature_id.clone());
        }
    }

    let (mut output, output_cant_subdivide) =
        subdivide_helper_rust(multi_first_names, multi_signature_ids, maximum_size, 2);
    telemetry.first_name_dead_end_block_count = output_cant_subdivide.len();
    telemetry.first_name_dead_end_signature_count = output_cant_subdivide
        .iter()
        .map(|(_key, values)| values.len())
        .sum();

    let mut output_for_specter = OrderedSubblocks::default();
    for (key, sig_ids_loop) in output_cant_subdivide.entries {
        let middle_names_loop: Vec<String> = sig_ids_loop
            .iter()
            .filter_map(|signature_id| row_by_signature_id.get(signature_id))
            .map(|row| row.middle.clone())
            .collect();
        let (output_loop, output_cant_subdivide_loop) =
            subdivide_helper_rust(middle_names_loop, sig_ids_loop, maximum_size, 1);
        for (key_loop, values) in output_loop.entries {
            output.insert(format!("{key}|middle={key_loop}"), values);
        }
        for (key_loop, values) in output_cant_subdivide_loop.entries {
            output_for_specter.insert(format!("{key}|middle={key_loop}"), values);
        }
    }

    if single_signature_ids.len() < maximum_size {
        if !single_signature_ids.is_empty() {
            output.insert(first_letter.clone(), single_signature_ids);
        }
    } else {
        let (output_single_letter_first_name, output_cant_subdivide_single_letter_first_name) =
            subdivide_helper_rust(single_middle_names, single_signature_ids, maximum_size, 1);
        for (key, values) in output_single_letter_first_name.entries {
            output.insert(format!("{first_letter}|middle={key}"), values);
        }
        for (key, values) in output_cant_subdivide_single_letter_first_name.entries {
            output_for_specter.insert(format!("{first_letter}|middle={key}"), values);
        }
    }

    telemetry.specter_fallback_candidate_block_count = output_for_specter.len();
    telemetry.specter_fallback_candidate_signature_count = output_for_specter
        .iter()
        .map(|(_key, values)| values.len())
        .sum();

    let fallback_signature_groups: Vec<Vec<String>> = output_for_specter
        .iter()
        .filter_map(|(_key, values)| {
            if values.len() > maximum_size {
                Some(values.clone())
            } else {
                None
            }
        })
        .collect();
    if !fallback_signature_groups.is_empty() {
        if let Ok(prepare_fallback) = fallback_cluster_fn.getattr("prepare") {
            if prepare_fallback.is_callable() {
                prepare_fallback.call1((fallback_signature_groups.clone(),))?;
            }
        }
    }

    for (key, sig_ids_loop) in output_for_specter.entries {
        if sig_ids_loop.len() <= maximum_size {
            telemetry.specter_non_invoked_candidate_block_count += 1;
            telemetry.specter_non_invoked_candidate_signature_count += sig_ids_loop.len();
            output.insert(key, sig_ids_loop);
        } else {
            telemetry.specter_invocation_count += 1;
            telemetry.specter_input_signature_count += sig_ids_loop.len();
            let kwargs = PyDict::new(py);
            kwargs.set_item("target_subblock_size", maximum_size)?;
            if let Some(compute_block_fn_value) = compute_block_fn {
                kwargs.set_item("compute_block_fn", compute_block_fn_value)?;
            }
            let specter_clustering =
                fallback_cluster_fn.call((sig_ids_loop, anddata), Some(&kwargs))?;
            for (key_loop, values) in extract_string_vec_entries(&specter_clustering)? {
                output.insert(format!("{key}|specter={key_loop}"), values);
            }
        }
    }

    let (pre_merge_specter_subblock_count, pre_merge_specter_signature_count) =
        specter_labeled_subblock_stats(&output);
    telemetry.pre_merge_subblock_count = output.len();
    telemetry.pre_merge_specter_labeled_subblock_count = pre_merge_specter_subblock_count;
    telemetry.pre_merge_specter_labeled_signature_count = pre_merge_specter_signature_count;

    merge_small_subblocks(&mut output, maximum_size, &first_k_letter_counts_sorted)?;

    if use_orcid_subblocking {
        apply_orcid_subblocking(
            &mut output,
            &row_by_signature_id,
            maximum_size,
            &mut telemetry,
        );
    }

    let input_set: HashSet<String> = signature_ids.into_iter().collect();
    let output_set: HashSet<String> = output
        .iter()
        .flat_map(|(_key, values)| values.iter().cloned())
        .collect();
    if input_set != output_set {
        return Err(pyo3::exceptions::PyAssertionError::new_err(
            "Subblocking did not produce a complete partition",
        ));
    }

    let (final_specter_subblock_count, final_specter_signature_count) =
        specter_labeled_subblock_stats(&output);
    telemetry.final_subblock_count = output.len();
    telemetry.final_specter_labeled_subblock_count = final_specter_subblock_count;
    telemetry.final_specter_labeled_signature_count = final_specter_signature_count;
    Ok((output.to_hashmap(), telemetry.to_dict(py)?))
}

#[pyfunction]
#[pyo3(signature = (
    paths,
    signature_ids,
    maximum_size,
    first_k_letter_counts_sorted,
    fallback_cluster_fn,
    anddata,
    compute_block_fn = None,
    use_orcid_subblocking = true,
    full_scan_without_index = false
))]
fn make_subblocks_with_telemetry_arrow(
    py: Python<'_>,
    paths: &Bound<'_, PyAny>,
    signature_ids: Vec<String>,
    maximum_size: usize,
    first_k_letter_counts_sorted: HashMap<String, HashMap<String, f64>>,
    fallback_cluster_fn: &Bound<'_, PyAny>,
    anddata: &Bound<'_, PyAny>,
    compute_block_fn: Option<&Bound<'_, PyAny>>,
    use_orcid_subblocking: bool,
    full_scan_without_index: bool,
) -> PyResult<(HashMap<String, Vec<String>>, Py<PyDict>)> {
    let signatures_path = extract_path_mapping_string(paths, "signatures", true)?
        .expect("required signatures path exists");
    let signatures_index_path =
        extract_path_mapping_string(paths, "signatures_batch_index", false)?;
    let mut seen_signature_ids = HashSet::<String>::new();
    let mut requested_signature_ids = Vec::<String>::new();
    for signature_id in signature_ids {
        if seen_signature_ids.insert(signature_id.clone()) {
            requested_signature_ids.push(signature_id);
        }
    }
    let keep_signature_ids: HashSet<String> = requested_signature_ids.iter().cloned().collect();
    let (subblocking_rows, _read_stats) = read_subblocking_signature_rows_with_optional_index(
        &signatures_path,
        signatures_index_path.as_deref(),
        Some(&keep_signature_ids),
        full_scan_without_index,
    )?;
    let missing_signature_ids: Vec<String> = requested_signature_ids
        .iter()
        .filter(|signature_id| !subblocking_rows.contains_key(*signature_id))
        .take(10)
        .cloned()
        .collect();
    if !missing_signature_ids.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "signatures Arrow is missing subblocking signature ids: {missing_signature_ids:?}"
        )));
    }

    let mut signature_rows: Vec<SubblockingSignatureRow> = requested_signature_ids
        .iter()
        .map(|signature_id| {
            subblocking_rows
                .get(signature_id)
                .expect("requested signature exists after missing check")
                .clone()
        })
        .collect();
    let text_module = py.import("s2and.text")?;
    let unidecode = text_module.getattr("unidecode")?;
    let name_prefixes = extract_required_string_set(&text_module.getattr("NAME_PREFIXES")?)?;
    let mut unidecode_char_map: HashMap<char, String> = HashMap::new();
    for row in signature_rows.iter() {
        ensure_unidecode_for_text(&unidecode, &row.first, &mut unidecode_char_map)?;
        ensure_unidecode_for_text(&unidecode, &row.middle, &mut unidecode_char_map)?;
    }
    normalize_subblocking_signature_rows(&mut signature_rows, &name_prefixes, &unidecode_char_map);

    make_subblocks_with_telemetry_from_rows(
        py,
        signature_rows,
        maximum_size,
        first_k_letter_counts_sorted,
        fallback_cluster_fn,
        anddata,
        compute_block_fn,
        use_orcid_subblocking,
    )
}

fn extract_string_string_map(obj: &Bound<'_, PyAny>) -> PyResult<HashMap<String, String>> {
    let dict = obj.downcast::<PyDict>()?;
    let mut out = HashMap::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        out.insert(key.extract()?, value.extract()?);
    }
    Ok(out)
}

fn extract_string_vec_map(obj: &Bound<'_, PyAny>) -> PyResult<HashMap<String, Vec<String>>> {
    let dict = obj.downcast::<PyDict>()?;
    let mut out = HashMap::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let key_text: String = key.extract()?;
        let mut values = Vec::new();
        for item in PyIterator::from_object(&value)? {
            values.push(item?.extract()?);
        }
        out.insert(key_text, values);
    }
    Ok(out)
}

fn filter_text_for_char_ngrams(text: &str, stopwords: Option<&HashSet<String>>) -> String {
    let Some(stopwords_set) = stopwords else {
        return text.to_string();
    };
    text.split(' ')
        .filter(|word| !stopwords_set.contains(*word) && py_len(word) > 2)
        .collect::<Vec<_>>()
        .join(" ")
}

fn char_ngrams_counter_python_compat(
    text: &str,
    use_unigrams: bool,
    use_bigrams: bool,
    stopwords: Option<&HashSet<String>>,
) -> HashMap<String, usize> {
    if text.is_empty() {
        return HashMap::new();
    }
    let filtered_text = filter_text_for_char_ngrams(text, stopwords);
    if filtered_text.is_empty() {
        return HashMap::new();
    }
    let chars: Vec<char> = filtered_text.chars().collect();
    if chars.is_empty() {
        return HashMap::new();
    }

    let mut out: HashMap<String, usize> = HashMap::new();
    if use_unigrams {
        for ch in chars.iter() {
            if *ch != ' ' {
                *out.entry(ch.to_string()).or_insert(0) += 1;
            }
        }
    }
    if use_bigrams && chars.len() >= 2 {
        for idx in 0..(chars.len() - 1) {
            if chars[idx] == ' ' || chars[idx + 1] == ' ' {
                continue;
            }
            let gram = format!("{}{}", chars[idx], chars[idx + 1]);
            *out.entry(gram).or_insert(0) += 1;
        }
    }
    if chars.len() >= 3 {
        for idx in 0..(chars.len() - 2) {
            if chars[idx] == ' ' || chars[idx + 1] == ' ' || chars[idx + 2] == ' ' {
                continue;
            }
            let gram = format!("{}{}{}", chars[idx], chars[idx + 1], chars[idx + 2]);
            *out.entry(gram).or_insert(0) += 1;
        }
    }
    if chars.len() >= 4 {
        for idx in 0..(chars.len() - 3) {
            if chars[idx] == ' '
                || chars[idx + 1] == ' '
                || chars[idx + 2] == ' '
                || chars[idx + 3] == ' '
            {
                continue;
            }
            let gram = format!(
                "{}{}{}{}",
                chars[idx],
                chars[idx + 1],
                chars[idx + 2],
                chars[idx + 3]
            );
            *out.entry(gram).or_insert(0) += 1;
        }
    }
    out
}

fn word_ngrams_counter_python_compat(
    text: &str,
    stopwords: &HashSet<String>,
) -> HashMap<String, usize> {
    if text.is_empty() {
        return HashMap::new();
    }
    let text_split: Vec<&str> = text
        .split_whitespace()
        .filter(|word| !stopwords.contains(*word) && py_len(word) > 1)
        .collect();
    if text_split.is_empty() {
        return HashMap::new();
    }
    let mut out: HashMap<String, usize> = HashMap::new();
    for token in text_split.iter() {
        *out.entry((*token).to_string()).or_insert(0) += 1;
    }
    if text_split.len() >= 2 {
        for pair in text_split.windows(2) {
            let gram = format!("{} {}", pair[0], pair[1]);
            *out.entry(gram).or_insert(0) += 1;
        }
    }
    if text_split.len() >= 3 {
        for tri in text_split.windows(3) {
            let gram = format!("{} {} {}", tri[0], tri[1], tri[2]);
            *out.entry(gram).or_insert(0) += 1;
        }
    }
    out
}

fn char_ngrams_counter(text: &str) -> HashMap<String, usize> {
    if text.is_empty() {
        return HashMap::new();
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return HashMap::new();
    }

    let mut out: HashMap<String, usize> = HashMap::new();
    for width in [2usize, 3usize, 4usize] {
        if chars.len() < width {
            continue;
        }
        for idx in 0..=(chars.len() - width) {
            let window = &chars[idx..idx + width];
            if window.iter().any(|ch| *ch == ' ') {
                continue;
            }
            let gram: String = window.iter().collect();
            *out.entry(gram).or_insert(0) += 1;
        }
    }
    out
}

fn word_ngrams_counter(text: &str) -> HashMap<String, usize> {
    if text.is_empty() {
        return HashMap::new();
    }

    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.is_empty() {
        return HashMap::new();
    }

    let mut out: HashMap<String, usize> = HashMap::new();
    for tok in tokens.iter() {
        *out.entry((*tok).to_string()).or_insert(0) += 1;
    }
    if tokens.len() >= 2 {
        for pair in tokens.windows(2) {
            let gram = format!("{} {}", pair[0], pair[1]);
            *out.entry(gram).or_insert(0) += 1;
        }
    }
    if tokens.len() >= 3 {
        for tri in tokens.windows(3) {
            let gram = format!("{} {} {}", tri[0], tri[1], tri[2]);
            *out.entry(gram).or_insert(0) += 1;
        }
    }
    out
}

#[pyfunction]
#[pyo3(signature = (coauthor_texts, affiliation_texts, num_threads = None))]
fn signature_ngrams_batch(
    py: Python<'_>,
    coauthor_texts: Vec<String>,
    affiliation_texts: Vec<String>,
    num_threads: Option<usize>,
) -> PyResult<(Vec<HashMap<String, usize>>, Vec<HashMap<String, usize>>)> {
    if coauthor_texts.len() != affiliation_texts.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "coauthor_texts and affiliation_texts must have equal length",
        ));
    }
    let n = coauthor_texts.len();
    let text_module = py.import("s2and.text")?;
    let affiliation_stopwords =
        extract_required_string_set(&text_module.getattr("AFFILIATIONS_STOP_WORDS")?)?;

    let pairs = py.allow_threads(|| {
        let compute = || {
            (0..n)
                .into_par_iter()
                .map(|idx| {
                    let coauthors = char_ngrams_counter(&coauthor_texts[idx]);
                    let affiliations = word_ngrams_counter_python_compat(
                        &affiliation_texts[idx],
                        &affiliation_stopwords,
                    );
                    (coauthors, affiliations)
                })
                .collect::<Vec<_>>()
        };
        install_with_optional_rayon_pool(num_threads, compute)
    });

    let mut coauthor_out: Vec<HashMap<String, usize>> = Vec::with_capacity(n);
    let mut affiliation_out: Vec<HashMap<String, usize>> = Vec::with_capacity(n);
    for (coauthors, affiliations) in pairs {
        coauthor_out.push(coauthors);
        affiliation_out.push(affiliations);
    }
    Ok((coauthor_out, affiliation_out))
}

fn counter_jaccard_data(
    counter1: &Option<CounterData>,
    counter2: &Option<CounterData>,
    denom_max: f64,
) -> f64 {
    let (Some(c1), Some(c2)) = (counter1.as_ref(), counter2.as_ref()) else {
        return f64::NAN;
    };
    if c1.entries.is_empty() || c2.entries.is_empty() {
        return f64::NAN;
    }
    let (small, large) = if c1.entries.len() <= c2.entries.len() {
        (c1, c2)
    } else {
        (c2, c1)
    };
    let mut intersection = 0.0f64;
    if large.entries.len() < small.entries.len().saturating_mul(4) {
        let mut small_idx = 0usize;
        let mut large_idx = 0usize;
        while small_idx < small.entries.len() && large_idx < large.entries.len() {
            let (small_hash, small_value) = small.entries[small_idx];
            let (large_hash, large_value) = large.entries[large_idx];
            if small_hash == large_hash {
                intersection += small_value.min(large_value) as f64;
                small_idx += 1;
                large_idx += 1;
            } else if small_hash < large_hash {
                small_idx += 1;
            } else {
                large_idx += 1;
            }
        }
    } else {
        for (h, v1) in small.entries.iter() {
            if let Ok(idx) = large.entries.binary_search_by_key(h, |e| e.0) {
                intersection += (*v1).min(large.entries[idx].1) as f64;
            }
        }
    }
    let union = c1.sum as f64 + c2.sum as f64 - intersection;
    if union == 0.0 {
        return f64::NAN;
    }
    let denom = if denom_max.is_infinite() {
        union
    } else {
        union.min(denom_max)
    };
    let score = intersection / denom;
    if score > 1.0 {
        1.0
    } else {
        score
    }
}

fn set_jaccard_data<T: Eq + std::hash::Hash>(
    set1: &Option<HashSet<T>>,
    set2: &Option<HashSet<T>>,
) -> f64 {
    let (Some(s1), Some(s2)) = (set1.as_ref(), set2.as_ref()) else {
        return f64::NAN;
    };
    if s1.is_empty() || s2.is_empty() {
        return f64::NAN;
    }
    let intersection = s1.intersection(s2).count();
    let union = s1.len() + s2.len() - intersection;
    if union == 0 {
        return f64::NAN;
    }
    (intersection as f64) / (union as f64)
}

fn refs_jaccard<T: Eq + std::hash::Hash>(set1: &HashSet<T>, set2: &HashSet<T>) -> f64 {
    if set1.is_empty() || set2.is_empty() {
        return f64::NAN;
    }
    let intersection = set1.intersection(set2).count();
    let union = set1.len() + set2.len() - intersection;
    if union == 0 {
        return f64::NAN;
    }
    (intersection as f64) / (union as f64)
}

fn nanmin(a: f64, b: f64) -> f64 {
    if a.is_nan() && b.is_nan() {
        f64::NAN
    } else if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else {
        a.min(b)
    }
}

fn max_propagate_nan(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

fn compute_name_counts_data(
    counts1: Option<&NameCountsData>,
    counts2: Option<&NameCountsData>,
) -> [f64; 6] {
    let (Some(c1), Some(c2)) = (counts1, counts2) else {
        return [f64::NAN; 6];
    };
    let min_first = nanmin(c1.first, c2.first);
    let min_first_last = nanmin(c1.first_last, c2.first_last);
    let min_last = nanmin(c1.last, c2.last);
    let min_last_first_initial = nanmin(c1.last_first_initial, c2.last_first_initial);
    let max_first = max_propagate_nan(c1.first, c2.first);
    let max_first_last = max_propagate_nan(c1.first_last, c2.first_last);
    [
        min_first,
        min_first_last,
        min_last,
        min_last_first_initial,
        max_first,
        max_first_last,
    ]
}

fn first_names_equal(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let (Some(n1), Some(n2)) = (name1, name2) else {
        return f64::NAN;
    };
    if py_len(n1) == 0 || py_len(n2) == 0 {
        return f64::NAN;
    }
    if n1 == "-" || n2 == "-" {
        return f64::NAN;
    }
    let n1_norm = n1.trim().to_lowercase();
    let n2_norm = n2.trim().to_lowercase();
    if n1_norm == n2_norm {
        1.0
    } else {
        0.0
    }
}

fn middle_initials_overlap(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let s1 = name1.unwrap_or("");
    let s2 = name2.unwrap_or("");
    let c1 = count_initials(s1);
    let c2 = count_initials(s2);
    if c1.is_empty() || c2.is_empty() {
        return f64::NAN;
    }
    let mut intersection_sum: usize = 0;
    for (k, v1) in c1.iter() {
        if let Some(v2) = c2.get(k) {
            intersection_sum += std::cmp::min(*v1, *v2);
        }
    }
    let sum1: usize = c1.values().sum();
    let sum2: usize = c2.values().sum();
    let union_sum = sum1 + sum2 - intersection_sum;
    if union_sum == 0 {
        return f64::NAN;
    }
    let score = (intersection_sum as f64) / (union_sum as f64);
    if score > 1.0 {
        1.0
    } else {
        score
    }
}

fn middle_names_equal(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let (Some(n1), Some(n2)) = (name1, name2) else {
        return f64::NAN;
    };
    if py_len(n1) == 0 || py_len(n2) == 0 {
        return f64::NAN;
    }
    if py_len(n1) == 1 || py_len(n2) == 1 {
        let (Some(c1), Some(c2)) = (n1.chars().next(), n2.chars().next()) else {
            return f64::NAN;
        };
        return if c1 == c2 { 1.0 } else { 0.0 };
    }
    if n1 == n2 {
        1.0
    } else {
        0.0
    }
}

fn middle_one_missing(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let n1 = name1.unwrap_or("");
    let n2 = name2.unwrap_or("");
    let len1 = py_len(n1);
    let len2 = py_len(n2);
    let val = (len1 == 0 && len2 != 0) || (len2 == 0 && len1 != 0);
    if val {
        1.0
    } else {
        0.0
    }
}

fn single_char_first(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let n1 = name1.unwrap_or("");
    let n2 = name2.unwrap_or("");
    let val = py_len(n1) == 1 || py_len(n2) == 1;
    if val {
        1.0
    } else {
        0.0
    }
}

fn single_char_middle(name1: Option<&str>, name2: Option<&str>) -> f64 {
    let n1 = name1.unwrap_or("");
    let n2 = name2.unwrap_or("");
    let mut val = false;
    for part in n1.split(' ') {
        if py_len(part) == 1 {
            val = true;
            break;
        }
    }
    if !val {
        for part in n2.split(' ') {
            if py_len(part) == 1 {
                val = true;
                break;
            }
        }
    }
    if val {
        1.0
    } else {
        0.0
    }
}

fn email_parts(email: &str) -> (String, String) {
    let (prefix_raw, suffix_raw) = if let Some((before_last, after_last)) = email.rsplit_once('@') {
        let mut merged_prefix = String::with_capacity(before_last.len());
        for ch in before_last.chars() {
            if ch != '@' {
                merged_prefix.push(ch);
            }
        }
        (merged_prefix, after_last.to_string())
    } else {
        (email.to_string(), "MISSING".to_string())
    };
    let prefix = prefix_raw.trim_matches('.').to_lowercase();
    let suffix = suffix_raw.trim_matches('.').to_lowercase();
    (prefix, suffix)
}

fn email_pair_parts(
    email1: Option<&str>,
    email2: Option<&str>,
) -> Option<((String, String), (String, String))> {
    let (Some(e1), Some(e2)) = (email1, email2) else {
        return None;
    };
    if py_len(e1) == 0 || py_len(e2) == 0 {
        return None;
    }
    Some((email_parts(e1), email_parts(e2)))
}

fn year_diff(year1: Option<i64>, year2: Option<i64>) -> f64 {
    let (Some(y1_raw), Some(y2_raw)) = (year1, year2) else {
        return f64::NAN;
    };
    let y1 = y1_raw as f64;
    let y2 = y2_raw as f64;
    let diff = (y1 - y2).abs();
    diff.min(50.0)
}

fn position_diff(p1: i64, p2: i64) -> f64 {
    let diff = (p1 - p2).abs() as f64;
    diff.min(50.0)
}

fn cosine_sim_vec_f32(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::NAN;
    }
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..a.len() {
        let av = a[i] as f64;
        let bv = b[i] as f64;
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

fn cosine_sim_with_norms(a: &[f32], norm_a: f64, b: &[f32], norm_b: f64) -> f64 {
    if a.len() != b.len() {
        return f64::NAN;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    let mut dot = 0.0;
    for i in 0..a.len() {
        dot += (a[i] as f64) * (b[i] as f64);
    }
    dot / (norm_a * norm_b)
}

fn levenshtein_distance(a: &str, b: &str) -> usize {
    if a == b {
        return 0;
    }

    if a.is_ascii() && b.is_ascii() {
        return levenshtein_distance_bytes(a.as_bytes(), b.as_bytes());
    }

    let b_chars: Vec<char> = b.chars().collect();
    let len_b = b_chars.len();
    if a.is_empty() {
        return len_b;
    }
    if len_b == 0 {
        return a.chars().count();
    }
    let mut prev: Vec<usize> = (0..=len_b).collect();
    let mut cur: Vec<usize> = vec![0; len_b + 1];
    for (i, a_char) in a.chars().enumerate() {
        cur[0] = i + 1;
        for j in 1..=len_b {
            let deletion = prev[j] + 1;
            let insertion = cur[j - 1] + 1;
            let edit = prev[j - 1] + if a_char == b_chars[j - 1] { 0 } else { 1 };
            cur[j] = deletion.min(insertion).min(edit);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[len_b]
}

fn levenshtein_distance_bytes(a: &[u8], b: &[u8]) -> usize {
    let len_a = a.len();
    let len_b = b.len();
    if len_a == 0 {
        return len_b;
    }
    if len_b == 0 {
        return len_a;
    }
    let mut prev: Vec<usize> = (0..=len_b).collect();
    let mut cur: Vec<usize> = vec![0; len_b + 1];
    for i in 1..=len_a {
        cur[0] = i;
        for j in 1..=len_b {
            let deletion = prev[j] + 1;
            let insertion = cur[j - 1] + 1;
            let edit = prev[j - 1] + if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = deletion.min(insertion).min(edit);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[len_b]
}

fn prefix_dist(a: &str, b: &str) -> f64 {
    if a == b {
        return 0.0;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let (min_chars, max_chars) = if a_chars.len() < b_chars.len() {
        (&a_chars, &b_chars)
    } else {
        (&b_chars, &a_chars)
    };
    let min_len = min_chars.len();
    for i in (1..=min_len).rev() {
        if min_chars[..i] == max_chars[..i] {
            return 1.0 - (i as f64 / min_len as f64);
        }
    }
    1.0
}

fn lcs_length(a: &str, b: &str) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }

    if a.is_ascii() && b.is_ascii() {
        return lcs_length_bytes(a.as_bytes(), b.as_bytes());
    }

    let b_chars: Vec<char> = b.chars().collect();
    let len_b = b_chars.len();
    let mut prev = vec![0usize; len_b + 1];
    let mut cur = vec![0usize; len_b + 1];
    for a_char in a.chars() {
        cur[0] = 0;
        for j in 1..=len_b {
            if a_char == b_chars[j - 1] {
                cur[j] = prev[j - 1] + 1;
            } else {
                cur[j] = cur[j - 1].max(prev[j]);
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[len_b]
}

fn lcs_length_bytes(a: &[u8], b: &[u8]) -> usize {
    let len_a = a.len();
    let len_b = b.len();
    if len_a == 0 || len_b == 0 {
        return 0;
    }
    let mut prev = vec![0usize; len_b + 1];
    let mut cur = vec![0usize; len_b + 1];
    for i in 1..=len_a {
        cur[0] = 0;
        for j in 1..=len_b {
            if a[i - 1] == b[j - 1] {
                cur[j] = prev[j - 1] + 1;
            } else {
                cur[j] = cur[j - 1].max(prev[j]);
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[len_b]
}

fn metric_lcs_distance(a: &str, b: &str) -> f64 {
    if a == b {
        return 0.0;
    }
    let len_a = py_len(a);
    let len_b = py_len(b);
    let max_len = len_a.max(len_b);
    if max_len == 0 {
        return 0.0;
    }
    let lcs = lcs_length(a, b);
    1.0 - (lcs as f64 / max_len as f64)
}

fn jaro_winkler_similarity(a: &str, b: &str, long_tolerance: bool) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();
    if a_len == 0 || b_len == 0 {
        return 0.0;
    }
    let min_len = a_len.min(b_len);
    let mut search_range = a_len.max(b_len) / 2;
    if search_range > 0 {
        search_range -= 1;
    }
    let mut a_flags = vec![false; a_len];
    let mut b_flags = vec![false; b_len];
    let mut common_chars = 0usize;
    for i in 0..a_len {
        let low = if i > search_range {
            i - search_range
        } else {
            0
        };
        let mut hi = i + search_range;
        if hi >= b_len {
            hi = b_len.saturating_sub(1);
        }
        for j in low..=hi {
            if !b_flags[j] && a_chars[i] == b_chars[j] {
                a_flags[i] = true;
                b_flags[j] = true;
                common_chars += 1;
                break;
            }
        }
    }
    if common_chars == 0 {
        return 0.0;
    }
    let mut k = 0usize;
    let mut trans_count = 0usize;
    for i in 0..a_len {
        if a_flags[i] {
            while k < b_len {
                if b_flags[k] {
                    break;
                }
                k += 1;
            }
            if k < b_len && a_chars[i] != b_chars[k] {
                trans_count += 1;
            }
            k += 1;
        }
    }
    trans_count /= 2;
    let common_f = common_chars as f64;
    let weight = (common_f / a_len as f64
        + common_f / b_len as f64
        + (common_f - trans_count as f64) / common_f)
        / 3.0;
    let mut weight = weight;
    if weight > 0.7 {
        let j = min_len.min(4);
        let mut i = 0usize;
        while i < j && a_chars[i] == b_chars[i] {
            i += 1;
        }
        if i > 0 {
            weight += (i as f64) * 0.1 * (1.0 - weight);
        }
        if long_tolerance && min_len > 4 && common_chars > i + 1 && 2 * common_chars >= min_len + i
        {
            weight += (1.0 - weight)
                * ((common_chars - i - 1) as f64 / (a_len + b_len - i * 2 + 2) as f64);
        }
    }
    weight
}

fn name_text_features(name1: Option<&str>, name2: Option<&str>) -> [f64; 4] {
    let (Some(n1), Some(n2)) = (name1, name2) else {
        return [f64::NAN; 4];
    };
    if py_len(n1) == 0 || py_len(n2) == 0 {
        return [f64::NAN; 4];
    }
    let lev = levenshtein_distance(n1, n2) as f64 / (py_len(n1).max(py_len(n2)) as f64);
    let pref = prefix_dist(n1, n2);
    let lcs = metric_lcs_distance(n1, n2);
    let jaro = jaro_winkler_similarity(n1, n2, false);
    [lev, pref, lcs, jaro]
}

const PAPER_IDX_TITLE: usize = 0;
const PAPER_IDX_HAS_ABSTRACT: usize = 1;
const PAPER_IDX_IN_SIGNATURES: usize = 2;
const PAPER_IDX_IS_RELIABLE: usize = 4;
const PAPER_IDX_PREDICTED_LANGUAGE: usize = 5;
const PAPER_IDX_TITLE_NGRAMS_WORDS: usize = 6;
const PAPER_IDX_AUTHORS: usize = 7;
const PAPER_IDX_VENUE: usize = 8;
const PAPER_IDX_JOURNAL_NAME: usize = 9;
const PAPER_IDX_TITLE_NGRAMS_CHARS: usize = 10;
const PAPER_IDX_VENUE_NGRAMS: usize = 11;
const PAPER_IDX_JOURNAL_NGRAMS: usize = 12;
const PAPER_IDX_REFERENCE_DETAILS: usize = 13;
const PAPER_IDX_YEAR: usize = 14;
const PAPER_IDX_REFERENCES: usize = 15;
const PAPER_IDX_PAPER_ID: usize = 16;
const FROM_DATASET_PAPER_PREPROCESS_CHUNK_SIZE: usize = 4096;
const PAPER_FASTPATH_REQUIRED_FIELDS: [(usize, &str); 16] = [
    (PAPER_IDX_TITLE, "title"),
    (PAPER_IDX_HAS_ABSTRACT, "has_abstract"),
    (PAPER_IDX_IN_SIGNATURES, "in_signatures"),
    (PAPER_IDX_IS_RELIABLE, "is_reliable"),
    (PAPER_IDX_PREDICTED_LANGUAGE, "predicted_language"),
    (PAPER_IDX_TITLE_NGRAMS_WORDS, "title_ngrams_words"),
    (PAPER_IDX_AUTHORS, "authors"),
    (PAPER_IDX_VENUE, "venue"),
    (PAPER_IDX_JOURNAL_NAME, "journal_name"),
    (PAPER_IDX_TITLE_NGRAMS_CHARS, "title_ngrams_chars"),
    (PAPER_IDX_VENUE_NGRAMS, "venue_ngrams"),
    (PAPER_IDX_JOURNAL_NGRAMS, "journal_ngrams"),
    (PAPER_IDX_REFERENCE_DETAILS, "reference_details"),
    (PAPER_IDX_YEAR, "year"),
    (PAPER_IDX_REFERENCES, "references"),
    (PAPER_IDX_PAPER_ID, "paper_id"),
];

const SIG_IDX_FIRST_RAW: usize = 0;
const SIG_IDX_FIRST_NORMALIZED_NO_APOSTROPHE: usize = 1;
const SIG_IDX_MIDDLE_RAW: usize = 2;
const SIG_IDX_MIDDLE_NORMALIZED_NO_APOSTROPHE: usize = 3;
const SIG_IDX_LAST_NORMALIZED: usize = 4;
const SIG_IDX_LAST_RAW: usize = 5;
const SIG_IDX_COAUTHORS: usize = 9;
const SIG_IDX_COAUTHOR_BLOCKS: usize = 10;
const SIG_IDX_AFFILIATIONS: usize = 12;
const SIG_IDX_AFFILIATIONS_NGRAMS: usize = 13;
const SIG_IDX_COAUTHOR_NGRAMS: usize = 14;
const SIG_IDX_EMAIL: usize = 15;
const SIG_IDX_ORCID: usize = 16;
const SIG_IDX_NAME_COUNTS: usize = 17;
const SIG_IDX_POSITION: usize = 18;
const SIG_IDX_PAPER_ID: usize = 23;
const FULL_FEATURE_COUNT: usize = 39;
const SIGNATURE_FASTPATH_REQUIRED_FIELDS: [(usize, &str); 13] = [
    (
        SIG_IDX_FIRST_NORMALIZED_NO_APOSTROPHE,
        "author_info_first_normalized_without_apostrophe",
    ),
    (
        SIG_IDX_MIDDLE_NORMALIZED_NO_APOSTROPHE,
        "author_info_middle_normalized_without_apostrophe",
    ),
    (SIG_IDX_LAST_NORMALIZED, "author_info_last_normalized"),
    (SIG_IDX_COAUTHORS, "author_info_coauthors"),
    (SIG_IDX_COAUTHOR_BLOCKS, "author_info_coauthor_blocks"),
    (SIG_IDX_AFFILIATIONS, "author_info_affiliations"),
    (
        SIG_IDX_AFFILIATIONS_NGRAMS,
        "author_info_affiliations_n_grams",
    ),
    (SIG_IDX_COAUTHOR_NGRAMS, "author_info_coauthor_n_grams"),
    (SIG_IDX_EMAIL, "author_info_email"),
    (SIG_IDX_ORCID, "author_info_orcid"),
    (SIG_IDX_NAME_COUNTS, "author_info_name_counts"),
    (SIG_IDX_POSITION, "author_info_position"),
    (SIG_IDX_PAPER_ID, "paper_id"),
];

struct MatrixAggregateIndexSelection {
    matrix_indices: Vec<usize>,
    aggregate_indices: Vec<usize>,
    aggregate_matrix_positions: Vec<usize>,
}

/// Resolve optional feature indices to concrete feature columns.
///
/// Contract: `None` expands to `0..full_cols`; explicit lists are returned
/// unchanged, including caller order and duplicate indices; every index must be
/// strictly less than `full_cols`.
fn resolve_feature_indices(
    argument_name: &str,
    indices: Option<Vec<usize>>,
    full_cols: usize,
) -> PyResult<Vec<usize>> {
    let resolved = indices.unwrap_or_else(|| (0..full_cols).collect());
    validate_feature_indices(argument_name, &resolved, full_cols)?;
    Ok(resolved)
}

fn validate_feature_indices(
    argument_name: &str,
    indices: &[usize],
    full_cols: usize,
) -> PyResult<()> {
    for idx in indices.iter() {
        if *idx >= full_cols {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{argument_name} contains out-of-range index {} for {} columns",
                idx, full_cols
            )));
        }
    }
    Ok(())
}

/// Map aggregate feature indices onto already materialized matrix columns.
///
/// Contract: one output position is produced for each aggregate index, in
/// aggregate-index order. Matrix indices keep caller order and duplicates;
/// duplicate matrix features map to their first matrix position because all
/// duplicate columns contain the same feature value for a row.
fn matrix_positions_for_feature_indices(
    matrix_indices: &[usize],
    aggregate_indices: &[usize],
) -> PyResult<Vec<usize>> {
    let mut first_position_by_feature: HashMap<usize, usize> =
        HashMap::with_capacity(matrix_indices.len());
    for (position, feature_index) in matrix_indices.iter().enumerate() {
        first_position_by_feature
            .entry(*feature_index)
            .or_insert(position);
    }

    let mut positions = Vec::with_capacity(aggregate_indices.len());
    for feature_index in aggregate_indices.iter() {
        let Some(matrix_position) = first_position_by_feature.get(feature_index) else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "aggregate index {} is not present in matrix_indices; include it to avoid recomputation",
                feature_index
            )));
        };
        positions.push(*matrix_position);
    }
    Ok(positions)
}

/// Resolve paired matrix and aggregate feature-index contracts.
///
/// Contract: `aggregate_indices=None` mirrors the resolved matrix columns; when
/// explicit, aggregates must be valid feature indices and already present in
/// `matrix_indices` so downstream aggregation reads the computed matrix instead
/// of recomputing feature rows.
fn resolve_matrix_aggregate_indices(
    matrix_indices: Option<Vec<usize>>,
    aggregate_indices: Option<Vec<usize>>,
    full_cols: usize,
) -> PyResult<MatrixAggregateIndexSelection> {
    let resolved_matrix_indices =
        resolve_feature_indices("matrix_indices", matrix_indices, full_cols)?;
    let resolved_aggregate_indices = match aggregate_indices {
        Some(indices) => {
            validate_feature_indices("aggregate_indices", &indices, full_cols)?;
            indices
        }
        None => resolved_matrix_indices.clone(),
    };
    let aggregate_matrix_positions = matrix_positions_for_feature_indices(
        &resolved_matrix_indices,
        &resolved_aggregate_indices,
    )?;
    Ok(MatrixAggregateIndexSelection {
        matrix_indices: resolved_matrix_indices,
        aggregate_indices: resolved_aggregate_indices,
        aggregate_matrix_positions,
    })
}

impl RustFeaturizer {
    fn cluster_seeds_disallow_index(&self) -> &HashMap<String, HashSet<String>> {
        self.cluster_seeds_disallow_index.get_or_init(|| {
            let mut index: HashMap<String, HashSet<String>> = HashMap::new();
            for (left, right) in self.cluster_seeds_disallow.iter() {
                index
                    .entry(left.clone())
                    .or_insert_with(HashSet::new)
                    .insert(right.clone());
            }
            index
        })
    }

    fn cluster_seeds_disallow_contains(&self, sig_id1: &str, sig_id2: &str) -> bool {
        let (left, right) = canonical_signature_pair_ref(sig_id1, sig_id2);
        self.cluster_seeds_disallow_index()
            .get(left)
            .is_some_and(|rights| rights.contains(right))
    }

    fn featurize_pair_data(
        &self,
        s1: &SignatureData,
        s2: &SignatureData,
        p1: &PaperData,
        p2: &PaperData,
    ) -> [f64; FULL_FEATURE_COUNT] {
        let mut feats = [f64::NAN; FULL_FEATURE_COUNT];
        let mut feat_i: usize = 0;
        macro_rules! push_feat {
            ($value:expr) => {{
                feats[feat_i] = $value;
                feat_i += 1;
            }};
        }

        let first1 = s1.first_without_apostrophe();
        let first2 = s2.first_without_apostrophe();
        let middle1 = s1.middle.as_deref();
        let middle2 = s2.middle.as_deref();

        push_feat!(first_names_equal(first1, first2));
        push_feat!(middle_initials_overlap(middle1, middle2));
        push_feat!(middle_names_equal(middle1, middle2));
        push_feat!(middle_one_missing(middle1, middle2));
        push_feat!(single_char_first(first1, first2));
        push_feat!(single_char_middle(middle1, middle2));

        push_feat!(counter_jaccard_data(
            &s1.affiliations,
            &s2.affiliations,
            f64::INFINITY,
        ));

        let (email_prefix, email_suffix) =
            match email_pair_parts(s1.email.as_deref(), s2.email.as_deref()) {
                Some(((p1, sfx1), (p2, sfx2))) => (
                    if p1 == p2 { 1.0 } else { 0.0 },
                    if sfx1 == sfx2 { 1.0 } else { 0.0 },
                ),
                None => (f64::NAN, f64::NAN),
            };
        push_feat!(email_prefix);
        push_feat!(email_suffix);

        push_feat!(set_jaccard_data(&s1.coauthor_blocks, &s2.coauthor_blocks));
        push_feat!(counter_jaccard_data(
            &s1.coauthor_ngrams,
            &s2.coauthor_ngrams,
            5000.0,
        ));
        push_feat!(set_jaccard_data(&s1.coauthors, &s2.coauthors));

        push_feat!(counter_jaccard_data(
            &p1.venue_ngrams,
            &p2.venue_ngrams,
            f64::INFINITY,
        ));
        push_feat!(year_diff(p1.year, p2.year));

        push_feat!(counter_jaccard_data(
            &p1.title_words,
            &p2.title_words,
            f64::INFINITY,
        ));
        push_feat!(counter_jaccard_data(
            &p1.title_chars,
            &p2.title_chars,
            f64::INFINITY,
        ));

        if self.compute_reference_features && p1.ref_details_present && p2.ref_details_present {
            push_feat!(counter_jaccard_data(
                &p1.ref_authors,
                &p2.ref_authors,
                5000.0,
            ));
            push_feat!(counter_jaccard_data(
                &p1.ref_titles,
                &p2.ref_titles,
                f64::INFINITY,
            ));
            push_feat!(counter_jaccard_data(
                &p1.ref_venues,
                &p2.ref_venues,
                f64::INFINITY,
            ));
            push_feat!(counter_jaccard_data(
                &p1.ref_blocks,
                &p2.ref_blocks,
                f64::INFINITY,
            ));
            let self_cite =
                if p1.references.contains(&s2.paper_id) || p2.references.contains(&s1.paper_id) {
                    1.0
                } else {
                    0.0
                };
            push_feat!(self_cite);
            push_feat!(refs_jaccard(&p1.references, &p2.references));
        } else {
            for _ in 0..6 {
                push_feat!(f64::NAN);
            }
        }

        let english_or_unknown_count = {
            let mut count = 0i64;
            if let Some(l1) = p1.predicted_language.as_deref() {
                if l1 == "en" || l1 == "un" {
                    count += 1;
                }
            }
            if let Some(l2) = p2.predicted_language.as_deref() {
                if l2 == "en" || l2 == "un" {
                    count += 1;
                }
            }
            count
        };

        push_feat!(position_diff(s1.position, s2.position));
        push_feat!((p1.has_abstract as i64 + p2.has_abstract as i64) as f64);
        push_feat!(english_or_unknown_count as f64);
        let same_lang = match (
            p1.predicted_language.as_deref(),
            p2.predicted_language.as_deref(),
        ) {
            (None, None) => true,
            (Some(a), Some(b)) => a == b,
            _ => false,
        };
        push_feat!(if same_lang { 1.0 } else { 0.0 });
        push_feat!((p1.is_reliable as i64 + p2.is_reliable as i64) as f64);

        let counts = compute_name_counts_data(s1.name_counts.as_ref(), s2.name_counts.as_ref());
        for value in counts.iter() {
            push_feat!(*value);
        }

        let specter_sim = if english_or_unknown_count == 2 {
            if let (Some(specter_a), Some(specter_b)) = (p1.specter.as_ref(), p2.specter.as_ref()) {
                let score = match (p1.specter_norm, p2.specter_norm) {
                    (Some(norm_a), Some(norm_b)) if specter_a.len() == specter_b.len() => {
                        cosine_sim_with_norms(specter_a, norm_a, specter_b, norm_b)
                    }
                    _ => cosine_sim_vec_f32(specter_a, specter_b),
                };
                score + 1.0
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };
        push_feat!(specter_sim);

        push_feat!(counter_jaccard_data(
            &p1.journal_ngrams,
            &p2.journal_ngrams,
            f64::INFINITY,
        ));

        let advanced = name_text_features(s1.adv_name_for_features(), s2.adv_name_for_features());
        for value in advanced.iter() {
            push_feat!(*value);
        }

        debug_assert_eq!(feat_i, FULL_FEATURE_COUNT);
        feats
    }

    fn constraint_value_from_records(
        &self,
        sig_id1: &str,
        sig_id2: &str,
        s1: &SignatureData,
        s2: &SignatureData,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        suppress_orcid: bool,
    ) -> Option<f64> {
        if self.cluster_seeds_disallow_contains(sig_id1, sig_id2) {
            return Some(self.cluster_seed_disallow_value);
        }

        if !incremental_dont_use_cluster_seeds {
            if let (Some(c1), Some(c2)) = (
                self.cluster_seeds_require.get(sig_id1),
                self.cluster_seeds_require.get(sig_id2),
            ) {
                if c1 == c2 {
                    return Some(self.cluster_seed_require_value);
                }
            }
        }

        if dont_merge_cluster_seeds && !incremental_dont_use_cluster_seeds {
            if let (Some(c1), Some(c2)) = (
                self.cluster_seeds_require.get(sig_id1),
                self.cluster_seeds_require.get(sig_id2),
            ) {
                if c1 != c2 {
                    return Some(self.cluster_seed_disallow_value);
                }
            }
        }

        if !suppress_orcid {
            if let (Some(o1), Some(o2)) = (s1.orcid.as_deref(), s2.orcid.as_deref()) {
                if o1 == o2 {
                    return Some(low_value);
                }
            }
        }

        let last1 = s1.last_normalized.as_deref().unwrap_or("");
        let last2 = s2.last_normalized.as_deref().unwrap_or("");
        if !lasts_equivalent_for_constraint(last1, last2) {
            return Some(high_value);
        }

        let first1 = s1.first_without_apostrophe().unwrap_or("");
        let first2 = s2.first_without_apostrophe().unwrap_or("");
        if !first1.is_empty() && !first2.is_empty() {
            if let (Some(c1), Some(c2)) = (first1.chars().next(), first2.chars().next()) {
                if c1 != c2 {
                    return Some(high_value);
                }
            }
        }

        let f1_join: String = first1.split_whitespace().collect();
        let f2_join: String = first2.split_whitespace().collect();
        let f1_tok = first1.split_whitespace().next().unwrap_or(first1);
        let f2_tok = first2.split_whitespace().next().unwrap_or(first2);
        let known_alias = name_tuple_contains(&self.name_tuples, first1, first2)
            || name_tuple_contains(&self.name_tuples, &f1_join, &f2_join)
            || name_tuple_contains(&self.name_tuples, f1_tok, f2_tok);

        let prefix = same_prefix_tokens(first1, first2);
        if !prefix && !known_alias {
            return Some(high_value);
        }

        let middle1_str = s1.middle.as_deref().unwrap_or("");
        let middle1_tokens: Vec<&str> = middle1_str.split_whitespace().collect();
        if !middle1_tokens.is_empty() {
            let middle2_str = s2.middle.as_deref().unwrap_or("");
            let middle2_tokens: Vec<&str> = middle2_str.split_whitespace().collect();
            if !middle2_tokens.is_empty() {
                let middle1_set: HashSet<&str> = middle1_tokens.iter().copied().collect();
                let middle2_set: HashSet<&str> = middle2_tokens.iter().copied().collect();
                let mut overlapping_affixes: HashSet<&str> = HashSet::new();
                for token in middle1_set.intersection(&middle2_set) {
                    if is_dropped_affix(token) {
                        overlapping_affixes.insert(*token);
                    }
                }

                let middle_1_all: Vec<&str> = middle1_tokens
                    .iter()
                    .copied()
                    .filter(|w| !w.is_empty() && !overlapping_affixes.contains(w))
                    .collect();
                let middle_2_all: Vec<&str> = middle2_tokens
                    .iter()
                    .copied()
                    .filter(|w| !w.is_empty() && !overlapping_affixes.contains(w))
                    .collect();

                let middle_1_words: HashSet<&str> = middle_1_all
                    .iter()
                    .copied()
                    .filter(|w| py_len(w) > 1)
                    .collect();
                let middle_2_words: HashSet<&str> = middle_2_all
                    .iter()
                    .copied()
                    .filter(|w| py_len(w) > 1)
                    .collect();

                let mut middle_1_firsts: HashSet<char> = HashSet::new();
                for word in middle_1_all.iter() {
                    if let Some(ch) = word.chars().next() {
                        middle_1_firsts.insert(ch);
                    }
                }
                let mut middle_2_firsts: HashSet<char> = HashSet::new();
                for word in middle_2_all.iter() {
                    if let Some(ch) = word.chars().next() {
                        middle_2_firsts.insert(ch);
                    }
                }

                let conflicting_initials = !middle_1_firsts.is_empty()
                    && !middle_2_firsts.is_empty()
                    && middle_1_firsts.is_disjoint(&middle_2_firsts);

                let mut middle_1_chars: HashSet<char> = HashSet::new();
                for word in middle_1_words.iter() {
                    for ch in word.chars() {
                        middle_1_chars.insert(ch);
                    }
                }
                let mut middle_2_chars: HashSet<char> = HashSet::new();
                for word in middle_2_words.iter() {
                    for ch in word.chars() {
                        middle_2_chars.insert(ch);
                    }
                }

                let conflicting_full_names = !middle_1_words.is_empty()
                    && !middle_2_words.is_empty()
                    && middle_1_words.is_disjoint(&middle_2_words)
                    && middle_1_chars != middle_2_chars;

                if conflicting_initials || conflicting_full_names {
                    return Some(high_value);
                }
            }
        }
        None
    }

    fn get_constraint_value_for_pair(
        &self,
        sig_id1: &str,
        sig_id2: &str,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        suppress_orcid: bool,
    ) -> PyResult<Option<f64>> {
        let s1 = self
            .signatures
            .get(sig_id1)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id1.to_string()))?;
        let s2 = self
            .signatures
            .get(sig_id2)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id2.to_string()))?;
        Ok(self.constraint_value_from_records(
            sig_id1,
            sig_id2,
            s1,
            s2,
            low_value,
            high_value,
            dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds,
            suppress_orcid,
        ))
    }

    fn signature_id_order(&self) -> &[String] {
        if !self.signature_ids.is_empty() {
            return self.signature_ids.as_slice();
        }
        self.cached_signature_id_order
            .get_or_init(|| {
                let mut ids: Vec<String> = self.signatures.keys().cloned().collect();
                ids.sort_unstable();
                ids
            })
            .as_slice()
    }

    fn full_feature_count(&self) -> usize {
        FULL_FEATURE_COUNT
    }

    fn signature_lookup(&self) -> PyResult<Vec<&SignatureData>> {
        let signature_ids = self.signature_id_order();
        let mut lookup: Vec<&SignatureData> = Vec::with_capacity(signature_ids.len());
        for signature_id in signature_ids.iter() {
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            lookup.push(signature);
        }
        Ok(lookup)
    }

    fn sparse_signature_paper_lookup_for_indices(
        &self,
        left_indices: &[u32],
        right_indices: &[u32],
    ) -> PyResult<Vec<Option<(&SignatureData, &PaperData)>>> {
        let signature_ids = self.signature_id_order();
        let signature_count = signature_ids.len();
        let mut used_indices = HashSet::<usize>::new();
        let mut max_index = 0usize;
        for (left_idx, right_idx) in left_indices.iter().zip(right_indices.iter()) {
            for raw_index in [*left_idx, *right_idx] {
                let index = raw_index as usize;
                if index >= signature_count {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "pair index out of range: index={} signature_count={}",
                        index, signature_count
                    )));
                }
                max_index = max_index.max(index);
                used_indices.insert(index);
            }
        }
        if used_indices.is_empty() {
            return Ok(Vec::new());
        }
        let mut lookup = vec![None; max_index + 1];
        for index in used_indices {
            let signature_id = &signature_ids[index];
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
            })?;
            lookup[index] = Some((signature, paper));
        }
        Ok(lookup)
    }

    fn sparse_signature_paper_lookup_for_pair_tuples(
        &self,
        pairs: &[(u32, u32)],
    ) -> PyResult<Vec<Option<(&SignatureData, &PaperData)>>> {
        let mut left_indices = Vec::with_capacity(pairs.len());
        let mut right_indices = Vec::with_capacity(pairs.len());
        for (left_idx, right_idx) in pairs.iter() {
            left_indices.push(*left_idx);
            right_indices.push(*right_idx);
        }
        self.sparse_signature_paper_lookup_for_indices(&left_indices, &right_indices)
    }

    fn pair_aggregate_row_ranges(owner_row_indices: &[u32]) -> Option<Vec<PairAggregateRowRange>> {
        if owner_row_indices.is_empty() {
            return Some(Vec::new());
        }
        let mut ranges = Vec::new();
        let mut start = 0usize;
        let mut previous = owner_row_indices[0];
        for (offset, row_index) in owner_row_indices.iter().enumerate().skip(1) {
            if *row_index < previous {
                return None;
            }
            if *row_index != previous {
                ranges.push(PairAggregateRowRange {
                    row_offset: previous as usize,
                    start,
                    stop: offset,
                });
                start = offset;
                previous = *row_index;
            }
        }
        ranges.push(PairAggregateRowRange {
            row_offset: previous as usize,
            start,
            stop: owner_row_indices.len(),
        });
        Some(ranges)
    }

    fn empty_pair_aggregate_buffers(
        row_count: usize,
        aggregate_cols: usize,
    ) -> PairAggregateBuffers {
        PairAggregateBuffers {
            counts: vec![0_u32; row_count],
            valid_counts: vec![0_u64; row_count * aggregate_cols],
            sums: vec![0.0_f64; row_count * aggregate_cols],
            mins: vec![f64::INFINITY; row_count * aggregate_cols],
            maxs: vec![f64::NEG_INFINITY; row_count * aggregate_cols],
        }
    }

    fn aggregate_pair_index_arrays_grouped(
        &self,
        left_indices: &[u32],
        right_indices: &[u32],
        row_ranges: &[PairAggregateRowRange],
        row_count: usize,
        aggregate_indices: &[usize],
        nan_value: f64,
        lookup: &[Option<(&SignatureData, &PaperData)>],
    ) -> PairAggregateBuffers {
        let aggregate_cols = aggregate_indices.len();
        let mut out = Self::empty_pair_aggregate_buffers(row_count, aggregate_cols);
        if aggregate_cols == 0 {
            for range in row_ranges.iter() {
                out.counts[range.row_offset] =
                    (range.stop - range.start).min(u32::MAX as usize) as u32;
            }
            return out;
        }

        let group_count = row_ranges.len();
        let mut group_counts = vec![0_u32; group_count];
        let mut group_valid_counts = vec![0_u64; group_count * aggregate_cols];
        let mut group_sums = vec![0.0_f64; group_count * aggregate_cols];
        let mut group_mins = vec![f64::INFINITY; group_count * aggregate_cols];
        let mut group_maxs = vec![f64::NEG_INFINITY; group_count * aggregate_cols];
        group_counts
            .par_iter_mut()
            .zip(group_valid_counts.par_chunks_mut(aggregate_cols))
            .zip(group_sums.par_chunks_mut(aggregate_cols))
            .zip(group_mins.par_chunks_mut(aggregate_cols))
            .zip(group_maxs.par_chunks_mut(aggregate_cols))
            .zip(row_ranges.par_iter())
            .for_each(
                |(((((count, valid_counts_row), sums_row), mins_row), maxs_row), range)| {
                    for pair_offset in range.start..range.stop {
                        *count = count.saturating_add(1);
                        let (s1, p1) = lookup[left_indices[pair_offset] as usize]
                            .expect("left signature index was validated before aggregation");
                        let (s2, p2) = lookup[right_indices[pair_offset] as usize]
                            .expect("right signature index was validated before aggregation");
                        let row = self.featurize_pair_data(s1, s2, p1, p2);
                        for (aggregate_position, feature_index) in
                            aggregate_indices.iter().enumerate()
                        {
                            let mut value = row[*feature_index];
                            if value.is_nan() && !nan_value.is_nan() {
                                value = nan_value;
                            }
                            if value.is_nan() {
                                continue;
                            }
                            valid_counts_row[aggregate_position] =
                                valid_counts_row[aggregate_position].saturating_add(1);
                            sums_row[aggregate_position] += value;
                            if value < mins_row[aggregate_position] {
                                mins_row[aggregate_position] = value;
                            }
                            if value > maxs_row[aggregate_position] {
                                maxs_row[aggregate_position] = value;
                            }
                        }
                    }
                },
            );

        for (group_offset, range) in row_ranges.iter().enumerate() {
            out.counts[range.row_offset] = group_counts[group_offset];
            let source_start = group_offset * aggregate_cols;
            let target_start = range.row_offset * aggregate_cols;
            out.valid_counts[target_start..target_start + aggregate_cols]
                .copy_from_slice(&group_valid_counts[source_start..source_start + aggregate_cols]);
            out.sums[target_start..target_start + aggregate_cols]
                .copy_from_slice(&group_sums[source_start..source_start + aggregate_cols]);
            out.mins[target_start..target_start + aggregate_cols]
                .copy_from_slice(&group_mins[source_start..source_start + aggregate_cols]);
            out.maxs[target_start..target_start + aggregate_cols]
                .copy_from_slice(&group_maxs[source_start..source_start + aggregate_cols]);
        }
        out
    }

    fn aggregate_pair_index_arrays_sequential(
        &self,
        left_indices: &[u32],
        right_indices: &[u32],
        owner_row_indices: &[u32],
        row_count: usize,
        aggregate_indices: &[usize],
        nan_value: f64,
        lookup: &[Option<(&SignatureData, &PaperData)>],
    ) -> PairAggregateBuffers {
        let aggregate_cols = aggregate_indices.len();
        let mut out = Self::empty_pair_aggregate_buffers(row_count, aggregate_cols);
        if aggregate_cols == 0 {
            for row_index in owner_row_indices.iter() {
                out.counts[*row_index as usize] = out.counts[*row_index as usize].saturating_add(1);
            }
            return out;
        }

        for (pair_offset, row_index) in owner_row_indices.iter().enumerate() {
            let row_offset = *row_index as usize;
            out.counts[row_offset] = out.counts[row_offset].saturating_add(1);
            let aggregate_row_start = row_offset * aggregate_cols;
            let (s1, p1) = lookup[left_indices[pair_offset] as usize]
                .expect("left signature index was validated before aggregation");
            let (s2, p2) = lookup[right_indices[pair_offset] as usize]
                .expect("right signature index was validated before aggregation");
            let row = self.featurize_pair_data(s1, s2, p1, p2);
            for (aggregate_position, feature_index) in aggregate_indices.iter().enumerate() {
                let mut value = row[*feature_index];
                if value.is_nan() && !nan_value.is_nan() {
                    value = nan_value;
                }
                if value.is_nan() {
                    continue;
                }
                let stats_index = aggregate_row_start + aggregate_position;
                out.valid_counts[stats_index] = out.valid_counts[stats_index].saturating_add(1);
                out.sums[stats_index] += value;
                if value < out.mins[stats_index] {
                    out.mins[stats_index] = value;
                }
                if value > out.maxs[stats_index] {
                    out.maxs[stats_index] = value;
                }
            }
        }
        out
    }

    fn update_top5_distance(row: &mut [f64], value: f64) {
        if value >= row[4] {
            return;
        }
        row[4] = value;
        row.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    }
}

#[pymethods]
impl RustFeaturizer {
    #[classattr]
    const SUPPORTS_FROM_DATASET_PAPER_PREPROCESS: bool = true;

    fn json_ingest_telemetry(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.json_ingest_telemetry
            .as_ref()
            .map(|telemetry| json_ingest_telemetry_to_py(py, telemetry))
            .transpose()
    }

    #[staticmethod]
    #[pyo3(signature = (dataset, cluster_seed_require_value = 0.0, cluster_seed_disallow_value = 10000.0, num_threads = None))]
    fn from_dataset(
        py: Python<'_>,
        dataset: &Bound<'_, PyAny>,
        cluster_seed_require_value: f64,
        cluster_seed_disallow_value: f64,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        let compute_reference_features: bool = dataset
            .getattr("compute_reference_features")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let preprocess: bool = dataset
            .getattr("preprocess")
            .and_then(|v| v.extract())
            .unwrap_or(true);

        let text_module = py.import("s2and.text")?;
        let unidecode = text_module.getattr("unidecode")?;
        let stop_words = extract_required_string_set(&text_module.getattr("STOPWORDS")?)?;
        let venue_stop_words =
            extract_required_string_set(&text_module.getattr("VENUE_STOP_WORDS")?)?;
        let name_prefixes = extract_required_string_set(&text_module.getattr("NAME_PREFIXES")?)?;
        let affiliation_stopwords = extract_affiliation_stopwords(py)?;
        let mut unidecode_char_map: HashMap<char, String> = HashMap::new();
        let mut language_detector: Option<LanguageDetectorCompat> = None;

        #[derive(Clone)]
        struct PaperInput {
            paper_id: PaperId,
            raw_title: String,
            raw_venue: String,
            raw_journal_name: String,
            raw_authors: Vec<(i64, String)>,
            need_title_words: bool,
            need_title_chars: bool,
            need_venue_ngrams: bool,
            need_journal_ngrams: bool,
            need_author_normalization: bool,
        }

        #[derive(Clone)]
        struct PaperComputed {
            paper_id: PaperId,
            normalized_authors: Vec<(i64, String)>,
            title_words: Option<CounterData>,
            title_chars: Option<CounterData>,
            venue_ngrams: Option<CounterData>,
            journal_ngrams: Option<CounterData>,
        }

        let papers_obj = dataset.getattr("papers")?;
        let papers_dict = papers_obj.downcast::<PyDict>()?;
        let use_paper_tuple_fastpath = validate_dict_namedtuple_fastpath_contract(
            &papers_dict,
            &PAPER_FASTPATH_REQUIRED_FIELDS,
            "Paper",
        )?;
        let specter_obj = dataset.getattr("specter_embeddings").ok();
        let specter_dict = match specter_obj.as_ref() {
            Some(value) if !value.is_none() => Some(specter_payload_to_dict(py, value)?),
            _ => None,
        };

        let mut papers = HashMap::with_capacity(papers_dict.len());
        let mut paper_authors_by_id: HashMap<PaperId, Vec<(i64, String)>> =
            HashMap::with_capacity(papers_dict.len());
        let mut paper_inputs: Vec<PaperInput> = Vec::new();
        for (_paper_id_obj, paper_obj) in papers_dict.iter() {
            let paper_id = extract_id_string(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_PAPER_ID,
                "paper_id",
            )?)?;
            let raw_title = extract_string_opt(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_TITLE,
                "title",
            )?)?
            .unwrap_or_default();
            let raw_venue = extract_string_opt(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_VENUE,
                "venue",
            )?)?
            .unwrap_or_default();
            let raw_journal_name = extract_string_opt(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_JOURNAL_NAME,
                "journal_name",
            )?)?
            .unwrap_or_default();
            let in_signatures: Option<bool> = get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_IN_SIGNATURES,
                "in_signatures",
            )?
            .extract()?;
            let in_signatures = in_signatures.unwrap_or(false);
            let venue_ngrams = extract_counter(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_VENUE_NGRAMS,
                "venue_ngrams",
            )?)?;
            let title_words = extract_counter(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_TITLE_NGRAMS_WORDS,
                "title_ngrams_words",
            )?)?;
            let title_chars = extract_counter(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_TITLE_NGRAMS_CHARS,
                "title_ngrams_chars",
            )?)?;
            let journal_ngrams = extract_counter(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_JOURNAL_NGRAMS,
                "journal_ngrams",
            )?)?;
            let paper_authors =
                extract_paper_authors_with_positions(&get_namedtuple_item_or_attr(
                    &paper_obj,
                    use_paper_tuple_fastpath,
                    PAPER_IDX_AUTHORS,
                    "authors",
                )?)?;

            let ref_details_present;
            let mut ref_authors = None;
            let mut ref_titles = None;
            let mut ref_venues = None;
            let mut ref_blocks = None;
            if compute_reference_features {
                let ref_details_obj = get_namedtuple_item_or_attr(
                    &paper_obj,
                    use_paper_tuple_fastpath,
                    PAPER_IDX_REFERENCE_DETAILS,
                    "reference_details",
                )?;
                ref_details_present = !ref_details_obj.is_none();
                if ref_details_present {
                    (ref_authors, ref_titles, ref_venues, ref_blocks) =
                        extract_reference_details_counters(py, &ref_details_obj)?;
                }
            } else {
                ref_details_present = false;
            }

            let references = extract_set_id_string(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_REFERENCES,
                "references",
            )?)?;
            let year: Option<i64> = get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_YEAR,
                "year",
            )?
            .extract()?;
            let year = match year {
                Some(v) if v > 0 => Some(v),
                _ => None,
            };
            let has_abstract: bool = get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_HAS_ABSTRACT,
                "has_abstract",
            )?
            .extract()?;
            let mut predicted_language = extract_string_opt(&get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_PREDICTED_LANGUAGE,
                "predicted_language",
            )?)?;
            let is_reliable_raw: Option<bool> = get_namedtuple_item_or_attr(
                &paper_obj,
                use_paper_tuple_fastpath,
                PAPER_IDX_IS_RELIABLE,
                "is_reliable",
            )?
            .extract()?;
            let mut is_reliable = is_reliable_raw.unwrap_or(false);

            let need_title_words = title_words.is_none();
            let need_title_chars = preprocess && in_signatures && title_chars.is_none();
            let need_venue_ngrams = preprocess && in_signatures && venue_ngrams.is_none();
            let need_journal_ngrams = preprocess && in_signatures && journal_ngrams.is_none();
            let need_language = in_signatures && predicted_language.is_none();
            if need_language {
                if language_detector.is_none() {
                    language_detector = Some(LanguageDetectorCompat::new(py));
                }
                let detector = language_detector.as_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("missing language detector")
                })?;
                let (reliable, _is_english, language) = detector.detect(&raw_title);
                predicted_language = Some(language);
                is_reliable = reliable;
            }

            let need_author_normalization = need_title_words
                || need_title_chars
                || need_venue_ngrams
                || need_journal_ngrams
                || need_language;
            if need_author_normalization {
                ensure_unidecode_for_text(&unidecode, &raw_title, &mut unidecode_char_map)?;
                if preprocess {
                    ensure_unidecode_for_text(&unidecode, &raw_venue, &mut unidecode_char_map)?;
                    ensure_unidecode_for_text(
                        &unidecode,
                        &raw_journal_name,
                        &mut unidecode_char_map,
                    )?;
                }
                for (_, author_name) in paper_authors.iter() {
                    ensure_unidecode_for_text(&unidecode, author_name, &mut unidecode_char_map)?;
                }
                paper_inputs.push(PaperInput {
                    paper_id: paper_id.clone(),
                    raw_title,
                    raw_venue,
                    raw_journal_name,
                    raw_authors: paper_authors.clone(),
                    need_title_words,
                    need_title_chars,
                    need_venue_ngrams,
                    need_journal_ngrams,
                    need_author_normalization,
                });
            }
            paper_authors_by_id.insert(paper_id.clone(), paper_authors);

            let specter = if let Some(spec_dict) = &specter_dict {
                extract_specter_for_paper_id(spec_dict, &paper_id)?
            } else {
                None
            };
            let specter_norm = specter.as_ref().map(|values| {
                values
                    .iter()
                    .map(|value| {
                        let value_f64 = *value as f64;
                        value_f64 * value_f64
                    })
                    .sum::<f64>()
                    .sqrt()
            });

            papers.insert(
                paper_id,
                PaperData {
                    venue_ngrams,
                    title_words,
                    title_chars,
                    ref_authors,
                    ref_titles,
                    ref_venues,
                    ref_blocks,
                    references,
                    year,
                    has_abstract,
                    predicted_language,
                    is_reliable,
                    journal_ngrams,
                    specter,
                    specter_norm,
                    ref_details_present,
                },
            );
        }

        for paper_input_chunk in paper_inputs.chunks(FROM_DATASET_PAPER_PREPROCESS_CHUNK_SIZE) {
            let computed_chunk: Vec<PaperComputed> = py.allow_threads(|| {
                let compute = || {
                    paper_input_chunk
                        .par_iter()
                        .map(|paper_input| {
                            let normalized_title = normalize_text_compat_from_map(
                                &paper_input.raw_title,
                                false,
                                &unidecode_char_map,
                            );
                            let normalized_venue = if paper_input.need_venue_ngrams {
                                Some(normalize_text_compat_from_map(
                                    &paper_input.raw_venue,
                                    false,
                                    &unidecode_char_map,
                                ))
                            } else {
                                None
                            };
                            let normalized_journal = if paper_input.need_journal_ngrams {
                                Some(normalize_text_compat_from_map(
                                    &paper_input.raw_journal_name,
                                    false,
                                    &unidecode_char_map,
                                ))
                            } else {
                                None
                            };
                            let normalized_authors = if paper_input.need_author_normalization {
                                paper_input
                                    .raw_authors
                                    .iter()
                                    .map(|(position, raw_name)| {
                                        (
                                            *position,
                                            normalize_text_compat_from_map(
                                                raw_name,
                                                false,
                                                &unidecode_char_map,
                                            ),
                                        )
                                    })
                                    .collect()
                            } else {
                                paper_input.raw_authors.clone()
                            };
                            let title_words = if paper_input.need_title_words {
                                counter_data_from_usize_map(word_ngrams_counter_python_compat(
                                    &normalized_title,
                                    &stop_words,
                                ))
                            } else {
                                None
                            };
                            let title_chars = if paper_input.need_title_chars {
                                counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                    &normalized_title,
                                    false,
                                    true,
                                    Some(&stop_words),
                                ))
                            } else {
                                None
                            };
                            let venue_ngrams = if paper_input.need_venue_ngrams {
                                counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                    normalized_venue.as_deref().unwrap_or(""),
                                    false,
                                    true,
                                    Some(&venue_stop_words),
                                ))
                            } else {
                                None
                            };
                            let journal_ngrams = if paper_input.need_journal_ngrams {
                                counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                    normalized_journal.as_deref().unwrap_or(""),
                                    false,
                                    true,
                                    Some(&venue_stop_words),
                                ))
                            } else {
                                None
                            };
                            PaperComputed {
                                paper_id: paper_input.paper_id.clone(),
                                normalized_authors,
                                title_words,
                                title_chars,
                                venue_ngrams,
                                journal_ngrams,
                            }
                        })
                        .collect::<Vec<_>>()
                };
                install_with_optional_rayon_pool(num_threads, compute)
            });
            for computed in computed_chunk {
                let PaperComputed {
                    paper_id,
                    normalized_authors,
                    title_words,
                    title_chars,
                    venue_ngrams,
                    journal_ngrams,
                } = computed;
                if let Some(paper) = papers.get_mut(&paper_id) {
                    if let Some(counter) = title_words {
                        paper.title_words = Some(counter);
                    }
                    if let Some(counter) = title_chars {
                        paper.title_chars = Some(counter);
                    }
                    if let Some(counter) = venue_ngrams {
                        paper.venue_ngrams = Some(counter);
                    }
                    if let Some(counter) = journal_ngrams {
                        paper.journal_ngrams = Some(counter);
                    }
                }
                paper_authors_by_id.insert(paper_id, normalized_authors);
            }
        }

        let signatures_obj = dataset.getattr("signatures")?;
        let signatures_dict = signatures_obj.downcast::<PyDict>()?;
        let use_signature_tuple_fastpath = validate_dict_namedtuple_fastpath_contract(
            &signatures_dict,
            &SIGNATURE_FASTPATH_REQUIRED_FIELDS,
            "Signature",
        )?;
        let mut signature_rows: Vec<(String, SignatureData, Option<String>, Option<String>)> =
            Vec::with_capacity(signatures_dict.len());
        for (sig_id_obj, sig_obj) in signatures_dict.iter() {
            let sig_id: String = sig_id_obj.extract()?;
            let raw_first = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_FIRST_RAW,
                "author_info_first",
            )?)?
            .unwrap_or_default();
            let raw_middle = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_MIDDLE_RAW,
                "author_info_middle",
            )?)?
            .unwrap_or_default();
            let raw_last = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_LAST_RAW,
                "author_info_last",
            )?)?
            .unwrap_or_default();

            let mut first = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_FIRST_NORMALIZED_NO_APOSTROPHE,
                "author_info_first_normalized_without_apostrophe",
            )?)?;
            let mut middle = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_MIDDLE_NORMALIZED_NO_APOSTROPHE,
                "author_info_middle_normalized_without_apostrophe",
            )?)?;
            let mut last_normalized = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_LAST_NORMALIZED,
                "author_info_last_normalized",
            )?)?;
            if first.is_none() || middle.is_none() || last_normalized.is_none() {
                ensure_unidecode_for_text(&unidecode, &raw_first, &mut unidecode_char_map)?;
                ensure_unidecode_for_text(&unidecode, &raw_middle, &mut unidecode_char_map)?;
                ensure_unidecode_for_text(&unidecode, &raw_last, &mut unidecode_char_map)?;

                let (first_without_apostrophe, middle_without_apostrophe) =
                    split_first_middle_hyphen_aware_compat(
                        &raw_first,
                        &raw_middle,
                        &name_prefixes,
                        &unidecode_char_map,
                    );
                if first.is_none() {
                    first = Some(first_without_apostrophe);
                }
                if middle.is_none() {
                    middle = Some(middle_without_apostrophe);
                }
                if last_normalized.is_none() {
                    last_normalized = Some(normalize_text_compat_from_map(
                        &raw_last,
                        false,
                        &unidecode_char_map,
                    ));
                }
            }

            let raw_orcid = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_ORCID,
                "author_info_orcid",
            )?)?;
            let orcid = raw_orcid.and_then(|value| normalize_orcid_compact_owned(&value));
            let email = extract_string_opt(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_EMAIL,
                "author_info_email",
            )?)?;
            let affiliations = extract_counter(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_AFFILIATIONS_NGRAMS,
                "author_info_affiliations_n_grams",
            )?)?;
            let mut coauthor_blocks = extract_optional_string_set(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_COAUTHOR_BLOCKS,
                "author_info_coauthor_blocks",
            )?)?;
            let coauthor_ngrams = extract_counter(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_COAUTHOR_NGRAMS,
                "author_info_coauthor_n_grams",
            )?)?;
            let mut coauthors = extract_optional_string_set(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_COAUTHORS,
                "author_info_coauthors",
            )?)?;
            let position: i64 = get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_POSITION,
                "author_info_position",
            )?
            .extract()?;
            let paper_id = extract_id_string(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_PAPER_ID,
                "paper_id",
            )?)?;
            let name_counts = extract_name_counts_data(&get_namedtuple_item_or_attr(
                &sig_obj,
                use_signature_tuple_fastpath,
                SIG_IDX_NAME_COUNTS,
                "author_info_name_counts",
            )?)?;

            let mut coauthor_text_for_compute: Option<String> = None;
            let mut affiliation_text_for_compute: Option<String> = None;

            let need_coauthor_from_papers =
                coauthor_ngrams.is_none() || coauthors.is_none() || coauthor_blocks.is_none();
            if need_coauthor_from_papers {
                if let Some(paper_authors) = paper_authors_by_id.get(&paper_id) {
                    let coauthor_names: Vec<String> = paper_authors
                        .iter()
                        .filter(|(author_position, _)| *author_position != position)
                        .map(|(_, author_name)| author_name.clone())
                        .collect();
                    if !coauthor_names.is_empty() {
                        if coauthor_ngrams.is_none() {
                            coauthor_text_for_compute = Some(coauthor_names.join(" "));
                        }
                        if coauthors.is_none() {
                            coauthors = Some(coauthor_names.iter().cloned().collect());
                        }
                        if coauthor_blocks.is_none() {
                            let mut coauthor_blocks_set: HashSet<String> =
                                HashSet::with_capacity(coauthor_names.len());
                            for coauthor in coauthor_names.iter() {
                                coauthor_blocks_set.insert(compute_block_compat(coauthor));
                            }
                            if !coauthor_blocks_set.is_empty() {
                                coauthor_blocks = Some(coauthor_blocks_set);
                            }
                        }
                    }
                }
            }

            if affiliations.is_none() {
                let affiliation_list = extract_string_list(&get_namedtuple_item_or_attr(
                    &sig_obj,
                    use_signature_tuple_fastpath,
                    SIG_IDX_AFFILIATIONS,
                    "author_info_affiliations",
                )?)?;
                let mut normalized_affiliation_list: Vec<String> =
                    Vec::with_capacity(affiliation_list.len());
                for affiliation in affiliation_list.iter() {
                    ensure_unidecode_for_text(&unidecode, affiliation, &mut unidecode_char_map)?;
                    normalized_affiliation_list.push(normalize_text_compat_from_map(
                        affiliation,
                        false,
                        &unidecode_char_map,
                    ));
                }
                let prefiltered = prefilter_affiliation_text(
                    &normalized_affiliation_list,
                    &affiliation_stopwords,
                );
                if !prefiltered.is_empty() {
                    affiliation_text_for_compute = Some(prefiltered);
                }
            }
            let adv_name = first.clone();

            signature_rows.push((
                sig_id,
                SignatureData {
                    first,
                    middle,
                    last_normalized,
                    orcid,
                    email,
                    affiliations,
                    coauthor_blocks,
                    coauthor_ngrams,
                    coauthors,
                    position,
                    paper_id,
                    name_counts,
                    adv_name,
                },
                coauthor_text_for_compute,
                affiliation_text_for_compute,
            ));
        }

        let computed_rows = py.allow_threads(|| {
            let compute = || {
                signature_rows
                    .into_par_iter()
                    .map(
                        |(
                            sig_id,
                            mut signature,
                            coauthor_text_for_compute,
                            affiliation_text_for_compute,
                        )| {
                            if signature.coauthor_ngrams.is_none() {
                                if let Some(coauthor_text) = coauthor_text_for_compute.as_deref() {
                                    signature.coauthor_ngrams = counter_data_from_usize_map(
                                        char_ngrams_counter(coauthor_text),
                                    );
                                }
                            }
                            if signature.affiliations.is_none() {
                                if let Some(affiliation_text) =
                                    affiliation_text_for_compute.as_deref()
                                {
                                    signature.affiliations = counter_data_from_usize_map(
                                        word_ngrams_counter(affiliation_text),
                                    );
                                }
                            }
                            (sig_id, signature)
                        },
                    )
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let mut signatures = HashMap::with_capacity(computed_rows.len());
        for (sig_id, signature) in computed_rows {
            signatures.insert(sig_id, signature);
        }
        let mut signature_ids: Vec<String> = signatures.keys().cloned().collect();
        signature_ids.sort_unstable();

        let name_tuples = extract_name_tuples_map(&dataset.getattr("name_tuples")?)?;
        let cluster_seeds_disallow = extract_pair_set(&dataset.getattr("cluster_seeds_disallow")?)?;
        let cluster_seeds_require =
            extract_cluster_seeds_require(&dataset.getattr("cluster_seeds_require")?)?;

        Ok(RustFeaturizer {
            signatures,
            signature_ids,
            papers,
            name_tuples,
            cluster_seeds_disallow,
            cluster_seeds_require,
            compute_reference_features,
            cluster_seed_require_value,
            cluster_seed_disallow_value,
            json_ingest_telemetry: None,
            cached_signature_id_order: OnceLock::new(),
            cluster_seeds_disallow_index: OnceLock::new(),
        })
    }

    #[staticmethod]
    #[pyo3(
        signature = (
            paths,
            signature_ids = None,
            name_tuples = None,
            name_counts_path = None,
            preprocess = true,
            compute_reference_features = false,
            cluster_seed_require_value = 0.0,
            cluster_seed_disallow_value = 10000.0,
            num_threads = None
        )
    )]
    fn from_arrow_paths(
        py: Python<'_>,
        paths: &Bound<'_, PyAny>,
        signature_ids: Option<&Bound<'_, PyAny>>,
        name_tuples: Option<&Bound<'_, PyAny>>,
        name_counts_path: Option<&str>,
        preprocess: bool,
        compute_reference_features: bool,
        cluster_seed_require_value: f64,
        cluster_seed_disallow_value: f64,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        if compute_reference_features {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "RustFeaturizer.from_arrow_paths does not support reference features",
            ));
        }

        let signatures_path =
            extract_path_mapping_string(paths, "signatures", true)?.ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err("missing signatures Arrow path")
            })?;
        let papers_path = extract_path_mapping_string(paths, "papers", true)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing papers Arrow path"))?;
        let paper_authors_path = extract_path_mapping_string(paths, "paper_authors", true)?
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err("missing paper_authors Arrow path")
            })?;
        let cluster_seeds_path = extract_path_mapping_string(paths, "cluster_seeds", false)?;
        let cluster_seed_disallows_path =
            extract_path_mapping_string(paths, "cluster_seed_disallows", false)?;
        let specter_path = extract_path_mapping_string(paths, "specter", false)?;
        let name_counts_arrow_path = extract_path_mapping_string(paths, "name_counts", false)?;
        let name_counts_index_path = extract_name_counts_index_path(paths)?;
        let name_tuples_arrow_path = match extract_path_mapping_string(paths, "name_pairs", false)?
        {
            Some(path) => Some(path),
            None => extract_path_mapping_string(paths, "name_tuples", false)?,
        };
        let signatures_batch_index_path =
            extract_path_mapping_string(paths, "signatures_batch_index", false)?;
        let papers_batch_index_path =
            extract_path_mapping_string(paths, "papers_batch_index", false)?;
        let paper_authors_batch_index_path =
            extract_path_mapping_string(paths, "paper_authors_batch_index", false)?;
        let specter_batch_index_path =
            extract_path_mapping_string(paths, "specter_batch_index", false)?;

        let requested_signature_ids = match signature_ids {
            Some(obj) if !obj.is_none() => Some(
                PyIterator::from_object(obj)?
                    .map(|item| item.and_then(|value| value.extract::<String>()))
                    .collect::<PyResult<Vec<_>>>()?,
            ),
            _ => None,
        };
        let keep_signature_ids: Option<HashSet<String>> = requested_signature_ids
            .as_ref()
            .map(|ids| ids.iter().cloned().collect());

        let parse_start = Instant::now();
        let (raw_signatures, _) = read_raw_arrow_signatures_with_optional_index(
            &signatures_path,
            signatures_batch_index_path.as_deref(),
            keep_signature_ids.as_ref(),
            true,
        )?;
        let mut signature_ids = match requested_signature_ids {
            Some(ids) => ids,
            None => {
                let mut ids = raw_signatures.keys().cloned().collect::<Vec<_>>();
                ids.sort_unstable();
                ids
            }
        };
        let mut seen_signature_ids = HashSet::<String>::with_capacity(signature_ids.len());
        signature_ids.retain(|signature_id| seen_signature_ids.insert(signature_id.clone()));
        let missing_signature_ids = signature_ids
            .iter()
            .filter(|signature_id| !raw_signatures.contains_key(*signature_id))
            .take(10)
            .cloned()
            .collect::<Vec<_>>();
        if !missing_signature_ids.is_empty() {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Arrow signatures input is missing requested signature ids: {missing_signature_ids:?}"
            )));
        }
        let selected_signature_id_set = signature_ids.iter().cloned().collect::<HashSet<_>>();
        let needed_paper_ids = signature_ids
            .iter()
            .filter_map(|signature_id| raw_signatures.get(signature_id))
            .map(|signature| signature.paper_id.clone())
            .collect::<HashSet<_>>();
        let (raw_papers, _) = read_raw_arrow_papers_with_optional_index(
            &papers_path,
            papers_batch_index_path.as_deref(),
            &needed_paper_ids,
            true,
        )?;
        let (mut raw_authors_by_paper, _) = read_raw_arrow_paper_authors_with_optional_index(
            &paper_authors_path,
            paper_authors_batch_index_path.as_deref(),
            &needed_paper_ids,
            true,
        )?;
        let specter_by_paper = match specter_path.as_ref() {
            Some(path) => {
                read_raw_arrow_specter_with_optional_index(
                    path,
                    specter_batch_index_path.as_deref(),
                    &needed_paper_ids,
                    true,
                )?
                .0
            }
            None => HashMap::new(),
        };
        let mut cluster_seeds_require = HashMap::<String, ClusterId>::new();
        if let Some(path) = cluster_seeds_path.as_ref() {
            let (_component_order, members_by_component) = read_raw_arrow_cluster_seeds(path)?;
            for (component_key, members) in members_by_component {
                for signature_id in members {
                    if selected_signature_id_set.contains(&signature_id) {
                        cluster_seeds_require
                            .insert(signature_id, ClusterId::Str(component_key.clone()));
                    }
                }
            }
        }
        let mut cluster_seeds_disallow = HashSet::<(String, String)>::new();
        if let Some(path) = cluster_seed_disallows_path.as_ref() {
            for (left, right) in read_raw_arrow_cluster_seed_disallows(path)? {
                if selected_signature_id_set.contains(&left)
                    && selected_signature_id_set.contains(&right)
                {
                    cluster_seeds_disallow.insert((left, right));
                }
            }
        }
        let parse_seconds = parse_start.elapsed().as_secs_f64();

        let text_module = py.import("s2and.text")?;
        let unidecode = text_module.getattr("unidecode")?;
        let stop_words = extract_required_string_set(&text_module.getattr("STOPWORDS")?)?;
        let venue_stop_words =
            extract_required_string_set(&text_module.getattr("VENUE_STOP_WORDS")?)?;
        let name_prefixes = extract_required_string_set(&text_module.getattr("NAME_PREFIXES")?)?;
        let affiliation_stopwords = extract_affiliation_stopwords(py)?;
        let raw_name_counts = match name_counts_index_path.as_ref() {
            Some(path) => read_raw_name_counts_index(path)?,
            None => match name_counts_arrow_path.as_ref() {
                Some(path) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "name_counts Arrow path '{path}' requires name_counts_index; refusing slow Arrow fallback"
                    )));
                }
                None => load_raw_name_counts_from_json_path(name_counts_path, None, false)?,
            },
        };
        let language_detector = Some(LanguageDetectorCompat::new(py));

        let mut unidecode_char_map: HashMap<char, String> = HashMap::new();
        ensure_unidecode_for_raw_arrow_inputs(
            &unidecode,
            &raw_signatures,
            &raw_papers,
            &raw_authors_by_paper,
            &mut unidecode_char_map,
        )?;
        let mut defaulted_signature_author_position_count = 0usize;
        let mut signature_inputs = Vec::<StageSignatureInput>::with_capacity(signature_ids.len());
        for signature_id in signature_ids.iter() {
            let raw_signature = raw_signatures.get(signature_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "Arrow signatures input is missing signature_id '{signature_id}'"
                ))
            })?;
            let position = raw_signature.position.unwrap_or_else(|| {
                defaulted_signature_author_position_count += 1;
                0
            });
            signature_inputs.push(StageSignatureInput {
                sig_id: signature_id.clone(),
                paper_id: raw_signature.paper_id.clone(),
                raw_first: raw_signature.author_first.clone(),
                raw_middle: raw_signature.author_middle.clone(),
                raw_last: raw_signature.author_last.clone(),
                email: raw_signature.email.clone(),
                position,
                affiliation_values: raw_signature.affiliations.clone(),
                orcid: raw_signature.orcid.clone(),
            });
        }

        let mut paper_inputs = Vec::<StagePaperInput>::with_capacity(needed_paper_ids.len());
        for paper_id in needed_paper_ids.iter() {
            let Some(raw_paper) = raw_papers.get(paper_id) else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Arrow signatures reference missing paper_id '{paper_id}'"
                )));
            };
            let raw_authors = raw_authors_by_paper.remove(paper_id).unwrap_or_default();
            let (is_reliable, predicted_language) = if raw_paper.predicted_language.is_some() {
                (
                    raw_paper.is_reliable.unwrap_or(false),
                    raw_paper.predicted_language.clone(),
                )
            } else {
                let detector = language_detector.as_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("missing language detector")
                })?;
                let (reliable, _is_english, language) = detector.detect(&raw_paper.title);
                (reliable, Some(language))
            };
            paper_inputs.push(StagePaperInput {
                paper_id: paper_id.clone(),
                raw_title: raw_paper.title.clone(),
                raw_venue: raw_paper.venue.clone(),
                raw_journal: raw_paper.journal_name.clone(),
                raw_authors,
                year: raw_paper.year.filter(|year| *year > 0),
                has_abstract: !raw_paper.abstract_text.is_empty(),
                predicted_language,
                is_reliable,
            });
        }

        let paper_preprocess_start = Instant::now();
        let computed_papers = py.allow_threads(|| {
            let compute = || {
                preprocess_stage_papers(
                    &paper_inputs,
                    preprocess,
                    &unidecode_char_map,
                    &stop_words,
                    &venue_stop_words,
                )
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        let mut preprocessed_papers: HashMap<PaperId, StagePaperPreprocessed> =
            HashMap::with_capacity(computed_papers.len());
        for (paper_id, preprocessed) in computed_papers {
            preprocessed_papers.insert(paper_id, preprocessed);
        }
        let paper_preprocess_seconds = paper_preprocess_start.elapsed().as_secs_f64();

        let signature_preprocess_start = Instant::now();
        let computed_signatures = py.allow_threads(|| {
            let compute = || {
                preprocess_stage_signatures(
                    &signature_inputs,
                    &preprocessed_papers,
                    &raw_name_counts,
                    &name_prefixes,
                    &affiliation_stopwords,
                    &unidecode_char_map,
                    preprocess,
                )
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        let mut signatures: HashMap<String, SignatureData> =
            HashMap::with_capacity(computed_signatures.len());
        for (sig_id, signature) in computed_signatures {
            signatures.insert(sig_id, signature);
        }
        let signature_preprocess_seconds = signature_preprocess_start.elapsed().as_secs_f64();

        let reference_counter_start = Instant::now();
        let mut missing_specter_paper_count = 0usize;
        let mut papers: HashMap<PaperId, PaperData> =
            HashMap::with_capacity(preprocessed_papers.len());
        for (paper_id, paper) in preprocessed_papers.into_iter() {
            let specter = specter_by_paper.get(&paper_id).cloned();
            if specter.is_none() {
                missing_specter_paper_count += 1;
            }
            let specter_norm = specter.as_ref().map(|values| {
                values
                    .iter()
                    .map(|value| {
                        let value_f64 = *value as f64;
                        value_f64 * value_f64
                    })
                    .sum::<f64>()
                    .sqrt()
            });
            papers.insert(
                paper_id,
                PaperData {
                    venue_ngrams: paper.venue_ngrams,
                    title_words: paper.title_words,
                    title_chars: paper.title_chars,
                    ref_authors: None,
                    ref_titles: None,
                    ref_venues: None,
                    ref_blocks: None,
                    ref_details_present: false,
                    references: HashSet::new(),
                    year: paper.year,
                    has_abstract: paper.has_abstract,
                    predicted_language: paper.predicted_language,
                    is_reliable: paper.is_reliable,
                    journal_ngrams: paper.journal_ngrams,
                    specter,
                    specter_norm,
                },
            );
        }
        let reference_counter_seconds = reference_counter_start.elapsed().as_secs_f64();

        let name_tuples = match name_tuples_arrow_path.as_ref() {
            Some(path) => read_raw_arrow_name_tuples(path)?,
            None => extract_feature_block_name_tuples(py, name_tuples)?,
        };
        let json_ingest_telemetry = JsonIngestTelemetry {
            json_parse_seconds: parse_seconds,
            paper_preprocess_seconds,
            reference_counter_seconds,
            signature_preprocess_seconds,
            cluster_seed_seconds: 0.0,
            missing_specter_paper_count,
            defaulted_name_count_signature_count: 0,
            defaulted_name_count_first_count: 0,
            defaulted_name_count_first_last_count: 0,
            defaulted_name_count_last_count: 0,
            defaulted_name_count_last_first_initial_count: 0,
            defaulted_signature_author_position_count,
            defaulted_paper_author_position_count: 0,
        };

        Ok(RustFeaturizer {
            signatures,
            signature_ids,
            papers,
            name_tuples,
            cluster_seeds_disallow,
            cluster_seeds_require,
            compute_reference_features,
            cluster_seed_require_value,
            cluster_seed_disallow_value,
            json_ingest_telemetry: Some(json_ingest_telemetry),
            cached_signature_id_order: OnceLock::new(),
            cluster_seeds_disallow_index: OnceLock::new(),
        })
    }

    #[staticmethod]
    #[pyo3(
        signature = (
            feature_block,
            name_tuples = None,
            name_counts_path = None,
            preprocess = true,
            compute_reference_features = false,
            cluster_seed_require_value = 0.0,
            cluster_seed_disallow_value = 10000.0,
            num_threads = None
        )
    )]
    fn from_feature_block(
        py: Python<'_>,
        feature_block: &Bound<'_, PyAny>,
        name_tuples: Option<&Bound<'_, PyAny>>,
        name_counts_path: Option<&str>,
        preprocess: bool,
        compute_reference_features: bool,
        cluster_seed_require_value: f64,
        cluster_seed_disallow_value: f64,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        if compute_reference_features {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "RustFeaturizer.from_feature_block does not support reference features",
            ));
        }

        let text_module = py.import("s2and.text")?;
        let unidecode = text_module.getattr("unidecode")?;
        let stop_words = extract_required_string_set(&text_module.getattr("STOPWORDS")?)?;
        let venue_stop_words =
            extract_required_string_set(&text_module.getattr("VENUE_STOP_WORDS")?)?;
        let name_prefixes = extract_required_string_set(&text_module.getattr("NAME_PREFIXES")?)?;
        let affiliation_stopwords = extract_affiliation_stopwords(py)?;
        let raw_name_counts = load_raw_name_counts_from_json_path(name_counts_path, None, false)?;
        let language_detector = Some(LanguageDetectorCompat::new(py));

        let parse_start = Instant::now();
        let mut unidecode_char_map: HashMap<char, String> = HashMap::new();
        let signatures_obj = feature_block.getattr("signatures")?;
        let mut signature_inputs: Vec<StageSignatureInput> = Vec::new();
        let mut needed_paper_ids: HashSet<PaperId> = HashSet::new();
        let mut defaulted_signature_author_position_count = 0usize;
        for item in PyIterator::from_object(&signatures_obj)? {
            let signature = item?;
            let sig_id: String = signature.getattr("signature_id")?.extract()?;
            let paper_id = extract_id_string(&signature.getattr("paper_id")?)?;
            needed_paper_ids.insert(paper_id.clone());
            let raw_first: String = signature
                .getattr("author_first")?
                .extract::<Option<String>>()?
                .unwrap_or_default();
            let raw_middle: String = signature
                .getattr("author_middle")?
                .extract::<Option<String>>()?
                .unwrap_or_default();
            let raw_last: String = signature
                .getattr("author_last")?
                .extract::<Option<String>>()?
                .unwrap_or_default();
            let email: Option<String> = signature.getattr("author_email")?.extract()?;
            let position = match signature
                .getattr("author_position")?
                .extract::<Option<i64>>()?
            {
                Some(value) => value,
                None => {
                    defaulted_signature_author_position_count += 1;
                    0
                }
            };
            let affiliation_values =
                extract_string_list(&signature.getattr("author_affiliations")?)?;
            let orcid: Option<String> = signature.getattr("author_orcid")?.extract()?;

            signature_inputs.push(StageSignatureInput {
                sig_id,
                paper_id,
                raw_first,
                raw_middle,
                raw_last,
                email,
                position,
                affiliation_values,
                orcid,
            });
        }
        ensure_unidecode_for_signature_texts(
            &unidecode,
            signature_inputs
                .iter()
                .map(|signature| SignatureTextFields {
                    author_first: &signature.raw_first,
                    author_middle: &signature.raw_middle,
                    author_last: &signature.raw_last,
                    author_suffix: "",
                    affiliations: &signature.affiliation_values,
                }),
            &mut unidecode_char_map,
        )?;

        let mut raw_authors_by_paper: HashMap<PaperId, Vec<(i64, String)>> = HashMap::new();
        let paper_authors_obj = feature_block.getattr("paper_authors")?;
        let mut defaulted_paper_author_position_count = 0usize;
        for item in PyIterator::from_object(&paper_authors_obj)? {
            let author = item?;
            let paper_id = extract_id_string(&author.getattr("paper_id")?)?;
            let position = match author.getattr("position")?.extract::<Option<i64>>()? {
                Some(value) => value,
                None => {
                    defaulted_paper_author_position_count += 1;
                    0
                }
            };
            let author_name: String = author.getattr("author_name")?.extract()?;
            raw_authors_by_paper
                .entry(paper_id)
                .or_insert_with(Vec::new)
                .push((position, author_name));
        }
        for authors in raw_authors_by_paper.values_mut() {
            authors.sort_by_key(|(position, _name)| *position);
        }
        ensure_unidecode_for_paper_author_texts(
            &unidecode,
            raw_authors_by_paper.values().map(Vec::as_slice),
            &mut unidecode_char_map,
        )?;

        let papers_obj = feature_block.getattr("papers")?;
        let mut paper_inputs: Vec<StagePaperInput> = Vec::new();
        for item in PyIterator::from_object(&papers_obj)? {
            let paper = item?;
            let paper_id = extract_id_string(&paper.getattr("paper_id")?)?;
            if !needed_paper_ids.contains(&paper_id) {
                continue;
            }
            let raw_title: String = paper
                .getattr("title")?
                .extract::<Option<String>>()?
                .unwrap_or_default();
            let raw_venue: String = paper
                .getattr("venue")?
                .extract::<Option<String>>()?
                .unwrap_or_default();
            let raw_journal: String = paper
                .getattr("journal_name")?
                .extract::<Option<String>>()?
                .unwrap_or_default();
            let raw_authors = raw_authors_by_paper.remove(&paper_id).unwrap_or_default();
            let year: Option<i64> = match paper.getattr("year")?.extract::<Option<i64>>()? {
                Some(value) if value > 0 => Some(value),
                _ => None,
            };
            let abstract_text: Option<String> = paper.getattr("abstract")?.extract()?;
            let has_abstract = abstract_text
                .as_ref()
                .map_or(false, |value| !value.is_empty());
            let supplied_predicted_language: Option<String> =
                paper.getattr("predicted_language")?.extract()?;
            let supplied_is_reliable: Option<bool> = paper.getattr("is_reliable")?.extract()?;
            let (is_reliable, predicted_language) = if supplied_predicted_language.is_some() {
                (
                    supplied_is_reliable.unwrap_or(false),
                    supplied_predicted_language,
                )
            } else {
                let detector = language_detector.as_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("missing language detector")
                })?;
                let (reliable, _is_english, language) = detector.detect(&raw_title);
                (reliable, Some(language))
            };
            paper_inputs.push(StagePaperInput {
                paper_id,
                raw_title,
                raw_venue,
                raw_journal,
                raw_authors,
                year,
                has_abstract,
                predicted_language,
                is_reliable,
            });
        }
        ensure_unidecode_for_paper_texts(
            &unidecode,
            paper_inputs.iter().map(|paper| PaperTextFields {
                title: &paper.raw_title,
                venue: &paper.raw_venue,
                journal_name: &paper.raw_journal,
            }),
            &mut unidecode_char_map,
        )?;
        let parse_seconds = parse_start.elapsed().as_secs_f64();

        let paper_preprocess_start = Instant::now();
        let computed_papers = py.allow_threads(|| {
            let compute = || {
                preprocess_stage_papers(
                    &paper_inputs,
                    preprocess,
                    &unidecode_char_map,
                    &stop_words,
                    &venue_stop_words,
                )
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        let mut preprocessed_papers: HashMap<PaperId, StagePaperPreprocessed> =
            HashMap::with_capacity(computed_papers.len());
        for (paper_id, preprocessed) in computed_papers {
            preprocessed_papers.insert(paper_id, preprocessed);
        }
        let paper_preprocess_seconds = paper_preprocess_start.elapsed().as_secs_f64();

        let missing_paper_ids: Vec<String> = signature_inputs
            .iter()
            .filter(|entry| !preprocessed_papers.contains_key(&entry.paper_id))
            .map(|entry| entry.paper_id.to_string())
            .collect();
        if !missing_paper_ids.is_empty() {
            let examples = missing_paper_ids
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "FeatureBlock signatures reference {} missing papers; examples: {}",
                missing_paper_ids.len(),
                examples
            )));
        }

        let signature_preprocess_start = Instant::now();
        let computed_signatures = py.allow_threads(|| {
            let compute = || {
                preprocess_stage_signatures(
                    &signature_inputs,
                    &preprocessed_papers,
                    &raw_name_counts,
                    &name_prefixes,
                    &affiliation_stopwords,
                    &unidecode_char_map,
                    preprocess,
                )
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        let mut signatures: HashMap<String, SignatureData> =
            HashMap::with_capacity(computed_signatures.len());
        for (sig_id, signature) in computed_signatures {
            signatures.insert(sig_id, signature);
        }
        let signature_preprocess_seconds = signature_preprocess_start.elapsed().as_secs_f64();

        let specter_by_paper = extract_feature_block_specter_by_paper(feature_block)?;
        let reference_counter_start = Instant::now();
        let mut missing_specter_paper_count = 0usize;
        let mut papers: HashMap<PaperId, PaperData> =
            HashMap::with_capacity(preprocessed_papers.len());
        for (paper_id, paper) in preprocessed_papers.into_iter() {
            let specter = specter_by_paper.get(&paper_id).cloned();
            if specter.is_none() {
                missing_specter_paper_count += 1;
            }
            let specter_norm = specter.as_ref().map(|values| {
                values
                    .iter()
                    .map(|value| {
                        let value_f64 = *value as f64;
                        value_f64 * value_f64
                    })
                    .sum::<f64>()
                    .sqrt()
            });
            papers.insert(
                paper_id,
                PaperData {
                    venue_ngrams: paper.venue_ngrams,
                    title_words: paper.title_words,
                    title_chars: paper.title_chars,
                    ref_authors: None,
                    ref_titles: None,
                    ref_venues: None,
                    ref_blocks: None,
                    ref_details_present: false,
                    references: HashSet::new(),
                    year: paper.year,
                    has_abstract: paper.has_abstract,
                    predicted_language: paper.predicted_language,
                    is_reliable: paper.is_reliable,
                    journal_ngrams: paper.journal_ngrams,
                    specter,
                    specter_norm,
                },
            );
        }
        let reference_counter_seconds = reference_counter_start.elapsed().as_secs_f64();

        let mut signature_ids: Vec<String> =
            PyIterator::from_object(&feature_block.getattr("signature_ids")?)?
                .map(|item| item.and_then(|value| value.extract::<String>()))
                .collect::<PyResult<Vec<_>>>()?;
        if signature_ids.is_empty() {
            signature_ids = signatures.keys().cloned().collect();
            signature_ids.sort_unstable();
        }
        let name_tuples = extract_feature_block_name_tuples(py, name_tuples)?;
        let cluster_seed_start = Instant::now();
        let cluster_seeds_require = extract_feature_block_cluster_seeds_require(feature_block)?;
        let cluster_seeds_disallow = extract_feature_block_cluster_seeds_disallow(feature_block)?;
        let cluster_seed_seconds = cluster_seed_start.elapsed().as_secs_f64();

        let json_ingest_telemetry = JsonIngestTelemetry {
            json_parse_seconds: parse_seconds,
            paper_preprocess_seconds,
            reference_counter_seconds,
            signature_preprocess_seconds,
            cluster_seed_seconds,
            missing_specter_paper_count,
            defaulted_name_count_signature_count: 0,
            defaulted_name_count_first_count: 0,
            defaulted_name_count_first_last_count: 0,
            defaulted_name_count_last_count: 0,
            defaulted_name_count_last_first_initial_count: 0,
            defaulted_signature_author_position_count,
            defaulted_paper_author_position_count,
        };

        Ok(RustFeaturizer {
            signatures,
            signature_ids,
            papers,
            name_tuples,
            cluster_seeds_disallow,
            cluster_seeds_require,
            compute_reference_features,
            cluster_seed_require_value,
            cluster_seed_disallow_value,
            json_ingest_telemetry: Some(json_ingest_telemetry),
            cached_signature_id_order: OnceLock::new(),
            cluster_seeds_disallow_index: OnceLock::new(),
        })
    }

    #[staticmethod]
    #[pyo3(
        signature = (
            signatures_path,
            papers_path,
            cluster_seeds_path = None,
            specter_embeddings = None,
            name_tuples_path = None,
            name_counts_path = None,
            preprocess = true,
            compute_reference_features = false,
            cluster_seed_require_value = 0.0,
            cluster_seed_disallow_value = 10000.0,
            num_threads = None,
            expected_normalization_version = None,
            allow_normalization_version_mismatch = false
        )
    )]
    fn from_json_paths(
        py: Python<'_>,
        signatures_path: &str,
        papers_path: &str,
        cluster_seeds_path: Option<&str>,
        specter_embeddings: Option<&Bound<'_, PyAny>>,
        name_tuples_path: Option<&str>,
        name_counts_path: Option<&str>,
        preprocess: bool,
        compute_reference_features: bool,
        cluster_seed_require_value: f64,
        cluster_seed_disallow_value: f64,
        num_threads: Option<usize>,
        expected_normalization_version: Option<&str>,
        allow_normalization_version_mismatch: bool,
    ) -> PyResult<Self> {
        let text_module = py.import("s2and.text")?;
        let unidecode = text_module.getattr("unidecode")?;
        let stop_words_obj = text_module.getattr("STOPWORDS")?;
        let venue_stop_words_obj = text_module.getattr("VENUE_STOP_WORDS")?;
        let name_prefixes_obj = text_module.getattr("NAME_PREFIXES")?;

        let stop_words = extract_required_string_set(&stop_words_obj)?;
        let venue_stop_words = extract_required_string_set(&venue_stop_words_obj)?;
        let name_prefixes = extract_required_string_set(&name_prefixes_obj)?;

        let language_detector = Some(LanguageDetectorCompat::new(py));

        let json_parse_start = Instant::now();
        let signatures_json = load_json_value(signatures_path)?;
        let signatures_obj = json_as_object(&signatures_json, "signatures payload")?;
        let raw_name_counts = load_raw_name_counts_from_json_path(
            name_counts_path,
            expected_normalization_version,
            allow_normalization_version_mismatch,
        )?;
        let name_tuples = load_name_tuples_from_text_path(py, name_tuples_path)?;
        let affiliation_stopwords = extract_affiliation_stopwords(py)?;

        #[derive(Clone)]
        struct SignatureInput {
            sig_id: String,
            paper_id: PaperId,
            raw_first: String,
            raw_middle: String,
            raw_last: String,
            email: Option<String>,
            position: i64,
            affiliation_values: Vec<String>,
            source_id_source: Option<String>,
            source_ids: Vec<String>,
        }

        #[derive(Clone)]
        struct PaperInput {
            paper_id: PaperId,
            raw_title: String,
            raw_venue: String,
            raw_journal: String,
            raw_authors: Vec<(i64, String)>,
            references: HashSet<PaperId>,
            year: Option<i64>,
            has_abstract: bool,
            predicted_language: Option<String>,
            is_reliable: bool,
        }

        let paper_preprocess_start = Instant::now();
        let mut needed_paper_ids: HashSet<PaperId> = HashSet::new();
        let mut signature_inputs: Vec<SignatureInput> = Vec::with_capacity(signatures_obj.len());
        let mut paper_inputs: Vec<PaperInput> = Vec::new();
        let mut unidecode_char_map: HashMap<char, String> = HashMap::new();
        let mut defaulted_signature_author_position_count = 0usize;
        for (sig_id, sig_value) in signatures_obj.iter() {
            let sig_dict = json_as_object(sig_value, "signature entry")?;
            let paper_id_value = json_get_required(sig_dict, "paper_id", "signature entry")?;
            let paper_id = json_value_to_id(paper_id_value).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("signature paper_id must be string/int")
            })?;
            needed_paper_ids.insert(paper_id.clone());

            let author_info_value = json_get_required(sig_dict, "author_info", "signature entry")?;
            let author_info = json_as_object(author_info_value, "signature author_info")?;
            let raw_first = json_get_string(author_info, "first", "");
            let raw_middle = json_get_string(author_info, "middle", "");
            let raw_last = json_get_string(author_info, "last", "");
            let email = json_get_optional_string(author_info, "email");
            let position =
                match json_get_i64_optional(author_info, "position", "signature author_info")? {
                    Some(value) => value,
                    None => {
                        defaulted_signature_author_position_count += 1;
                        0
                    }
                };
            let affiliation_values = json_get_string_list(author_info.get("affiliations"));
            let source_id_source = json_get_optional_string(author_info, "source_id_source");
            let source_ids = if source_id_source.as_deref() == Some("ORCID") {
                json_get_string_list(author_info.get("source_ids"))
            } else {
                Vec::new()
            };

            ensure_unidecode_for_text(&unidecode, &raw_first, &mut unidecode_char_map)?;
            ensure_unidecode_for_text(&unidecode, &raw_middle, &mut unidecode_char_map)?;
            ensure_unidecode_for_text(&unidecode, &raw_last, &mut unidecode_char_map)?;
            for affiliation in affiliation_values.iter() {
                ensure_unidecode_for_text(&unidecode, affiliation, &mut unidecode_char_map)?;
            }

            signature_inputs.push(SignatureInput {
                sig_id: sig_id.clone(),
                paper_id,
                raw_first,
                raw_middle,
                raw_last,
                email,
                position,
                affiliation_values,
                source_id_source,
                source_ids,
            });
        }
        drop(signatures_json);

        #[derive(Clone)]
        struct PaperPreprocessed {
            title: String,
            venue: String,
            journal_name: String,
            authors: Vec<(i64, String)>,
            references: HashSet<PaperId>,
            year: Option<i64>,
            has_abstract: bool,
            predicted_language: Option<String>,
            is_reliable: bool,
            title_words: Option<CounterData>,
            title_chars: Option<CounterData>,
            venue_ngrams: Option<CounterData>,
            journal_ngrams: Option<CounterData>,
        }

        let papers_json = load_json_value(papers_path)?;
        let papers_obj = json_as_object(&papers_json, "papers payload")?;
        let mut defaulted_paper_author_position_count = 0usize;
        for (_paper_key, paper_value) in papers_obj.iter() {
            let paper_dict = json_as_object(paper_value, "paper entry")?;
            let paper_id_value = json_get_required(paper_dict, "paper_id", "paper entry")?;
            let paper_id = json_value_to_id(paper_id_value).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("paper paper_id must be string/int")
            })?;
            if !needed_paper_ids.contains(&paper_id) {
                continue;
            }

            let raw_title = json_get_string(paper_dict, "title", "");
            let raw_venue = json_get_string(paper_dict, "venue", "");
            let raw_journal = json_get_string(paper_dict, "journal_name", "");

            ensure_unidecode_for_text(&unidecode, &raw_title, &mut unidecode_char_map)?;
            ensure_unidecode_for_text(&unidecode, &raw_venue, &mut unidecode_char_map)?;
            ensure_unidecode_for_text(&unidecode, &raw_journal, &mut unidecode_char_map)?;

            let mut raw_authors: Vec<(i64, String)> = Vec::new();
            if let Some(author_values) = paper_dict.get("authors").and_then(JsonValue::as_array) {
                for author_value in author_values {
                    let Some(author_dict) = author_value.as_object() else {
                        continue;
                    };
                    let position =
                        match json_get_i64_optional(author_dict, "position", "paper author entry")?
                        {
                            Some(value) => value,
                            None => {
                                defaulted_paper_author_position_count += 1;
                                0
                            }
                        };
                    let raw_author_name = json_get_string(author_dict, "author_name", "");
                    ensure_unidecode_for_text(
                        &unidecode,
                        &raw_author_name,
                        &mut unidecode_char_map,
                    )?;
                    raw_authors.push((position, raw_author_name));
                }
            }

            let references = json_get_id_set(paper_dict.get("references"));

            let year = match json_get_i64_optional(paper_dict, "year", "paper entry")? {
                Some(v) if v > 0 => Some(v),
                _ => None,
            };

            let has_abstract = match paper_dict.get("abstract") {
                None | Some(JsonValue::Null) => false,
                Some(JsonValue::String(s)) => !s.is_empty(),
                Some(_) => true,
            };

            let detector = language_detector.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("missing language detector")
            })?;
            let (is_reliable, _is_english, language) = detector.detect(&raw_title);
            let predicted_language = Some(language);

            paper_inputs.push(PaperInput {
                paper_id,
                raw_title,
                raw_venue,
                raw_journal,
                raw_authors,
                references,
                year,
                has_abstract,
                predicted_language,
                is_reliable,
            });
        }
        drop(needed_paper_ids);
        drop(papers_json);
        let json_parse_seconds = json_parse_start.elapsed().as_secs_f64();

        let computed_papers = py.allow_threads(|| {
            let compute = || {
                paper_inputs
                    .par_iter()
                    .map(|paper_input| {
                        let title = normalize_text_compat_from_map(
                            &paper_input.raw_title,
                            false,
                            &unidecode_char_map,
                        );
                        let venue = if preprocess {
                            normalize_text_compat_from_map(
                                &paper_input.raw_venue,
                                false,
                                &unidecode_char_map,
                            )
                        } else {
                            paper_input.raw_venue.clone()
                        };
                        let journal_name = if preprocess {
                            normalize_text_compat_from_map(
                                &paper_input.raw_journal,
                                false,
                                &unidecode_char_map,
                            )
                        } else {
                            paper_input.raw_journal.clone()
                        };
                        let authors = paper_input
                            .raw_authors
                            .iter()
                            .map(|(position, raw_name)| {
                                (
                                    *position,
                                    normalize_text_compat_from_map(
                                        raw_name,
                                        false,
                                        &unidecode_char_map,
                                    ),
                                )
                            })
                            .collect::<Vec<_>>();

                        let title_words = counter_data_from_usize_map(
                            word_ngrams_counter_python_compat(&title, &stop_words),
                        );

                        let title_chars = if preprocess {
                            counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                &title,
                                false,
                                true,
                                Some(&stop_words),
                            ))
                        } else {
                            None
                        };

                        let venue_ngrams = if preprocess {
                            counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                &venue,
                                false,
                                true,
                                Some(&venue_stop_words),
                            ))
                        } else {
                            None
                        };

                        let journal_ngrams = if preprocess {
                            counter_data_from_usize_map(char_ngrams_counter_python_compat(
                                &journal_name,
                                false,
                                true,
                                Some(&venue_stop_words),
                            ))
                        } else {
                            None
                        };

                        (
                            paper_input.paper_id.clone(),
                            PaperPreprocessed {
                                title,
                                venue,
                                journal_name,
                                authors,
                                references: paper_input.references.clone(),
                                year: paper_input.year,
                                has_abstract: paper_input.has_abstract,
                                predicted_language: paper_input.predicted_language.clone(),
                                is_reliable: paper_input.is_reliable,
                                title_words,
                                title_chars,
                                venue_ngrams,
                                journal_ngrams,
                            },
                        )
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        drop(paper_inputs);

        let mut preprocessed_papers: HashMap<PaperId, PaperPreprocessed> =
            HashMap::with_capacity(computed_papers.len());
        for (paper_id, preprocessed) in computed_papers {
            preprocessed_papers.insert(paper_id, preprocessed);
        }
        let paper_preprocess_seconds = paper_preprocess_start.elapsed().as_secs_f64();

        let signature_preprocess_start = Instant::now();
        let missing_paper_ids: Vec<String> = signature_inputs
            .iter()
            .filter(|entry| !preprocessed_papers.contains_key(&entry.paper_id))
            .map(|entry| entry.paper_id.to_string())
            .collect();
        if !missing_paper_ids.is_empty() {
            let examples = missing_paper_ids
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "signatures reference {} missing papers; examples: {}",
                missing_paper_ids.len(),
                examples
            )));
        }
        let computed_signatures = py.allow_threads(|| {
            let compute = || {
                signature_inputs
                    .par_iter()
                    .map(|entry| {
                        let middle_normalized = normalize_text_compat_from_map(
                            &entry.raw_middle,
                            false,
                            &unidecode_char_map,
                        );
                        let first_normalized = normalize_text_compat_from_map(
                            &entry.raw_first,
                            false,
                            &unidecode_char_map,
                        );
                        let first_normalized_token = first_normalized_token_python_compat(
                            &first_normalized,
                            &middle_normalized,
                            &name_prefixes,
                        );
                        let (first_without_apostrophe, middle_without_apostrophe) =
                            split_first_middle_hyphen_aware_compat(
                                &entry.raw_first,
                                &entry.raw_middle,
                                &name_prefixes,
                                &unidecode_char_map,
                            );
                        let last_normalized = normalize_text_compat_from_map(
                            &entry.raw_last,
                            false,
                            &unidecode_char_map,
                        );

                        let mut coauthor_list: Vec<String> = Vec::new();
                        if let Some(preprocessed_paper) = preprocessed_papers.get(&entry.paper_id) {
                            for (author_position, author_name) in preprocessed_paper.authors.iter()
                            {
                                if *author_position != entry.position {
                                    coauthor_list.push(author_name.clone());
                                }
                            }
                        }
                        let coauthors = if coauthor_list.is_empty() {
                            None
                        } else {
                            Some(coauthor_list.iter().cloned().collect::<HashSet<String>>())
                        };

                        let mut coauthor_blocks_set: HashSet<String> = HashSet::new();
                        for coauthor in coauthor_list.iter() {
                            coauthor_blocks_set.insert(compute_block_compat(coauthor));
                        }
                        let coauthor_blocks = if coauthor_blocks_set.is_empty() {
                            None
                        } else {
                            Some(coauthor_blocks_set)
                        };

                        let normalized_affiliations: Vec<String> = if preprocess {
                            entry
                                .affiliation_values
                                .iter()
                                .filter_map(|affiliation| {
                                    let normalized = normalize_text_compat_from_map(
                                        affiliation,
                                        false,
                                        &unidecode_char_map,
                                    );
                                    if normalized.is_empty() {
                                        None
                                    } else {
                                        Some(normalized)
                                    }
                                })
                                .collect()
                        } else {
                            entry.affiliation_values.clone()
                        };

                        let affiliation_text = if preprocess {
                            prefilter_affiliation_text(
                                &normalized_affiliations,
                                &affiliation_stopwords,
                            )
                        } else {
                            String::new()
                        };
                        let coauthor_text = if preprocess {
                            coauthor_list.join(" ")
                        } else {
                            String::new()
                        };

                        let affiliations = if preprocess && !affiliation_text.is_empty() {
                            counter_data_from_usize_map(word_ngrams_counter(&affiliation_text))
                        } else {
                            None
                        };
                        let coauthor_ngrams = if preprocess && !coauthor_text.is_empty() {
                            counter_data_from_usize_map(char_ngrams_counter(&coauthor_text))
                        } else {
                            None
                        };

                        let normalized_orcid = if entry.source_id_source.as_deref() == Some("ORCID")
                        {
                            entry
                                .source_ids
                                .first()
                                .and_then(|source_id| normalize_orcid_compact_owned(source_id))
                        } else {
                            None
                        };

                        let name_counts_result = build_name_counts_data_from_artifact(
                            &raw_name_counts,
                            &entry.raw_first,
                            &first_normalized_token,
                            &first_without_apostrophe,
                            &entry.raw_last,
                            &last_normalized,
                        );

                        (
                            entry.sig_id.clone(),
                            SignatureData {
                                first: Some(first_without_apostrophe.clone()),
                                middle: Some(middle_without_apostrophe),
                                last_normalized: Some(last_normalized),
                                orcid: normalized_orcid,
                                email: entry.email.clone(),
                                affiliations,
                                coauthor_blocks,
                                coauthor_ngrams,
                                coauthors,
                                position: entry.position,
                                paper_id: entry.paper_id.clone(),
                                name_counts: name_counts_result.data,
                                adv_name: Some(first_without_apostrophe),
                            },
                            name_counts_result.telemetry,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        let mut signatures: HashMap<String, SignatureData> =
            HashMap::with_capacity(computed_signatures.len());
        let mut defaulted_name_count_signature_count = 0usize;
        let mut defaulted_name_count_first_count = 0usize;
        let mut defaulted_name_count_first_last_count = 0usize;
        let mut defaulted_name_count_last_count = 0usize;
        let mut defaulted_name_count_last_first_initial_count = 0usize;
        for (sig_id, signature, name_count_telemetry) in computed_signatures {
            if name_count_telemetry.any() {
                defaulted_name_count_signature_count += 1;
            }
            if name_count_telemetry.first {
                defaulted_name_count_first_count += 1;
            }
            if name_count_telemetry.first_last {
                defaulted_name_count_first_last_count += 1;
            }
            if name_count_telemetry.last {
                defaulted_name_count_last_count += 1;
            }
            if name_count_telemetry.last_first_initial {
                defaulted_name_count_last_first_initial_count += 1;
            }
            signatures.insert(sig_id, signature);
        }
        drop(signature_inputs);
        drop(unidecode_char_map);
        let mut signature_ids: Vec<String> = signatures.keys().cloned().collect();
        signature_ids.sort_unstable();
        let signature_preprocess_seconds = signature_preprocess_start.elapsed().as_secs_f64();

        let specter_dict = match specter_embeddings {
            Some(obj) => {
                if let Ok(path) = obj.extract::<&str>() {
                    load_pickle_dict(py, path)?
                } else if let Ok(dict) = obj.downcast::<PyDict>() {
                    Some(dict.clone())
                } else if obj.is_none() {
                    None
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "specter_embeddings must be str, dict, or None; got {}",
                        obj.get_type().name()?
                    )));
                }
            }
            None => None,
        };

        let reference_counter_start = Instant::now();
        let mut missing_specter_paper_count = 0usize;
        let mut papers: HashMap<PaperId, PaperData> =
            HashMap::with_capacity(preprocessed_papers.len());
        if compute_reference_features {
            for (paper_id, paper) in preprocessed_papers.iter() {
                let mut ref_authors = None;
                let mut ref_titles = None;
                let mut ref_venues = None;
                let mut ref_blocks = None;
                let ref_details_present = true;

                let mut titles: Vec<String> = Vec::new();
                let mut venues: Vec<String> = Vec::new();
                let mut journals: Vec<String> = Vec::new();
                let mut authors: Vec<String> = Vec::new();
                let mut blocks: Vec<String> = Vec::new();

                for reference_id in paper.references.iter() {
                    if let Some(reference_paper) = preprocessed_papers.get(reference_id) {
                        if !reference_paper.title.is_empty() {
                            titles.push(reference_paper.title.clone());
                        }
                        if !reference_paper.venue.is_empty() {
                            venues.push(reference_paper.venue.clone());
                        }
                        if !reference_paper.journal_name.is_empty() {
                            journals.push(reference_paper.journal_name.clone());
                        }
                        for (_, author_name) in reference_paper.authors.iter() {
                            if author_name.is_empty() {
                                continue;
                            }
                            authors.push(author_name.clone());
                            let block = compute_block_compat(author_name);
                            blocks.push(block);
                        }
                    }
                }

                let author_names = authors.join(" ");
                let reference_titles = titles.join(" ");
                let venues_joined = venues.join(" ");
                let journals_joined = journals.join(" ");
                let reference_venues = if venues_joined == journals_joined {
                    venues_joined
                } else {
                    format!("{} {}", venues_joined, journals_joined)
                        .trim()
                        .to_string()
                };

                if !author_names.is_empty() {
                    ref_authors = counter_data_from_usize_map(char_ngrams_counter_python_compat(
                        &author_names,
                        false,
                        true,
                        None,
                    ));
                }

                if !reference_titles.is_empty() {
                    ref_titles = counter_data_from_usize_map(char_ngrams_counter_python_compat(
                        &reference_titles,
                        false,
                        true,
                        Some(&stop_words),
                    ));
                }

                if !reference_venues.is_empty() {
                    ref_venues = counter_data_from_usize_map(char_ngrams_counter_python_compat(
                        &reference_venues,
                        false,
                        true,
                        Some(&venue_stop_words),
                    ));
                }

                if !blocks.is_empty() {
                    let mut block_counter: HashMap<String, usize> = HashMap::new();
                    for block in blocks {
                        *block_counter.entry(block).or_insert(0) += 1;
                    }
                    ref_blocks = counter_data_from_usize_map(block_counter);
                }
                let specter = if let Some(spec_dict) = &specter_dict {
                    extract_specter_for_paper_id(spec_dict, paper_id)?
                } else {
                    None
                };
                if specter.is_none() {
                    missing_specter_paper_count += 1;
                }
                let specter_norm = specter.as_ref().map(|values| {
                    values
                        .iter()
                        .map(|value| {
                            let value_f64 = *value as f64;
                            value_f64 * value_f64
                        })
                        .sum::<f64>()
                        .sqrt()
                });

                papers.insert(
                    paper_id.clone(),
                    PaperData {
                        venue_ngrams: paper.venue_ngrams.clone(),
                        title_words: paper.title_words.clone(),
                        title_chars: paper.title_chars.clone(),
                        ref_authors,
                        ref_titles,
                        ref_venues,
                        ref_blocks,
                        ref_details_present,
                        references: paper.references.clone(),
                        year: paper.year,
                        has_abstract: paper.has_abstract,
                        predicted_language: paper.predicted_language.clone(),
                        is_reliable: paper.is_reliable,
                        journal_ngrams: paper.journal_ngrams.clone(),
                        specter,
                        specter_norm,
                    },
                );
            }
        } else {
            for (paper_id, paper) in preprocessed_papers.into_iter() {
                let specter = if let Some(spec_dict) = &specter_dict {
                    extract_specter_for_paper_id(spec_dict, &paper_id)?
                } else {
                    None
                };
                if specter.is_none() {
                    missing_specter_paper_count += 1;
                }
                let specter_norm = specter.as_ref().map(|values| {
                    values
                        .iter()
                        .map(|value| {
                            let value_f64 = *value as f64;
                            value_f64 * value_f64
                        })
                        .sum::<f64>()
                        .sqrt()
                });
                papers.insert(
                    paper_id,
                    PaperData {
                        venue_ngrams: paper.venue_ngrams,
                        title_words: paper.title_words,
                        title_chars: paper.title_chars,
                        ref_authors: None,
                        ref_titles: None,
                        ref_venues: None,
                        ref_blocks: None,
                        ref_details_present: false,
                        references: paper.references,
                        year: paper.year,
                        has_abstract: paper.has_abstract,
                        predicted_language: paper.predicted_language,
                        is_reliable: paper.is_reliable,
                        journal_ngrams: paper.journal_ngrams,
                        specter,
                        specter_norm,
                    },
                );
            }
        }
        let reference_counter_seconds = reference_counter_start.elapsed().as_secs_f64();

        let cluster_seed_start = Instant::now();
        let mut cluster_seeds_disallow: HashSet<(String, String)> = HashSet::new();
        let mut cluster_seeds_require: HashMap<String, ClusterId> = HashMap::new();
        if let Some(path) = cluster_seeds_path {
            let cluster_seeds_json = load_json_value(path)?;
            let cluster_seeds_obj = json_as_object(&cluster_seeds_json, "cluster_seeds payload")?;
            let mut cluster_num = 0_i64;
            for (signature_id_a, values_value) in cluster_seeds_obj.iter() {
                let values_obj = json_as_object(values_value, "cluster seed entry")?;
                let mut root_added = false;
                for (signature_id_b, constraint_value) in values_obj.iter() {
                    let constraint = match constraint_value {
                        JsonValue::String(value) => value.as_str(),
                        _ => {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "cluster seed constraint for ({signature_id_a:?}, {signature_id_b:?}) must be a string"
                            )));
                        }
                    };
                    match constraint {
                        "disallow" => {
                            cluster_seeds_disallow.insert(canonical_signature_pair_cloned(
                                signature_id_a,
                                signature_id_b,
                            ));
                        }
                        "require" => {
                            if !root_added {
                                cluster_seeds_require
                                    .insert(signature_id_a.clone(), ClusterId::Int(cluster_num));
                                root_added = true;
                            }
                            cluster_seeds_require
                                .insert(signature_id_b.clone(), ClusterId::Int(cluster_num));
                        }
                        _ => {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "unknown cluster seed constraint {constraint:?} for ({signature_id_a:?}, {signature_id_b:?}); expected 'require' or 'disallow'"
                            )));
                        }
                    }
                }
                cluster_num += 1;
            }
        }
        let cluster_seed_seconds = cluster_seed_start.elapsed().as_secs_f64();

        let json_ingest_telemetry = JsonIngestTelemetry {
            json_parse_seconds,
            paper_preprocess_seconds,
            reference_counter_seconds,
            signature_preprocess_seconds,
            cluster_seed_seconds,
            missing_specter_paper_count,
            defaulted_name_count_signature_count,
            defaulted_name_count_first_count,
            defaulted_name_count_first_last_count,
            defaulted_name_count_last_count,
            defaulted_name_count_last_first_initial_count,
            defaulted_signature_author_position_count,
            defaulted_paper_author_position_count,
        };

        Ok(RustFeaturizer {
            signatures,
            signature_ids,
            papers,
            name_tuples,
            cluster_seeds_disallow,
            cluster_seeds_require,
            compute_reference_features,
            cluster_seed_require_value,
            cluster_seed_disallow_value,
            json_ingest_telemetry: Some(json_ingest_telemetry),
            cached_signature_id_order: OnceLock::new(),
            cluster_seeds_disallow_index: OnceLock::new(),
        })
    }

    fn update_cluster_seeds(
        &mut self,
        cluster_seeds_require: &Bound<'_, PyAny>,
        cluster_seeds_disallow: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.cluster_seeds_require = extract_cluster_seeds_require(cluster_seeds_require)?;
        self.cluster_seeds_disallow = extract_pair_set(cluster_seeds_disallow)?;
        self.cluster_seeds_disallow_index = OnceLock::new();
        Ok(())
    }

    #[pyo3(
        signature = (
            sig_id1,
            sig_id2,
            low_value = 0.0,
            high_value = 10000.0,
            dont_merge_cluster_seeds = true,
            incremental_dont_use_cluster_seeds = false,
            suppress_orcid = false
        )
    )]
    fn get_constraint(
        &self,
        sig_id1: &str,
        sig_id2: &str,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        suppress_orcid: bool,
    ) -> PyResult<Option<f64>> {
        self.get_constraint_value_for_pair(
            sig_id1,
            sig_id2,
            low_value,
            high_value,
            dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds,
            suppress_orcid,
        )
    }

    #[pyo3(
        signature = (
            pairs,
            low_value = 0.0,
            high_value = 10000.0,
            dont_merge_cluster_seeds = true,
            incremental_dont_use_cluster_seeds = false,
            num_threads = None,
            suppress_orcid = false
        )
    )]
    fn get_constraints_matrix_indexed(
        &self,
        py: Python<'_>,
        pairs: Vec<(u32, u32)>,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: Option<usize>,
        suppress_orcid: bool,
    ) -> PyResult<Vec<Option<f64>>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        let signature_ids = self.signature_id_order();
        let signature_count = signature_ids.len();
        for (left_idx, right_idx) in pairs.iter() {
            let left = *left_idx as usize;
            let right = *right_idx as usize;
            if left >= signature_count || right >= signature_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "pair index out of range: left={} right={} signature_count={}",
                    left, right, signature_count
                )));
            }
        }

        let mut lookup: Vec<(&String, &SignatureData)> = Vec::with_capacity(signature_count);
        for signature_id in signature_ids.iter() {
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            lookup.push((signature_id, signature));
        }

        let values = py.allow_threads(|| {
            let compute = || {
                pairs
                    .par_iter()
                    .map(|(left_idx, right_idx)| {
                        let (left_id, s1) = lookup[*left_idx as usize];
                        let (right_id, s2) = lookup[*right_idx as usize];
                        self.constraint_value_from_records(
                            left_id,
                            right_id,
                            s1,
                            s2,
                            low_value,
                            high_value,
                            dont_merge_cluster_seeds,
                            incremental_dont_use_cluster_seeds,
                            suppress_orcid,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        Ok(values)
    }

    #[pyo3(
        signature = (
            left_signature_indices,
            right_signature_indices,
            low_value = 0.0,
            high_value = 10000.0,
            dont_merge_cluster_seeds = true,
            incremental_dont_use_cluster_seeds = false,
            num_threads = None,
            suppress_orcid = false,
            large_integer = 100000.0
        )
    )]
    fn linker_pair_index_arrays_constraint_labels<'py>(
        &self,
        py: Python<'py>,
        left_signature_indices: PyReadonlyArray1<'py, u32>,
        right_signature_indices: PyReadonlyArray1<'py, u32>,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: Option<usize>,
        suppress_orcid: bool,
        large_integer: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let left_indices = left_signature_indices.as_slice()?;
        let right_indices = right_signature_indices.as_slice()?;
        let pair_count = left_indices.len();
        if right_indices.len() != pair_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "left_signature_indices and right_signature_indices must have equal length: left={} right={}",
                left_indices.len(),
                right_indices.len()
            )));
        }

        let signature_ids = self.signature_id_order();
        for (left_idx, right_idx) in left_indices.iter().zip(right_indices.iter()) {
            let left = *left_idx as usize;
            let right = *right_idx as usize;
            if left >= signature_ids.len() || right >= signature_ids.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "pair index out of range: left={} right={} signature_count={}",
                    left,
                    right,
                    signature_ids.len()
                )));
            }
        }

        let lookup = self.signature_lookup()?;
        let labels = py.allow_threads(|| {
            let compute = || {
                left_indices
                    .par_iter()
                    .zip(right_indices.par_iter())
                    .map(|(left_idx, right_idx)| {
                        let left = *left_idx as usize;
                        let right = *right_idx as usize;
                        let sig_id1 = signature_ids[left].as_str();
                        let sig_id2 = signature_ids[right].as_str();
                        let s1 = lookup[left];
                        let s2 = lookup[right];
                        match self.constraint_value_from_records(
                            sig_id1,
                            sig_id2,
                            s1,
                            s2,
                            low_value,
                            high_value,
                            dont_merge_cluster_seeds,
                            incremental_dont_use_cluster_seeds,
                            suppress_orcid,
                        ) {
                            Some(value) => value - large_integer,
                            None => f64::NAN,
                        }
                    })
                    .collect::<Vec<f64>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        Ok(numpy::ndarray::Array1::from_vec(labels).to_pyarray(py))
    }

    #[pyo3(
        signature = (
            row_indices,
            row_count,
            pair_distances,
            pair_labels = None,
            num_threads = None,
            large_integer = 100000.0,
            hard_disallow_distance = 10000.0
        )
    )]
    fn linker_pair_distance_accumulators<'py>(
        &self,
        py: Python<'py>,
        row_indices: PyReadonlyArray1<'py, u32>,
        row_count: usize,
        pair_distances: PyReadonlyArray1<'py, f64>,
        pair_labels: Option<PyReadonlyArray1<'py, f64>>,
        num_threads: Option<usize>,
        large_integer: f64,
        hard_disallow_distance: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
        u64,
    )> {
        let owner_row_indices = row_indices.as_slice()?;
        let model_distances = pair_distances.as_slice()?;
        let pair_count = owner_row_indices.len();
        if model_distances.len() != pair_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "row_indices and pair_distances must have equal length: rows={} distances={}",
                owner_row_indices.len(),
                model_distances.len()
            )));
        }
        let labels = match pair_labels.as_ref() {
            Some(values) => {
                let slice = values.as_slice()?;
                if slice.len() != pair_count {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "pair_labels length must match row_indices length: labels={} rows={}",
                        slice.len(),
                        pair_count
                    )));
                }
                Some(slice)
            }
            None => None,
        };
        for row_index in owner_row_indices.iter() {
            let bounded = *row_index as usize;
            if bounded >= row_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "row index out of range: row_index={} row_count={}",
                    bounded, row_count
                )));
            }
        }

        let accumulate_range =
            |start: usize, end: usize| -> Result<LinkerPairDistanceAccumulator, String> {
                let mut accumulator = LinkerPairDistanceAccumulator::new(row_count);
                for pair_offset in start..end {
                    let label = labels.map(|values| values[pair_offset]).unwrap_or(f64::NAN);
                    let value = if label.is_nan() {
                        model_distances[pair_offset]
                    } else {
                        label + large_integer
                    };
                    if value.is_nan() {
                        return Err("pairwise model returned NaN distance".to_string());
                    }
                    let row = owner_row_indices[pair_offset] as usize;
                    accumulator.counts[row] = accumulator.counts[row].saturating_add(1);
                    accumulator.sums[row] += value;
                    if value < accumulator.mins[row] {
                        accumulator.mins[row] = value;
                    }
                    if value >= hard_disallow_distance {
                        accumulator.hard_disallow_pair_count =
                            accumulator.hard_disallow_pair_count.saturating_add(1);
                    }
                    let top_start = row * 5;
                    Self::update_top5_distance(
                        &mut accumulator.top_distances[top_start..top_start + 5],
                        value,
                    );
                }
                Ok(accumulator)
            };

        let accumulator = if num_threads.is_some_and(|threads| threads > 1) && pair_count > 1 {
            py.allow_threads(|| {
                let compute = || {
                    let requested_threads = num_threads.unwrap_or(1).max(1);
                    let shard_count = requested_threads.min(pair_count);
                    let chunk_size = pair_count.div_ceil(shard_count);
                    let partials = (0..pair_count)
                        .step_by(chunk_size)
                        .collect::<Vec<_>>()
                        .into_par_iter()
                        .map(|start| accumulate_range(start, (start + chunk_size).min(pair_count)))
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut merged = LinkerPairDistanceAccumulator::new(row_count);
                    for partial in partials {
                        merged.merge_from(partial);
                    }
                    Ok::<LinkerPairDistanceAccumulator, String>(merged)
                };
                install_with_optional_rayon_pool(num_threads, compute)
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?
        } else {
            accumulate_range(0, pair_count).map_err(pyo3::exceptions::PyValueError::new_err)?
        };

        let top_array =
            numpy::ndarray::Array2::from_shape_vec((row_count, 5), accumulator.top_distances)
                .map_err(|err| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to build top-distance matrix: {}",
                        err
                    ))
                })?;
        Ok((
            numpy::ndarray::Array1::from_vec(accumulator.counts).to_pyarray(py),
            numpy::ndarray::Array1::from_vec(accumulator.sums).to_pyarray(py),
            numpy::ndarray::Array1::from_vec(accumulator.mins).to_pyarray(py),
            top_array.to_pyarray(py),
            accumulator.hard_disallow_pair_count,
        ))
    }

    #[pyo3(
        signature = (
            block_signature_indices,
            start_offset = 0,
            max_pairs = None,
            low_value = 0.0,
            high_value = 10000.0,
            dont_merge_cluster_seeds = true,
            incremental_dont_use_cluster_seeds = false,
            num_threads = None,
            suppress_orcid = false
        )
    )]
    fn get_constraints_block_upper_triangle_indexed(
        &self,
        py: Python<'_>,
        block_signature_indices: Vec<u32>,
        start_offset: usize,
        max_pairs: Option<usize>,
        low_value: f64,
        high_value: f64,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: Option<usize>,
        suppress_orcid: bool,
    ) -> PyResult<(Vec<u32>, Vec<u32>, Vec<Option<f64>>)> {
        if block_signature_indices.len() <= 1 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let signature_ids = self.signature_id_order();
        let signature_count = signature_ids.len();
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            if global_idx >= signature_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "block signature index out of range: index={} signature_count={}",
                    global_idx, signature_count
                )));
            }
        }

        let mut block_lookup: Vec<(&String, &SignatureData)> =
            Vec::with_capacity(block_signature_indices.len());
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            let signature_id = &signature_ids[global_idx];
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            block_lookup.push((signature_id, signature));
        }

        let local_pairs =
            upper_triangle_pairs_for_range(block_lookup.len(), start_offset, max_pairs)?;
        if local_pairs.is_empty() {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let left_indices: Vec<u32> = local_pairs.iter().map(|(left, _)| *left as u32).collect();
        let right_indices: Vec<u32> = local_pairs.iter().map(|(_, right)| *right as u32).collect();
        let values = py.allow_threads(|| {
            let compute = || {
                local_pairs
                    .par_iter()
                    .map(|(left_idx, right_idx)| {
                        let (left_id, s1) = block_lookup[*left_idx];
                        let (right_id, s2) = block_lookup[*right_idx];
                        self.constraint_value_from_records(
                            left_id,
                            right_id,
                            s1,
                            s2,
                            low_value,
                            high_value,
                            dont_merge_cluster_seeds,
                            incremental_dont_use_cluster_seeds,
                            suppress_orcid,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        Ok((left_indices, right_indices, values))
    }

    fn featurize_pair(&self, sig_id1: &str, sig_id2: &str) -> PyResult<Vec<f64>> {
        let s1 = self
            .signatures
            .get(sig_id1)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id1.to_string()))?;
        let s2 = self
            .signatures
            .get(sig_id2)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id2.to_string()))?;
        let p1 = self
            .papers
            .get(&s1.paper_id)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(s1.paper_id.to_string()))?;
        let p2 = self
            .papers
            .get(&s2.paper_id)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(s2.paper_id.to_string()))?;
        Ok(self.featurize_pair_data(s1, s2, p1, p2).to_vec())
    }

    #[pyo3(signature = (pairs, num_threads = None))]
    fn featurize_pairs(
        &self,
        py: Python<'_>,
        pairs: Vec<(String, String)>,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<Vec<f64>>> {
        for (sig_id1, sig_id2) in pairs.iter() {
            let s1 = self
                .signatures
                .get(sig_id1)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id1.to_string()))?;
            let s2 = self
                .signatures
                .get(sig_id2)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id2.to_string()))?;
            if self.papers.get(&s1.paper_id).is_none() {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    s1.paper_id.to_string(),
                ));
            }
            if self.papers.get(&s2.paper_id).is_none() {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    s2.paper_id.to_string(),
                ));
            }
        }
        let feats = py.allow_threads(|| {
            let compute = || {
                pairs
                    .par_iter()
                    .map(|(sig_id1, sig_id2)| -> PyResult<Vec<f64>> {
                        let s1 = self.signatures.get(sig_id1).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(sig_id1.to_string())
                        })?;
                        let s2 = self.signatures.get(sig_id2).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(sig_id2.to_string())
                        })?;
                        let p1 = self.papers.get(&s1.paper_id).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(s1.paper_id.to_string())
                        })?;
                        let p2 = self.papers.get(&s2.paper_id).ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(s2.paper_id.to_string())
                        })?;
                        Ok(self.featurize_pair_data(s1, s2, p1, p2).to_vec())
                    })
                    .collect::<PyResult<Vec<_>>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        feats
    }

    fn signature_ids(&self) -> Vec<String> {
        self.signature_id_order().to_vec()
    }

    fn signature_rule_metadata(&self) -> Vec<(String, Option<String>, Option<String>)> {
        self.signature_id_order()
            .iter()
            .filter_map(|signature_id| {
                self.signatures.get(signature_id).map(|signature| {
                    (
                        signature_id.clone(),
                        signature.first_without_apostrophe().map(str::to_owned),
                        signature.orcid.clone(),
                    )
                })
            })
            .collect()
    }

    fn signature_name_counts_present(&self) -> Vec<(String, bool)> {
        self.signature_id_order()
            .iter()
            .filter_map(|signature_id| {
                self.signatures
                    .get(signature_id)
                    .map(|signature| (signature_id.clone(), signature.name_counts.is_some()))
            })
            .collect()
    }

    fn cluster_seeds_require(&self) -> Vec<(String, String)> {
        let mut pairs: Vec<(String, String)> = self
            .cluster_seeds_require
            .iter()
            .map(|(signature_id, cluster_id)| {
                (signature_id.clone(), cluster_id_to_string(cluster_id))
            })
            .collect();
        pairs.sort_by(|left, right| left.0.cmp(&right.0));
        pairs
    }

    fn update_signature_name_counts(&mut self, signatures: &Bound<'_, PyAny>) -> PyResult<usize> {
        let signatures_dict = signatures.downcast::<PyDict>()?;
        let mut updated = 0usize;
        for (sig_id_obj, sig_obj) in signatures_dict.iter() {
            let sig_id: String = sig_id_obj.extract()?;
            let Some(signature) = self.signatures.get_mut(&sig_id) else {
                continue;
            };
            let counts_obj = sig_obj.getattr("author_info_name_counts")?;
            let counts = extract_name_counts_data(&counts_obj)?;
            if counts.is_some() {
                signature.name_counts = counts;
                updated += 1;
            }
        }
        Ok(updated)
    }

    #[pyo3(signature = (pairs, selected_indices = None, num_threads = None, nan_value = f64::NAN))]
    fn featurize_pairs_matrix<'py>(
        &self,
        py: Python<'py>,
        pairs: Vec<(String, String)>,
        selected_indices: Option<Vec<usize>>,
        num_threads: Option<usize>,
        nan_value: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let row_count = pairs.len();
        if row_count == 0 {
            let empty = numpy::ndarray::Array2::<f64>::zeros((0, 0));
            return Ok(empty.to_pyarray(py));
        }

        for (sig_id1, sig_id2) in pairs.iter() {
            let s1 = self
                .signatures
                .get(sig_id1)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id1.to_string()))?;
            let s2 = self
                .signatures
                .get(sig_id2)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id2.to_string()))?;
            if self.papers.get(&s1.paper_id).is_none() {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    s1.paper_id.to_string(),
                ));
            }
            if self.papers.get(&s2.paper_id).is_none() {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    s2.paper_id.to_string(),
                ));
            }
        }

        let mut id_to_lookup_idx: HashMap<&str, usize> =
            HashMap::with_capacity(row_count.saturating_mul(2));
        let mut lookup: Vec<(&SignatureData, &PaperData)> = Vec::new();
        for (sig_id1, sig_id2) in pairs.iter() {
            for sig_id in [sig_id1.as_str(), sig_id2.as_str()] {
                if id_to_lookup_idx.contains_key(sig_id) {
                    continue;
                }
                let signature = self
                    .signatures
                    .get(sig_id)
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id.to_string()))?;
                let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
                })?;
                let lookup_idx = lookup.len();
                lookup.push((signature, paper));
                id_to_lookup_idx.insert(sig_id, lookup_idx);
            }
        }

        let mut indexed_pairs: Vec<(usize, usize)> = Vec::with_capacity(row_count);
        for (sig_id1, sig_id2) in pairs.iter() {
            let left_idx = *id_to_lookup_idx
                .get(sig_id1.as_str())
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id1.clone()))?;
            let right_idx = *id_to_lookup_idx
                .get(sig_id2.as_str())
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(sig_id2.clone()))?;
            indexed_pairs.push((left_idx, right_idx));
        }

        let full_cols = self.full_feature_count();
        let indices = resolve_feature_indices("selected_indices", selected_indices, full_cols)?;
        let out_cols = indices.len();
        if out_cols == 0 {
            let empty_cols = numpy::ndarray::Array2::<f64>::zeros((row_count, 0));
            return Ok(empty_cols.to_pyarray(py));
        }
        let out = py.allow_threads(|| {
            let compute = || {
                let mut buffer = vec![0.0_f64; row_count * out_cols];
                buffer
                    .par_chunks_mut(out_cols)
                    .zip(indexed_pairs.par_iter())
                    .for_each(|(out_row, (left_idx, right_idx))| {
                        let (s1, p1) = lookup[*left_idx];
                        let (s2, p2) = lookup[*right_idx];
                        let row = self.featurize_pair_data(s1, s2, p1, p2);
                        for (dest, idx) in out_row.iter_mut().zip(indices.iter()) {
                            let mut value = row[*idx];
                            if value.is_nan() && !nan_value.is_nan() {
                                value = nan_value;
                            }
                            *dest = value;
                        }
                    });
                buffer
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let array =
            numpy::ndarray::Array2::from_shape_vec((row_count, out_cols), out).map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build output matrix: {}",
                    err
                ))
            })?;
        Ok(array.to_pyarray(py))
    }

    #[pyo3(signature = (pairs, selected_indices = None, num_threads = None, nan_value = f64::NAN))]
    fn featurize_pairs_matrix_indexed<'py>(
        &self,
        py: Python<'py>,
        pairs: Vec<(u32, u32)>,
        selected_indices: Option<Vec<usize>>,
        num_threads: Option<usize>,
        nan_value: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let row_count = pairs.len();
        if row_count == 0 {
            let empty = numpy::ndarray::Array2::<f64>::zeros((0, 0));
            return Ok(empty.to_pyarray(py));
        }

        let lookup = self.sparse_signature_paper_lookup_for_pair_tuples(&pairs)?;

        let full_cols = self.full_feature_count();
        let indices = resolve_feature_indices("selected_indices", selected_indices, full_cols)?;
        let out_cols = indices.len();
        if out_cols == 0 {
            let empty_cols = numpy::ndarray::Array2::<f64>::zeros((row_count, 0));
            return Ok(empty_cols.to_pyarray(py));
        }

        let out = py.allow_threads(|| {
            let compute = || {
                let mut buffer = vec![0.0_f64; row_count * out_cols];
                buffer
                    .par_chunks_mut(out_cols)
                    .zip(pairs.par_iter())
                    .for_each(|(out_row, (left_idx, right_idx))| {
                        let (s1, p1) = lookup[*left_idx as usize]
                            .expect("left signature index was validated before featurization");
                        let (s2, p2) = lookup[*right_idx as usize]
                            .expect("right signature index was validated before featurization");
                        let row = self.featurize_pair_data(s1, s2, p1, p2);
                        for (dest, idx) in out_row.iter_mut().zip(indices.iter()) {
                            let mut value = row[*idx];
                            if value.is_nan() && !nan_value.is_nan() {
                                value = nan_value;
                            }
                            *dest = value;
                        }
                    });
                buffer
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let array =
            numpy::ndarray::Array2::from_shape_vec((row_count, out_cols), out).map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build output matrix: {}",
                    err
                ))
            })?;
        Ok(array.to_pyarray(py))
    }

    #[pyo3(
        signature = (
            left_signature_indices,
            right_signature_indices,
            row_indices,
            row_count,
            matrix_indices = None,
            aggregate_indices = None,
            num_threads = None,
            nan_value = f64::NAN,
            aggregate_nan_value = None,
            emit_matrix = true
        )
    )]
    fn linker_pair_index_arrays_and_aggregate_stats<'py>(
        &self,
        py: Python<'py>,
        left_signature_indices: PyReadonlyArray1<'py, u32>,
        right_signature_indices: PyReadonlyArray1<'py, u32>,
        row_indices: PyReadonlyArray1<'py, u32>,
        row_count: usize,
        matrix_indices: Option<Vec<usize>>,
        aggregate_indices: Option<Vec<usize>>,
        num_threads: Option<usize>,
        nan_value: f64,
        aggregate_nan_value: Option<f64>,
        emit_matrix: bool,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray2<u64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        let left_indices = left_signature_indices.as_slice()?;
        let right_indices = right_signature_indices.as_slice()?;
        let owner_row_indices = row_indices.as_slice()?;
        let pair_count = left_indices.len();
        if right_indices.len() != pair_count || owner_row_indices.len() != pair_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "left_signature_indices, right_signature_indices, and row_indices must have equal length: left={} right={} rows={}",
                left_indices.len(),
                right_indices.len(),
                owner_row_indices.len()
            )));
        }

        let lookup = self.sparse_signature_paper_lookup_for_indices(left_indices, right_indices)?;
        for row_index in owner_row_indices.iter() {
            let bounded = *row_index as usize;
            if bounded >= row_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "row index out of range: row_index={} row_count={}",
                    bounded, row_count
                )));
            }
        }

        let full_cols = self.full_feature_count();
        if !emit_matrix {
            let resolved_aggregate_indices =
                resolve_feature_indices("aggregate_indices", aggregate_indices, full_cols)?;
            let row_ranges = Self::pair_aggregate_row_ranges(owner_row_indices);
            let aggregate_cols = resolved_aggregate_indices.len();
            let aggregate_buffers = py.allow_threads(|| {
                let compute = || match row_ranges.as_ref() {
                    Some(ranges) => self.aggregate_pair_index_arrays_grouped(
                        left_indices,
                        right_indices,
                        ranges,
                        row_count,
                        &resolved_aggregate_indices,
                        nan_value,
                        &lookup,
                    ),
                    None => self.aggregate_pair_index_arrays_sequential(
                        left_indices,
                        right_indices,
                        owner_row_indices,
                        row_count,
                        &resolved_aggregate_indices,
                        nan_value,
                        &lookup,
                    ),
                };
                install_with_optional_rayon_pool(num_threads, compute)
            });
            let matrix_array = numpy::ndarray::Array2::<f64>::zeros((pair_count, 0));
            let valid_counts_array = numpy::ndarray::Array2::from_shape_vec(
                (row_count, aggregate_cols),
                aggregate_buffers.valid_counts,
            )
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate valid counts matrix: {}",
                    err
                ))
            })?;
            let sums_array = numpy::ndarray::Array2::from_shape_vec(
                (row_count, aggregate_cols),
                aggregate_buffers.sums,
            )
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate sums matrix: {}",
                    err
                ))
            })?;
            let mins_array = numpy::ndarray::Array2::from_shape_vec(
                (row_count, aggregate_cols),
                aggregate_buffers.mins,
            )
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate mins matrix: {}",
                    err
                ))
            })?;
            let maxs_array = numpy::ndarray::Array2::from_shape_vec(
                (row_count, aggregate_cols),
                aggregate_buffers.maxs,
            )
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate maxs matrix: {}",
                    err
                ))
            })?;
            return Ok((
                matrix_array.to_pyarray(py),
                numpy::ndarray::Array1::from_vec(aggregate_buffers.counts).to_pyarray(py),
                valid_counts_array.to_pyarray(py),
                sums_array.to_pyarray(py),
                mins_array.to_pyarray(py),
                maxs_array.to_pyarray(py),
            ));
        }

        let index_selection =
            resolve_matrix_aggregate_indices(matrix_indices, aggregate_indices, full_cols)?;
        let resolved_matrix_indices = index_selection.matrix_indices;
        let resolved_aggregate_indices = index_selection.aggregate_indices;
        let aggregate_matrix_positions = index_selection.aggregate_matrix_positions;

        let out_cols = resolved_matrix_indices.len();
        let aggregate_cols = resolved_aggregate_indices.len();
        let resolved_aggregate_nan_value = aggregate_nan_value.unwrap_or(nan_value);
        let matrix_buffer = py.allow_threads(|| {
            let compute = || {
                let mut buffer = vec![0.0_f64; pair_count * out_cols];
                if out_cols == 0 {
                    return buffer;
                }
                buffer
                    .par_chunks_mut(out_cols)
                    .zip(left_indices.par_iter().zip(right_indices.par_iter()))
                    .for_each(|(out_row, (left_idx, right_idx))| {
                        let (s1, p1) = lookup[*left_idx as usize]
                            .expect("left signature index was validated before featurization");
                        let (s2, p2) = lookup[*right_idx as usize]
                            .expect("right signature index was validated before featurization");
                        let row = self.featurize_pair_data(s1, s2, p1, p2);
                        for (dest, idx) in out_row.iter_mut().zip(resolved_matrix_indices.iter()) {
                            let mut value = row[*idx];
                            if value.is_nan() && !nan_value.is_nan() {
                                value = nan_value;
                            }
                            *dest = value;
                        }
                    });
                buffer
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let mut counts = vec![0_u32; row_count];
        let mut valid_counts = vec![0_u64; row_count * aggregate_cols];
        let mut sums = vec![0.0_f64; row_count * aggregate_cols];
        let mut mins = vec![f64::INFINITY; row_count * aggregate_cols];
        let mut maxs = vec![f64::NEG_INFINITY; row_count * aggregate_cols];
        if aggregate_cols > 0 {
            for (pair_offset, row_index) in owner_row_indices.iter().enumerate() {
                let row_offset = *row_index as usize;
                counts[row_offset] = counts[row_offset].saturating_add(1);
                let matrix_row_start = pair_offset * out_cols;
                let aggregate_row_start = row_offset * aggregate_cols;
                for (aggregate_position, matrix_position) in
                    aggregate_matrix_positions.iter().enumerate()
                {
                    let mut value = matrix_buffer[matrix_row_start + *matrix_position];
                    if value.is_nan() {
                        if resolved_aggregate_nan_value.is_nan() {
                            continue;
                        }
                        value = resolved_aggregate_nan_value;
                    }
                    let stats_index = aggregate_row_start + aggregate_position;
                    valid_counts[stats_index] = valid_counts[stats_index].saturating_add(1);
                    sums[stats_index] += value;
                    if value < mins[stats_index] {
                        mins[stats_index] = value;
                    }
                    if value > maxs[stats_index] {
                        maxs[stats_index] = value;
                    }
                }
            }
        } else {
            for row_index in owner_row_indices.iter() {
                counts[*row_index as usize] = counts[*row_index as usize].saturating_add(1);
            }
        }

        let matrix_array =
            numpy::ndarray::Array2::from_shape_vec((pair_count, out_cols), matrix_buffer).map_err(
                |err| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to build pair feature matrix: {}",
                        err
                    ))
                },
            )?;
        let valid_counts_array =
            numpy::ndarray::Array2::from_shape_vec((row_count, aggregate_cols), valid_counts)
                .map_err(|err| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to build aggregate valid counts matrix: {}",
                        err
                    ))
                })?;
        let sums_array = numpy::ndarray::Array2::from_shape_vec((row_count, aggregate_cols), sums)
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate sums matrix: {}",
                    err
                ))
            })?;
        let mins_array = numpy::ndarray::Array2::from_shape_vec((row_count, aggregate_cols), mins)
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate mins matrix: {}",
                    err
                ))
            })?;
        let maxs_array = numpy::ndarray::Array2::from_shape_vec((row_count, aggregate_cols), maxs)
            .map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build aggregate maxs matrix: {}",
                    err
                ))
            })?;
        Ok((
            matrix_array.to_pyarray(py),
            numpy::ndarray::Array1::from_vec(counts).to_pyarray(py),
            valid_counts_array.to_pyarray(py),
            sums_array.to_pyarray(py),
            mins_array.to_pyarray(py),
            maxs_array.to_pyarray(py),
        ))
    }

    #[pyo3(
        signature = (
            block_signature_indices,
            start_offset = 0,
            max_pairs = None,
            selected_indices = None,
            num_threads = None,
            nan_value = f64::NAN
        )
    )]
    fn featurize_block_upper_triangle_matrix_indexed<'py>(
        &self,
        py: Python<'py>,
        block_signature_indices: Vec<u32>,
        start_offset: usize,
        max_pairs: Option<usize>,
        selected_indices: Option<Vec<usize>>,
        num_threads: Option<usize>,
        nan_value: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if block_signature_indices.len() <= 1 {
            let empty = numpy::ndarray::Array2::<f64>::zeros((0, 0));
            return Ok(empty.to_pyarray(py));
        }

        let signature_ids = self.signature_id_order();
        let signature_count = signature_ids.len();
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            if global_idx >= signature_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "block signature index out of range: index={} signature_count={}",
                    global_idx, signature_count
                )));
            }
        }

        let mut block_lookup: Vec<(&SignatureData, &PaperData)> =
            Vec::with_capacity(block_signature_indices.len());
        for signature_index in block_signature_indices.iter() {
            let global_idx = *signature_index as usize;
            let signature_id = &signature_ids[global_idx];
            let signature = self
                .signatures
                .get(signature_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(signature_id.clone()))?;
            let paper = self.papers.get(&signature.paper_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(signature.paper_id.to_string())
            })?;
            block_lookup.push((signature, paper));
        }

        let local_pairs =
            upper_triangle_pairs_for_range(block_lookup.len(), start_offset, max_pairs)?;
        let row_count = local_pairs.len();
        if row_count == 0 {
            let empty = numpy::ndarray::Array2::<f64>::zeros((0, 0));
            return Ok(empty.to_pyarray(py));
        }

        let full_cols = self.full_feature_count();
        let indices = resolve_feature_indices("selected_indices", selected_indices, full_cols)?;
        let out_cols = indices.len();
        if out_cols == 0 {
            let empty_cols = numpy::ndarray::Array2::<f64>::zeros((row_count, 0));
            return Ok(empty_cols.to_pyarray(py));
        }

        let out = py.allow_threads(|| {
            let compute = || {
                let mut buffer = vec![0.0_f64; row_count * out_cols];
                buffer
                    .par_chunks_mut(out_cols)
                    .zip(local_pairs.par_iter())
                    .for_each(|(out_row, (left_idx, right_idx))| {
                        let (s1, p1) = block_lookup[*left_idx];
                        let (s2, p2) = block_lookup[*right_idx];
                        let row = self.featurize_pair_data(s1, s2, p1, p2);
                        for (dest, idx) in out_row.iter_mut().zip(indices.iter()) {
                            let mut value = row[*idx];
                            if value.is_nan() && !nan_value.is_nan() {
                                value = nan_value;
                            }
                            *dest = value;
                        }
                    });
                buffer
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let array =
            numpy::ndarray::Array2::from_shape_vec((row_count, out_cols), out).map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build output matrix: {}",
                    err
                ))
            })?;
        Ok(array.to_pyarray(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let file =
            File::create(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let file =
            File::open(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let reader = BufReader::new(file);
        let featurizer: RustFeaturizer = bincode::deserialize_from(reader)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(featurizer)
    }
}

impl RustNameCompatibleSubblockSelector {
    fn from_py(
        py: Python<'_>,
        retrieval_subblock_index: &Bound<'_, PyAny>,
        name_tuples_path: Option<String>,
    ) -> PyResult<Self> {
        let signature_to_subblock = extract_string_string_map(
            &retrieval_subblock_index.get_item("signature_to_subblock")?,
        )?;
        let subblock_to_components =
            extract_string_vec_map(&retrieval_subblock_index.get_item("subblock_to_components")?)?;
        let subblock_tokens_by_subblock =
            match retrieval_subblock_index.get_item("subblock_tokens_by_subblock") {
                Ok(tokens_obj) => extract_string_vec_map(&tokens_obj)?,
                Err(_) => subblock_to_components
                    .keys()
                    .map(|subblock| (subblock.clone(), subblock_tokens_from_key(subblock)))
                    .collect(),
            };
        let name_tuples = load_name_tuples_from_text_path(py, name_tuples_path.as_deref())?;
        Ok(Self {
            signature_to_subblock,
            subblock_to_components,
            subblock_tokens_by_subblock,
            name_tuples,
        })
    }

    fn allowed_component_keys(
        &self,
        query_signature_id: &str,
        query_first: &str,
    ) -> Option<HashSet<String>> {
        let query_subblock = self.signature_to_subblock.get(query_signature_id)?;
        let mut allowed_components: HashSet<String> = HashSet::new();
        if let Some(components) = self.subblock_to_components.get(query_subblock) {
            allowed_components.extend(components.iter().cloned());
        }
        for (subblock, tokens) in self.subblock_tokens_by_subblock.iter() {
            if tokens
                .iter()
                .any(|token| first_names_name_compatible(query_first, token, &self.name_tuples))
            {
                if let Some(components) = self.subblock_to_components.get(subblock) {
                    allowed_components.extend(components.iter().cloned());
                }
            }
        }
        Some(allowed_components)
    }

    fn select_ordered_component_keys(
        &self,
        query_signature_id: &str,
        query_first: &str,
        ordered_component_keys: Vec<String>,
        global_backfill_count: usize,
    ) -> Option<Vec<String>> {
        let allowed_components = self.allowed_component_keys(query_signature_id, query_first)?;
        let mut selected: Vec<String> = ordered_component_keys
            .iter()
            .filter(|component_key| allowed_components.contains(*component_key))
            .cloned()
            .collect();
        if selected.is_empty() {
            return None;
        }
        if global_backfill_count > 0 {
            let mut selected_set: HashSet<String> = selected.iter().cloned().collect();
            let mut remaining = global_backfill_count;
            for component_key in ordered_component_keys {
                if remaining == 0 {
                    break;
                }
                if selected_set.insert(component_key.clone()) {
                    selected.push(component_key);
                    remaining -= 1;
                }
            }
        }
        Some(selected)
    }

    fn select_candidate_indices_for_summaries(
        &self,
        query_signature_id: &str,
        query_first: &str,
        summaries: &[RetrievalSummaryData],
        base_candidate_indices: Option<&[usize]>,
        global_backfill_count: usize,
    ) -> Option<Vec<usize>> {
        let allowed_components = self.allowed_component_keys(query_signature_id, query_first)?;
        let ordered_indices: Vec<usize> = base_candidate_indices
            .map_or_else(|| (0..summaries.len()).collect(), |values| values.to_vec());
        let mut selected: Vec<usize> = ordered_indices
            .iter()
            .copied()
            .filter(|index| allowed_components.contains(&summaries[*index].component_key))
            .collect();
        if selected.is_empty() {
            return None;
        }
        if global_backfill_count > 0 {
            let mut selected_set: HashSet<String> = selected
                .iter()
                .map(|index| summaries[*index].component_key.clone())
                .collect();
            let mut remaining = global_backfill_count;
            for index in ordered_indices {
                if remaining == 0 {
                    break;
                }
                if selected_set.insert(summaries[index].component_key.clone()) {
                    selected.push(index);
                    remaining -= 1;
                }
            }
        }
        Some(selected)
    }
}

#[pymethods]
impl RustNameCompatibleSubblockSelector {
    #[new]
    #[pyo3(signature = (retrieval_subblock_index, name_tuples_path = None))]
    fn new(
        py: Python<'_>,
        retrieval_subblock_index: &Bound<'_, PyAny>,
        name_tuples_path: Option<String>,
    ) -> PyResult<Self> {
        Self::from_py(py, retrieval_subblock_index, name_tuples_path)
    }

    #[pyo3(signature = (query_signature_id, query_first, component_keys, global_backfill_count = 0))]
    fn select(
        &self,
        query_signature_id: &str,
        query_first: &str,
        component_keys: &Bound<'_, PyAny>,
        global_backfill_count: usize,
    ) -> PyResult<Option<Vec<String>>> {
        let ordered_component_keys: Vec<String> = PyIterator::from_object(component_keys)?
            .map(|item| item.and_then(|value| value.extract()))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(self.select_ordered_component_keys(
            query_signature_id,
            query_first,
            ordered_component_keys,
            global_backfill_count,
        ))
    }
}

#[pymethods]
impl RustHybridCentroidRetriever {
    #[new]
    #[pyo3(signature = (summaries, include_exemplars = false))]
    fn new(summaries: &Bound<'_, PyAny>, include_exemplars: bool) -> PyResult<Self> {
        let mut packed_summaries = Vec::new();
        let mut component_index_by_key = HashMap::new();
        let mut coauthor_cluster_df = HashMap::new();
        let mut non_mega_coauthor_cluster_df = HashMap::new();
        let mut affiliation_cluster_df = HashMap::new();
        for item in PyIterator::from_object(summaries)? {
            let summary_obj = item?;
            update_cluster_df_from_counter(
                &summary_obj.getattr("coauthor_counts")?,
                &mut coauthor_cluster_df,
            )?;
            update_cluster_df_from_counter(
                &summary_obj.getattr("non_mega_coauthor_counts")?,
                &mut non_mega_coauthor_cluster_df,
            )?;
            update_cluster_df_from_counter(
                &summary_obj.getattr("affiliation_counts")?,
                &mut affiliation_cluster_df,
            )?;
            let summary = extract_retrieval_summary(&summary_obj, include_exemplars)?;
            component_index_by_key.insert(summary.component_key.clone(), packed_summaries.len());
            packed_summaries.push(summary);
        }
        Ok(Self {
            summaries: packed_summaries,
            component_index_by_key,
            coauthor_cluster_df,
            non_mega_coauthor_cluster_df,
            affiliation_cluster_df,
        })
    }

    #[pyo3(signature = (query, top_k, num_threads = None))]
    fn top_k_hybrid_centroid(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        top_k: usize,
        num_threads: Option<usize>,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        validate_positive_top_k(top_k)?;
        let query_data = extract_retrieval_query(query)?;

        let selection = self.hard_filtered_candidate_indices_for_query(
            &query_data,
            (0..self.summaries.len()).collect(),
        );

        if selection.indices.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let effective_top_k = if selection.return_all {
            selection.indices.len()
        } else {
            top_k
        };

        self.score_top_k_candidate_indices_experimental(
            py,
            &query_data,
            &selection.indices,
            effective_top_k,
            num_threads,
            None,
            None,
            Self::default_hybrid_weights_for_query(&query_data),
            Self::default_experimental_config_for_query(&query_data),
        )
    }

    #[pyo3(signature = (
        queries,
        query_signature_indices,
        component_member_indices_by_key,
        top_k,
        num_threads = None,
        query_signature_ids = None,
        retrieval_subblock_index = None,
        query_candidate_component_keys_by_signature_id = None,
        full_first_global_backfill_count = 0
    ))]
    fn top_k_hybrid_centroid_pair_plan<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        query_signature_indices: PyReadonlyArray1<'py, u32>,
        component_member_indices_by_key: &Bound<'py, PyAny>,
        top_k: usize,
        num_threads: Option<usize>,
        query_signature_ids: Option<&Bound<'py, PyAny>>,
        retrieval_subblock_index: Option<&Bound<'py, PyAny>>,
        query_candidate_component_keys_by_signature_id: Option<&Bound<'py, PyAny>>,
        full_first_global_backfill_count: usize,
    ) -> PyResult<Py<PyDict>> {
        validate_retrieval_rank_top_k(top_k)?;
        let mut query_data = Vec::new();
        for item in PyIterator::from_object(queries)? {
            query_data.push(extract_retrieval_query(&item?)?);
        }
        let query_indices_slice = query_signature_indices.as_slice()?;
        if query_data.len() != query_indices_slice.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "queries and query_signature_indices must have equal length: {} != {}",
                query_data.len(),
                query_indices_slice.len()
            )));
        }
        let query_indices = query_indices_slice.to_vec();
        let query_signature_ids = query_signature_ids
            .map(|values| {
                PyIterator::from_object(values)?
                    .map(|item| item.and_then(|value| value.extract::<String>()))
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        if let Some(values) = query_signature_ids.as_ref() {
            if values.len() != query_data.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "queries and query_signature_ids must have equal length: {} != {}",
                    query_data.len(),
                    values.len()
                )));
            }
        }
        if retrieval_subblock_index.is_some() && query_signature_ids.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "query_signature_ids are required when retrieval_subblock_index is provided",
            ));
        }
        if query_candidate_component_keys_by_signature_id.is_some() && query_signature_ids.is_none()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "query_signature_ids are required when query candidate component keys are provided",
            ));
        }
        let selector = retrieval_subblock_index
            .map(|index| RustNameCompatibleSubblockSelector::from_py(py, index, None))
            .transpose()?;
        let query_candidate_indices_by_signature_id =
            query_candidate_component_keys_by_signature_id
                .map(|mapping| self.extract_candidate_indices_by_query_signature_id(mapping))
                .transpose()?;
        let component_member_indices =
            extract_component_member_indices(component_member_indices_by_key)?;

        let mut row_query_signature_indices = Vec::<u32>::new();
        let mut row_component_keys = Vec::<String>::new();
        let mut row_retrieval_scores = Vec::<f32>::new();
        let mut row_retrieval_ranks = Vec::<u16>::new();
        let mut row_component_sizes = Vec::<u32>::new();
        let mut row_named_signature_counts = Vec::<u32>::new();
        let mut row_dominant_first_names = Vec::<String>::new();
        let mut row_candidate_year_min = Vec::<i32>::new();
        let mut row_candidate_year_max = Vec::<i32>::new();
        let mut row_candidate_year_range_missing = Vec::<u8>::new();
        let mut row_query_first_tokens = Vec::<String>::new();
        let mut row_query_years = Vec::<i32>::new();
        let mut row_query_year_missing = Vec::<u8>::new();
        let mut row_query_has_affiliations = Vec::<u8>::new();
        let mut row_query_has_coauthors = Vec::<u8>::new();
        let mut row_orcid_match = Vec::<u8>::new();
        let mut row_middle_initial_compatibility = Vec::<f32>::new();
        let mut row_affiliation_overlap = Vec::<f32>::new();
        let mut row_coauthor_overlap = Vec::<f32>::new();
        let mut row_venue_overlap = Vec::<f32>::new();
        let mut row_year_compatibility = Vec::<f32>::new();
        let mut row_title_overlap = Vec::<f32>::new();
        let mut row_specter_centroid_similarity = Vec::<f32>::new();
        let mut row_specter_exemplar_similarity = Vec::<f32>::new();
        let mut left_signature_indices = Vec::<u32>::new();
        let mut right_signature_indices = Vec::<u32>::new();
        let mut pair_row_indices = Vec::<u32>::new();

        let query_results: Vec<Result<RetrievalPairPlanQueryResult, String>> =
            py.allow_threads(|| {
                let compute = || {
                    query_data
                        .par_iter()
                        .enumerate()
                        .map(|(query_offset, current_query)| {
                            let query_signature_id = query_signature_ids
                                .as_ref()
                                .map(|values| values[query_offset].as_str());
                            let base_candidate_indices =
                                query_signature_id.and_then(|signature_id| {
                                    query_candidate_indices_by_signature_id.as_ref().and_then(
                                        |mapping| mapping.get(signature_id).map(Vec::as_slice),
                                    )
                                });
                            self.build_pair_plan_query_result(
                                current_query,
                                current_query.first.as_str(),
                                query_indices[query_offset],
                                base_candidate_indices,
                                None,
                                query_signature_id,
                                &component_member_indices,
                                top_k,
                                selector.as_ref(),
                                full_first_global_backfill_count,
                                true,
                            )
                        })
                        .collect::<Vec<_>>()
                };
                install_with_optional_rayon_pool(num_threads, compute)
            });

        for query_result in query_results {
            let mut query_result = query_result.map_err(pyo3::exceptions::PyKeyError::new_err)?;
            let base_row_index = u32::try_from(row_component_keys.len()).map_err(|_| {
                pyo3::exceptions::PyOverflowError::new_err(
                    "retrieved candidate row count exceeds u32",
                )
            })?;
            for (local_row_index, member_indices) in query_result
                .right_signature_indices_by_row
                .iter()
                .enumerate()
            {
                let row_index = checked_retrieved_row_index(base_row_index, local_row_index)?;
                let query_signature_index =
                    query_result.row_query_signature_indices[local_row_index];
                for member_index in member_indices.iter() {
                    left_signature_indices.push(query_signature_index);
                    right_signature_indices.push(*member_index);
                    pair_row_indices.push(row_index);
                }
            }
            row_query_signature_indices.append(&mut query_result.row_query_signature_indices);
            row_component_keys.append(&mut query_result.row_component_keys);
            row_retrieval_scores.append(&mut query_result.row_retrieval_scores);
            row_retrieval_ranks.append(&mut query_result.row_retrieval_ranks);
            row_component_sizes.append(&mut query_result.row_component_sizes);
            row_named_signature_counts.append(&mut query_result.row_named_signature_counts);
            row_dominant_first_names.append(&mut query_result.row_dominant_first_names);
            row_candidate_year_min.append(&mut query_result.row_candidate_year_min);
            row_candidate_year_max.append(&mut query_result.row_candidate_year_max);
            row_candidate_year_range_missing
                .append(&mut query_result.row_candidate_year_range_missing);
            row_query_first_tokens.append(&mut query_result.row_query_first_tokens);
            row_query_years.append(&mut query_result.row_query_years);
            row_query_year_missing.append(&mut query_result.row_query_year_missing);
            row_query_has_affiliations.append(&mut query_result.row_query_has_affiliations);
            row_query_has_coauthors.append(&mut query_result.row_query_has_coauthors);
            row_orcid_match.append(&mut query_result.row_orcid_match);
            row_middle_initial_compatibility
                .append(&mut query_result.row_middle_initial_compatibility);
            row_affiliation_overlap.append(&mut query_result.row_affiliation_overlap);
            row_coauthor_overlap.append(&mut query_result.row_coauthor_overlap);
            row_venue_overlap.append(&mut query_result.row_venue_overlap);
            row_year_compatibility.append(&mut query_result.row_year_compatibility);
            row_title_overlap.append(&mut query_result.row_title_overlap);
            row_specter_centroid_similarity
                .append(&mut query_result.row_specter_centroid_similarity);
            row_specter_exemplar_similarity
                .append(&mut query_result.row_specter_exemplar_similarity);
        }

        let payload = PyDict::new(py);
        payload.set_item("row_count", row_component_keys.len())?;
        payload.set_item(
            "left_signature_indices",
            left_signature_indices.to_pyarray(py),
        )?;
        payload.set_item(
            "right_signature_indices",
            right_signature_indices.to_pyarray(py),
        )?;
        payload.set_item("pair_row_indices", pair_row_indices.to_pyarray(py))?;
        payload.set_item(
            "row_query_signature_indices",
            row_query_signature_indices.to_pyarray(py),
        )?;
        payload.set_item("row_component_keys", row_component_keys)?;
        payload.set_item("retrieval_scores", row_retrieval_scores.to_pyarray(py))?;
        payload.set_item("retrieval_ranks", row_retrieval_ranks.to_pyarray(py))?;
        payload.set_item("row_component_sizes", row_component_sizes.to_pyarray(py))?;
        payload.set_item(
            "row_named_signature_counts",
            row_named_signature_counts.to_pyarray(py),
        )?;
        payload.set_item("row_dominant_first_names", row_dominant_first_names)?;
        payload.set_item(
            "row_candidate_year_min",
            row_candidate_year_min.to_pyarray(py),
        )?;
        payload.set_item(
            "row_candidate_year_max",
            row_candidate_year_max.to_pyarray(py),
        )?;
        payload.set_item(
            "row_candidate_year_range_missing",
            row_candidate_year_range_missing.to_pyarray(py),
        )?;
        payload.set_item("row_query_first_tokens", row_query_first_tokens)?;
        payload.set_item("row_query_years", row_query_years.to_pyarray(py))?;
        payload.set_item(
            "row_query_year_missing",
            row_query_year_missing.to_pyarray(py),
        )?;
        payload.set_item(
            "row_query_has_affiliations",
            row_query_has_affiliations.to_pyarray(py),
        )?;
        payload.set_item(
            "row_query_has_coauthors",
            row_query_has_coauthors.to_pyarray(py),
        )?;
        payload.set_item("row_orcid_match", row_orcid_match.to_pyarray(py))?;
        payload.set_item(
            "middle_initial_compatibility",
            row_middle_initial_compatibility.to_pyarray(py),
        )?;
        payload.set_item(
            "affiliation_overlap",
            row_affiliation_overlap.to_pyarray(py),
        )?;
        payload.set_item("coauthor_overlap", row_coauthor_overlap.to_pyarray(py))?;
        payload.set_item("venue_overlap", row_venue_overlap.to_pyarray(py))?;
        payload.set_item("year_compatibility", row_year_compatibility.to_pyarray(py))?;
        payload.set_item("title_overlap", row_title_overlap.to_pyarray(py))?;
        payload.set_item(
            "specter_centroid_similarity",
            row_specter_centroid_similarity.to_pyarray(py),
        )?;
        payload.set_item(
            "specter_exemplar_similarity",
            row_specter_exemplar_similarity.to_pyarray(py),
        )?;
        Ok(payload.unbind())
    }

    #[pyo3(
        signature = (
            query,
            component_keys,
            top_k,
            weights,
            first_name_mode = "prefix",
            specter_mode = "centroid",
            coauthor_use_idf = false,
            coauthor_per_term_cap = None,
            coauthor_total_cap = None,
            drop_candidate_mega_coauthors = false,
            mega_coauthor_rescue_query_coverage = None,
            mega_coauthor_rescue_min_shared_blocks = 3,
            affiliation_use_idf = false,
            affiliation_per_term_cap = None,
            affiliation_total_cap = None,
            affiliation_min_token_count = 1,
            affiliation_unigram_weight = 1.0,
            affiliation_multi_token_weight = 1.0,
            num_threads = None,
            override_summary = None
        )
    )]
    fn top_k_experimental_weighted_hybrid_centroid_subset(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        component_keys: &Bound<'_, PyAny>,
        top_k: usize,
        weights: Vec<f64>,
        first_name_mode: &str,
        specter_mode: &str,
        coauthor_use_idf: bool,
        coauthor_per_term_cap: Option<f64>,
        coauthor_total_cap: Option<f64>,
        drop_candidate_mega_coauthors: bool,
        mega_coauthor_rescue_query_coverage: Option<f64>,
        mega_coauthor_rescue_min_shared_blocks: usize,
        affiliation_use_idf: bool,
        affiliation_per_term_cap: Option<f64>,
        affiliation_total_cap: Option<f64>,
        affiliation_min_token_count: usize,
        affiliation_unigram_weight: f64,
        affiliation_multi_token_weight: f64,
        num_threads: Option<usize>,
        override_summary: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        validate_positive_top_k(top_k)?;

        let query_data = extract_retrieval_query(query)?;
        let weights_data = extract_retrieval_weights(weights)?;
        let config = build_experimental_config(
            first_name_mode,
            specter_mode,
            coauthor_use_idf,
            coauthor_per_term_cap,
            coauthor_total_cap,
            drop_candidate_mega_coauthors,
            mega_coauthor_rescue_query_coverage,
            mega_coauthor_rescue_min_shared_blocks,
            affiliation_use_idf,
            affiliation_per_term_cap,
            affiliation_total_cap,
            affiliation_min_token_count,
            affiliation_unigram_weight,
            affiliation_multi_token_weight,
        )?;
        let mut candidate_indices = Vec::new();
        for item in PyIterator::from_object(component_keys)? {
            let component_key: String = item?.extract()?;
            let Some(candidate_index) = self.component_index_by_key.get(&component_key) else {
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "Unknown component_key for RustHybridCentroidRetriever: {component_key}"
                )));
            };
            candidate_indices.push(*candidate_index);
        }

        let override_data = override_summary
            .map(|value| {
                extract_retrieval_summary(
                    value,
                    !matches!(config.specter_mode, RetrievalSpecterMode::Centroid),
                )
            })
            .transpose()?;
        let override_index = if let Some(override_summary_data) = override_data.as_ref() {
            let Some(candidate_index) = self
                .component_index_by_key
                .get(&override_summary_data.component_key)
            else {
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "Unknown override component_key for RustHybridCentroidRetriever: {}",
                    override_summary_data.component_key
                )));
            };
            if !candidate_indices.contains(candidate_index) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "override_summary component_key {} was not present in component_keys",
                    override_summary_data.component_key
                )));
            }
            Some(*candidate_index)
        } else {
            None
        };

        self.score_top_k_candidate_indices_experimental(
            py,
            &query_data,
            &candidate_indices,
            top_k,
            num_threads,
            override_index,
            override_data.as_ref(),
            weights_data,
            config,
        )
    }

    #[pyo3(signature = (query, component_keys, num_threads = None, override_summary = None))]
    fn chooser_feature_rows_subset(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        component_keys: &Bound<'_, PyAny>,
        num_threads: Option<usize>,
        override_summary: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDict>> {
        let query_data = extract_retrieval_query(query)?;
        let mut candidate_indices = Vec::new();
        for item in PyIterator::from_object(component_keys)? {
            let component_key: String = item?.extract()?;
            let Some(candidate_index) = self.component_index_by_key.get(&component_key) else {
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "Unknown component_key for RustHybridCentroidRetriever: {component_key}"
                )));
            };
            candidate_indices.push(*candidate_index);
        }

        let override_data = override_summary
            .map(|value| extract_retrieval_summary(value, true))
            .transpose()?;
        let override_index = if let Some(override_summary_data) = override_data.as_ref() {
            let Some(candidate_index) = self
                .component_index_by_key
                .get(&override_summary_data.component_key)
            else {
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "Unknown override component_key for RustHybridCentroidRetriever: {}",
                    override_summary_data.component_key
                )));
            };
            if !candidate_indices.contains(candidate_index) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "override_summary component_key {} was not present in component_keys",
                    override_summary_data.component_key
                )));
            }
            Some(*candidate_index)
        } else {
            None
        };

        let feature_rows = py.allow_threads(|| {
            let compute = || {
                candidate_indices
                    .par_iter()
                    .map(|candidate_index| {
                        let summary = if override_index == Some(*candidate_index) {
                            override_data.as_ref().unwrap_or_else(|| {
                                unreachable!("override_index implies override_data")
                            })
                        } else {
                            &self.summaries[*candidate_index]
                        };
                        (
                            summary.component_key.clone(),
                            chooser_summary_features(&query_data, summary),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });

        let payload = PyDict::new(py);
        for (component_key, feature_values) in feature_rows {
            let feature_dict = PyDict::new(py);
            feature_dict.set_item("middle_initial_compatibility", feature_values[0])?;
            feature_dict.set_item("affiliation_overlap", feature_values[1])?;
            feature_dict.set_item("coauthor_overlap", feature_values[2])?;
            feature_dict.set_item("venue_overlap", feature_values[3])?;
            feature_dict.set_item("year_compatibility", feature_values[4])?;
            feature_dict.set_item("title_overlap", feature_values[5])?;
            feature_dict.set_item("specter_centroid_similarity", feature_values[6])?;
            feature_dict.set_item("specter_exemplar_similarity", feature_values[7])?;
            payload.set_item(component_key.as_str(), feature_dict)?;
        }
        Ok(payload.unbind())
    }
}

const LINKER_GENERIC_FAMILY_MIN_COUNT: f32 = 3.0;
const LINKER_GENERIC_FAMILY_MIN_RATIO: f32 = 0.6;

fn linker_round(value: f32, scale: f32) -> f32 {
    (value * scale).round() / scale
}

fn linker_clip01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn linker_bool(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn linker_dict_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?.ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!("Missing linker row signal: {key}"))
    })
}

fn linker_extract_f32_vec(
    dict: &Bound<'_, PyDict>,
    key: &str,
    row_count: usize,
) -> PyResult<Vec<f32>> {
    let obj = linker_dict_item(dict, key)?;
    let values = if let Ok(arr) = obj.downcast::<PyArray1<f32>>() {
        arr.readonly().as_slice()?.to_vec()
    } else if let Ok(arr) = obj.downcast::<PyArray1<f64>>() {
        arr.readonly()
            .as_slice()?
            .iter()
            .map(|value| *value as f32)
            .collect()
    } else if let Ok(arr) = obj.downcast::<PyArray1<u16>>() {
        arr.readonly()
            .as_slice()?
            .iter()
            .map(|value| *value as f32)
            .collect()
    } else if let Ok(arr) = obj.downcast::<PyArray1<u32>>() {
        arr.readonly()
            .as_slice()?
            .iter()
            .map(|value| *value as f32)
            .collect()
    } else if let Ok(arr) = obj.downcast::<PyArray1<i32>>() {
        arr.readonly()
            .as_slice()?
            .iter()
            .map(|value| *value as f32)
            .collect()
    } else if let Ok(arr) = obj.downcast::<PyArray1<u8>>() {
        arr.readonly()
            .as_slice()?
            .iter()
            .map(|value| *value as f32)
            .collect()
    } else {
        let mut out = Vec::with_capacity(row_count);
        for item in PyIterator::from_object(&obj)? {
            out.push(item?.extract::<f64>()? as f32);
        }
        out
    };
    if values.len() != row_count {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Signal {key:?} must have row_count={row_count}, got {}",
            values.len()
        )));
    }
    if values.iter().any(|value| value.is_nan()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Signal {key:?} contains NaN values"
        )));
    }
    Ok(values)
}

fn linker_extract_string_vec(
    dict: &Bound<'_, PyDict>,
    key: &str,
    row_count: usize,
) -> PyResult<Vec<String>> {
    let obj = linker_dict_item(dict, key)?;
    let mut values = Vec::with_capacity(row_count);
    for item in PyIterator::from_object(&obj)? {
        let current = item?;
        if current.is_none() {
            values.push(String::new());
        } else {
            values.push(current.extract::<String>()?);
        }
    }
    if values.len() != row_count {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Signal {key:?} must have row_count={row_count}, got {}",
            values.len()
        )));
    }
    Ok(values)
}

fn linker_optional_string_vec(
    dict: &Bound<'_, PyDict>,
    key: &str,
    row_count: usize,
) -> PyResult<Option<Vec<String>>> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => {
            Ok(Some(linker_extract_string_vec(dict, key, row_count)?))
        }
        _ => Ok(None),
    }
}

fn linker_groups(row_query_signature_indices: &[u32]) -> Vec<Vec<usize>> {
    let mut ordered: Vec<usize> = (0..row_query_signature_indices.len()).collect();
    ordered.sort_by_key(|index| row_query_signature_indices[*index]);
    let mut groups = Vec::new();
    let mut start = 0usize;
    while start < ordered.len() {
        let query_index = row_query_signature_indices[ordered[start]];
        let mut end = start + 1;
        while end < ordered.len() && row_query_signature_indices[ordered[end]] == query_index {
            end += 1;
        }
        groups.push(ordered[start..end].to_vec());
        start = end;
    }
    groups
}

fn linker_retrieval_ordered_groups(
    groups: &[Vec<usize>],
    retrieval_score: &[f32],
    retrieval_rank: &[f32],
    component_keys: &[String],
) -> Vec<Vec<usize>> {
    let mut out = Vec::with_capacity(groups.len());
    for group in groups {
        let mut ordered = group.clone();
        ordered.sort_by(|left, right| {
            retrieval_score[*right]
                .total_cmp(&retrieval_score[*left])
                .then_with(|| (retrieval_rank[*left] as i64).cmp(&(retrieval_rank[*right] as i64)))
                .then_with(|| component_keys[*left].cmp(&component_keys[*right]))
        });
        out.push(ordered);
    }
    out
}

fn linker_normalize_alpha(value: &str, unidecode_char_map: &HashMap<char, String>) -> String {
    normalize_text_compat_from_map(value, false, unidecode_char_map)
        .chars()
        .filter(|character| character.is_ascii_alphabetic())
        .collect()
}

fn linker_normalized_alpha_vec(
    values: &[String],
    unidecode_char_map: &HashMap<char, String>,
) -> Vec<String> {
    let mut cache = HashMap::<&str, String>::new();
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        if let Some(normalized) = cache.get(value.as_str()) {
            out.push(normalized.clone());
        } else {
            let normalized = linker_normalize_alpha(value, unidecode_char_map);
            cache.insert(value.as_str(), normalized.clone());
            out.push(normalized);
        }
    }
    out
}

fn linker_family_ids(
    component_keys: &[String],
    dominant_first_names: &[String],
    named_signature_count: &[f32],
    cluster_size: &[f32],
) -> Vec<String> {
    let mut out = component_keys.to_vec();
    for index in 0..component_keys.len() {
        let dominant = dominant_first_names[index].as_str();
        let named_count = named_signature_count[index];
        let dominance_ratio = named_count / cluster_size[index].max(1.0);
        if !dominant.is_empty()
            && named_count >= LINKER_GENERIC_FAMILY_MIN_COUNT
            && dominance_ratio >= LINKER_GENERIC_FAMILY_MIN_RATIO
        {
            out[index] = dominant.to_string();
        }
    }
    out
}

struct LinkerGroupFeatures {
    retrieval_score_gap_vs_best_competitor: Vec<f32>,
    same_family_as_top1: Vec<f32>,
    same_family_as_heuristic_choice: Vec<f32>,
    same_dominant_first_as_best_top5: Vec<f32>,
    current_retrieval_rank: Vec<f32>,
}

fn linker_derive_group_features(
    ordered_groups: &[Vec<usize>],
    retrieval_score: &[f32],
    retrieval_rank: &[f32],
    component_keys: &[String],
    family_ids: &[String],
    dominant_first_alpha: &[String],
    top5_mean_distance: &[f32],
) -> LinkerGroupFeatures {
    let row_count = retrieval_score.len();
    let mut retrieval_score_gap_vs_best_competitor = vec![0.0f32; row_count];
    let mut same_family_as_top1 = vec![0.0f32; row_count];
    let mut same_family_as_heuristic_choice = vec![0.0f32; row_count];
    let mut dominant_first_top1_match = vec![0.0f32; row_count];
    let mut same_dominant_first_as_best_top5 = vec![0.0f32; row_count];
    let mut current_retrieval_rank = vec![0.0f32; row_count];

    for ordered in ordered_groups {
        let top1 = ordered[0];
        let runner_up = if ordered.len() > 1 {
            ordered[1]
        } else {
            ordered[0]
        };
        let mut best_top5 = ordered[0];
        for index in ordered.iter().copied().skip(1) {
            let current_key = (
                top5_mean_distance[index],
                retrieval_rank[index] as i64,
                component_keys[index].as_str(),
            );
            let best_key = (
                top5_mean_distance[best_top5],
                retrieval_rank[best_top5] as i64,
                component_keys[best_top5].as_str(),
            );
            if current_key < best_key {
                best_top5 = index;
            }
        }
        for (current_rank, index) in ordered.iter().enumerate() {
            let competitor = if *index == top1 { runner_up } else { top1 };
            current_retrieval_rank[*index] = (current_rank + 1) as f32;
            retrieval_score_gap_vs_best_competitor[*index] = linker_round(
                retrieval_score[*index] - retrieval_score[competitor],
                1_000_000.0,
            );
            same_family_as_top1[*index] = linker_bool(
                !family_ids[*index].is_empty() && family_ids[*index] == family_ids[top1],
            );
            dominant_first_top1_match[*index] = linker_bool(
                !dominant_first_alpha[*index].is_empty()
                    && dominant_first_alpha[*index] == dominant_first_alpha[top1],
            );
            same_dominant_first_as_best_top5[*index] = linker_bool(
                !dominant_first_alpha[*index].is_empty()
                    && dominant_first_alpha[*index] == dominant_first_alpha[best_top5],
            );
            same_family_as_heuristic_choice[*index] = linker_round(
                dominant_first_top1_match[*index] * retrieval_score[*index]
                    + same_dominant_first_as_best_top5[*index] * (1.0 - top5_mean_distance[*index]),
                1_000_000.0,
            );
        }
    }

    LinkerGroupFeatures {
        retrieval_score_gap_vs_best_competitor,
        same_family_as_top1,
        same_family_as_heuristic_choice,
        same_dominant_first_as_best_top5,
        current_retrieval_rank,
    }
}

fn linker_year_gap_features(
    query_year: &[f32],
    query_year_missing: &[f32],
    candidate_year_min: &[f32],
    candidate_year_max: &[f32],
    candidate_year_range_missing: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut gap = vec![0.0f32; query_year.len()];
    let mut signed_gap = vec![0.0f32; query_year.len()];
    let mut span = vec![0.0f32; query_year.len()];
    for index in 0..query_year.len() {
        if candidate_year_range_missing[index] == 0.0 {
            span[index] = (candidate_year_max[index] - candidate_year_min[index]).max(0.0);
        }
        if query_year_missing[index] != 0.0 || candidate_year_range_missing[index] != 0.0 {
            continue;
        }
        if query_year[index] < candidate_year_min[index] {
            let current_gap = candidate_year_min[index] - query_year[index];
            gap[index] = linker_round(current_gap, 1_000_000.0);
            signed_gap[index] = linker_round(-current_gap, 1_000_000.0);
        } else if query_year[index] > candidate_year_max[index] {
            let current_gap = query_year[index] - candidate_year_max[index];
            gap[index] = linker_round(current_gap, 1_000_000.0);
            signed_gap[index] = linker_round(current_gap, 1_000_000.0);
        }
    }
    (gap, signed_gap, span)
}

fn linker_alpha_length(value: &str) -> f32 {
    py_len(value) as f32
}

fn linker_set_f32_array<'py>(
    py: Python<'py>,
    payload: &Bound<'py, PyDict>,
    key: &str,
    values: Vec<f32>,
) -> PyResult<()> {
    payload.set_item(key, values.to_pyarray(py))?;
    Ok(())
}

#[pyfunction]
fn promoted_linker_non_pairwise_features<'py>(
    py: Python<'py>,
    signals: &Bound<'py, PyDict>,
) -> PyResult<Py<PyDict>> {
    let row_query_signature_indices_obj = linker_dict_item(signals, "row_query_signature_indices")?;
    let row_query_signature_indices =
        if let Ok(arr) = row_query_signature_indices_obj.downcast::<PyArray1<u32>>() {
            arr.readonly().as_slice()?.to_vec()
        } else {
            let mut out = Vec::new();
            for item in PyIterator::from_object(&row_query_signature_indices_obj)? {
                out.push(item?.extract::<u32>()?);
            }
            out
        };
    let row_count = row_query_signature_indices.len();

    let retrieval_score = linker_extract_f32_vec(signals, "retrieval_score", row_count)?;
    let retrieval_rank = linker_extract_f32_vec(signals, "retrieval_rank", row_count)?;
    let component_keys = linker_extract_string_vec(signals, "candidate_component_key", row_count)?;
    let cluster_size = linker_extract_f32_vec(signals, "cluster_size", row_count)?;
    let named_signature_count =
        linker_extract_f32_vec(signals, "named_signature_count", row_count)?;
    let dominant_first_name = linker_extract_string_vec(signals, "dominant_first_name", row_count)?;
    let candidate_year_min = linker_extract_f32_vec(signals, "candidate_year_min", row_count)?;
    let candidate_year_max = linker_extract_f32_vec(signals, "candidate_year_max", row_count)?;
    let candidate_year_range_missing =
        linker_extract_f32_vec(signals, "candidate_year_range_missing", row_count)?;
    let query_first_token = linker_extract_string_vec(signals, "query_first_token", row_count)?;
    let query_year = linker_extract_f32_vec(signals, "query_year", row_count)?;
    let query_year_missing = linker_extract_f32_vec(signals, "query_year_missing", row_count)?;
    let query_has_affiliations =
        linker_extract_f32_vec(signals, "query_has_affiliations", row_count)?;
    let affiliation_overlap = linker_extract_f32_vec(signals, "affiliation_overlap", row_count)?;
    let coauthor_overlap = linker_extract_f32_vec(signals, "coauthor_overlap", row_count)?;
    let year_compatibility = linker_extract_f32_vec(signals, "year_compatibility", row_count)?;
    let specter_exemplar_similarity =
        linker_extract_f32_vec(signals, "specter_exemplar_similarity", row_count)?;
    let min_distance = linker_extract_f32_vec(signals, "min_distance", row_count)?;
    let top5_mean_distance = linker_extract_f32_vec(signals, "top5_mean_distance", row_count)?;
    let last_name_count_min_rarity =
        linker_extract_f32_vec(signals, "last_name_count_min_rarity", row_count)?;
    let last_first_name_count_min_rarity =
        linker_extract_f32_vec(signals, "last_first_name_count_min_rarity", row_count)?;
    let candidate_cluster_max_paper_author_count = linker_extract_f32_vec(
        signals,
        "candidate_cluster_max_paper_author_count",
        row_count,
    )?;
    let paper_author_list_max_jaccard =
        linker_extract_f32_vec(signals, "paper_author_list_max_jaccard", row_count)?;
    let paper_author_list_max_containment =
        linker_extract_f32_vec(signals, "paper_author_list_max_containment", row_count)?;
    let paper_author_list_max_overlap_count =
        linker_extract_f32_vec(signals, "paper_author_list_max_overlap_count", row_count)?;
    let local_author_window10_jaccard_max =
        linker_extract_f32_vec(signals, "local_author_window10_jaccard_max", row_count)?;
    let local_author_window10_overlap_count_max = linker_extract_f32_vec(
        signals,
        "local_author_window10_overlap_count_max",
        row_count,
    )?;
    let best_author_count_log_absdiff =
        linker_extract_f32_vec(signals, "best_author_count_log_absdiff", row_count)?;

    let groups = linker_groups(&row_query_signature_indices);
    let ordered_groups = linker_retrieval_ordered_groups(
        &groups,
        &retrieval_score,
        &retrieval_rank,
        &component_keys,
    );
    let family_ids_from_signal = linker_optional_string_vec(signals, "family_id", row_count)?;
    let generated_family_id_count = if family_ids_from_signal.is_some() {
        0usize
    } else {
        row_count
    };
    let family_ids = family_ids_from_signal.unwrap_or_else(|| {
        linker_family_ids(
            &component_keys,
            &dominant_first_name,
            &named_signature_count,
            &cluster_size,
        )
    });
    let generic_family_override_count = family_ids
        .iter()
        .zip(component_keys.iter())
        .filter(|(family, component)| !family.is_empty() && *family != *component)
        .count();
    let text_module = py.import("s2and.text")?;
    let unidecode = text_module.getattr("unidecode")?;
    let mut linker_unidecode_char_map = HashMap::<char, String>::new();
    for value in query_first_token.iter().chain(dominant_first_name.iter()) {
        ensure_unidecode_for_text(&unidecode, value, &mut linker_unidecode_char_map)?;
    }
    let query_first_alpha =
        linker_normalized_alpha_vec(&query_first_token, &linker_unidecode_char_map);
    let dominant_first_alpha =
        linker_normalized_alpha_vec(&dominant_first_name, &linker_unidecode_char_map);
    let group_features = linker_derive_group_features(
        &ordered_groups,
        &retrieval_score,
        &retrieval_rank,
        &component_keys,
        &family_ids,
        &dominant_first_alpha,
        &top5_mean_distance,
    );
    let (year_gap_to_candidate_range, year_gap_signed_to_candidate_range, candidate_year_span) =
        linker_year_gap_features(
            &query_year,
            &query_year_missing,
            &candidate_year_min,
            &candidate_year_max,
            &candidate_year_range_missing,
        );

    let mut affiliation_contradiction_severity = vec![0.0f32; row_count];
    let mut anchor_evidence_count = vec![0.0f32; row_count];
    let mut strong_positive_anchor_score = vec![0.0f32; row_count];
    let mut weak_residual_anchor_score = vec![0.0f32; row_count];
    let mut sparse_relative_winner_score = vec![0.0f32; row_count];
    let mut query_first_prefix_match_any_length = vec![0.0f32; row_count];
    let mut candidate_dominant_first_name_length = vec![0.0f32; row_count];
    let mut cluster_size_log = vec![0.0f32; row_count];
    let mut retrieval_reciprocal_rank = vec![0.0f32; row_count];

    for index in 0..row_count {
        if query_has_affiliations[index] > 0.0 {
            affiliation_contradiction_severity[index] =
                linker_round((1.0 - affiliation_overlap[index]).max(0.0), 1_000_000.0);
        }
        let query_first = &query_first_alpha[index];
        let dominant_first = &dominant_first_alpha[index];
        let retrieval_gap = group_features.retrieval_score_gap_vs_best_competitor[index];
        anchor_evidence_count[index] =
            linker_bool(min_distance[index] <= 0.15) + linker_bool(retrieval_gap >= 0.02);
        let distance_signal = 1.0 - linker_clip01(min_distance[index]);
        let support_strength = 0.20 * distance_signal;
        let same_top1 = group_features.same_family_as_top1[index];
        strong_positive_anchor_score[index] = linker_round(
            linker_clip01(support_strength) * (0.5 + 0.5 * linker_clip01(same_top1)),
            1_000_000.0,
        );
        let retrieval_gap_scaled = linker_clip01((retrieval_gap.clamp(-0.2, 0.3) + 0.2) / 0.5);
        let residual_support = 0.28 * distance_signal + 0.08 * retrieval_gap_scaled;
        weak_residual_anchor_score[index] =
            linker_round(same_top1 * linker_clip01(residual_support), 1_000_000.0);
        sparse_relative_winner_score[index] = linker_round(
            linker_bool(group_features.current_retrieval_rank[index] <= 1.0)
                * same_top1
                * linker_clip01(retrieval_gap.clamp(0.0, 0.3) / 0.3)
                * linker_clip01(residual_support),
            1_000_000.0,
        );
        query_first_prefix_match_any_length[index] = linker_bool(
            !query_first.is_empty()
                && !dominant_first.is_empty()
                && (query_first.starts_with(dominant_first)
                    || dominant_first.starts_with(query_first)),
        );
        candidate_dominant_first_name_length[index] = linker_alpha_length(dominant_first);
        cluster_size_log[index] = (1.0 + cluster_size[index].max(0.0)).ln();
        retrieval_reciprocal_rank[index] = linker_round(
            1.0 / group_features.current_retrieval_rank[index].max(1.0),
            1_000_000.0,
        );
    }

    let payload = PyDict::new(py);
    linker_set_f32_array(py, &payload, "min_distance", min_distance.clone())?;
    linker_set_f32_array(
        py,
        &payload,
        "affiliation_contradiction_severity",
        affiliation_contradiction_severity,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "same_family_as_heuristic_choice",
        group_features.same_family_as_heuristic_choice,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "same_dominant_first_as_best_top5",
        group_features.same_dominant_first_as_best_top5,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "specter_exemplar_similarity",
        specter_exemplar_similarity,
    )?;
    linker_set_f32_array(py, &payload, "coauthor_overlap", coauthor_overlap)?;
    linker_set_f32_array(py, &payload, "affiliation_overlap", affiliation_overlap)?;
    linker_set_f32_array(py, &payload, "year_compatibility", year_compatibility)?;
    linker_set_f32_array(
        py,
        &payload,
        "retrieval_rank",
        group_features.current_retrieval_rank,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "retrieval_reciprocal_rank",
        retrieval_reciprocal_rank,
    )?;
    linker_set_f32_array(py, &payload, "cluster_size_log", cluster_size_log)?;
    linker_set_f32_array(py, &payload, "candidate_year_span", candidate_year_span)?;
    linker_set_f32_array(
        py,
        &payload,
        "year_gap_to_candidate_range",
        year_gap_to_candidate_range,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "year_gap_signed_to_candidate_range",
        year_gap_signed_to_candidate_range,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "candidate_dominant_first_name_length",
        candidate_dominant_first_name_length,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "query_first_prefix_match_any_length",
        query_first_prefix_match_any_length,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "candidate_cluster_max_paper_author_count",
        candidate_cluster_max_paper_author_count,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "paper_author_list_max_jaccard",
        paper_author_list_max_jaccard,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "paper_author_list_max_containment",
        paper_author_list_max_containment,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "paper_author_list_max_overlap_count",
        paper_author_list_max_overlap_count,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "local_author_window10_jaccard_max",
        local_author_window10_jaccard_max,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "local_author_window10_overlap_count_max",
        local_author_window10_overlap_count_max,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "best_author_count_log_absdiff",
        best_author_count_log_absdiff,
    )?;
    linker_set_f32_array(py, &payload, "anchor_evidence_count", anchor_evidence_count)?;
    linker_set_f32_array(
        py,
        &payload,
        "strong_positive_anchor_score",
        strong_positive_anchor_score,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "weak_residual_anchor_score",
        weak_residual_anchor_score,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "sparse_relative_winner_score",
        sparse_relative_winner_score,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "last_name_count_min_rarity",
        last_name_count_min_rarity,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "last_first_name_count_min_rarity",
        last_first_name_count_min_rarity,
    )?;
    linker_set_f32_array(
        py,
        &payload,
        "top5_mean_distance",
        top5_mean_distance.clone(),
    )?;
    let telemetry = PyDict::new(py);
    telemetry.set_item("generated_family_id_count", generated_family_id_count)?;
    telemetry.set_item(
        "generic_family_override_count",
        generic_family_override_count,
    )?;
    payload.set_item("telemetry", telemetry)?;
    Ok(payload.unbind())
}

#[pyfunction]
fn get_build_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let build_info = PyDict::new(py);
    build_info.set_item("crate_version", env!("CARGO_PKG_VERSION"))?;
    build_info.set_item("profile", option_env!("PROFILE").unwrap_or("unknown"))?;
    build_info.set_item("debug_assertions", cfg!(debug_assertions))?;
    build_info.set_item("debug", option_env!("DEBUG").unwrap_or("unknown"))?;
    build_info.set_item("opt_level", option_env!("OPT_LEVEL").unwrap_or("unknown"))?;
    build_info.set_item("target", option_env!("TARGET").unwrap_or("unknown"))?;
    build_info.set_item("host", option_env!("HOST").unwrap_or("unknown"))?;
    build_info.set_item("rustc", option_env!("RUSTC").unwrap_or("unknown"))?;
    build_info.set_item(
        "incremental_linking_pair_plan_row_signals",
        INCREMENTAL_LINKING_PAIR_PLAN_ROW_SIGNALS.to_vec(),
    )?;
    Ok(build_info.unbind())
}

fn required_path_from_py_dict(paths: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    match paths.get_item(key)? {
        Some(value) => value.extract(),
        None => Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "paths must include '{key}'"
        ))),
    }
}

fn optional_path_from_py_dict(paths: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    paths
        .get_item(key)?
        .map(|value| value.extract())
        .transpose()
}

fn optional_name_counts_index_path_from_py_dict(
    paths: &Bound<'_, PyDict>,
) -> PyResult<Option<String>> {
    optional_path_from_py_dict(paths, "name_counts_index")
}

fn py_dict_usize_or_zero(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<usize> {
    dict.get_item(key)?
        .map(|value| value.extract::<usize>())
        .transpose()
        .map(|value| value.unwrap_or(0))
}

fn py_dict_f64_or_zero(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    dict.get_item(key)?
        .map(|value| value.extract::<f64>())
        .transpose()
        .map(|value| value.unwrap_or(0.0))
}

fn merge_raw_arrow_planner_build_telemetry(
    py: Python<'_>,
    raw_plan: &Py<PyDict>,
    build_telemetry: &Py<PyDict>,
) -> PyResult<()> {
    let telemetry_obj = raw_plan
        .bind(py)
        .get_item("telemetry")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("raw plan is missing telemetry"))?;
    let telemetry = telemetry_obj.downcast::<PyDict>()?;
    let build = build_telemetry.bind(py);
    for key in [
        "signature_batches_read",
        "signature_rows_scanned",
        "paper_batches_read",
        "paper_rows_scanned",
        "paper_author_batches_read",
        "paper_author_rows_scanned",
        "specter_batches_read",
        "specter_rows_scanned",
    ] {
        telemetry.set_item(
            key,
            py_dict_usize_or_zero(telemetry, key)? + py_dict_usize_or_zero(build, key)?,
        )?;
    }
    let timings_obj = telemetry.get_item("timings")?.ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err("raw plan telemetry is missing timings")
    })?;
    let timings = timings_obj.downcast::<PyDict>()?;
    let build_timings_obj = build.get_item("timings")?.ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err("planner build telemetry is missing timings")
    })?;
    let build_timings = build_timings_obj.downcast::<PyDict>()?;
    for key in [
        "read_cluster_seeds_secs",
        "read_signatures_secs",
        "read_papers_secs",
        "read_paper_authors_secs",
        "read_specter_secs",
        "read_name_counts_secs",
        "metadata_reads_parallel_secs",
        "text_context_secs",
        "feature_secs",
        "summary_secs",
        "component_members_secs",
    ] {
        timings.set_item(
            key,
            py_dict_f64_or_zero(timings, key)? + py_dict_f64_or_zero(build_timings, key)?,
        )?;
    }
    telemetry.set_item("planner_seed_state_reused", 0)?;
    telemetry.set_item("planner_seed_state_built", 1)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (
    paths,
    query_signature_ids,
    top_k = 25,
    query_view = "auto",
    orcid_enabled = true,
    num_threads = None,
    max_exemplars = 4,
    include_pair_signature_ids = true,
    include_component_members = true,
    full_scan_without_index = true
))]
fn raw_block_query_candidate_plan_arrow<'py>(
    py: Python<'py>,
    paths: &Bound<'py, PyDict>,
    query_signature_ids: Vec<String>,
    top_k: usize,
    query_view: &str,
    orcid_enabled: bool,
    num_threads: Option<usize>,
    max_exemplars: usize,
    include_pair_signature_ids: bool,
    include_component_members: bool,
    full_scan_without_index: bool,
) -> PyResult<Py<PyDict>> {
    let mut planner = RawBlockQueryCandidatePlanner::new(
        py,
        paths,
        query_signature_ids.clone(),
        top_k,
        query_view,
        orcid_enabled,
        num_threads,
        max_exemplars,
        include_pair_signature_ids,
        include_component_members,
        full_scan_without_index,
    )?;
    let plan = planner.plan(
        py,
        query_signature_ids,
        Some(top_k),
        Some(query_view.to_string()),
        Some(include_pair_signature_ids),
        Some(include_component_members),
    )?;
    let build_telemetry = planner.build_telemetry(py)?;
    merge_raw_arrow_planner_build_telemetry(py, &plan, &build_telemetry)?;
    Ok(plan)
}

struct RawArrowPlannerPaths {
    signatures_path: String,
    papers_path: String,
    paper_authors_path: String,
    cluster_seeds_path: String,
    cluster_seed_disallows_path: Option<String>,
    specter_path: Option<String>,
    name_counts_arrow_path: Option<String>,
    name_counts_index_path: Option<String>,
    signatures_batch_index_path: Option<String>,
    papers_batch_index_path: Option<String>,
    paper_authors_batch_index_path: Option<String>,
    specter_batch_index_path: Option<String>,
}

impl RawArrowPlannerPaths {
    fn from_py_dict(paths: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            signatures_path: required_path_from_py_dict(paths, "signatures")?,
            papers_path: required_path_from_py_dict(paths, "papers")?,
            paper_authors_path: required_path_from_py_dict(paths, "paper_authors")?,
            cluster_seeds_path: required_path_from_py_dict(paths, "cluster_seeds")?,
            cluster_seed_disallows_path: optional_path_from_py_dict(
                paths,
                "cluster_seed_disallows",
            )?,
            specter_path: optional_path_from_py_dict(paths, "specter")?,
            name_counts_arrow_path: optional_path_from_py_dict(paths, "name_counts")?,
            name_counts_index_path: optional_name_counts_index_path_from_py_dict(paths)?,
            signatures_batch_index_path: optional_path_from_py_dict(
                paths,
                "signatures_batch_index",
            )?,
            papers_batch_index_path: optional_path_from_py_dict(paths, "papers_batch_index")?,
            paper_authors_batch_index_path: optional_path_from_py_dict(
                paths,
                "paper_authors_batch_index",
            )?,
            specter_batch_index_path: optional_path_from_py_dict(paths, "specter_batch_index")?,
        })
    }
}

fn raw_arrow_feature_paths_from_py_dict(
    paths: &Bound<'_, PyDict>,
) -> PyResult<RawArrowPlannerPaths> {
    Ok(RawArrowPlannerPaths {
        signatures_path: required_path_from_py_dict(paths, "signatures")?,
        papers_path: required_path_from_py_dict(paths, "papers")?,
        paper_authors_path: required_path_from_py_dict(paths, "paper_authors")?,
        cluster_seeds_path: String::new(),
        cluster_seed_disallows_path: optional_path_from_py_dict(paths, "cluster_seed_disallows")?,
        specter_path: optional_path_from_py_dict(paths, "specter")?,
        name_counts_arrow_path: optional_path_from_py_dict(paths, "name_counts")?,
        name_counts_index_path: optional_name_counts_index_path_from_py_dict(paths)?,
        signatures_batch_index_path: optional_path_from_py_dict(paths, "signatures_batch_index")?,
        papers_batch_index_path: optional_path_from_py_dict(paths, "papers_batch_index")?,
        paper_authors_batch_index_path: optional_path_from_py_dict(
            paths,
            "paper_authors_batch_index",
        )?,
        specter_batch_index_path: optional_path_from_py_dict(paths, "specter_batch_index")?,
    })
}

struct RawArrowPlannerBuildTelemetry {
    read_cluster_seeds_secs: f64,
    read_signatures_secs: f64,
    read_papers_secs: f64,
    read_paper_authors_secs: f64,
    read_specter_secs: f64,
    read_name_counts_secs: f64,
    metadata_reads_parallel_secs: f64,
    text_context_secs: f64,
    feature_secs: f64,
    summary_secs: f64,
    component_members_secs: f64,
    signature_index_stats: IndexedArrowReadStats,
    paper_index_stats: IndexedArrowReadStats,
    paper_author_index_stats: IndexedArrowReadStats,
    specter_index_stats: IndexedArrowReadStats,
    indexed_arrow_candidate_plan: bool,
}

struct ReusableRawArrowCandidatePlanState {
    features_by_signature_id: HashMap<String, RawArrowFeature>,
    signatures: HashMap<String, RawArrowSignature>,
    paper_authors: HashMap<String, Vec<(i64, String)>>,
    raw_name_counts: RawNameCountMaps,
    members_by_component: HashMap<String, Vec<String>>,
    component_keys_by_member: HashMap<String, Vec<String>>,
    summary_signals_by_component: HashMap<String, RawArrowSummarySignalData>,
    retriever: RustHybridCentroidRetriever,
    component_order: Vec<String>,
    seed_signature_ids: Vec<String>,
    seed_signature_id_set: HashSet<String>,
    cluster_seed_disallows: HashSet<(String, String)>,
    unidecode_char_map: HashMap<char, String>,
    name_prefixes: HashSet<String>,
    affiliation_stopwords: HashSet<String>,
    seed_paper_count: usize,
    seed_specter_count: usize,
    build_telemetry: RawArrowPlannerBuildTelemetry,
}

struct RawArrowQueryInputReadResult {
    signatures: HashMap<String, RawArrowSignature>,
    papers: HashMap<String, RawArrowPaper>,
    paper_authors: HashMap<String, Vec<(i64, String)>>,
    specter_by_paper_id: Option<HashMap<String, Arc<Vec<f32>>>>,
    signature_index_stats: IndexedArrowReadStats,
    paper_index_stats: IndexedArrowReadStats,
    paper_author_index_stats: IndexedArrowReadStats,
    specter_index_stats: IndexedArrowReadStats,
    read_signatures_secs: f64,
    read_papers_secs: f64,
    read_paper_authors_secs: f64,
    read_specter_secs: f64,
    metadata_reads_parallel_secs: f64,
}

fn validate_raw_arrow_query_signature_ids(query_signature_ids: &[String]) -> PyResult<()> {
    if query_signature_ids.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "query_signature_ids must be non-empty",
        ));
    }
    let mut seen = HashSet::<&str>::with_capacity(query_signature_ids.len());
    for signature_id in query_signature_ids.iter() {
        if !seen.insert(signature_id.as_str()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "query_signature_ids must be unique; duplicate signature_id={signature_id:?}"
            )));
        }
    }
    Ok(())
}

fn build_retriever_from_raw_arrow_components(
    py: Python<'_>,
    component_order: &[String],
    members_by_component: &HashMap<String, Vec<String>>,
    features_by_signature_id: &HashMap<String, RawArrowFeature>,
    max_exemplars: usize,
    num_threads: Option<usize>,
) -> PyResult<RustHybridCentroidRetriever> {
    let summary_results: Vec<Result<RetrievalSummaryData, String>> = py.allow_threads(|| {
        let compute = || {
            component_order
                .par_iter()
                .map(|component_key| {
                    let members = members_by_component.get(component_key).ok_or_else(|| {
                        format!(
                            "component_key '{}' disappeared while building summaries",
                            component_key
                        )
                    })?;
                    build_raw_arrow_summary(
                        component_key,
                        members,
                        features_by_signature_id,
                        max_exemplars,
                    )
                })
                .collect::<Vec<_>>()
        };
        install_with_optional_rayon_pool(num_threads, compute)
    });
    let mut summaries = Vec::<RetrievalSummaryData>::with_capacity(summary_results.len());
    for result in summary_results {
        summaries.push(result.map_err(pyo3::exceptions::PyKeyError::new_err)?);
    }
    let mut component_index_by_key = HashMap::with_capacity(summaries.len());
    let mut coauthor_cluster_df = HashMap::new();
    let mut non_mega_coauthor_cluster_df = HashMap::new();
    let mut affiliation_cluster_df = HashMap::new();
    for (index, summary) in summaries.iter().enumerate() {
        component_index_by_key.insert(summary.component_key.clone(), index);
        increment_df_from_counter(&summary.coauthor_counts, &mut coauthor_cluster_df);
        increment_df_from_counter(
            &summary.non_mega_coauthor_counts,
            &mut non_mega_coauthor_cluster_df,
        );
        increment_df_from_counter(&summary.affiliation_counts, &mut affiliation_cluster_df);
    }
    Ok(RustHybridCentroidRetriever {
        summaries,
        component_index_by_key,
        coauthor_cluster_df,
        non_mega_coauthor_cluster_df,
        affiliation_cluster_df,
    })
}

fn raw_arrow_component_member_indices_for_batch(
    component_order: &[String],
    members_by_component: &HashMap<String, Vec<String>>,
    query_count: usize,
) -> PyResult<(HashMap<String, Vec<u32>>, Vec<String>, Vec<String>)> {
    let mut component_member_indices = HashMap::<String, Vec<u32>>::new();
    let mut seed_signature_ids = Vec::<String>::new();
    let mut seed_component_keys = Vec::<String>::new();
    let query_count_u32 = u32::try_from(query_count)
        .map_err(|_| pyo3::exceptions::PyOverflowError::new_err("query count exceeds u32"))?;
    for component_key in component_order.iter() {
        let mut member_indices = Vec::<u32>::new();
        if let Some(members) = members_by_component.get(component_key) {
            for signature_id in members.iter() {
                let seed_offset = u32::try_from(seed_signature_ids.len()).map_err(|_| {
                    pyo3::exceptions::PyOverflowError::new_err("seed signature count exceeds u32")
                })?;
                let member_index = query_count_u32.checked_add(seed_offset).ok_or_else(|| {
                    pyo3::exceptions::PyOverflowError::new_err("signature index exceeds u32")
                })?;
                member_indices.push(member_index);
                seed_signature_ids.push(signature_id.clone());
                seed_component_keys.push(component_key.clone());
            }
        }
        component_member_indices.insert(component_key.clone(), member_indices);
    }
    Ok((
        component_member_indices,
        seed_signature_ids,
        seed_component_keys,
    ))
}

fn raw_arrow_excluded_candidate_indices_by_query(
    query_signature_ids: &[String],
    component_order: &[String],
    component_keys_by_member: &HashMap<String, Vec<String>>,
    cluster_seed_disallows: &HashSet<(String, String)>,
) -> PyResult<(Option<Vec<Option<HashSet<usize>>>>, usize)> {
    let query_signature_id_set: HashSet<&str> =
        query_signature_ids.iter().map(String::as_str).collect();
    let mut disallowed_members_by_query = HashMap::<String, HashSet<String>>::new();
    for (left, right) in cluster_seed_disallows.iter() {
        let left_is_query = query_signature_id_set.contains(left.as_str());
        let right_is_query = query_signature_id_set.contains(right.as_str());
        let left_is_seed = component_keys_by_member.contains_key(left);
        let right_is_seed = component_keys_by_member.contains_key(right);
        if left_is_query && !right_is_query && !right_is_seed {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cluster_seed_disallows pair references unknown seed endpoint for query {left:?}: {right:?}"
            )));
        }
        if right_is_query && !left_is_query && !left_is_seed {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cluster_seed_disallows pair references unknown seed endpoint for query {right:?}: {left:?}"
            )));
        }
        if left_is_query {
            disallowed_members_by_query
                .entry(left.clone())
                .or_default()
                .insert(right.clone());
        }
        if right_is_query {
            disallowed_members_by_query
                .entry(right.clone())
                .or_default()
                .insert(left.clone());
        }
    }
    let excluded_indices_by_query = if disallowed_members_by_query.is_empty() {
        None
    } else {
        let mut component_index_by_key =
            HashMap::<&str, usize>::with_capacity(component_order.len());
        for (component_index, component_key) in component_order.iter().enumerate() {
            component_index_by_key.insert(component_key.as_str(), component_index);
        }
        Some(
            query_signature_ids
                .iter()
                .map(|query_signature_id| {
                    let Some(disallowed_members) =
                        disallowed_members_by_query.get(query_signature_id)
                    else {
                        return None;
                    };
                    let mut excluded_indices = HashSet::<usize>::new();
                    for disallowed_member in disallowed_members.iter() {
                        if let Some(component_keys) =
                            component_keys_by_member.get(disallowed_member.as_str())
                        {
                            for component_key in component_keys {
                                if let Some(component_index) =
                                    component_index_by_key.get(component_key.as_str())
                                {
                                    excluded_indices.insert(*component_index);
                                }
                            }
                        }
                    }
                    Some(excluded_indices)
                })
                .collect::<Vec<_>>(),
        )
    };
    let cluster_seed_disallowed_candidate_count =
        excluded_indices_by_query
            .as_ref()
            .map_or(0usize, |indices_by_query| {
                indices_by_query
                    .iter()
                    .filter_map(|excluded| excluded.as_ref())
                    .map(HashSet::len)
                    .sum()
            });
    Ok((
        excluded_indices_by_query,
        cluster_seed_disallowed_candidate_count,
    ))
}

fn raw_arrow_summary_signals_cached<'a>(
    cache: &'a mut HashMap<String, RawArrowSummarySignalData>,
    component_key: &str,
    members_by_component: &HashMap<String, Vec<String>>,
    features_by_signature_id: &HashMap<String, RawArrowFeature>,
    signatures: &HashMap<String, RawArrowSignature>,
    paper_authors: &HashMap<String, Vec<(i64, String)>>,
    unidecode_char_map: &HashMap<char, String>,
) -> PyResult<&'a RawArrowSummarySignalData> {
    if let Entry::Vacant(entry) = cache.entry(component_key.to_string()) {
        let members = members_by_component.get(component_key).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "component_key '{}' disappeared while building row signals",
                component_key
            ))
        })?;
        let summary_signals = build_raw_arrow_summary_signals(
            members,
            features_by_signature_id,
            signatures,
            paper_authors,
            unidecode_char_map,
        )
        .map_err(pyo3::exceptions::PyKeyError::new_err)?;
        entry.insert(summary_signals);
    }
    cache.get(component_key).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!(
            "component_key '{}' is missing raw row signal summary",
            component_key
        ))
    })
}

fn raw_arrow_summary_signals_for_members_cached<'a>(
    cache: &'a mut HashMap<String, RawArrowSummarySignalData>,
    cache_key: &str,
    members: &[String],
    features_by_signature_id: &HashMap<String, RawArrowFeature>,
    signatures: &HashMap<String, RawArrowSignature>,
    paper_authors: &HashMap<String, Vec<(i64, String)>>,
    unidecode_char_map: &HashMap<char, String>,
) -> PyResult<&'a RawArrowSummarySignalData> {
    if let Entry::Vacant(entry) = cache.entry(cache_key.to_string()) {
        let summary_signals = build_raw_arrow_summary_signals(
            members,
            features_by_signature_id,
            signatures,
            paper_authors,
            unidecode_char_map,
        )
        .map_err(pyo3::exceptions::PyKeyError::new_err)?;
        entry.insert(summary_signals);
    }
    cache.get(cache_key).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!(
            "cache_key '{}' is missing raw row signal summary",
            cache_key
        ))
    })
}

fn read_reusable_raw_arrow_query_inputs(
    py: Python<'_>,
    paths: &RawArrowPlannerPaths,
    query_signature_ids: &[String],
    num_threads: Option<usize>,
    full_scan_without_index: bool,
) -> PyResult<RawArrowQueryInputReadResult> {
    let query_signature_id_set: HashSet<String> = query_signature_ids.iter().cloned().collect();
    let read_signatures_start = Instant::now();
    let (signatures, signature_index_stats) = read_raw_arrow_signatures_with_optional_index(
        &paths.signatures_path,
        paths.signatures_batch_index_path.as_deref(),
        Some(&query_signature_id_set),
        full_scan_without_index,
    )?;
    let read_signatures_secs = read_signatures_start.elapsed().as_secs_f64();
    for signature_id in query_signature_ids.iter() {
        if !signatures.contains_key(signature_id) {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "query signature_id '{}' is missing from signatures Arrow input",
                signature_id
            )));
        }
    }
    let needed_paper_ids: HashSet<String> = query_signature_ids
        .iter()
        .filter_map(|signature_id| signatures.get(signature_id))
        .map(|signature| signature.paper_id.clone())
        .collect();
    let metadata_reads_parallel_start = Instant::now();
    let ((papers_result, paper_authors_result), raw_specter_by_paper_id_result) =
        py.allow_threads(|| {
            let compute = || {
                rayon::join(
                    || {
                        rayon::join(
                            || -> PyResult<(
                                HashMap<String, RawArrowPaper>,
                                IndexedArrowReadStats,
                                f64,
                            )> {
                                let start = Instant::now();
                                let (loaded, stats) = read_raw_arrow_papers_with_optional_index(
                                    &paths.papers_path,
                                    paths.papers_batch_index_path.as_deref(),
                                    &needed_paper_ids,
                                    full_scan_without_index,
                                )?;
                                Ok((loaded, stats, start.elapsed().as_secs_f64()))
                            },
                            || -> PyResult<(
                                HashMap<String, Vec<(i64, String)>>,
                                IndexedArrowReadStats,
                                f64,
                            )> {
                                let start = Instant::now();
                                let (loaded, stats) =
                                    read_raw_arrow_paper_authors_with_optional_index(
                                        &paths.paper_authors_path,
                                        paths.paper_authors_batch_index_path.as_deref(),
                                        &needed_paper_ids,
                                        full_scan_without_index,
                                    )?;
                                Ok((loaded, stats, start.elapsed().as_secs_f64()))
                            },
                        )
                    },
                    || -> PyResult<(
                        Option<HashMap<String, Vec<f32>>>,
                        IndexedArrowReadStats,
                        f64,
                    )> {
                        let start = Instant::now();
                        let (loaded, stats) = match paths.specter_path.as_ref() {
                            Some(path) => {
                                let (loaded, stats) = read_raw_arrow_specter_with_optional_index(
                                    path,
                                    paths.specter_batch_index_path.as_deref(),
                                    &needed_paper_ids,
                                    full_scan_without_index,
                                )?;
                                (loaded, stats)
                            }
                            None => (HashMap::new(), IndexedArrowReadStats::default()),
                        };
                        Ok((
                            if paths.specter_path.is_some() {
                                Some(loaded)
                            } else {
                                None
                            },
                            stats,
                            start.elapsed().as_secs_f64(),
                        ))
                    },
                )
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
    let (papers, paper_index_stats, read_papers_secs) = papers_result?;
    let (paper_authors, paper_author_index_stats, read_paper_authors_secs) = paper_authors_result?;
    let (raw_specter_by_paper_id, specter_index_stats, read_specter_secs) =
        raw_specter_by_paper_id_result?;
    let specter_by_paper_id = raw_specter_by_paper_id.map(|values| {
        values
            .into_iter()
            .map(|(paper_id, vector)| (paper_id, Arc::new(vector)))
            .collect::<HashMap<_, _>>()
    });
    Ok(RawArrowQueryInputReadResult {
        signatures,
        papers,
        paper_authors,
        specter_by_paper_id,
        signature_index_stats,
        paper_index_stats,
        paper_author_index_stats,
        specter_index_stats,
        read_signatures_secs,
        read_papers_secs,
        read_paper_authors_secs,
        read_specter_secs,
        metadata_reads_parallel_secs: metadata_reads_parallel_start.elapsed().as_secs_f64(),
    })
}

#[pyclass]
struct RawBlockQueryCandidatePlanner {
    paths: RawArrowPlannerPaths,
    state: ReusableRawArrowCandidatePlanState,
    planner_query_signature_count: usize,
    planner_query_signature_id_set: HashSet<String>,
    top_k: usize,
    query_view: String,
    orcid_enabled: bool,
    num_threads: Option<usize>,
    max_exemplars: usize,
    include_pair_signature_ids: bool,
    include_component_members: bool,
    full_scan_without_index: bool,
}

#[pymethods]
impl RawBlockQueryCandidatePlanner {
    #[new]
    #[pyo3(signature = (
        paths,
        query_signature_ids,
        top_k,
        query_view = "auto",
        orcid_enabled = true,
        num_threads = None,
        max_exemplars = 4,
        include_pair_signature_ids = false,
        include_component_members = false,
        full_scan_without_index = false
    ))]
    fn new(
        py: Python<'_>,
        paths: &Bound<'_, PyDict>,
        query_signature_ids: Vec<String>,
        top_k: usize,
        query_view: &str,
        orcid_enabled: bool,
        num_threads: Option<usize>,
        max_exemplars: usize,
        include_pair_signature_ids: bool,
        include_component_members: bool,
        full_scan_without_index: bool,
    ) -> PyResult<Self> {
        validate_retrieval_rank_top_k(top_k)?;
        validate_raw_arrow_query_signature_ids(&query_signature_ids)?;
        let planner_query_signature_id_set = query_signature_ids.iter().cloned().collect();
        let paths = RawArrowPlannerPaths::from_py_dict(paths)?;

        let read_cluster_seeds_start = Instant::now();
        let (component_order, members_by_component) =
            read_raw_arrow_cluster_seeds(&paths.cluster_seeds_path)?;
        let read_cluster_seeds_secs = read_cluster_seeds_start.elapsed().as_secs_f64();
        let cluster_seed_disallows = match paths.cluster_seed_disallows_path.as_ref() {
            Some(path) => read_raw_arrow_cluster_seed_disallows(path)?,
            None => HashSet::new(),
        };

        let mut seed_signature_ids = Vec::<String>::new();
        let mut seed_seen = HashSet::<String>::new();
        let mut component_keys_by_member = HashMap::<String, Vec<String>>::new();
        for component_key in component_order.iter() {
            if let Some(members) = members_by_component.get(component_key) {
                for signature_id in members {
                    component_keys_by_member
                        .entry(signature_id.clone())
                        .or_default()
                        .push(component_key.clone());
                    if seed_seen.insert(signature_id.clone()) {
                        seed_signature_ids.push(signature_id.clone());
                    }
                }
            }
        }
        let seed_signature_id_set: HashSet<String> = seed_signature_ids.iter().cloned().collect();

        let read_signatures_start = Instant::now();
        let (signatures, signature_index_stats) = if seed_signature_id_set.is_empty() {
            (HashMap::new(), IndexedArrowReadStats::default())
        } else {
            read_raw_arrow_signatures_with_optional_index(
                &paths.signatures_path,
                paths.signatures_batch_index_path.as_deref(),
                Some(&seed_signature_id_set),
                full_scan_without_index,
            )?
        };
        let read_signatures_secs = read_signatures_start.elapsed().as_secs_f64();
        for signature_id in seed_signature_ids.iter() {
            if !signatures.contains_key(signature_id) {
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "cluster seed signature_id '{}' is missing from signatures Arrow input",
                    signature_id
                )));
            }
        }

        let needed_paper_ids: HashSet<String> = seed_signature_ids
            .iter()
            .filter_map(|signature_id| signatures.get(signature_id))
            .map(|signature| signature.paper_id.clone())
            .collect();
        let metadata_reads_parallel_start = Instant::now();
        let (
            (papers_result, paper_authors_result),
            (raw_specter_by_paper_id_result, raw_name_counts_result),
        ) = py.allow_threads(|| {
            let compute = || {
                rayon::join(
                    || {
                        rayon::join(
                            || -> PyResult<(
                                HashMap<String, RawArrowPaper>,
                                IndexedArrowReadStats,
                                f64,
                            )> {
                                let start = Instant::now();
                                let (loaded, stats) = if needed_paper_ids.is_empty() {
                                    (HashMap::new(), IndexedArrowReadStats::default())
                                } else {
                                    read_raw_arrow_papers_with_optional_index(
                                        &paths.papers_path,
                                        paths.papers_batch_index_path.as_deref(),
                                        &needed_paper_ids,
                                        full_scan_without_index,
                                    )?
                                };
                                Ok((loaded, stats, start.elapsed().as_secs_f64()))
                            },
                            || -> PyResult<(
                                HashMap<String, Vec<(i64, String)>>,
                                IndexedArrowReadStats,
                                f64,
                            )> {
                                let start = Instant::now();
                                let (loaded, stats) = if needed_paper_ids.is_empty() {
                                    (HashMap::new(), IndexedArrowReadStats::default())
                                } else {
                                    read_raw_arrow_paper_authors_with_optional_index(
                                        &paths.paper_authors_path,
                                        paths.paper_authors_batch_index_path.as_deref(),
                                        &needed_paper_ids,
                                        full_scan_without_index,
                                    )?
                                };
                                Ok((loaded, stats, start.elapsed().as_secs_f64()))
                            },
                        )
                    },
                    || {
                        rayon::join(
                            || -> PyResult<(
                                Option<HashMap<String, Vec<f32>>>,
                                IndexedArrowReadStats,
                                f64,
                            )> {
                                let start = Instant::now();
                                let (loaded, stats) = match paths.specter_path.as_ref() {
                                    Some(path) if !needed_paper_ids.is_empty() => {
                                        let (loaded, stats) =
                                            read_raw_arrow_specter_with_optional_index(
                                                path,
                                                paths.specter_batch_index_path.as_deref(),
                                                &needed_paper_ids,
                                                full_scan_without_index,
                                            )?;
                                        (loaded, stats)
                                    }
                                    Some(_) | None => {
                                        (HashMap::new(), IndexedArrowReadStats::default())
                                    }
                                };
                                Ok((
                                    if paths.specter_path.is_some() {
                                        Some(loaded)
                                    } else {
                                        None
                                    },
                                    stats,
                                    start.elapsed().as_secs_f64(),
                                ))
                            },
                            || -> PyResult<(RawNameCountMaps, f64)> {
                                let start = Instant::now();
                                let loaded = match paths.name_counts_index_path.as_ref() {
                                    Some(path) => read_raw_name_counts_index(path)?,
                                    None => match paths.name_counts_arrow_path.as_ref() {
                                        Some(path) => {
                                            return Err(pyo3::exceptions::PyValueError::new_err(
                                                format!(
                                                    "name_counts Arrow path '{path}' requires name_counts_index; refusing slow Arrow fallback"
                                                ),
                                            ));
                                        }
                                        None => RawNameCountMaps::default(),
                                    },
                                };
                                Ok((loaded, start.elapsed().as_secs_f64()))
                            },
                        )
                    },
                )
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
        let (papers, paper_index_stats, read_papers_secs) = papers_result?;
        let (paper_authors, paper_author_index_stats, read_paper_authors_secs) =
            paper_authors_result?;
        let (raw_specter_by_paper_id, specter_index_stats, read_specter_secs) =
            raw_specter_by_paper_id_result?;
        let (raw_name_counts, read_name_counts_secs) = raw_name_counts_result?;
        let specter_by_paper_id = raw_specter_by_paper_id.map(|values| {
            values
                .into_iter()
                .map(|(paper_id, vector)| (paper_id, Arc::new(vector)))
                .collect::<HashMap<_, _>>()
        });

        let text_context_start = Instant::now();
        let text_module = py.import("s2and.text")?;
        let unidecode = text_module.getattr("unidecode")?;
        let name_prefixes = extract_required_string_set(&text_module.getattr("NAME_PREFIXES")?)?;
        let affiliation_stopwords = extract_affiliation_stopwords(py)?;
        let mut unidecode_char_map: HashMap<char, String> = HashMap::new();
        ensure_unidecode_for_raw_arrow_inputs(
            &unidecode,
            &signatures,
            &papers,
            &paper_authors,
            &mut unidecode_char_map,
        )?;
        let text_context_secs = text_context_start.elapsed().as_secs_f64();

        let feature_start = Instant::now();
        let raw_feature_results: Vec<Result<(String, RawArrowFeature), String>> =
            py.allow_threads(|| {
                let compute = || {
                    seed_signature_ids
                        .par_iter()
                        .map(|signature_id| {
                            let signature = signatures.get(signature_id).ok_or_else(|| {
                                format!(
                                    "signature_id '{}' is missing from signatures",
                                    signature_id
                                )
                            })?;
                            let paper = papers.get(&signature.paper_id);
                            let authors = paper_authors.get(&signature.paper_id);
                            Ok((
                                signature_id.clone(),
                                build_raw_arrow_feature(
                                    signature,
                                    paper,
                                    authors,
                                    specter_by_paper_id.as_ref(),
                                    &raw_name_counts,
                                    &name_prefixes,
                                    &affiliation_stopwords,
                                    &unidecode_char_map,
                                    orcid_enabled,
                                ),
                            ))
                        })
                        .collect::<Vec<_>>()
                };
                install_with_optional_rayon_pool(num_threads, compute)
            });
        let mut features_by_signature_id = HashMap::with_capacity(raw_feature_results.len());
        for result in raw_feature_results {
            let (signature_id, feature) = result.map_err(pyo3::exceptions::PyKeyError::new_err)?;
            features_by_signature_id.insert(signature_id, feature);
        }
        let feature_secs = feature_start.elapsed().as_secs_f64();

        let summary_start = Instant::now();
        let retriever = build_retriever_from_raw_arrow_components(
            py,
            &component_order,
            &members_by_component,
            &features_by_signature_id,
            max_exemplars,
            num_threads,
        )?;
        let summary_secs = summary_start.elapsed().as_secs_f64();

        let component_members_start = Instant::now();
        let (_component_member_indices, flat_seed_signature_ids, _seed_component_keys) =
            raw_arrow_component_member_indices_for_batch(
                &component_order,
                &members_by_component,
                0,
            )?;
        let component_members_secs = component_members_start.elapsed().as_secs_f64();

        let indexed_arrow_candidate_plan = paths.signatures_batch_index_path.is_some()
            || paths.papers_batch_index_path.is_some()
            || paths.paper_authors_batch_index_path.is_some()
            || paths.specter_batch_index_path.is_some();

        Ok(Self {
            paths,
            state: ReusableRawArrowCandidatePlanState {
                features_by_signature_id,
                signatures,
                paper_authors,
                raw_name_counts,
                members_by_component,
                component_keys_by_member,
                summary_signals_by_component: HashMap::new(),
                retriever,
                component_order,
                seed_signature_ids: flat_seed_signature_ids,
                seed_signature_id_set,
                cluster_seed_disallows,
                unidecode_char_map,
                name_prefixes,
                affiliation_stopwords,
                seed_paper_count: needed_paper_ids.len(),
                seed_specter_count: specter_by_paper_id.as_ref().map_or(0usize, HashMap::len),
                build_telemetry: RawArrowPlannerBuildTelemetry {
                    read_cluster_seeds_secs,
                    read_signatures_secs,
                    read_papers_secs,
                    read_paper_authors_secs,
                    read_specter_secs,
                    read_name_counts_secs,
                    metadata_reads_parallel_secs: metadata_reads_parallel_start
                        .elapsed()
                        .as_secs_f64(),
                    text_context_secs,
                    feature_secs,
                    summary_secs,
                    component_members_secs,
                    signature_index_stats,
                    paper_index_stats,
                    paper_author_index_stats,
                    specter_index_stats,
                    indexed_arrow_candidate_plan,
                },
            },
            planner_query_signature_count: query_signature_ids.len(),
            planner_query_signature_id_set,
            top_k,
            query_view: query_view.to_string(),
            orcid_enabled,
            num_threads,
            max_exemplars,
            include_pair_signature_ids,
            include_component_members,
            full_scan_without_index,
        })
    }

    fn build_telemetry(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let timings = PyDict::new(py);
        let telemetry = &self.state.build_telemetry;
        timings.set_item("read_cluster_seeds_secs", telemetry.read_cluster_seeds_secs)?;
        timings.set_item("read_signatures_secs", telemetry.read_signatures_secs)?;
        timings.set_item("read_papers_secs", telemetry.read_papers_secs)?;
        timings.set_item("read_paper_authors_secs", telemetry.read_paper_authors_secs)?;
        timings.set_item("read_specter_secs", telemetry.read_specter_secs)?;
        timings.set_item("read_name_counts_secs", telemetry.read_name_counts_secs)?;
        timings.set_item(
            "metadata_reads_parallel_secs",
            telemetry.metadata_reads_parallel_secs,
        )?;
        timings.set_item("text_context_secs", telemetry.text_context_secs)?;
        timings.set_item("feature_secs", telemetry.feature_secs)?;
        timings.set_item("summary_secs", telemetry.summary_secs)?;
        timings.set_item("component_members_secs", telemetry.component_members_secs)?;

        let payload = PyDict::new(py);
        payload.set_item("signature_count", self.state.signatures.len())?;
        payload.set_item("paper_count", self.state.seed_paper_count)?;
        payload.set_item("paper_author_paper_count", self.state.paper_authors.len())?;
        payload.set_item("cluster_count", self.state.component_order.len())?;
        payload.set_item("seed_signature_count", self.state.seed_signature_ids.len())?;
        payload.set_item("query_signature_count", self.planner_query_signature_count)?;
        payload.set_item(
            "cluster_seed_disallow_pair_count",
            self.state.cluster_seed_disallows.len(),
        )?;
        payload.set_item("specter_count", self.state.seed_specter_count)?;
        payload.set_item(
            "indexed_arrow_candidate_plan",
            telemetry.indexed_arrow_candidate_plan,
        )?;
        payload.set_item(
            "signature_batches_read",
            telemetry.signature_index_stats.batches_read,
        )?;
        payload.set_item(
            "signature_rows_scanned",
            telemetry.signature_index_stats.rows_scanned,
        )?;
        payload.set_item(
            "paper_batches_read",
            telemetry.paper_index_stats.batches_read,
        )?;
        payload.set_item(
            "paper_rows_scanned",
            telemetry.paper_index_stats.rows_scanned,
        )?;
        payload.set_item(
            "paper_author_batches_read",
            telemetry.paper_author_index_stats.batches_read,
        )?;
        payload.set_item(
            "paper_author_rows_scanned",
            telemetry.paper_author_index_stats.rows_scanned,
        )?;
        payload.set_item(
            "specter_batches_read",
            telemetry.specter_index_stats.batches_read,
        )?;
        payload.set_item(
            "specter_rows_scanned",
            telemetry.specter_index_stats.rows_scanned,
        )?;
        payload.set_item("unidecode_char_count", self.state.unidecode_char_map.len())?;
        payload.set_item("planner_seed_state", 1)?;
        payload.set_item("timings", timings)?;
        Ok(payload.unbind())
    }

    #[pyo3(signature = (
        query_signature_ids,
        top_k = None,
        query_view = None,
        include_pair_signature_ids = None,
        include_component_members = None
    ))]
    fn plan(
        &mut self,
        py: Python<'_>,
        query_signature_ids: Vec<String>,
        top_k: Option<usize>,
        query_view: Option<String>,
        include_pair_signature_ids: Option<bool>,
        include_component_members: Option<bool>,
    ) -> PyResult<Py<PyDict>> {
        validate_raw_arrow_query_signature_ids(&query_signature_ids)?;
        if let Some(value) = top_k {
            if value != self.top_k {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "RawBlockQueryCandidatePlanner was built with top_k={}, got plan top_k={value}",
                    self.top_k
                )));
            }
        }
        if let Some(value) = query_view.as_ref() {
            if value != &self.query_view {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "RawBlockQueryCandidatePlanner was built with query_view={:?}, got plan query_view={value:?}",
                    self.query_view
                )));
            }
        }
        if let Some(value) = include_pair_signature_ids {
            if value != self.include_pair_signature_ids {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "RawBlockQueryCandidatePlanner was built with include_pair_signature_ids={}, got {value}",
                    self.include_pair_signature_ids
                )));
            }
        }
        if let Some(value) = include_component_members {
            if value != self.include_component_members {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "RawBlockQueryCandidatePlanner was built with include_component_members={}, got {value}",
                    self.include_component_members
                )));
            }
        }
        let missing: Vec<&String> = query_signature_ids
            .iter()
            .filter(|signature_id| !self.planner_query_signature_id_set.contains(*signature_id))
            .collect();
        if !missing.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "RawBlockQueryCandidatePlanner plan requested query_signature_ids outside the planner query set: {:?}",
                missing.into_iter().take(10).collect::<Vec<_>>()
            )));
        }

        let total_start = Instant::now();
        let query_inputs = read_reusable_raw_arrow_query_inputs(
            py,
            &self.paths,
            &query_signature_ids,
            self.num_threads,
            self.full_scan_without_index,
        )?;

        let text_context_start = Instant::now();
        let text_module = py.import("s2and.text")?;
        let unidecode = text_module.getattr("unidecode")?;
        ensure_unidecode_for_raw_arrow_inputs(
            &unidecode,
            &query_inputs.signatures,
            &query_inputs.papers,
            &query_inputs.paper_authors,
            &mut self.state.unidecode_char_map,
        )?;
        let text_context_secs = text_context_start.elapsed().as_secs_f64();

        let feature_start = Instant::now();
        let raw_feature_results: Vec<Result<(String, RawArrowFeature), String>> =
            py.allow_threads(|| {
                let compute = || {
                    query_signature_ids
                        .par_iter()
                        .map(|signature_id| {
                            let signature =
                                query_inputs.signatures.get(signature_id).ok_or_else(|| {
                                    format!(
                                        "signature_id '{}' is missing from signatures",
                                        signature_id
                                    )
                                })?;
                            let paper = query_inputs.papers.get(&signature.paper_id);
                            let authors = query_inputs.paper_authors.get(&signature.paper_id);
                            Ok((
                                signature_id.clone(),
                                build_raw_arrow_feature(
                                    signature,
                                    paper,
                                    authors,
                                    query_inputs.specter_by_paper_id.as_ref(),
                                    &self.state.raw_name_counts,
                                    &self.state.name_prefixes,
                                    &self.state.affiliation_stopwords,
                                    &self.state.unidecode_char_map,
                                    self.orcid_enabled,
                                ),
                            ))
                        })
                        .collect::<Vec<_>>()
                };
                install_with_optional_rayon_pool(self.num_threads, compute)
            });
        let mut query_features_by_signature_id = HashMap::with_capacity(raw_feature_results.len());
        for result in raw_feature_results {
            let (signature_id, feature) = result.map_err(pyo3::exceptions::PyKeyError::new_err)?;
            query_features_by_signature_id.insert(signature_id, feature);
        }
        let feature_secs = feature_start.elapsed().as_secs_f64();

        let query_start = Instant::now();
        let mut queries = Vec::<RetrievalQueryData>::with_capacity(query_signature_ids.len());
        let mut query_views = Vec::<String>::with_capacity(query_signature_ids.len());
        let mut query_first_tokens = Vec::<String>::with_capacity(query_signature_ids.len());
        let mut query_authors = Vec::<String>::with_capacity(query_signature_ids.len());
        for signature_id in query_signature_ids.iter() {
            let base_feature = &query_features_by_signature_id[signature_id];
            let base = &base_feature.query;
            let (masked, resolved_view) = mask_raw_arrow_query(base, &self.query_view)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            queries.push(masked);
            query_views.push(resolved_view);
            query_first_tokens.push(base.first.clone());
            query_authors.push(base_feature.query_author.clone());
        }
        let query_secs = query_start.elapsed().as_secs_f64();

        let query_signature_id_set: HashSet<&str> =
            query_signature_ids.iter().map(String::as_str).collect();
        let overlapping_query_seed_ids = query_signature_ids
            .iter()
            .filter(|signature_id| self.state.seed_signature_id_set.contains(*signature_id))
            .take(10)
            .cloned()
            .collect::<Vec<_>>();
        if query_signature_ids.len() > 1 && !overlapping_query_seed_ids.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "RawBlockQueryCandidatePlanner.plan requires singleton query windows when query ids are also \
                 cluster seed members; overlapping query_signature_ids={overlapping_query_seed_ids:?}"
            )));
        }
        let mut excluded_query_seed_count = 0usize;
        let mut filtered_component_order = Vec::<String>::new();
        let mut filtered_members_by_component = HashMap::<String, Vec<String>>::new();
        let needs_filtered_retriever = !overlapping_query_seed_ids.is_empty();
        let summary_start = Instant::now();
        let filtered_retriever = if needs_filtered_retriever {
            for component_key in self.state.component_order.iter() {
                let members = self
                    .state
                    .members_by_component
                    .get(component_key)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "component_key '{}' disappeared while filtering query seeds",
                            component_key
                        ))
                    })?;
                let mut filtered_members = Vec::with_capacity(members.len());
                for signature_id in members {
                    if query_signature_id_set.contains(signature_id.as_str()) {
                        excluded_query_seed_count += 1;
                    } else {
                        filtered_members.push(signature_id.clone());
                    }
                }
                if !filtered_members.is_empty() {
                    filtered_component_order.push(component_key.clone());
                    filtered_members_by_component.insert(component_key.clone(), filtered_members);
                }
            }
            Some(build_retriever_from_raw_arrow_components(
                py,
                &filtered_component_order,
                &filtered_members_by_component,
                &self.state.features_by_signature_id,
                self.max_exemplars,
                self.num_threads,
            )?)
        } else {
            None
        };
        let summary_secs = summary_start.elapsed().as_secs_f64();
        let (component_order, members_by_component, retriever) =
            if let Some(retriever) = filtered_retriever.as_ref() {
                (
                    &filtered_component_order,
                    &filtered_members_by_component,
                    retriever,
                )
            } else {
                (
                    &self.state.component_order,
                    &self.state.members_by_component,
                    &self.state.retriever,
                )
            };

        let component_members_start = Instant::now();
        let (component_member_indices, seed_signature_ids, _seed_component_keys) =
            raw_arrow_component_member_indices_for_batch(
                component_order,
                members_by_component,
                query_signature_ids.len(),
            )?;
        let component_members_secs = component_members_start.elapsed().as_secs_f64();

        let (excluded_candidate_indices_by_query, cluster_seed_disallowed_candidate_count) =
            raw_arrow_excluded_candidate_indices_by_query(
                &query_signature_ids,
                component_order,
                &self.state.component_keys_by_member,
                &self.state.cluster_seed_disallows,
            )?;

        let retrieval_start = Instant::now();
        let query_results: Vec<Result<RetrievalPairPlanQueryResult, String>> =
            py.allow_threads(|| {
                let compute = || {
                    queries
                        .par_iter()
                        .enumerate()
                        .map(|(query_offset, current_query)| {
                            let query_index = u32::try_from(query_offset)
                                .map_err(|_| "query index exceeds u32".to_string())?;
                            let excluded_candidate_indices = excluded_candidate_indices_by_query
                                .as_ref()
                                .and_then(|values| values[query_offset].as_ref());
                            retriever.build_pair_plan_query_result(
                                current_query,
                                query_first_tokens[query_offset].as_str(),
                                query_index,
                                None,
                                excluded_candidate_indices,
                                Some(query_signature_ids[query_offset].as_str()),
                                &component_member_indices,
                                self.top_k,
                                None,
                                0,
                                true,
                            )
                        })
                        .collect::<Vec<_>>()
                };
                install_with_optional_rayon_pool(self.num_threads, compute)
            });

        let mut row_query_signature_indices = Vec::<u32>::new();
        let mut row_component_keys = Vec::<String>::new();
        let mut row_retrieval_scores = Vec::<f32>::new();
        let mut row_retrieval_ranks = Vec::<u16>::new();
        let mut row_component_sizes = Vec::<u32>::new();
        let mut row_named_signature_counts = Vec::<u32>::new();
        let mut row_dominant_first_names = Vec::<String>::new();
        let mut row_candidate_year_min = Vec::<i32>::new();
        let mut row_candidate_year_max = Vec::<i32>::new();
        let mut row_candidate_year_range_missing = Vec::<u8>::new();
        let mut row_query_first_tokens = Vec::<String>::new();
        let mut row_query_years = Vec::<i32>::new();
        let mut row_query_year_missing = Vec::<u8>::new();
        let mut row_query_has_affiliations = Vec::<u8>::new();
        let mut row_query_has_coauthors = Vec::<u8>::new();
        let mut row_orcid_match = Vec::<u8>::new();
        let mut row_middle_initial_compatibility = Vec::<f32>::new();
        let mut row_affiliation_overlap = Vec::<f32>::new();
        let mut row_coauthor_overlap = Vec::<f32>::new();
        let mut row_venue_overlap = Vec::<f32>::new();
        let mut row_year_compatibility = Vec::<f32>::new();
        let mut row_title_overlap = Vec::<f32>::new();
        let mut row_specter_centroid_similarity = Vec::<f32>::new();
        let mut row_specter_exemplar_similarity = Vec::<f32>::new();
        let mut row_last_name_count_min_rarity = Vec::<f32>::new();
        let mut row_candidate_last_name_count_min_rarity = Vec::<f32>::new();
        let mut row_candidate_last_first_name_count_min_rarity = Vec::<f32>::new();
        let mut row_last_first_name_count_min_rarity = Vec::<f32>::new();
        let mut row_first_prefix_x_last_first_name_count_min_rarity = Vec::<f32>::new();
        let mut row_candidate_cluster_max_paper_author_count = Vec::<f32>::new();
        let mut row_paper_author_list_max_jaccard = Vec::<f32>::new();
        let mut row_paper_author_list_max_containment = Vec::<f32>::new();
        let mut row_paper_author_list_max_overlap_count = Vec::<f32>::new();
        let mut row_local_author_window10_jaccard_max = Vec::<f32>::new();
        let mut row_local_author_window10_overlap_count_max = Vec::<f32>::new();
        let mut row_best_author_count_log_absdiff = Vec::<f32>::new();
        let mut left_signature_indices = Vec::<u32>::new();
        let mut right_signature_indices = Vec::<u32>::new();
        let mut pair_row_indices = Vec::<u32>::new();
        let mut filtered_summary_signals_by_component =
            HashMap::<String, RawArrowSummarySignalData>::new();
        let mut author_signals_by_query_signature_id =
            HashMap::<String, RawArrowAuthorSignalData>::new();
        for query_result in query_results {
            let mut query_result = query_result.map_err(pyo3::exceptions::PyKeyError::new_err)?;
            let base_row_index = u32::try_from(row_component_keys.len()).map_err(|_| {
                pyo3::exceptions::PyOverflowError::new_err(
                    "retrieved candidate row count exceeds u32",
                )
            })?;
            for (local_row_index, member_indices) in query_result
                .right_signature_indices_by_row
                .iter()
                .enumerate()
            {
                let row_index = checked_retrieved_row_index(base_row_index, local_row_index)?;
                let query_signature_index =
                    query_result.row_query_signature_indices[local_row_index];
                for member_index in member_indices.iter() {
                    left_signature_indices.push(query_signature_index);
                    right_signature_indices.push(*member_index);
                    pair_row_indices.push(row_index);
                }
            }
            for (local_row_index, component_key) in
                query_result.row_component_keys.iter().enumerate()
            {
                let query_offset =
                    query_result.row_query_signature_indices[local_row_index] as usize;
                let query_signature_id =
                    query_signature_ids.get(query_offset).ok_or_else(|| {
                        pyo3::exceptions::PyIndexError::new_err(format!(
                            "row query signature offset {} is outside query_signature_ids",
                            query_offset
                        ))
                    })?;
                let query_feature = query_features_by_signature_id
                    .get(query_signature_id)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "query signature_id '{}' is missing from computed raw Arrow features",
                            query_signature_id
                        ))
                    })?;
                let query = queries.get(query_offset).ok_or_else(|| {
                    pyo3::exceptions::PyIndexError::new_err(format!(
                        "row query signature offset {} is outside query feature table",
                        query_offset
                    ))
                })?;
                let component_index = retriever
                    .component_index_by_key
                    .get(component_key)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "component_key '{}' disappeared while building row signals",
                            component_key
                        ))
                    })?;
                let summary = retriever.summaries.get(*component_index).ok_or_else(|| {
                    pyo3::exceptions::PyIndexError::new_err(format!(
                        "component index {} is outside summary table",
                        component_index
                    ))
                })?;
                let summary_signals = if needs_filtered_retriever {
                    raw_arrow_summary_signals_cached(
                        &mut filtered_summary_signals_by_component,
                        component_key,
                        members_by_component,
                        &self.state.features_by_signature_id,
                        &self.state.signatures,
                        &self.state.paper_authors,
                        &self.state.unidecode_char_map,
                    )?
                } else {
                    raw_arrow_summary_signals_cached(
                        &mut self.state.summary_signals_by_component,
                        component_key,
                        members_by_component,
                        &self.state.features_by_signature_id,
                        &self.state.signatures,
                        &self.state.paper_authors,
                        &self.state.unidecode_char_map,
                    )?
                };
                let rarity = raw_arrow_name_count_rarity_row(
                    query,
                    &query_feature.name_counts,
                    summary,
                    summary_signals,
                );
                if let Entry::Vacant(entry) =
                    author_signals_by_query_signature_id.entry(query_signature_id.clone())
                {
                    let query_signature = query_inputs
                        .signatures
                        .get(query_signature_id)
                        .ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err(format!(
                                "query signature_id '{}' is missing from signatures",
                                query_signature_id
                            ))
                        })?;
                    let author_signals = build_raw_arrow_author_signal_data(
                        query_signature,
                        query_inputs.paper_authors.get(&query_signature.paper_id),
                        &self.state.unidecode_char_map,
                    );
                    entry.insert(author_signals);
                }
                let query_author_signals = author_signals_by_query_signature_id
                    .get(query_signature_id)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "query signature_id '{}' is missing raw author signals",
                            query_signature_id
                        ))
                    })?;
                let evidence = raw_arrow_paper_evidence_row(
                    query_signature_id,
                    query_feature.paper_author_count,
                    query_author_signals,
                    summary_signals,
                );
                row_last_name_count_min_rarity.push(rarity.last_name_count_min_rarity);
                row_candidate_last_name_count_min_rarity
                    .push(rarity.candidate_last_name_count_min_rarity);
                row_candidate_last_first_name_count_min_rarity
                    .push(rarity.candidate_last_first_name_count_min_rarity);
                row_last_first_name_count_min_rarity.push(rarity.last_first_name_count_min_rarity);
                row_first_prefix_x_last_first_name_count_min_rarity
                    .push(rarity.first_prefix_x_last_first_name_count_min_rarity);
                row_candidate_cluster_max_paper_author_count
                    .push(summary.max_paper_author_count as f32);
                row_paper_author_list_max_jaccard.push(evidence.paper_author_list_max_jaccard);
                row_paper_author_list_max_containment
                    .push(evidence.paper_author_list_max_containment);
                row_paper_author_list_max_overlap_count
                    .push(evidence.paper_author_list_max_overlap_count);
                row_local_author_window10_jaccard_max
                    .push(evidence.local_author_window10_jaccard_max);
                row_local_author_window10_overlap_count_max
                    .push(evidence.local_author_window10_overlap_count_max);
                row_best_author_count_log_absdiff.push(evidence.best_author_count_log_absdiff);
            }
            row_query_signature_indices.append(&mut query_result.row_query_signature_indices);
            row_component_keys.append(&mut query_result.row_component_keys);
            row_retrieval_scores.append(&mut query_result.row_retrieval_scores);
            row_retrieval_ranks.append(&mut query_result.row_retrieval_ranks);
            row_component_sizes.append(&mut query_result.row_component_sizes);
            row_named_signature_counts.append(&mut query_result.row_named_signature_counts);
            row_dominant_first_names.append(&mut query_result.row_dominant_first_names);
            row_candidate_year_min.append(&mut query_result.row_candidate_year_min);
            row_candidate_year_max.append(&mut query_result.row_candidate_year_max);
            row_candidate_year_range_missing
                .append(&mut query_result.row_candidate_year_range_missing);
            row_query_first_tokens.append(&mut query_result.row_query_first_tokens);
            row_query_years.append(&mut query_result.row_query_years);
            row_query_year_missing.append(&mut query_result.row_query_year_missing);
            row_query_has_affiliations.append(&mut query_result.row_query_has_affiliations);
            row_query_has_coauthors.append(&mut query_result.row_query_has_coauthors);
            row_orcid_match.append(&mut query_result.row_orcid_match);
            row_middle_initial_compatibility
                .append(&mut query_result.row_middle_initial_compatibility);
            row_affiliation_overlap.append(&mut query_result.row_affiliation_overlap);
            row_coauthor_overlap.append(&mut query_result.row_coauthor_overlap);
            row_venue_overlap.append(&mut query_result.row_venue_overlap);
            row_year_compatibility.append(&mut query_result.row_year_compatibility);
            row_title_overlap.append(&mut query_result.row_title_overlap);
            row_specter_centroid_similarity
                .append(&mut query_result.row_specter_centroid_similarity);
            row_specter_exemplar_similarity
                .append(&mut query_result.row_specter_exemplar_similarity);
        }
        let retrieval_secs = retrieval_start.elapsed().as_secs_f64();

        let pair_signature_ids_start = Instant::now();
        let mut left_signature_ids: Option<Vec<String>> = None;
        let mut right_signature_ids: Option<Vec<String>> = None;
        if self.include_pair_signature_ids {
            let mut left_ids = Vec::<String>::with_capacity(left_signature_indices.len());
            let mut right_ids = Vec::<String>::with_capacity(right_signature_indices.len());
            let signature_index_count = query_signature_ids.len() + seed_signature_ids.len();
            let signature_id_for_index = |index: u32| -> Option<&String> {
                let offset = index as usize;
                if offset < query_signature_ids.len() {
                    query_signature_ids.get(offset)
                } else {
                    seed_signature_ids.get(offset - query_signature_ids.len())
                }
            };
            for index in left_signature_indices.iter() {
                let Some(signature_id) = signature_id_for_index(*index) else {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "left signature index {} is outside signature id table of length {}",
                        index, signature_index_count
                    )));
                };
                left_ids.push(signature_id.clone());
            }
            for index in right_signature_indices.iter() {
                let Some(signature_id) = signature_id_for_index(*index) else {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "right signature index {} is outside signature id table of length {}",
                        index, signature_index_count
                    )));
                };
                right_ids.push(signature_id.clone());
            }
            left_signature_ids = Some(left_ids);
            right_signature_ids = Some(right_ids);
        }
        let pair_signature_ids_secs = pair_signature_ids_start.elapsed().as_secs_f64();

        let mut payload_seed_signature_ids = if self.include_pair_signature_ids {
            Vec::new()
        } else {
            seed_signature_ids.clone()
        };
        if !self.include_pair_signature_ids {
            let query_signature_count = query_signature_ids.len();
            let mut compact_seed_signature_ids = Vec::<String>::new();
            let mut seed_index_remap = HashMap::<usize, u32>::new();
            for signature_index in left_signature_indices
                .iter_mut()
                .chain(right_signature_indices.iter_mut())
            {
                let old_offset = *signature_index as usize;
                if old_offset < query_signature_count {
                    continue;
                }
                let old_seed_offset = old_offset - query_signature_count;
                if old_seed_offset >= seed_signature_ids.len() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "signature index {} is outside signature id table of length {}",
                        old_offset,
                        query_signature_count + seed_signature_ids.len()
                    )));
                }
                let new_offset = match seed_index_remap.entry(old_seed_offset) {
                    Entry::Occupied(entry) => *entry.get(),
                    Entry::Vacant(entry) => {
                        let new_offset =
                            u32::try_from(query_signature_count + compact_seed_signature_ids.len())
                                .map_err(|_| {
                                    pyo3::exceptions::PyOverflowError::new_err(
                                        "compact raw Arrow signature index exceeds u32",
                                    )
                                })?;
                        compact_seed_signature_ids
                            .push(seed_signature_ids[old_seed_offset].clone());
                        *entry.insert(new_offset)
                    }
                };
                *signature_index = new_offset;
            }
            payload_seed_signature_ids = compact_seed_signature_ids;
        }

        let component_members_payload_start = Instant::now();
        let component_members = if self.include_component_members {
            let component_members = PyDict::new(py);
            for component_key in component_order.iter() {
                component_members.set_item(
                    component_key,
                    members_by_component
                        .get(component_key)
                        .cloned()
                        .unwrap_or_default(),
                )?;
            }
            Some(component_members)
        } else {
            None
        };
        let component_members_payload_secs =
            component_members_payload_start.elapsed().as_secs_f64();

        let timings = PyDict::new(py);
        timings.set_item("read_cluster_seeds_secs", 0.0)?;
        timings.set_item("read_signatures_secs", query_inputs.read_signatures_secs)?;
        timings.set_item("read_papers_secs", query_inputs.read_papers_secs)?;
        timings.set_item(
            "read_paper_authors_secs",
            query_inputs.read_paper_authors_secs,
        )?;
        timings.set_item("read_specter_secs", query_inputs.read_specter_secs)?;
        timings.set_item("read_name_counts_secs", 0.0)?;
        timings.set_item(
            "metadata_reads_parallel_secs",
            query_inputs.metadata_reads_parallel_secs,
        )?;
        timings.set_item("text_context_secs", text_context_secs)?;
        timings.set_item("feature_secs", feature_secs)?;
        timings.set_item("query_secs", query_secs)?;
        timings.set_item("summary_secs", summary_secs)?;
        timings.set_item("component_members_secs", component_members_secs)?;
        timings.set_item("retrieval_secs", retrieval_secs)?;
        timings.set_item("pair_signature_ids_secs", pair_signature_ids_secs)?;
        timings.set_item(
            "component_members_payload_secs",
            component_members_payload_secs,
        )?;

        let telemetry = PyDict::new(py);
        telemetry.set_item(
            "signature_count",
            query_signature_ids.len() + seed_signature_ids.len(),
        )?;
        let mut telemetry_paper_ids = HashSet::<String>::new();
        for signature_id in query_signature_ids.iter() {
            if let Some(signature) = query_inputs.signatures.get(signature_id) {
                telemetry_paper_ids.insert(signature.paper_id.clone());
            }
        }
        for signature_id in seed_signature_ids.iter() {
            if let Some(signature) = self.state.signatures.get(signature_id) {
                telemetry_paper_ids.insert(signature.paper_id.clone());
            }
        }
        telemetry.set_item("paper_count", telemetry_paper_ids.len())?;
        let mut telemetry_paper_author_ids = HashSet::<String>::new();
        for paper_id in telemetry_paper_ids.iter() {
            if self.state.paper_authors.contains_key(paper_id)
                || query_inputs.paper_authors.contains_key(paper_id)
            {
                telemetry_paper_author_ids.insert(paper_id.clone());
            }
        }
        telemetry.set_item("paper_author_paper_count", telemetry_paper_author_ids.len())?;
        telemetry.set_item("cluster_count", component_order.len())?;
        telemetry.set_item("seed_signature_count", seed_signature_ids.len())?;
        telemetry.set_item("query_signature_count", query_signature_ids.len())?;
        telemetry.set_item("excluded_query_seed_count", excluded_query_seed_count)?;
        telemetry.set_item(
            "cluster_seed_disallow_pair_count",
            self.state.cluster_seed_disallows.len(),
        )?;
        telemetry.set_item(
            "cluster_seed_disallowed_candidate_count",
            cluster_seed_disallowed_candidate_count,
        )?;
        let mut telemetry_specter_ids = HashSet::<String>::new();
        for signature_id in seed_signature_ids.iter() {
            if self
                .state
                .features_by_signature_id
                .get(signature_id)
                .and_then(|feature| feature.query.specter.as_ref())
                .is_some()
            {
                if let Some(signature) = self.state.signatures.get(signature_id) {
                    telemetry_specter_ids.insert(signature.paper_id.clone());
                }
            }
        }
        if let Some(query_specter) = query_inputs.specter_by_paper_id.as_ref() {
            for paper_id in telemetry_paper_ids.iter() {
                if query_specter.contains_key(paper_id) {
                    telemetry_specter_ids.insert(paper_id.clone());
                }
            }
        }
        telemetry.set_item("specter_count", telemetry_specter_ids.len())?;
        telemetry.set_item(
            "indexed_arrow_candidate_plan",
            self.state.build_telemetry.indexed_arrow_candidate_plan,
        )?;
        telemetry.set_item(
            "signature_batches_read",
            query_inputs.signature_index_stats.batches_read,
        )?;
        telemetry.set_item(
            "signature_rows_scanned",
            query_inputs.signature_index_stats.rows_scanned,
        )?;
        telemetry.set_item(
            "paper_batches_read",
            query_inputs.paper_index_stats.batches_read,
        )?;
        telemetry.set_item(
            "paper_rows_scanned",
            query_inputs.paper_index_stats.rows_scanned,
        )?;
        telemetry.set_item(
            "paper_author_batches_read",
            query_inputs.paper_author_index_stats.batches_read,
        )?;
        telemetry.set_item(
            "paper_author_rows_scanned",
            query_inputs.paper_author_index_stats.rows_scanned,
        )?;
        telemetry.set_item(
            "specter_batches_read",
            query_inputs.specter_index_stats.batches_read,
        )?;
        telemetry.set_item(
            "specter_rows_scanned",
            query_inputs.specter_index_stats.rows_scanned,
        )?;
        telemetry.set_item("unidecode_char_count", self.state.unidecode_char_map.len())?;
        telemetry.set_item(
            "payload_seed_signature_count",
            payload_seed_signature_ids.len(),
        )?;
        telemetry.set_item("planner_seed_state_reused", 1)?;
        telemetry.set_item("timings", &timings)?;

        let payload_start = Instant::now();
        let payload = PyDict::new(py);
        payload.set_item("schema_version", "raw_arrow_candidate_plan_v1")?;
        payload.set_item("row_count", row_component_keys.len())?;
        payload.set_item("pair_count", left_signature_indices.len())?;
        payload.set_item("query_signature_ids", query_signature_ids)?;
        payload.set_item("query_views", query_views)?;
        payload.set_item("query_authors", query_authors)?;
        payload.set_item("seed_signature_ids", payload_seed_signature_ids)?;
        if let Some(component_members) = component_members {
            payload.set_item("component_members", component_members)?;
        }
        payload.set_item(
            "left_signature_indices",
            left_signature_indices.to_pyarray(py),
        )?;
        payload.set_item(
            "right_signature_indices",
            right_signature_indices.to_pyarray(py),
        )?;
        if let Some(left_signature_ids) = left_signature_ids {
            payload.set_item("left_signature_ids", left_signature_ids)?;
        }
        if let Some(right_signature_ids) = right_signature_ids {
            payload.set_item("right_signature_ids", right_signature_ids)?;
        }
        payload.set_item("pair_row_indices", pair_row_indices.to_pyarray(py))?;
        payload.set_item(
            "row_query_signature_indices",
            row_query_signature_indices.to_pyarray(py),
        )?;
        payload.set_item("row_component_keys", row_component_keys)?;
        payload.set_item("retrieval_scores", row_retrieval_scores.to_pyarray(py))?;
        payload.set_item("retrieval_ranks", row_retrieval_ranks.to_pyarray(py))?;
        payload.set_item("row_component_sizes", row_component_sizes.to_pyarray(py))?;
        payload.set_item(
            "row_named_signature_counts",
            row_named_signature_counts.to_pyarray(py),
        )?;
        payload.set_item("row_dominant_first_names", row_dominant_first_names)?;
        payload.set_item(
            "row_candidate_year_min",
            row_candidate_year_min.to_pyarray(py),
        )?;
        payload.set_item(
            "row_candidate_year_max",
            row_candidate_year_max.to_pyarray(py),
        )?;
        payload.set_item(
            "row_candidate_year_range_missing",
            row_candidate_year_range_missing.to_pyarray(py),
        )?;
        payload.set_item("row_query_first_tokens", row_query_first_tokens)?;
        payload.set_item("row_query_years", row_query_years.to_pyarray(py))?;
        payload.set_item(
            "row_query_year_missing",
            row_query_year_missing.to_pyarray(py),
        )?;
        payload.set_item(
            "row_query_has_affiliations",
            row_query_has_affiliations.to_pyarray(py),
        )?;
        payload.set_item(
            "row_query_has_coauthors",
            row_query_has_coauthors.to_pyarray(py),
        )?;
        payload.set_item("row_orcid_match", row_orcid_match.to_pyarray(py))?;
        payload.set_item(
            "middle_initial_compatibility",
            row_middle_initial_compatibility.to_pyarray(py),
        )?;
        payload.set_item(
            "affiliation_overlap",
            row_affiliation_overlap.to_pyarray(py),
        )?;
        payload.set_item("coauthor_overlap", row_coauthor_overlap.to_pyarray(py))?;
        payload.set_item("venue_overlap", row_venue_overlap.to_pyarray(py))?;
        payload.set_item("year_compatibility", row_year_compatibility.to_pyarray(py))?;
        payload.set_item("title_overlap", row_title_overlap.to_pyarray(py))?;
        payload.set_item(
            "specter_centroid_similarity",
            row_specter_centroid_similarity.to_pyarray(py),
        )?;
        payload.set_item(
            "specter_exemplar_similarity",
            row_specter_exemplar_similarity.to_pyarray(py),
        )?;
        payload.set_item(
            "row_last_name_count_min_rarity",
            row_last_name_count_min_rarity.to_pyarray(py),
        )?;
        payload.set_item(
            "row_candidate_last_name_count_min_rarity",
            row_candidate_last_name_count_min_rarity.to_pyarray(py),
        )?;
        payload.set_item(
            "row_candidate_last_first_name_count_min_rarity",
            row_candidate_last_first_name_count_min_rarity.to_pyarray(py),
        )?;
        payload.set_item(
            "row_last_first_name_count_min_rarity",
            row_last_first_name_count_min_rarity.to_pyarray(py),
        )?;
        payload.set_item(
            "row_first_prefix_x_last_first_name_count_min_rarity",
            row_first_prefix_x_last_first_name_count_min_rarity.to_pyarray(py),
        )?;
        payload.set_item(
            "row_candidate_cluster_max_paper_author_count",
            row_candidate_cluster_max_paper_author_count.to_pyarray(py),
        )?;
        payload.set_item(
            "row_paper_author_list_max_jaccard",
            row_paper_author_list_max_jaccard.to_pyarray(py),
        )?;
        payload.set_item(
            "row_paper_author_list_max_containment",
            row_paper_author_list_max_containment.to_pyarray(py),
        )?;
        payload.set_item(
            "row_paper_author_list_max_overlap_count",
            row_paper_author_list_max_overlap_count.to_pyarray(py),
        )?;
        payload.set_item(
            "row_local_author_window10_jaccard_max",
            row_local_author_window10_jaccard_max.to_pyarray(py),
        )?;
        payload.set_item(
            "row_local_author_window10_overlap_count_max",
            row_local_author_window10_overlap_count_max.to_pyarray(py),
        )?;
        payload.set_item(
            "row_best_author_count_log_absdiff",
            row_best_author_count_log_absdiff.to_pyarray(py),
        )?;
        payload.set_item("telemetry", telemetry)?;
        timings.set_item("payload_secs", payload_start.elapsed().as_secs_f64())?;
        timings.set_item("total_secs", total_start.elapsed().as_secs_f64())?;
        timings.set_item("drop_secs", 0.0)?;
        timings.set_item("wall_secs", total_start.elapsed().as_secs_f64())?;
        Ok(payload.unbind())
    }
}

fn raw_arrow_labeled_empty_plan(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let telemetry = PyDict::new(py);
    telemetry.set_item("signature_count", 0)?;
    telemetry.set_item("paper_count", 0)?;
    telemetry.set_item("paper_author_paper_count", 0)?;
    telemetry.set_item("component_count", 0)?;
    telemetry.set_item("query_signature_count", 0)?;
    telemetry.set_item("row_count", 0)?;
    telemetry.set_item("pair_count", 0)?;
    let timings = PyDict::new(py);
    timings.set_item("total_secs", 0.0)?;
    telemetry.set_item("timings", timings)?;

    let payload = PyDict::new(py);
    payload.set_item("schema_version", "raw_arrow_labeled_candidate_plan_v1")?;
    payload.set_item("row_count", 0)?;
    payload.set_item("pair_count", 0)?;
    payload.set_item("signature_ids", Vec::<String>::new())?;
    payload.set_item("query_signature_ids", Vec::<String>::new())?;
    payload.set_item("query_views", Vec::<String>::new())?;
    payload.set_item("query_authors", Vec::<String>::new())?;
    payload.set_item("row_query_signature_ids", Vec::<String>::new())?;
    payload.set_item("row_query_views", Vec::<String>::new())?;
    payload.set_item("row_query_authors", Vec::<String>::new())?;
    payload.set_item("row_query_group_ids", Vec::<String>::new())?;
    payload.set_item("row_component_keys", Vec::<String>::new())?;
    payload.set_item("left_signature_ids", Vec::<String>::new())?;
    payload.set_item("right_signature_ids", Vec::<String>::new())?;
    payload.set_item("pair_row_indices", Vec::<u32>::new().to_pyarray(py))?;
    payload.set_item("retrieval_scores", Vec::<f32>::new().to_pyarray(py))?;
    payload.set_item("retrieval_ranks", Vec::<u16>::new().to_pyarray(py))?;
    payload.set_item("row_component_sizes", Vec::<u32>::new().to_pyarray(py))?;
    payload.set_item(
        "row_named_signature_counts",
        Vec::<u32>::new().to_pyarray(py),
    )?;
    payload.set_item("row_dominant_first_names", Vec::<String>::new())?;
    payload.set_item("row_candidate_year_min", Vec::<i32>::new().to_pyarray(py))?;
    payload.set_item("row_candidate_year_max", Vec::<i32>::new().to_pyarray(py))?;
    payload.set_item(
        "row_candidate_year_range_missing",
        Vec::<u8>::new().to_pyarray(py),
    )?;
    payload.set_item("row_query_first_tokens", Vec::<String>::new())?;
    payload.set_item("row_query_years", Vec::<i32>::new().to_pyarray(py))?;
    payload.set_item("row_query_year_missing", Vec::<u8>::new().to_pyarray(py))?;
    payload.set_item(
        "row_query_has_affiliations",
        Vec::<u8>::new().to_pyarray(py),
    )?;
    payload.set_item("row_query_has_coauthors", Vec::<u8>::new().to_pyarray(py))?;
    payload.set_item("row_query_has_specter", Vec::<u8>::new().to_pyarray(py))?;
    payload.set_item("row_query_has_name_counts", Vec::<u8>::new().to_pyarray(py))?;
    payload.set_item(
        "row_candidate_has_affiliations",
        Vec::<u8>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_coauthors",
        Vec::<u8>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_specter_exemplars",
        Vec::<u8>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_name_counts",
        Vec::<u8>::new().to_pyarray(py),
    )?;
    payload.set_item("row_orcid_match", Vec::<u8>::new().to_pyarray(py))?;
    payload.set_item(
        "middle_initial_compatibility",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item("affiliation_overlap", Vec::<f32>::new().to_pyarray(py))?;
    payload.set_item("coauthor_overlap", Vec::<f32>::new().to_pyarray(py))?;
    payload.set_item("venue_overlap", Vec::<f32>::new().to_pyarray(py))?;
    payload.set_item("year_compatibility", Vec::<f32>::new().to_pyarray(py))?;
    payload.set_item("title_overlap", Vec::<f32>::new().to_pyarray(py))?;
    payload.set_item(
        "specter_centroid_similarity",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "specter_exemplar_similarity",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_last_name_count_min_rarity",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_last_name_count_min_rarity",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_last_first_name_count_min_rarity",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_last_first_name_count_min_rarity",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_first_prefix_x_last_first_name_count_min_rarity",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_cluster_max_paper_author_count",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_paper_author_list_max_jaccard",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_paper_author_list_max_containment",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_paper_author_list_max_overlap_count",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_local_author_window10_jaccard_max",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_local_author_window10_overlap_count_max",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item(
        "row_best_author_count_log_absdiff",
        Vec::<f32>::new().to_pyarray(py),
    )?;
    payload.set_item("telemetry", telemetry)?;
    Ok(payload.unbind())
}

fn raw_arrow_labeled_component_members(
    component_key: &str,
    raw_members: &[String],
    signatures: &HashMap<String, RawArrowSignature>,
    component_scope: &str,
) -> Vec<String> {
    if component_scope != "block-local" {
        return raw_members.to_vec();
    }
    let Some((block_key, _cluster_key)) = component_key.split_once("::") else {
        return raw_members.to_vec();
    };
    let filtered = raw_members
        .iter()
        .filter(|signature_id| {
            signatures
                .get(signature_id.as_str())
                .and_then(|signature| signature.author_block.as_deref())
                .is_some_and(|author_block| author_block == block_key)
        })
        .cloned()
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        raw_members.to_vec()
    } else {
        filtered
    }
}

fn raw_arrow_active_members_for_row(
    component_key: &str,
    query_signature_id: &str,
    members_by_component: &HashMap<String, Vec<String>>,
) -> PyResult<Vec<String>> {
    let members = members_by_component.get(component_key).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!(
            "candidate_component_key missing from members table: {component_key}"
        ))
    })?;
    Ok(members
        .iter()
        .filter(|signature_id| signature_id.as_str() != query_signature_id)
        .cloned()
        .collect())
}

fn raw_arrow_component_summary_for_members(
    component_key: &str,
    members: &[String],
    features_by_signature_id: &HashMap<String, RawArrowFeature>,
    max_exemplars: usize,
) -> PyResult<RetrievalSummaryData> {
    build_raw_arrow_summary(
        component_key,
        members,
        features_by_signature_id,
        max_exemplars,
    )
    .map_err(pyo3::exceptions::PyKeyError::new_err)
}

fn raw_arrow_counter_present(counter: &Option<CounterData>) -> bool {
    counter
        .as_ref()
        .is_some_and(|values| !values.entries.is_empty())
}

#[pyfunction]
#[pyo3(signature = (
    paths,
    row_query_signature_ids,
    row_query_views,
    row_query_group_ids,
    row_component_keys,
    stored_retrieval_ranks,
    component_members,
    component_scope = "block-local",
    orcid_enabled = false,
    num_threads = None,
    max_exemplars = 4,
    include_pair_signature_ids = true,
    full_scan_without_index = false
))]
fn raw_arrow_labeled_candidate_plan<'py>(
    py: Python<'py>,
    paths: &Bound<'py, PyDict>,
    row_query_signature_ids: Vec<String>,
    row_query_views: Vec<String>,
    row_query_group_ids: Vec<String>,
    row_component_keys: Vec<String>,
    stored_retrieval_ranks: Vec<u16>,
    component_members: &Bound<'py, PyAny>,
    component_scope: &str,
    orcid_enabled: bool,
    num_threads: Option<usize>,
    max_exemplars: usize,
    include_pair_signature_ids: bool,
    full_scan_without_index: bool,
) -> PyResult<Py<PyDict>> {
    let total_start = Instant::now();
    let row_count = row_component_keys.len();
    if row_query_signature_ids.len() != row_count
        || row_query_views.len() != row_count
        || row_query_group_ids.len() != row_count
        || stored_retrieval_ranks.len() != row_count
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "labeled candidate row arrays must have equal length: query_ids={} query_views={} query_groups={} component_keys={} ranks={}",
            row_query_signature_ids.len(),
            row_query_views.len(),
            row_query_group_ids.len(),
            row_count,
            stored_retrieval_ranks.len()
        )));
    }
    if component_scope != "block-local" && component_scope != "frozen" {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "component_scope must be 'block-local' or 'frozen', got {component_scope:?}"
        )));
    }
    if row_count == 0 {
        return raw_arrow_labeled_empty_plan(py);
    }
    if !include_pair_signature_ids {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "raw_arrow_labeled_candidate_plan requires include_pair_signature_ids=True for non-empty plans",
        ));
    }

    let paths = raw_arrow_feature_paths_from_py_dict(paths)?;
    let mut query_signature_ids = Vec::<String>::new();
    let mut query_seen = HashSet::<String>::new();
    for signature_id in row_query_signature_ids.iter() {
        if query_seen.insert(signature_id.clone()) {
            query_signature_ids.push(signature_id.clone());
        }
    }
    validate_raw_arrow_query_signature_ids(&query_signature_ids)?;

    let component_entries = extract_string_vec_entries(component_members)?;
    let row_component_key_set: HashSet<String> = row_component_keys.iter().cloned().collect();
    let mut raw_members_by_component = HashMap::<String, Vec<String>>::new();
    let mut component_order = Vec::<String>::new();
    for (component_key, members) in component_entries {
        if raw_members_by_component
            .insert(component_key.clone(), members)
            .is_some()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "component_members contains duplicate candidate_component_key: {component_key:?}"
            )));
        }
        component_order.push(component_key);
    }
    let missing_component_keys = row_component_key_set
        .iter()
        .filter(|component_key| !raw_members_by_component.contains_key(component_key.as_str()))
        .take(10)
        .cloned()
        .collect::<Vec<_>>();
    if !missing_component_keys.is_empty() {
        return Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "candidate rows reference component keys missing from component_members: {missing_component_keys:?}"
        )));
    }

    let mut needed_signature_ids = Vec::<String>::new();
    let mut needed_seen = HashSet::<String>::new();
    for signature_id in query_signature_ids.iter() {
        if needed_seen.insert(signature_id.clone()) {
            needed_signature_ids.push(signature_id.clone());
        }
    }
    for component_key in component_order.iter() {
        let members = raw_members_by_component.get(component_key).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "component_key '{}' disappeared while collecting needed signatures",
                component_key
            ))
        })?;
        for signature_id in members.iter() {
            if needed_seen.insert(signature_id.clone()) {
                needed_signature_ids.push(signature_id.clone());
            }
        }
    }

    let query_inputs = read_reusable_raw_arrow_query_inputs(
        py,
        &paths,
        &needed_signature_ids,
        num_threads,
        full_scan_without_index,
    )?;
    let raw_name_counts = match paths.name_counts_index_path.as_ref() {
        Some(path) => read_raw_name_counts_index(path)?,
        None => match paths.name_counts_arrow_path.as_ref() {
            Some(path) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "name_counts Arrow path '{path}' requires name_counts_index; refusing slow Arrow fallback"
                )));
            }
            None => RawNameCountMaps::default(),
        },
    };

    let text_context_start = Instant::now();
    let text_module = py.import("s2and.text")?;
    let unidecode = text_module.getattr("unidecode")?;
    let name_prefixes = extract_required_string_set(&text_module.getattr("NAME_PREFIXES")?)?;
    let affiliation_stopwords = extract_affiliation_stopwords(py)?;
    let mut unidecode_char_map: HashMap<char, String> = HashMap::new();
    ensure_unidecode_for_raw_arrow_inputs(
        &unidecode,
        &query_inputs.signatures,
        &query_inputs.papers,
        &query_inputs.paper_authors,
        &mut unidecode_char_map,
    )?;
    let text_context_secs = text_context_start.elapsed().as_secs_f64();

    let specter_by_paper_id = query_inputs.specter_by_paper_id.as_ref();
    let feature_start = Instant::now();
    let raw_feature_results: Vec<Result<(String, RawArrowFeature), String>> =
        py.allow_threads(|| {
            let compute = || {
                needed_signature_ids
                    .par_iter()
                    .map(|signature_id| {
                        let signature =
                            query_inputs.signatures.get(signature_id).ok_or_else(|| {
                                format!(
                                    "signature_id '{}' is missing from signatures",
                                    signature_id
                                )
                            })?;
                        let paper = query_inputs.papers.get(&signature.paper_id);
                        let authors = query_inputs.paper_authors.get(&signature.paper_id);
                        Ok((
                            signature_id.clone(),
                            build_raw_arrow_feature(
                                signature,
                                paper,
                                authors,
                                specter_by_paper_id,
                                &raw_name_counts,
                                &name_prefixes,
                                &affiliation_stopwords,
                                &unidecode_char_map,
                                orcid_enabled,
                            ),
                        ))
                    })
                    .collect::<Vec<_>>()
            };
            install_with_optional_rayon_pool(num_threads, compute)
        });
    let mut features_by_signature_id = HashMap::with_capacity(raw_feature_results.len());
    for result in raw_feature_results {
        let (signature_id, feature) = result.map_err(pyo3::exceptions::PyKeyError::new_err)?;
        features_by_signature_id.insert(signature_id, feature);
    }
    let feature_secs = feature_start.elapsed().as_secs_f64();

    let mut members_by_component = HashMap::<String, Vec<String>>::new();
    for component_key in component_order.iter() {
        let raw_members = raw_members_by_component.get(component_key).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "component_key '{}' disappeared while filtering members",
                component_key
            ))
        })?;
        let members = raw_arrow_labeled_component_members(
            component_key,
            raw_members,
            &query_inputs.signatures,
            component_scope,
        );
        for signature_id in members.iter() {
            if !features_by_signature_id.contains_key(signature_id) {
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "component member signature_id '{}' is missing from computed raw Arrow features",
                    signature_id
                )));
            }
        }
        members_by_component.insert(component_key.clone(), members);
    }

    let summary_start = Instant::now();
    let retriever = build_retriever_from_raw_arrow_components(
        py,
        &component_order,
        &members_by_component,
        &features_by_signature_id,
        max_exemplars,
        num_threads,
    )?;
    let summary_secs = summary_start.elapsed().as_secs_f64();

    let mut group_order = Vec::<String>::new();
    let mut group_seen = HashSet::<String>::new();
    let mut rows_by_group = HashMap::<String, Vec<usize>>::new();
    for (row_index, group_id) in row_query_group_ids.iter().enumerate() {
        if group_seen.insert(group_id.clone()) {
            group_order.push(group_id.clone());
        }
        rows_by_group
            .entry(group_id.clone())
            .or_default()
            .push(row_index);
    }

    let mut row_retrieval_scores = vec![0.0f32; row_count];
    let mut row_retrieval_ranks = vec![0u16; row_count];
    let mut resolved_row_query_views = vec![String::new(); row_count];
    for group_id in group_order.iter() {
        let row_indices = rows_by_group.get(group_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "query group '{}' disappeared while scoring labeled candidates",
                group_id
            ))
        })?;
        let first_row = row_indices[0];
        let query_signature_id = row_query_signature_ids[first_row].as_str();
        let requested_view = row_query_views[first_row].as_str();
        for row_index in row_indices.iter().copied() {
            if row_query_signature_ids[row_index] != query_signature_id
                || row_query_views[row_index] != requested_view
            {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "query group {group_id:?} is not a single query/view"
                )));
            }
        }
        let query_feature = features_by_signature_id
            .get(query_signature_id)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "query signature_id '{}' is missing from computed raw Arrow features",
                    query_signature_id
                ))
            })?;
        let (query, resolved_view) = mask_raw_arrow_query(&query_feature.query, requested_view)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let weights = RustHybridCentroidRetriever::default_hybrid_weights_for_query(&query);
        let config = RustHybridCentroidRetriever::default_experimental_config_for_query(&query);
        let mut scored_rows = Vec::<(usize, f32, u16, String)>::with_capacity(row_indices.len());
        for row_index in row_indices.iter().copied() {
            resolved_row_query_views[row_index] = resolved_view.clone();
            let component_key = row_component_keys[row_index].as_str();
            let active_members = raw_arrow_active_members_for_row(
                component_key,
                query_signature_id,
                &members_by_component,
            )?;
            let summary = if active_members.len()
                == members_by_component
                    .get(component_key)
                    .map_or(usize::MAX, Vec::len)
            {
                let component_index = retriever
                    .component_index_by_key
                    .get(component_key)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "Unknown component_key for raw Arrow labeled scoring: {component_key}"
                        ))
                    })?;
                retriever.summaries[*component_index].clone()
            } else {
                raw_arrow_component_summary_for_members(
                    component_key,
                    &active_members,
                    &features_by_signature_id,
                    max_exemplars,
                )?
            };
            let score = round_six(score_experimental_hybrid_centroid_query(
                &query,
                &summary,
                weights,
                config,
                &retriever.coauthor_cluster_df,
                &retriever.non_mega_coauthor_cluster_df,
                &retriever.affiliation_cluster_df,
                retriever.summaries.len(),
            ) as f64);
            scored_rows.push((
                row_index,
                score,
                stored_retrieval_ranks[row_index],
                component_key.to_string(),
            ));
        }
        scored_rows.sort_unstable_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.2.cmp(&right.2))
                .then_with(|| left.3.cmp(&right.3))
        });
        for (rank_offset, (row_index, score, _stored_rank, _component_key)) in
            scored_rows.into_iter().enumerate()
        {
            row_retrieval_scores[row_index] = score;
            row_retrieval_ranks[row_index] = (rank_offset + 1).min(u16::MAX as usize) as u16;
        }
    }

    let mut signature_ids = Vec::<String>::new();
    let mut signature_seen = HashSet::<String>::new();
    for signature_id in row_query_signature_ids.iter() {
        if signature_seen.insert(signature_id.clone()) {
            signature_ids.push(signature_id.clone());
        }
    }
    let mut left_signature_ids = Vec::<String>::new();
    let mut right_signature_ids = Vec::<String>::new();
    let mut pair_row_indices = Vec::<u32>::new();
    for (row_index, (query_signature_id, component_key)) in row_query_signature_ids
        .iter()
        .zip(row_component_keys.iter())
        .enumerate()
    {
        let active_members = raw_arrow_active_members_for_row(
            component_key,
            query_signature_id,
            &members_by_component,
        )?;
        for member_signature_id in active_members {
            if signature_seen.insert(member_signature_id.clone()) {
                signature_ids.push(member_signature_id.clone());
            }
            left_signature_ids.push(query_signature_id.clone());
            right_signature_ids.push(member_signature_id);
            pair_row_indices.push(u32::try_from(row_index).map_err(|_| {
                pyo3::exceptions::PyOverflowError::new_err(
                    "labeled candidate row index exceeds u32",
                )
            })?);
        }
    }

    let mut row_query_authors = Vec::<String>::with_capacity(row_count);
    let mut row_component_sizes = Vec::<u32>::with_capacity(row_count);
    let mut row_named_signature_counts = Vec::<u32>::with_capacity(row_count);
    let mut row_dominant_first_names = Vec::<String>::with_capacity(row_count);
    let mut row_candidate_year_min = Vec::<i32>::with_capacity(row_count);
    let mut row_candidate_year_max = Vec::<i32>::with_capacity(row_count);
    let mut row_candidate_year_range_missing = Vec::<u8>::with_capacity(row_count);
    let mut row_query_first_tokens = Vec::<String>::with_capacity(row_count);
    let mut row_query_years = Vec::<i32>::with_capacity(row_count);
    let mut row_query_year_missing = Vec::<u8>::with_capacity(row_count);
    let mut row_query_has_affiliations = Vec::<u8>::with_capacity(row_count);
    let mut row_query_has_coauthors = Vec::<u8>::with_capacity(row_count);
    let mut row_query_has_specter = Vec::<u8>::with_capacity(row_count);
    let mut row_query_has_name_counts = Vec::<u8>::with_capacity(row_count);
    let mut row_candidate_has_affiliations = Vec::<u8>::with_capacity(row_count);
    let mut row_candidate_has_coauthors = Vec::<u8>::with_capacity(row_count);
    let mut row_candidate_has_specter_exemplars = Vec::<u8>::with_capacity(row_count);
    let mut row_candidate_has_name_counts = Vec::<u8>::with_capacity(row_count);
    let mut row_orcid_match = Vec::<u8>::with_capacity(row_count);
    let mut row_middle_initial_compatibility = Vec::<f32>::with_capacity(row_count);
    let mut row_affiliation_overlap = Vec::<f32>::with_capacity(row_count);
    let mut row_coauthor_overlap = Vec::<f32>::with_capacity(row_count);
    let mut row_venue_overlap = Vec::<f32>::with_capacity(row_count);
    let mut row_year_compatibility = Vec::<f32>::with_capacity(row_count);
    let mut row_title_overlap = Vec::<f32>::with_capacity(row_count);
    let mut row_specter_centroid_similarity = Vec::<f32>::with_capacity(row_count);
    let mut row_specter_exemplar_similarity = Vec::<f32>::with_capacity(row_count);
    let mut row_last_name_count_min_rarity = Vec::<f32>::with_capacity(row_count);
    let mut row_candidate_last_name_count_min_rarity = Vec::<f32>::with_capacity(row_count);
    let mut row_candidate_last_first_name_count_min_rarity = Vec::<f32>::with_capacity(row_count);
    let mut row_last_first_name_count_min_rarity = Vec::<f32>::with_capacity(row_count);
    let mut row_first_prefix_x_last_first_name_count_min_rarity =
        Vec::<f32>::with_capacity(row_count);
    let mut row_candidate_cluster_max_paper_author_count = Vec::<f32>::with_capacity(row_count);
    let mut row_paper_author_list_max_jaccard = Vec::<f32>::with_capacity(row_count);
    let mut row_paper_author_list_max_containment = Vec::<f32>::with_capacity(row_count);
    let mut row_paper_author_list_max_overlap_count = Vec::<f32>::with_capacity(row_count);
    let mut row_local_author_window10_jaccard_max = Vec::<f32>::with_capacity(row_count);
    let mut row_local_author_window10_overlap_count_max = Vec::<f32>::with_capacity(row_count);
    let mut row_best_author_count_log_absdiff = Vec::<f32>::with_capacity(row_count);
    let mut full_summary_signal_cache = HashMap::<String, RawArrowSummarySignalData>::new();
    let mut residual_summary_signal_cache = HashMap::<String, RawArrowSummarySignalData>::new();
    let mut author_signals_by_query_signature_id =
        HashMap::<String, RawArrowAuthorSignalData>::new();
    for row_index in 0..row_count {
        let query_signature_id = row_query_signature_ids[row_index].as_str();
        let component_key = row_component_keys[row_index].as_str();
        let query_feature = features_by_signature_id
            .get(query_signature_id)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "query signature_id '{}' is missing from computed raw Arrow features",
                    query_signature_id
                ))
            })?;
        let (query, resolved_view) =
            mask_raw_arrow_query(&query_feature.query, row_query_views[row_index].as_str())
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
        resolved_row_query_views[row_index] = resolved_view;
        let active_members = raw_arrow_active_members_for_row(
            component_key,
            query_signature_id,
            &members_by_component,
        )?;
        let full_member_count = members_by_component
            .get(component_key)
            .map_or(usize::MAX, Vec::len);
        let (summary, summary_signals) = if active_members.len() == full_member_count {
            let component_index = retriever
                .component_index_by_key
                .get(component_key)
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "Unknown component_key for raw Arrow labeled row signals: {component_key}"
                    ))
                })?;
            let summary = retriever.summaries[*component_index].clone();
            let signals = raw_arrow_summary_signals_cached(
                &mut full_summary_signal_cache,
                component_key,
                &members_by_component,
                &features_by_signature_id,
                &query_inputs.signatures,
                &query_inputs.paper_authors,
                &unidecode_char_map,
            )?;
            (summary, signals)
        } else {
            let summary = raw_arrow_component_summary_for_members(
                component_key,
                &active_members,
                &features_by_signature_id,
                max_exemplars,
            )?;
            let cache_key = format!("{component_key}\u{0}{query_signature_id}");
            let signals = raw_arrow_summary_signals_for_members_cached(
                &mut residual_summary_signal_cache,
                &cache_key,
                &active_members,
                &features_by_signature_id,
                &query_inputs.signatures,
                &query_inputs.paper_authors,
                &unidecode_char_map,
            )?;
            (summary, signals)
        };

        if let Entry::Vacant(entry) =
            author_signals_by_query_signature_id.entry(query_signature_id.to_string())
        {
            let query_signature =
                query_inputs
                    .signatures
                    .get(query_signature_id)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "query signature_id '{}' is missing from signatures",
                            query_signature_id
                        ))
                    })?;
            entry.insert(build_raw_arrow_author_signal_data(
                query_signature,
                query_inputs.paper_authors.get(&query_signature.paper_id),
                &unidecode_char_map,
            ));
        }
        let query_author_signals = author_signals_by_query_signature_id
            .get(query_signature_id)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "query signature_id '{}' is missing raw author signals",
                    query_signature_id
                ))
            })?;
        let rarity = raw_arrow_name_count_rarity_row(
            &query,
            &query_feature.name_counts,
            &summary,
            summary_signals,
        );
        let evidence = raw_arrow_paper_evidence_row(
            query_signature_id,
            query_feature.paper_author_count,
            query_author_signals,
            summary_signals,
        );
        let chooser_features = chooser_summary_features(&query, &summary);
        let mut dominant_first_name = "";
        let mut dominant_first_count = 0.0f32;
        let mut named_signature_count = 0.0f32;
        for (first_name, count) in summary.first_name_counts.iter() {
            named_signature_count += *count;
            if *count > dominant_first_count
                || (*count == dominant_first_count && first_name.as_str() > dominant_first_name)
            {
                dominant_first_name = first_name.as_str();
                dominant_first_count = *count;
            }
        }
        let (candidate_year_min, candidate_year_min_missing) =
            year_signal_value(summary.year_min, "candidate year_min")
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let (candidate_year_max, candidate_year_max_missing) =
            year_signal_value(summary.year_max, "candidate year_max")
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let (query_year, query_year_missing) = year_signal_value(query.year, "query year")
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        row_query_authors.push(query_feature.query_author.clone());
        row_component_sizes.push(summary.size.min(u32::MAX as usize) as u32);
        row_named_signature_counts.push(named_signature_count.round().max(0.0) as u32);
        row_dominant_first_names.push(dominant_first_name.to_string());
        row_candidate_year_min.push(candidate_year_min);
        row_candidate_year_max.push(candidate_year_max);
        row_candidate_year_range_missing.push(u8::from(
            candidate_year_min_missing != 0 || candidate_year_max_missing != 0,
        ));
        row_query_first_tokens.push(query_feature.query.first.clone());
        row_query_years.push(query_year);
        row_query_year_missing.push(query_year_missing);
        row_query_has_affiliations.push(u8::from(!query.affiliation_hashes.is_empty()));
        row_query_has_coauthors.push(u8::from(!query.coauthor_hashes.is_empty()));
        row_query_has_specter.push(u8::from(query.specter.is_some()));
        row_query_has_name_counts.push(u8::from(query_feature.name_counts.is_some()));
        row_candidate_has_affiliations.push(u8::from(
            raw_arrow_counter_present(&summary.affiliation_counts) && summary.size > 0,
        ));
        row_candidate_has_coauthors.push(u8::from(
            raw_arrow_counter_present(&summary.coauthor_counts) && summary.size > 0,
        ));
        row_candidate_has_specter_exemplars.push(u8::from(!summary.exemplar_vectors.is_empty()));
        row_candidate_has_name_counts
            .push(u8::from(!summary_signals.name_counts_values.is_empty()));
        row_orcid_match.push(u8::from(query.orcid_hash.is_some_and(|orcid_hash| {
            contains_hashed_value(&summary.orcid_hashes, orcid_hash)
        })));
        row_middle_initial_compatibility.push(chooser_features[0]);
        row_affiliation_overlap.push(chooser_features[1]);
        row_coauthor_overlap.push(chooser_features[2]);
        row_venue_overlap.push(chooser_features[3]);
        row_year_compatibility.push(chooser_features[4]);
        row_title_overlap.push(chooser_features[5]);
        row_specter_centroid_similarity.push(chooser_features[6]);
        row_specter_exemplar_similarity.push(chooser_features[7]);
        row_last_name_count_min_rarity.push(rarity.last_name_count_min_rarity);
        row_candidate_last_name_count_min_rarity.push(rarity.candidate_last_name_count_min_rarity);
        row_candidate_last_first_name_count_min_rarity
            .push(rarity.candidate_last_first_name_count_min_rarity);
        row_last_first_name_count_min_rarity.push(rarity.last_first_name_count_min_rarity);
        row_first_prefix_x_last_first_name_count_min_rarity
            .push(rarity.first_prefix_x_last_first_name_count_min_rarity);
        row_candidate_cluster_max_paper_author_count.push(summary.max_paper_author_count as f32);
        row_paper_author_list_max_jaccard.push(evidence.paper_author_list_max_jaccard);
        row_paper_author_list_max_containment.push(evidence.paper_author_list_max_containment);
        row_paper_author_list_max_overlap_count.push(evidence.paper_author_list_max_overlap_count);
        row_local_author_window10_jaccard_max.push(evidence.local_author_window10_jaccard_max);
        row_local_author_window10_overlap_count_max
            .push(evidence.local_author_window10_overlap_count_max);
        row_best_author_count_log_absdiff.push(evidence.best_author_count_log_absdiff);
    }

    let timings = PyDict::new(py);
    timings.set_item("read_signatures_secs", query_inputs.read_signatures_secs)?;
    timings.set_item("read_papers_secs", query_inputs.read_papers_secs)?;
    timings.set_item(
        "read_paper_authors_secs",
        query_inputs.read_paper_authors_secs,
    )?;
    timings.set_item("read_specter_secs", query_inputs.read_specter_secs)?;
    timings.set_item(
        "metadata_reads_parallel_secs",
        query_inputs.metadata_reads_parallel_secs,
    )?;
    timings.set_item("text_context_secs", text_context_secs)?;
    timings.set_item("feature_secs", feature_secs)?;
    timings.set_item("summary_secs", summary_secs)?;
    timings.set_item("total_secs", total_start.elapsed().as_secs_f64())?;

    let telemetry = PyDict::new(py);
    telemetry.set_item("signature_count", signature_ids.len())?;
    telemetry.set_item("paper_count", query_inputs.papers.len())?;
    telemetry.set_item("paper_author_paper_count", query_inputs.paper_authors.len())?;
    telemetry.set_item("component_count", component_order.len())?;
    telemetry.set_item("query_signature_count", query_signature_ids.len())?;
    telemetry.set_item("row_count", row_count)?;
    telemetry.set_item("pair_count", left_signature_ids.len())?;
    telemetry.set_item("component_scope", component_scope)?;
    telemetry.set_item("orcid_enabled", orcid_enabled)?;
    telemetry.set_item(
        "signature_batches_read",
        query_inputs.signature_index_stats.batches_read,
    )?;
    telemetry.set_item(
        "signature_rows_scanned",
        query_inputs.signature_index_stats.rows_scanned,
    )?;
    telemetry.set_item(
        "paper_batches_read",
        query_inputs.paper_index_stats.batches_read,
    )?;
    telemetry.set_item(
        "paper_rows_scanned",
        query_inputs.paper_index_stats.rows_scanned,
    )?;
    telemetry.set_item(
        "paper_author_batches_read",
        query_inputs.paper_author_index_stats.batches_read,
    )?;
    telemetry.set_item(
        "paper_author_rows_scanned",
        query_inputs.paper_author_index_stats.rows_scanned,
    )?;
    telemetry.set_item(
        "specter_batches_read",
        query_inputs.specter_index_stats.batches_read,
    )?;
    telemetry.set_item(
        "specter_rows_scanned",
        query_inputs.specter_index_stats.rows_scanned,
    )?;
    telemetry.set_item("unidecode_char_count", unidecode_char_map.len())?;
    telemetry.set_item("timings", &timings)?;

    let payload = PyDict::new(py);
    payload.set_item("schema_version", "raw_arrow_labeled_candidate_plan_v1")?;
    payload.set_item("row_count", row_count)?;
    payload.set_item("pair_count", left_signature_ids.len())?;
    payload.set_item("signature_ids", signature_ids)?;
    payload.set_item("query_signature_ids", query_signature_ids)?;
    payload.set_item("row_query_signature_ids", row_query_signature_ids)?;
    payload.set_item("row_query_views", resolved_row_query_views.clone())?;
    payload.set_item("row_query_authors", row_query_authors)?;
    payload.set_item("row_query_group_ids", row_query_group_ids)?;
    payload.set_item("row_component_keys", row_component_keys)?;
    payload.set_item("query_views", Vec::<String>::new())?;
    payload.set_item("query_authors", Vec::<String>::new())?;
    if include_pair_signature_ids {
        payload.set_item("left_signature_ids", left_signature_ids)?;
        payload.set_item("right_signature_ids", right_signature_ids)?;
    }
    payload.set_item("pair_row_indices", pair_row_indices.to_pyarray(py))?;
    payload.set_item("retrieval_scores", row_retrieval_scores.to_pyarray(py))?;
    payload.set_item("retrieval_ranks", row_retrieval_ranks.to_pyarray(py))?;
    payload.set_item("row_component_sizes", row_component_sizes.to_pyarray(py))?;
    payload.set_item(
        "row_named_signature_counts",
        row_named_signature_counts.to_pyarray(py),
    )?;
    payload.set_item("row_dominant_first_names", row_dominant_first_names)?;
    payload.set_item(
        "row_candidate_year_min",
        row_candidate_year_min.to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_year_max",
        row_candidate_year_max.to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_year_range_missing",
        row_candidate_year_range_missing.to_pyarray(py),
    )?;
    payload.set_item("row_query_first_tokens", row_query_first_tokens)?;
    payload.set_item("row_query_years", row_query_years.to_pyarray(py))?;
    payload.set_item(
        "row_query_year_missing",
        row_query_year_missing.to_pyarray(py),
    )?;
    payload.set_item(
        "row_query_has_affiliations",
        row_query_has_affiliations.to_pyarray(py),
    )?;
    payload.set_item(
        "row_query_has_coauthors",
        row_query_has_coauthors.to_pyarray(py),
    )?;
    payload.set_item(
        "row_query_has_specter",
        row_query_has_specter.to_pyarray(py),
    )?;
    payload.set_item(
        "row_query_has_name_counts",
        row_query_has_name_counts.to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_affiliations",
        row_candidate_has_affiliations.to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_coauthors",
        row_candidate_has_coauthors.to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_specter_exemplars",
        row_candidate_has_specter_exemplars.to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_has_name_counts",
        row_candidate_has_name_counts.to_pyarray(py),
    )?;
    payload.set_item("row_orcid_match", row_orcid_match.to_pyarray(py))?;
    payload.set_item(
        "middle_initial_compatibility",
        row_middle_initial_compatibility.to_pyarray(py),
    )?;
    payload.set_item(
        "affiliation_overlap",
        row_affiliation_overlap.to_pyarray(py),
    )?;
    payload.set_item("coauthor_overlap", row_coauthor_overlap.to_pyarray(py))?;
    payload.set_item("venue_overlap", row_venue_overlap.to_pyarray(py))?;
    payload.set_item("year_compatibility", row_year_compatibility.to_pyarray(py))?;
    payload.set_item("title_overlap", row_title_overlap.to_pyarray(py))?;
    payload.set_item(
        "specter_centroid_similarity",
        row_specter_centroid_similarity.to_pyarray(py),
    )?;
    payload.set_item(
        "specter_exemplar_similarity",
        row_specter_exemplar_similarity.to_pyarray(py),
    )?;
    payload.set_item(
        "row_last_name_count_min_rarity",
        row_last_name_count_min_rarity.to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_last_name_count_min_rarity",
        row_candidate_last_name_count_min_rarity.to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_last_first_name_count_min_rarity",
        row_candidate_last_first_name_count_min_rarity.to_pyarray(py),
    )?;
    payload.set_item(
        "row_last_first_name_count_min_rarity",
        row_last_first_name_count_min_rarity.to_pyarray(py),
    )?;
    payload.set_item(
        "row_first_prefix_x_last_first_name_count_min_rarity",
        row_first_prefix_x_last_first_name_count_min_rarity.to_pyarray(py),
    )?;
    payload.set_item(
        "row_candidate_cluster_max_paper_author_count",
        row_candidate_cluster_max_paper_author_count.to_pyarray(py),
    )?;
    payload.set_item(
        "row_paper_author_list_max_jaccard",
        row_paper_author_list_max_jaccard.to_pyarray(py),
    )?;
    payload.set_item(
        "row_paper_author_list_max_containment",
        row_paper_author_list_max_containment.to_pyarray(py),
    )?;
    payload.set_item(
        "row_paper_author_list_max_overlap_count",
        row_paper_author_list_max_overlap_count.to_pyarray(py),
    )?;
    payload.set_item(
        "row_local_author_window10_jaccard_max",
        row_local_author_window10_jaccard_max.to_pyarray(py),
    )?;
    payload.set_item(
        "row_local_author_window10_overlap_count_max",
        row_local_author_window10_overlap_count_max.to_pyarray(py),
    )?;
    payload.set_item(
        "row_best_author_count_log_absdiff",
        row_best_author_count_log_absdiff.to_pyarray(py),
    )?;
    payload.set_item("telemetry", telemetry)?;
    Ok(payload.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyString;

    fn py_err_message(err: PyErr) -> String {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            err.value(py)
                .str()
                .expect("PyErr value should stringify")
                .to_str()
                .expect("test error messages are ASCII")
                .to_string()
        })
    }

    #[test]
    fn json_value_to_i64_rejects_lossy_numbers() {
        let float_value = serde_json::json!(1.5);
        let float_error =
            json_value_to_i64(&float_value, "signature author_info.position").unwrap_err();
        assert!(py_err_message(float_error).contains("must be an integer"));

        let overflow_value = serde_json::json!(u64::MAX);
        let overflow_error =
            json_value_to_i64(&overflow_value, "signature author_info.position").unwrap_err();
        assert!(py_err_message(overflow_error).contains("outside i64 range"));
    }

    #[test]
    fn validate_retrieval_top_k_rejects_uint16_rank_overflow() {
        let error = validate_retrieval_rank_top_k((u16::MAX as usize) + 1).unwrap_err();
        assert!(py_err_message(error).contains("retrieval_ranks are stored as uint16"));
    }

    #[test]
    fn feature_index_resolution_preserves_order_and_duplicates() {
        assert_eq!(
            resolve_feature_indices("selected_indices", Some(vec![2, 2, 3]), 5)
                .expect("indices are in range"),
            vec![2, 2, 3]
        );
        assert_eq!(
            resolve_feature_indices("selected_indices", None, 3).expect("default indices"),
            vec![0, 1, 2]
        );

        let result = resolve_feature_indices("selected_indices", Some(vec![0, 3]), 3);
        assert!(result.is_err());
        let message = py_err_message(result.err().expect("error was asserted"));
        assert!(message.contains("selected_indices contains out-of-range index 3 for 3 columns"));
    }

    #[test]
    fn aggregate_matrix_positions_preserve_aggregate_order_and_duplicate_mapping() {
        assert_eq!(
            matrix_positions_for_feature_indices(&[2, 2, 3], &[2, 3, 2])
                .expect("aggregate features are present"),
            vec![0, 2, 0]
        );

        let result = resolve_matrix_aggregate_indices(Some(vec![2, 2]), Some(vec![3]), 4);
        assert!(result.is_err());
        let message = py_err_message(result.err().expect("error was asserted"));
        assert!(message.contains("aggregate index 3 is not present in matrix_indices"));
    }

    #[test]
    fn sorted_subblock_merge_candidates_allows_nan_scores() {
        let mut output = OrderedSubblocks::default();
        output.insert("alice".to_string(), vec!["s1".to_string()]);
        output.insert("bob".to_string(), vec!["s2".to_string()]);
        let mut counts = HashMap::<String, HashMap<String, f64>>::new();
        counts.insert(
            "alice".to_string(),
            HashMap::from([("bob".to_string(), f64::NAN)]),
        );

        let result = sorted_subblock_merge_candidates(&output, 3, &counts);

        assert!(result.is_ok());
        let candidates = result.expect("NaN scores should sort without raising");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, ("alice".to_string(), "bob".to_string()));
        assert!(candidates[0].1.is_nan());
    }

    #[test]
    fn reference_details_extraction_errors_are_not_silenced() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let non_tuple = PyString::new(py, "not-a-tuple");
            let result = extract_reference_details_counters(py, non_tuple.as_any());
            assert!(result.is_err(), "non-tuple reference_details should raise");
            let err = result
                .err()
                .unwrap_or_else(|| unreachable!("assert above guarantees error"));
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
        });
    }

    #[test]
    fn orcid_normalization_canonicalizes_common_forms() {
        assert_eq!(
            normalize_orcid_owned(" https://orcid.org/0000-0002-1825-0097 "),
            Some("0000-0002-1825-0097".to_string())
        );
        assert_eq!(
            normalize_orcid_owned("ORCID: 000000021825009x"),
            Some("0000-0002-1825-009X".to_string())
        );
        assert_eq!(
            normalize_orcid_owned("https://orcid.org/0000\u{2010}0002\u{2010}1825\u{2010}0097"),
            Some("0000-0002-1825-0097".to_string())
        );
        for dash in [
            '-', '\u{2010}', '\u{2011}', '\u{2012}', '\u{2013}', '\u{2014}', '\u{2212}',
            '\u{FE58}', '\u{FE63}', '\u{FF0D}',
        ] {
            let value = format!("0000{dash}0002{dash}1825{dash}0097");
            assert_eq!(
                normalize_orcid_owned(&value),
                Some("0000-0002-1825-0097".to_string())
            );
        }
        assert_eq!(
            normalize_orcid_compact_owned("ORCID: 000000021825009x"),
            Some("000000021825009X".to_string())
        );
        assert_eq!(normalize_orcid_owned("s000-0000-1879-1075X"), None);
        assert_eq!(normalize_orcid_owned("0000-0002-1825"), None);
    }

    #[test]
    fn year_signal_value_rejects_out_of_range_and_reserved_sentinel() {
        assert_eq!(
            year_signal_value(None, "query year").expect("missing year"),
            (i32::MIN, 1)
        );
        assert_eq!(
            year_signal_value(Some(2024), "query year").expect("valid year"),
            (2024, 0)
        );
        assert!(year_signal_value(Some(i64::from(i32::MAX) + 1), "query year").is_err());
        assert!(year_signal_value(Some(i64::from(i32::MIN)), "query year").is_err());
    }

    #[test]
    fn i64_author_position_distance_handles_extreme_values() {
        assert!(i64::MIN.abs_diff(0) > 10);
        assert_eq!(10_i64.abs_diff(0), 10);
    }

    #[test]
    fn subblocking_arrow_rows_normalize_first_and_middle_names() {
        let mut rows = vec![
            SubblockingSignatureRow {
                signature_id: "s1".to_string(),
                first: "Alice".to_string(),
                middle: String::new(),
                orcid: None,
            },
            SubblockingSignatureRow {
                signature_id: "s2".to_string(),
                first: "alice".to_string(),
                middle: String::new(),
                orcid: None,
            },
            SubblockingSignatureRow {
                signature_id: "s3".to_string(),
                first: "Qi-Xin".to_string(),
                middle: "A.".to_string(),
                orcid: None,
            },
            SubblockingSignatureRow {
                signature_id: "s4".to_string(),
                first: "Arif\u{2010}ullah".to_string(),
                middle: String::new(),
                orcid: None,
            },
        ];
        let prefixes = HashSet::new();
        let unidecode_char_map = HashMap::from([('\u{2010}', "-".to_string())]);

        normalize_subblocking_signature_rows(&mut rows, &prefixes, &unidecode_char_map);

        assert_eq!(rows[0].first, "alice");
        assert_eq!(rows[1].first, "alice");
        assert_eq!(rows[2].first, "qi xin");
        assert_eq!(rows[2].middle, "a");
        assert_eq!(rows[3].first, "arif ullah");
        assert_eq!(rows[3].middle, "");
    }

    #[test]
    fn normalize_text_compat_drops_digits_like_python_reference() {
        let unidecode_char_map = HashMap::new();

        assert_eq!(
            normalize_text_compat_from_map("A1 B-2", false, &unidecode_char_map),
            "a b"
        );
        assert_eq!(
            normalize_text_compat_from_map("O'Neil2", true, &unidecode_char_map),
            "oneil"
        );
    }

    #[test]
    #[should_panic(expected = "missing unidecode mapping")]
    fn normalize_text_compat_requires_mapping_for_non_ascii() {
        let unidecode_char_map = HashMap::new();
        let _ = normalize_text_compat_from_map("\u{00C9}lodie", false, &unidecode_char_map);
    }

    #[test]
    fn first_normalized_token_uses_python_space_split() {
        let prefixes = HashSet::new();
        assert_eq!(
            first_normalized_token_python_compat("", "alan", &prefixes),
            ""
        );

        let prefixes = HashSet::from(["dr".to_string()]);
        assert_eq!(
            first_normalized_token_python_compat("dr", "alice", &prefixes),
            "alice"
        );
    }

    #[test]
    fn stage_papers_normalize_title_and_authors_without_full_preprocess() {
        let input = StagePaperInput {
            paper_id: "p1".to_string(),
            raw_title: "Some Title".to_string(),
            raw_venue: "My Venue".to_string(),
            raw_journal: "My Journal".to_string(),
            raw_authors: vec![(0, "ALICE-1".to_string()), (1, "Bob O'Neil".to_string())],
            year: Some(2024),
            has_abstract: false,
            predicted_language: None,
            is_reliable: false,
        };

        let papers = preprocess_stage_papers(
            &[input],
            false,
            &HashMap::new(),
            &HashSet::new(),
            &HashSet::new(),
        );

        let paper = &papers[0].1;
        assert_eq!(
            paper.authors,
            vec![(0, "alice".to_string()), (1, "bob o neil".to_string())]
        );
        assert!(paper.title_words.is_some());
        assert!(paper.title_chars.is_none());
        assert!(paper.venue_ngrams.is_none());
        assert!(paper.journal_ngrams.is_none());
    }

    #[test]
    fn name_tuple_compatibility_does_not_apply_extra_case_normalization() {
        let mut name_tuples = HashMap::new();
        insert_name_tuple_alias(&mut name_tuples, "Bill".to_string(), "William".to_string());

        assert!(first_names_name_compatible("Bill", "William", &name_tuples));
        assert!(!first_names_name_compatible(
            "bill",
            "william",
            &name_tuples
        ));
    }

    #[test]
    fn subblock_token_fallback_matches_python_case_preserving_parse() {
        assert_eq!(
            subblock_tokens_from_key("Ali|3,bob|2,a|1"),
            vec!["Ali".to_string(), "bob".to_string()]
        );
    }

    #[test]
    fn linker_alpha_normalization_uses_text_normalization() {
        let unidecode_char_map = HashMap::from([('É', "E".to_string())]);
        assert_eq!(
            linker_normalize_alpha("Élodie-2", &unidecode_char_map),
            "elodie"
        );
    }
}

#[pymodule]
fn _s2and_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("RETRIEVAL_FEATURE_ORDER", RETRIEVAL_FEATURE_ORDER.to_vec())?;
    m.add(
        "DEFAULT_HYBRID_CENTROID_POLICY_NAME",
        DEFAULT_HYBRID_CENTROID_POLICY_NAME,
    )?;
    m.add(
        "DEFAULT_HYBRID_CENTROID_WEIGHTS",
        DEFAULT_HYBRID_CENTROID_WEIGHTS.to_vec(),
    )?;
    m.add(
        "DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS",
        DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS.to_vec(),
    )?;
    m.add(
        "DEFAULT_HYBRID_EXEMPLAR_4_WEIGHTS",
        DEFAULT_HYBRID_EXEMPLAR_4_WEIGHTS.to_vec(),
    )?;
    m.add(
        "RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE",
        RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE,
    )?;
    m.add(
        "RETRIEVAL_YEAR_SCORE_DECAY_YEARS",
        RETRIEVAL_YEAR_SCORE_DECAY_YEARS,
    )?;
    m.add(
        "RETRIEVAL_YEAR_SCORE_RANGE_GAP",
        RETRIEVAL_YEAR_SCORE_RANGE_GAP,
    )?;
    m.add(
        "RETRIEVAL_YEAR_SCORE_RANGE_PENALTY",
        RETRIEVAL_YEAR_SCORE_RANGE_PENALTY,
    )?;
    m.add(
        "RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP",
        RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP,
    )?;
    m.add(
        "INCREMENTAL_LINKING_PAIR_PLAN_ROW_SIGNALS",
        INCREMENTAL_LINKING_PAIR_PLAN_ROW_SIGNALS.to_vec(),
    )?;
    m.add_function(wrap_pyfunction!(get_build_info, m)?)?;
    m.add_function(wrap_pyfunction!(raw_block_query_candidate_plan_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(raw_arrow_labeled_candidate_plan, m)?)?;
    m.add_function(wrap_pyfunction!(promoted_linker_non_pairwise_features, m)?)?;
    m.add_function(wrap_pyfunction!(signature_ngrams_batch, m)?)?;
    m.add_function(wrap_pyfunction!(make_subblocks_with_telemetry_arrow, m)?)?;
    m.add_class::<RustFeaturizer>()?;
    m.add_class::<RustHybridCentroidRetriever>()?;
    m.add_class::<RustNameCompatibleSubblockSelector>()?;
    m.add_class::<RawBlockQueryCandidatePlanner>()?;
    Ok(())
}
