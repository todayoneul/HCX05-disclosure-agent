PRAGMA foreign_keys = ON;

CREATE TABLE document (
    doc_id TEXT PRIMARY KEY,
    rcept_no TEXT NOT NULL UNIQUE,
    corp_code TEXT NOT NULL,
    corp_name TEXT NOT NULL,
    listed_name TEXT,
    stock_code TEXT,
    industry TEXT,
    sector TEXT,
    doc_group TEXT NOT NULL CHECK (doc_group IN ('periodic','exchange','major','holding')),
    doc_subtype TEXT,
    report_nm TEXT NOT NULL,
    is_correction INTEGER NOT NULL CHECK (is_correction IN (0,1)),
    rcept_dt TEXT NOT NULL,
    flr_nm TEXT,
    base_year INTEGER,
    base_month INTEGER CHECK (base_month IS NULL OR base_month BETWEEN 1 AND 12),
    file_path TEXT NOT NULL,
    file_format TEXT NOT NULL CHECK (file_format IN ('xml','pdf+html')),
    n_files INTEGER NOT NULL CHECK (n_files > 0)
) STRICT;

CREATE TABLE event (
    doc_id TEXT PRIMARY KEY REFERENCES document(doc_id),
    rcept_no TEXT NOT NULL UNIQUE REFERENCES document(rcept_no),
    rcept_dt TEXT, corp_code TEXT, corp_name TEXT, listed_name TEXT, sector TEXT, industry TEXT,
    doc_group TEXT, doc_subtype TEXT, event_type TEXT, report_nm TEXT,
    is_correction INTEGER CHECK (is_correction IS NULL OR is_correction IN (0,1)),
    title TEXT, amount TEXT, amount_type TEXT, ratio TEXT, ratio_base TEXT, counterparty TEXT,
    period_start TEXT, period_end TEXT, event_date TEXT, reserved_reason TEXT,
    extra_json TEXT CHECK (extra_json IS NULL OR json_valid(extra_json)),
    fields_json TEXT CHECK (fields_json IS NULL OR json_valid(fields_json)),
    corr_date TEXT, corr_reason TEXT, corr_target_doc TEXT, corr_target_date TEXT,
    corr_diffs_json TEXT CHECK (corr_diffs_json IS NULL OR json_valid(corr_diffs_json))
) STRICT;

CREATE TABLE chunk (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES document(doc_id),
    rcept_no TEXT NOT NULL REFERENCES document(rcept_no),
    src_file TEXT NOT NULL,
    path TEXT NOT NULL,
    part INTEGER NOT NULL CHECK (part > 0),
    document_sequence INTEGER NOT NULL CHECK (document_sequence > 0),
    section_start INTEGER NOT NULL CHECK (section_start >= 0),
    section_end INTEGER NOT NULL CHECK (section_end >= section_start),
    block_start INTEGER NOT NULL CHECK (block_start >= 0),
    block_end INTEGER NOT NULL CHECK (block_end > block_start),
    n_chars INTEGER NOT NULL CHECK (n_chars > 0 AND n_chars = length(text)),
    n_tables INTEGER NOT NULL CHECK (n_tables >= 0),
    text TEXT NOT NULL CHECK (length(text) > 0)
) STRICT;

CREATE TABLE correction_link (
    correction_rcept_no TEXT PRIMARY KEY REFERENCES document(rcept_no),
    predecessor_rcept_no TEXT REFERENCES document(rcept_no),
    status TEXT NOT NULL CHECK (status IN ('linked','ambiguous_candidate','unresolved_external_root')),
    method TEXT NOT NULL CHECK (method IN ('periodic_key','target_date','content_match','none')),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    evidence_json TEXT NOT NULL CHECK (json_valid(evidence_json)),
    candidates_json TEXT NOT NULL CHECK (json_valid(candidates_json)),
    CHECK ((status = 'linked' AND predecessor_rcept_no IS NOT NULL AND evidence_json <> '{}') OR
           (status <> 'linked' AND predecessor_rcept_no IS NULL))
) STRICT;

CREATE TABLE document_status (
    rcept_no TEXT PRIMARY KEY REFERENCES document(rcept_no),
    root_rcept_no TEXT NOT NULL REFERENCES document(rcept_no),
    latest_rcept_no TEXT NOT NULL REFERENCES document(rcept_no),
    is_latest INTEGER NOT NULL CHECK (is_latest IN (0,1)),
    n_corrections INTEGER NOT NULL CHECK (n_corrections >= 0)
) STRICT;

CREATE TABLE build_manifest (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL CHECK (json_valid(value_json))
) STRICT;

CREATE INDEX ix_document_company_date_type ON document(corp_code, rcept_dt, doc_group, doc_subtype);
CREATE INDEX ix_document_status_latest ON document_status(is_latest, latest_rcept_no);
CREATE INDEX ix_chunk_document_path ON chunk(doc_id, path, part);
CREATE INDEX ix_correction_predecessor_status ON correction_link(predecessor_rcept_no, status);
CREATE INDEX ix_event_company_date_type ON event(corp_code, event_date, event_type);
