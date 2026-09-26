"""
analyze_baseline_results.py
============================
Parses the pre-computed baseline_analysis.txt report and
sample_match_features.csv to produce a concise structured summary.

IMPORTANT: Does NOT re-run any expensive streaming evaluation.
           Reads only the pre-computed output files.

Outputs:
    output/baseline_summary.txt  - structured summary
    (also printed to terminal)
"""

import sys
import os
import re
import csv
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Locate project root and output files
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIRS = [
    PROJECT_ROOT / "output",
    PROJECT_ROOT.parent / "output",
]

DATASET_DIR = PROJECT_ROOT / "dataset" / "train"
RAW_TSV_FILES = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv",
]


def find_output_file(filename):
    for d in OUTPUT_DIRS:
        p = d / filename
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find '{filename}' in any of: {OUTPUT_DIRS}"
    )


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_files():
    """Return list of issues; empty list means all good."""
    issues = []

    for fname in ("baseline_analysis.txt", "sample_match_features.csv"):
        try:
            find_output_file(fname)
        except FileNotFoundError as e:
            issues.append(f"MISSING OUTPUT: {e}")

    for fname in RAW_TSV_FILES:
        p = DATASET_DIR / fname
        if not p.exists():
            issues.append(f"MISSING RAW FILE: {p}")
        elif p.stat().st_size == 0:
            issues.append(f"EMPTY RAW FILE: {p}")

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )
        dirty = result.stdout.strip()
        if dirty:
            issues.append(f"GIT WORKING TREE HAS UNCOMMITTED CHANGES:\n{dirty}")
    except Exception as e:
        issues.append(f"Could not check git status: {e}")

    return issues


# ---------------------------------------------------------------------------
# Parse baseline_analysis.txt
# ---------------------------------------------------------------------------

def parse_baseline_report(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    result = {
        "timestamp": None,
        "dataset_scale": None,
        "total_true_pairs": None,
        "candidate_volumes": {},
        "similarity_observations": {},
        "baseline_strategies": {},
        "missing_address_impact": {},
    }

    for line in lines:
        if line.startswith("Timestamp:"):
            result["timestamp"] = line.split(":", 1)[1].strip()
        elif line.startswith("Dataset Scale:"):
            result["dataset_scale"] = line.split(":", 1)[1].strip()
        elif "Full Ground Truth" in line and "true matches" in line:
            m = re.search(r"([\d,]+) true matches", line)
            if m:
                result["total_true_pairs"] = int(m.group(1).replace(",", ""))

    cv = result["candidate_volumes"]
    in_rule = None
    for line in lines:
        if "Total Unconstrained Search Space" in line:
            m = re.search(r"([\d,]+) pairs", line)
            if m:
                cv["total_unconstrained"] = int(m.group(1).replace(",", ""))
        elif "Total Country Candidate Pairs" in line:
            m = re.search(r"([\d,]+)", line)
            if m:
                cv["country_only_total"] = int(m.group(1).replace(",", ""))
        elif "Rule B:" in line and "Exact Raw Name" in line:
            in_rule = "B"
        elif "Rule C:" in line and "Normalized Name" in line:
            in_rule = "C"
        elif line.strip().startswith("- Total Candidate Pairs:") and in_rule:
            m = re.search(r"([\d,]+)", line)
            if m:
                val = int(m.group(1).replace(",", ""))
                if in_rule == "B":
                    cv["exact_name_country_total"] = val
                elif in_rule == "C":
                    cv["norm_name_country_total"] = val
                in_rule = None

    so = result["similarity_observations"]
    for line in lines:
        if "Exact Raw Business Name Match" in line:
            m = re.search(r"([\d.]+)%", line)
            if m:
                so["exact_raw_name_pct"] = float(m.group(1))
        elif "Exact Normalized Name Match" in line:
            m = re.search(r"([\d.]+)%", line)
            if m:
                so["exact_norm_name_pct"] = float(m.group(1))
        elif "Similar Name but Different Formatting" in line:
            m = re.search(r"([\d.]+)%", line)
            if m:
                so["similar_name_diff_format_pct"] = float(m.group(1))
        elif "Exact Country Match" in line:
            m = re.search(r"([\d.]+)%", line)
            if m:
                so["exact_country_pct"] = float(m.group(1))
        elif "Missing Secondary-Source Address" in line and "Source" not in line:
            m = re.search(r"([\d.]+)%", line)
            if m:
                so["missing_address_pct"] = float(m.group(1))
        elif "Source 2 missing address rate" in line:
            m = re.search(r"([\d.]+)%", line)
            if m:
                so["missing_address_s2_pct"] = float(m.group(1))
        elif "Source 3 missing address rate" in line:
            m = re.search(r"([\d.]+)%", line)
            if m:
                so["missing_address_s3_pct"] = float(m.group(1))
        elif "Similar Addresses (JW >= 0.70, all)" in line:
            m = re.search(r"([\d.]+)%", line)
            if m:
                so["addr_sim_jw70_all_pct"] = float(m.group(1))
        elif "Similar Addresses (JW >= 0.70, present)" in line:
            m = re.search(r"([\d.]+)%", line)
            if m:
                so["addr_sim_jw70_present_pct"] = float(m.group(1))

    dist_keys = {
        "Business Name Jaro-Winkler": "name_jw",
        "Business Address Jaro-Winkler (All Pairs)": "addr_jw_all",
        "Business Address Jaro-Winkler (When Present)": "addr_jw_present",
        "Combined Similarity": "combined_sim",
    }
    current_dist = None
    for line in lines:
        for label, key in dist_keys.items():
            if label in line:
                current_dist = key
                so.setdefault("distributions", {})[key] = {}
                break
        if current_dist and "Mean:" in line:
            parts = re.findall(r"(\w+):\s*([\d.]+)", line)
            for k, v in parts:
                so["distributions"][current_dist][k.lower()] = float(v)
        if current_dist and "p10:" in line:
            parts = re.findall(r"p(\d+):\s*([\d.]+)", line)
            for pct, val in parts:
                so["distributions"][current_dist][f"p{pct}"] = float(val)
            current_dist = None

    bs = result["baseline_strategies"]
    current_strategy = None
    for line in lines:
        m = re.match(r"---\s+Strategy:\s+(.+?)\s+---", line)
        if m:
            current_strategy = m.group(1).strip()
            bs[current_strategy] = {}
            continue
        if current_strategy:
            row_m = re.match(
                r"(S2|S3|Overall)\s+\|\s+([\d,]+)\s+\|\s+([\d,]+)\s+\|\s+([\d,]+)"
                r"\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)",
                line.strip()
            )
            if row_m:
                src = row_m.group(1)
                bs[current_strategy][src] = {
                    "tp": int(row_m.group(2).replace(",", "")),
                    "fp": int(row_m.group(3).replace(",", "")),
                    "fn": int(row_m.group(4).replace(",", "")),
                    "precision": float(row_m.group(5)),
                    "recall":    float(row_m.group(6)),
                    "f1":        float(row_m.group(7)),
                }

    ma = result["missing_address_impact"]
    for line in lines:
        if "Requiring a strict address similarity threshold" in line:
            m = re.search(r"([\d.]+)%.*\(([\d,]+) true matches", line)
            if m:
                ma["recall_drop_pct"] = float(m.group(1))
                ma["discarded_matches"] = int(m.group(2).replace(",", ""))
        if "Adding address filtering" in line:
            m = re.search(r"precision from ([\d.]+)% to ([\d.]+)%.*\+([\d.]+)%", line)
            if m:
                ma["addr_filter_precision_gain_pct"] = float(m.group(3))

    return result


# ---------------------------------------------------------------------------
# Analyze sample_match_features.csv
# ---------------------------------------------------------------------------

def analyze_sample_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    n = len(rows)
    if n == 0:
        return {"error": "Empty CSV", "n": 0}

    def to_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    def to_bool(v):
        return str(v).strip().lower() in ("true", "1", "yes")

    name_jw   = [to_float(r["name_jaro_winkler"]) for r in rows]
    addr_jw   = [to_float(r["address_jaro_winkler"]) for r in rows]
    combined  = [to_float(r["combined_similarity"]) for r in rows]
    name_jw_c = [x for x in name_jw if x is not None]
    addr_jw_c = [x for x in addr_jw if x is not None]
    comb_c    = [x for x in combined if x is not None]

    exact_norm_name = sum(1 for r in rows if to_bool(r["normalized_name_match"]))
    addr_missing    = sum(1 for r in rows if to_bool(r["address_is_missing"]))
    addr_present    = [r for r in rows if not to_bool(r["address_is_missing"])]

    high_name_low_addr = sum(
        1 for r in rows
        if (to_float(r["name_jaro_winkler"]) or 0) >= 0.92
        and not to_bool(r["address_is_missing"])
        and (to_float(r["address_jaro_winkler"]) or 1.0) < 0.70
    )
    high_addr_low_name = sum(
        1 for r in rows
        if (to_float(r["address_jaro_winkler"]) or 0) >= 0.90
        and not to_bool(r["address_is_missing"])
        and (to_float(r["name_jaro_winkler"]) or 1.0) < 0.80
    )

    def percentiles(vals, ps=(10, 25, 50, 75, 90, 95, 99)):
        if not vals:
            return {}
        sv = sorted(vals)
        out = {}
        for p in ps:
            idx = min(int(len(sv) * p / 100), len(sv) - 1)
            out[f"p{p}"] = round(sv[idx], 4)
        return out

    def dist_stats(vals):
        if not vals:
            return {}
        mean = sum(vals) / len(vals)
        std  = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
        return {
            "n": len(vals),
            "mean": round(mean, 4),
            "std":  round(std,  4),
            "min":  round(min(vals), 4),
            "max":  round(max(vals), 4),
            **percentiles(vals),
        }

    s2_rows = [r for r in rows if r.get("target_source", "").strip() == "S2"]
    s3_rows = [r for r in rows if r.get("target_source", "").strip() == "S3"]

    def pct(count, total):
        return round(100.0 * count / total, 2) if total > 0 else 0.0

    addr_present_jw = [
        to_float(r["address_jaro_winkler"])
        for r in addr_present
        if to_float(r["address_jaro_winkler"]) is not None
    ]

    return {
        "n": n,
        "n_s2": len(s2_rows),
        "n_s3": len(s3_rows),
        "exact_norm_name_pct":    pct(exact_norm_name, n),
        "missing_address_pct":    pct(addr_missing, n),
        "high_name_low_addr_pct": pct(high_name_low_addr, n),
        "high_addr_low_name_pct": pct(high_addr_low_name, n),
        "name_jw":        dist_stats(name_jw_c),
        "addr_jw_all":    dist_stats(addr_jw_c),
        "addr_jw_present":dist_stats(addr_present_jw),
        "combined_sim":   dist_stats(comb_c),
        "name_jw_brackets": {
            "< 0.70":         pct(sum(1 for v in name_jw_c if v < 0.70), n),
            "0.70 – 0.80":    pct(sum(1 for v in name_jw_c if 0.70 <= v < 0.80), n),
            "0.80 – 0.90":    pct(sum(1 for v in name_jw_c if 0.80 <= v < 0.90), n),
            "0.90 – 0.95":    pct(sum(1 for v in name_jw_c if 0.90 <= v < 0.95), n),
            ">= 0.95":        pct(sum(1 for v in name_jw_c if v >= 0.95), n),
        },
        "addr_jw_brackets": {
            "missing":        pct(addr_missing, n),
            "< 0.60":         pct(sum(1 for v in addr_jw_c if v < 0.60), n),
            "0.60 – 0.75":    pct(sum(1 for v in addr_jw_c if 0.60 <= v < 0.75), n),
            "0.75 – 0.90":    pct(sum(1 for v in addr_jw_c if 0.75 <= v < 0.90), n),
            ">= 0.90":        pct(sum(1 for v in addr_jw_c if v >= 0.90), n),
        },
    }


# ---------------------------------------------------------------------------
# Identify ML failure patterns
# ---------------------------------------------------------------------------

def identify_failure_patterns(parsed_report, sample_stats):
    patterns = []
    bs = parsed_report.get("baseline_strategies", {})
    so = parsed_report.get("similarity_observations", {})
    ma = parsed_report.get("missing_address_impact", {})

    simple_key = next((k for k in bs if "Simple Baseline" in k or
                       ("Normalized Name + Country" in k and "Address" not in k)), None)

    if simple_key:
        overall = bs[simple_key].get("Overall", {})
        fn = overall.get("fn", 0)
        recall = overall.get("recall", 0)
        patterns.append(
            f"HIGH FALSE-NEGATIVE RATE (Priority 1 — Blocking Gap): "
            f"Normalized Name + Country baseline misses {fn:,} true pairs "
            f"(Recall = {recall:.1%}). Most true matches fall below the exact "
            f"normalized-name blocking threshold. The ML pipeline must use fuzzy "
            f"name blocking (prefix, token index, n-gram) to recover these FNs."
        )

    exact_raw = so.get("exact_raw_name_pct", 0)
    exact_norm = so.get("exact_norm_name_pct", 0)
    patterns.append(
        f"NAME SURFACE-FORM VARIATION (Priority 2 — Feature Gap): "
        f"Only {exact_raw}% of true matches share an exact raw name; only "
        f"{exact_norm}% share an exact normalized name. The remaining "
        f"~{100 - exact_norm:.1f}% have legal suffix, abbreviation, casing, "
        f"word-order, or typo differences. ML needs: Jaro-Winkler, token Jaccard, "
        f"token-set ratio, length ratio, and prefix-match features."
    )

    hn_la = sample_stats.get("high_name_low_addr_pct", 0)
    patterns.append(
        f"HIGH-NAME / LOW-ADDRESS DISAGREEMENT ({hn_la:.1f}% of true-match sample): "
        f"True matches exist where name JW >= 0.92 but address JW < 0.70. "
        f"These arise from abbreviated vs. expanded address styles or location "
        f"data latency differences. A hard address threshold rejects them. "
        f"ML must learn to trust strong name signal when address disagreement is present."
    )

    miss_pct = so.get("missing_address_pct", 0)
    discarded = ma.get("discarded_matches", 0)
    patterns.append(
        f"MISSING ADDRESS HANDLING ({miss_pct:.1f}% of true pairs, "
        f"{discarded:,} discarded under strict rules): Secondary-source entities "
        f"frequently have empty address fields. Any rule requiring address similarity "
        f"unconditionally misses these. ML needs: is_address_missing binary feature "
        f"and a name-only pathway when address is absent."
    )

    addr_all = so.get("addr_sim_jw70_all_pct", 0)
    addr_pres = so.get("addr_sim_jw70_present_pct", 0)
    patterns.append(
        f"ADDRESS FORMULATION VARIATION ({100 - addr_pres:.1f}% of present-address "
        f"pairs have JW < 0.70): Street abbreviations, suite numbering, city spelling, "
        f"and postal-code differences cause true matches to score poorly on raw JW. "
        f"ML should use token-sort ratio, structured sub-field extraction (city, "
        f"postal code), and Levenshtein alongside JW."
    )

    if simple_key:
        overall = bs[simple_key].get("Overall", {})
        prec = overall.get("precision", 0)
        fp = overall.get("fp", 0)
        patterns.append(
            f"BLOCKING OVER-GENERATION / FALSE POSITIVES ({fp:,} FP pairs, "
            f"Precision = {prec:.1%} for norm-name baseline): Many businesses "
            f"share similar names within a country. ML must learn to distinguish "
            f"same-name-different-entity cases using address, token structure, "
            f"and any available auxiliary fields."
        )

    ha_ln = sample_stats.get("high_addr_low_name_pct", 0)
    if ha_ln > 0:
        patterns.append(
            f"HIGH-ADDRESS / LOW-NAME DISAGREEMENT ({ha_ln:.1f}% of true-match sample): "
            f"Some true matches share a high address similarity (JW >= 0.90) but "
            f"name JW < 0.80. These may be business-name-change scenarios or "
            f"franchises recorded differently. ML can use address as a recovery signal "
            f"when name similarity alone is insufficient."
        )

    return patterns


# ---------------------------------------------------------------------------
# Format summary
# ---------------------------------------------------------------------------

def format_summary(parsed_report, sample_stats, failure_patterns, verification_issues):
    W = 80
    lines = []

    def section(title):
        lines.append("=" * W)
        lines.append(title)
        lines.append("=" * W)

    def sub(title):
        lines.append("")
        lines.append("-" * W)
        lines.append(title)
        lines.append("-" * W)

    def row(label, value, width=42):
        lines.append(f"  {label:<{width}} {value}")

    section("AMAZON ML CHALLENGE 2026 - BASELINE RESULTS SUMMARY")
    lines.append(f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Source    : {parsed_report.get('timestamp', 'N/A')}")
    lines.append(f"Scale     : {parsed_report.get('dataset_scale', 'N/A')}")
    n_pairs = parsed_report.get("total_true_pairs", 0) or 0
    lines.append(f"Evaluation: Full Ground Truth - {n_pairs:,} true pairs")

    # 0. Verification
    sub("0. PRE-RUN VERIFICATION")
    if verification_issues:
        for issue in verification_issues:
            lines.append(f"  [FAIL] {issue}")
    else:
        lines.append("  [OK] baseline_analysis.txt exists")
        lines.append("  [OK] sample_match_features.csv exists")
        lines.append("  [OK] All 4 raw dataset TSV files present and non-empty")
        lines.append("  [OK] Git working tree is clean")

    # 1. Candidate volume
    sub("1. CANDIDATE VOLUME REDUCTION")
    cv = parsed_report.get("candidate_volumes", {})
    row("Total unconstrained search space:",
        f"{cv.get('total_unconstrained', 0):,} pairs  (~22.77 Trillion)")
    row("After country blocking only:",
        f"{cv.get('country_only_total', 0):,} pairs  [~48% reduction]")
    row("After exact name + country:",
        f"{cv.get('exact_name_country_total', 0):,} pairs  [99.999963% reduction]")
    row("After norm name + country:",
        f"{cv.get('norm_name_country_total', 0):,} pairs  [99.9999% reduction]")

    # 2. Baseline evaluation
    sub("2. BASELINE MATCHER EVALUATION (Full Ground Truth)")

    bs = parsed_report.get("baseline_strategies", {})

    strategy_display = [
        ("Exact Name + Country",                    "1. Exact Name + Country"),
        ("Normalized Name + Country (Simple",       "2. Normalized Name + Country (Simple Baseline)"),
        ("Normalized Name + Country + Address",     None),   # split into strict/tolerant below
    ]

    # Find all strategy keys and their display names
    strat_items = []
    for k in bs:
        if "Exact Name" in k and "Normalized" not in k:
            strat_items.append((k, "1. Exact Name + Country"))
        elif "Simple Baseline" in k or ("Normalized Name + Country" in k and "Address" not in k):
            strat_items.append((k, "2. Normalized Name + Country (Simple Baseline)"))
        elif "Strict" in k:
            strat_items.append((k, "3. Norm Name + Country + Address JW>=0.6 (Strict)"))
        elif "Missing-Tolerant" in k or "Tolerant" in k:
            strat_items.append((k, "4. Norm Name + Country + Address JW>=0.6 (Missing-Tolerant)"))

    for actual_key, display_name in strat_items:
        lines.append("")
        lines.append(f"  Strategy: {display_name}")
        hdr = (f"  {'Source':<10} {'TP':>12} {'FP':>12} {'FN':>12} "
               f"{'Precision':>10} {'Recall':>10} {'F1':>10}")
        lines.append(hdr)
        lines.append("  " + "-" * 72)
        for src in ("S2", "S3", "Overall"):
            d = bs.get(actual_key, {}).get(src)
            if d:
                lines.append(
                    f"  {src:<10} {d['tp']:>12,} {d['fp']:>12,} {d['fn']:>12,} "
                    f"{d['precision']:>10.4f} {d['recall']:>10.4f} {d['f1']:>10.4f}"
                )
            else:
                lines.append(f"  {src:<10} {'(data not found)':>52}")

    # 3. Sample feature analysis
    sub("3. SAMPLE TRUE-MATCH FEATURE ANALYSIS (N=5,000)")
    lines.append(f"  Sample: {sample_stats['n']:,} pairs  "
                 f"(S2: {sample_stats['n_s2']:,}, S3: {sample_stats['n_s3']:,})")
    lines.append("")

    lines.append("  Match Rates Among True Matches:")
    row("Exact normalized name match:",
        f"{sample_stats['exact_norm_name_pct']:.2f}%")
    row("Missing secondary address:",
        f"{sample_stats['missing_address_pct']:.2f}%")
    row("High name sim (>=0.92) AND low addr sim (<0.70):",
        f"{sample_stats['high_name_low_addr_pct']:.2f}%")
    row("High addr sim (>=0.90) AND low name sim (<0.80):",
        f"{sample_stats['high_addr_low_name_pct']:.2f}%")

    def fmt_dist(d):
        return (f"mean={d.get('mean', 0):.4f} std={d.get('std', 0):.4f}  "
                f"p10={d.get('p10',0):.4f} p25={d.get('p25',0):.4f} "
                f"p50={d.get('p50',0):.4f} p75={d.get('p75',0):.4f} "
                f"p90={d.get('p90',0):.4f} p95={d.get('p95',0):.4f}")

    lines.append("")
    lines.append("  Name Jaro-Winkler Distribution:")
    lines.append(f"    {fmt_dist(sample_stats.get('name_jw', {}))}")
    lines.append("    Brackets (% of N=5,000):")
    for label, val in sample_stats.get("name_jw_brackets", {}).items():
        lines.append(f"      JW {label:<14}: {val:>6.2f}%")

    lines.append("")
    lines.append("  Address Jaro-Winkler (All Pairs, missing addr scored 0.0):")
    lines.append(f"    {fmt_dist(sample_stats.get('addr_jw_all', {}))}")
    lines.append("    Brackets (% of N=5,000, including missing):")
    for label, val in sample_stats.get("addr_jw_brackets", {}).items():
        lines.append(f"      {label:<16}: {val:>6.2f}%")

    lines.append("")
    lines.append("  Address Jaro-Winkler (When Present Only):")
    lines.append(f"    {fmt_dist(sample_stats.get('addr_jw_present', {}))}")

    lines.append("")
    lines.append("  Combined Similarity (0.6*Name + 0.4*Addr | fallback to Name):")
    lines.append(f"    {fmt_dist(sample_stats.get('combined_sim', {}))}")

    # 4. Failure patterns
    sub("4. KEY FAILURE PATTERNS FOR ML TO LEARN (Priority Order)")
    for i, pattern in enumerate(failure_patterns, 1):
        lines.append("")
        # Word-wrap at 76 chars
        words = pattern.split()
        cur_line = f"  [{i}] "
        indent = "       "
        for word in words:
            if len(cur_line) + len(word) + 1 > 78:
                lines.append(cur_line)
                cur_line = indent + word + " "
            else:
                cur_line += word + " "
        if cur_line.strip():
            lines.append(cur_line.rstrip())

    # 5. Recommendations
    sub("5. BLOCKING & FEATURE ENGINEERING RECOMMENDATIONS")
    lines.append("  Blocking:")
    lines.append("    * Country match is mandatory (100% of true matches share country).")
    lines.append("    * Multi-Tier Blocking Union:")
    lines.append("        Tier 1: Exact Normalized Name + Country             (precision core)")
    lines.append("        Tier 2: Name First-5-Char Prefix + Country          (typos/abbrevs)")
    lines.append("        Tier 3: Name Token Inverted Index + Country         (word reordering)")
    lines.append("    * Union targets >90% recall at <35M candidates.")
    lines.append("")
    lines.append("  Features for ML model:")
    lines.append("    * is_address_missing               (binary)")
    lines.append("    * name_jaro_winkler")
    lines.append("    * name_token_jaccard")
    lines.append("    * name_levenshtein_similarity")
    lines.append("    * name_length_ratio")
    lines.append("    * name_prefix_match_length")
    lines.append("    * address_jaro_winkler             (when present)")
    lines.append("    * address_token_sort_ratio         (when present)")
    lines.append("    * address_levenshtein              (when present)")
    lines.append("    * country_match                    (should always be True post-blocking)")
    lines.append("")
    lines.append("  Decision Heuristics (from baseline report):")
    lines.append("    * Name JW >= 0.92 AND address missing -> HIGH CONFIDENCE MATCH")
    lines.append("    * Name JW in [0.75, 0.92] AND addr JW >= 0.70 -> HIGH CONFIDENCE MATCH")
    lines.append("    * Addr JW < 0.35 AND address present -> REJECT (conflicting location)")

    lines.append("")
    lines.append("=" * W)
    lines.append("END OF BASELINE SUMMARY")
    lines.append("=" * W)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("BASELINE RESULTS ANALYSIS")
    print("=" * 60)

    print("\n[1/4] Verifying required files and git state...")
    issues = verify_files()
    if issues:
        for issue in issues:
            print(f"  [FAIL] {issue}")
    else:
        print("  All checks passed.")

    print("\n[2/4] Parsing output/baseline_analysis.txt ...")
    report_path = find_output_file("baseline_analysis.txt")
    parsed = parse_baseline_report(report_path)
    print(f"  Parsed {len(parsed['baseline_strategies'])} baseline strategies.")

    print("\n[3/4] Analyzing output/sample_match_features.csv ...")
    csv_path = find_output_file("sample_match_features.csv")
    sample_stats = analyze_sample_csv(csv_path)
    print(f"  Analyzed {sample_stats['n']:,} sample pairs "
          f"(S2: {sample_stats['n_s2']:,}, S3: {sample_stats['n_s3']:,}).")

    print("\n[4/4] Identifying ML failure patterns ...")
    patterns = identify_failure_patterns(parsed, sample_stats)
    print(f"  Identified {len(patterns)} key failure patterns.")

    summary = format_summary(parsed, sample_stats, patterns, issues)

    written = []
    for d in OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        out_path = d / "baseline_summary.txt"
        out_path.write_text(summary, encoding="utf-8")
        written.append(str(out_path))

    print(f"\n  Saved:")
    for p in written:
        print(f"    {p}")

    print("\n" + "=" * 60)
    print("FULL SUMMARY")
    print("=" * 60)
    print(summary)


if __name__ == "__main__":
    main()
