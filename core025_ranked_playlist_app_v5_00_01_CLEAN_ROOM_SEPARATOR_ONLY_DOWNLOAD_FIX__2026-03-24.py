
import io
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

APP_VERSION_STR = "v5.00.01-clean-room-separator-only-download-fix"

MEMBERS = ["0025", "0225", "0255"]
MEMBER_COLS = {"0025": "score_0025", "0225": "score_0225", "0255": "score_0255"}

# -----------------------------
# Helpers
# -----------------------------

def normalize_member(x) -> str:
    s = str(x).strip().replace("'", "")
    if s == "" or s.lower() == "nan":
        return ""
    s = s.replace(".0", "")
    if s in {"25", "025", "0025"}:
        return "0025"
    if s in {"225", "0225"}:
        return "0225"
    if s in {"255", "0255"}:
        return "0255"
    return s.zfill(4)


def clean_seed_text(x) -> str:
    s = str(x).strip().replace("'", "")
    if s == "" or s.lower() == "nan":
        return ""
    s = s.replace(".0", "")
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits.zfill(4) if digits else ""


def to_int_or_none(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def parse_digit_set(value) -> Optional[set]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    s = s.strip("[](){}")
    parts = re.split(r"[\s,|;/]+", s)
    vals = []
    for p in parts:
        if p == "":
            continue
        try:
            vals.append(int(float(p)))
        except Exception:
            pass
    return set(vals) if vals else None


def parse_text_set(value) -> Optional[set]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    s = s.strip("[](){}")
    parts = [p.strip().upper() for p in re.split(r"[\s,|;/]+", s) if p.strip()]
    return set(parts) if parts else None


def sha256_of_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Normalize key identity columns
    if "date" not in out.columns:
        if "PlayDate" in out.columns:
            out["date"] = out["PlayDate"].astype(str)
        else:
            out["date"] = ""

    if "stream" not in out.columns:
        if "StreamKey" in out.columns:
            out["stream"] = out["StreamKey"].astype(str).str.strip().str.lower()
        elif "WinningStreamKey" in out.columns:
            out["stream"] = out["WinningStreamKey"].astype(str).str.strip().str.lower()
        else:
            out["stream"] = ""

    if "feat_seed" not in out.columns:
        if "PrevSeed_text" in out.columns:
            out["feat_seed"] = out["PrevSeed_text"].map(clean_seed_text)
        elif "PrevSeed" in out.columns:
            out["feat_seed"] = out["PrevSeed"].map(clean_seed_text)
        elif "seed" in out.columns:
            out["feat_seed"] = out["seed"].map(clean_seed_text)
        else:
            out["feat_seed"] = ""

    if "true_member" not in out.columns:
        if "WinningMember_text" in out.columns:
            out["true_member"] = out["WinningMember_text"].map(normalize_member)
        elif "WinningMember" in out.columns:
            out["true_member"] = out["WinningMember"].map(normalize_member)
        else:
            out["true_member"] = ""

    # If feature columns already exist, keep them. Otherwise derive.
    need_seed_derivation = "seed_sum" not in out.columns

    # Seed digits
    out["feat_seed"] = out["feat_seed"].fillna("").astype(str).map(clean_seed_text)
    digits = out["feat_seed"].apply(lambda s: [int(ch) for ch in s] if len(s) == 4 and s.isdigit() else [None, None, None, None])
    out["seed_pos1"] = [d[0] for d in digits]
    out["seed_pos2"] = [d[1] for d in digits]
    out["seed_pos3"] = [d[2] for d in digits]
    out["seed_pos4"] = [d[3] for d in digits]

    if need_seed_derivation:
        ds = digits
        out["seed_sum"] = [sum(d) if None not in d else np.nan for d in ds]
        out["seed_sum_lastdigit"] = out["seed_sum"] % 10
        out["seed_sum_mod3"] = out["seed_sum"] % 3
        out["seed_sum_mod4"] = out["seed_sum"] % 4
        out["seed_sum_mod5"] = out["seed_sum"] % 5
        out["seed_sum_mod6"] = out["seed_sum"] % 6
        out["seed_sum_mod9"] = out["seed_sum"] % 9
        out["seed_sum_mod10"] = out["seed_sum"] % 10
        out["seed_sum_mod11"] = out["seed_sum"] % 11
        out["seed_sum_mod12"] = out["seed_sum"] % 12
        out["seed_sum_mod13"] = out["seed_sum"] % 13
        out["seed_root"] = out["seed_sum"].apply(lambda x: int((x - 1) % 9 + 1) if pd.notna(x) and x > 0 else (0 if pd.notna(x) else np.nan))
        out["seed_spread"] = [max(d) - min(d) if None not in d else np.nan for d in ds]
        out["seed_unique_digits"] = [len(set(d)) if None not in d else np.nan for d in ds]
        out["seed_even_cnt"] = [sum(1 for x in d if x % 2 == 0) if None not in d else np.nan for d in ds]
        out["seed_odd_cnt"] = [sum(1 for x in d if x % 2 == 1) if None not in d else np.nan for d in ds]
        out["seed_high_cnt"] = [sum(1 for x in d if x >= 5) if None not in d else np.nan for d in ds]
        out["seed_low_cnt"] = [sum(1 for x in d if x <= 4) if None not in d else np.nan for d in ds]
        out["seed_first_last_sum"] = [d[0] + d[3] if None not in d else np.nan for d in ds]
        out["seed_middle_sum"] = [d[1] + d[2] if None not in d else np.nan for d in ds]
        out["seed_pairwise_absdiff_sum"] = [abs(d[0]-d[1]) + abs(d[2]-d[3]) if None not in d else np.nan for d in ds]
        out["seed_adj_absdiff_sum"] = [abs(d[0]-d[1]) + abs(d[1]-d[2]) + abs(d[2]-d[3]) if None not in d else np.nan for d in ds]
        out["seed_adj_absdiff_min"] = [min(abs(d[0]-d[1]), abs(d[1]-d[2]), abs(d[2]-d[3])) if None not in d else np.nan for d in ds]
        out["seed_highlow_pattern"] = ["".join("H" if x >= 5 else "L" for x in d) if None not in d else "" for d in ds]
        out["cnt_0_3"] = [sum(1 for x in d if 0 <= x <= 3) if None not in d else np.nan for d in ds]
        out["cnt_4_6"] = [sum(1 for x in d if 4 <= x <= 6) if None not in d else np.nan for d in ds]
        out["cnt_7_9"] = [sum(1 for x in d if 7 <= x <= 9) if None not in d else np.nan for d in ds]
        # Vtrac groups: 0/5,1/6,2/7,3/8,4/9
        def vtrac_group(x: int) -> int:
            return x % 5
        out["seed_vtrac_groups"] = [len(set(vtrac_group(x) for x in d)) if None not in d else np.nan for d in ds]
        for digit in range(10):
            out[f"seed_has{digit}"] = [1 if (None not in d and digit in d) else 0 for d in ds]
            out[f"seed_cnt{digit}"] = [sum(1 for x in d if x == digit) if None not in d else np.nan for d in ds]

    # Ensure all needed cols exist
    needed = [
        "seed_sum", "seed_sum_lastdigit", "seed_sum_mod3", "seed_sum_mod4", "seed_sum_mod5", "seed_sum_mod6",
        "seed_sum_mod9", "seed_sum_mod10", "seed_sum_mod11", "seed_sum_mod12", "seed_sum_mod13", "seed_root",
        "seed_spread", "seed_even_cnt", "seed_odd_cnt", "seed_high_cnt", "seed_low_cnt", "seed_vtrac_groups",
        "seed_unique_digits", "seed_pos1", "seed_pos2", "seed_pos3", "seed_pos4", "seed_first_last_sum",
        "seed_middle_sum", "seed_pairwise_absdiff_sum", "seed_adj_absdiff_sum", "seed_adj_absdiff_min",
        "seed_highlow_pattern", "cnt_0_3", "cnt_4_6", "cnt_7_9"
    ]
    for c in needed:
        if c not in out.columns:
            out[c] = np.nan

    return out


@dataclass
class OverlayRule:
    rule_id: str
    enabled: bool
    conditions: Dict[str, object]
    deltas: Dict[str, float]
    note: str


def parse_overlay_file(upload_bytes: bytes, upload_name: str) -> Tuple[List[OverlayRule], Dict]:
    if upload_bytes is None:
        return [], {"source": "none", "rows": 0, "sha256": "", "filename": ""}
    data = upload_bytes
    name = upload_name or "overlay.csv"
    sha = sha256_of_bytes(data)

    if name.lower().endswith(".txt"):
        # accept CSV-like TXT
        df = pd.read_csv(io.BytesIO(data))
    else:
        df = pd.read_csv(io.BytesIO(data))

    df.columns = [str(c).strip() for c in df.columns]
    rules: List[OverlayRule] = []
    for _, row in df.iterrows():
        enabled = str(row.get("enabled", "1")).strip()
        enabled_bool = False if enabled.lower() in {"0", "false", "no"} else True
        conditions = {}
        for col in df.columns:
            if col.startswith("when_") or col in {"rank_min", "rank_max"}:
                val = row.get(col)
                try:
                    if pd.isna(val):
                        continue
                except Exception:
                    pass
                sval = str(val).strip()
                if sval == "" or sval.lower() == "nan":
                    continue
                conditions[col] = val

        deltas = {
            "0025": float(row.get("delta_0025", 0) if not pd.isna(row.get("delta_0025", np.nan)) else 0),
            "0225": float(row.get("delta_0225", 0) if not pd.isna(row.get("delta_0225", np.nan)) else 0),
            "0255": float(row.get("delta_0255", 0) if not pd.isna(row.get("delta_0255", np.nan)) else 0),
        }
        rules.append(
            OverlayRule(
                rule_id=str(row.get("rule_id", f"rule_{len(rules)+1}")),
                enabled=enabled_bool,
                conditions=conditions,
                deltas=deltas,
                note=str(row.get("note", "")),
            )
        )
    meta = {"source": "upload", "rows": len(df), "sha256": sha, "filename": name}
    return rules, meta


def match_rule_to_row(rule: OverlayRule, row: pd.Series) -> bool:
    if not rule.enabled:
        return False

    # Helper getters
    def g(col, default=np.nan):
        return row[col] if col in row.index else default

    # Explicit top/rank conditions if present
    if "when_base_top1" in rule.conditions:
        if normalize_member(g("Top1_pred", "")) != normalize_member(rule.conditions["when_base_top1"]):
            return False
    if "when_base_top2" in rule.conditions:
        if normalize_member(g("Top2_pred", "")) != normalize_member(rule.conditions["when_base_top2"]):
            return False
    if "when_base_top3" in rule.conditions:
        if normalize_member(g("Top3_pred", "")) != normalize_member(rule.conditions["when_base_top3"]):
            return False

    # Numeric helpers
    numeric_exact_map = {
        "when_seed_sum_in": "seed_sum",
        "when_seed_sum_lastdigit_in": "seed_sum_lastdigit",
        "when_seed_sum_lastdigit_not_in": "seed_sum_lastdigit",
        "when_seed_sum_mod3_in": "seed_sum_mod3",
        "when_seed_sum_mod5_in": "seed_sum_mod5",
        "when_seed_sum_mod6_in": "seed_sum_mod6",
        "when_seed_sum_mod9_in": "seed_sum_mod9",
        "when_seed_sum_mod10_in": "seed_sum_mod10",
        "when_seed_sum_mod11_in": "seed_sum_mod11",
        "when_seed_sum_mod12_in": "seed_sum_mod12",
        "when_seed_sum_mod13_in": "seed_sum_mod13",
        "when_seed_root_in": "seed_root",
        "when_seed_root_not_in": "seed_root",
        "when_seed_first_last_sum_in": "seed_first_last_sum",
        "when_seed_first_last_sum_not_in": "seed_first_last_sum",
        "when_seed_middle_sum_in": "seed_middle_sum",
        "when_seed_middle_sum_not_in": "seed_middle_sum",
        "when_seed_pairwise_absdiff_sum_in": "seed_pairwise_absdiff_sum",
        "when_seed_pairwise_absdiff_sum_not_in": "seed_pairwise_absdiff_sum",
        "when_seed_adj_absdiff_sum_in": "seed_adj_absdiff_sum",
        "when_seed_adj_absdiff_sum_not_in": "seed_adj_absdiff_sum",
        "when_seed_adj_absdiff_min_in": "seed_adj_absdiff_min",
        "when_seed_adj_absdiff_min_not_in": "seed_adj_absdiff_min",
        "when_seed_pos1_in": "seed_pos1",
        "when_seed_pos1_not_in": "seed_pos1",
        "when_seed_pos2_in": "seed_pos2",
        "when_seed_pos2_not_in": "seed_pos2",
        "when_seed_pos3_in": "seed_pos3",
        "when_seed_pos3_not_in": "seed_pos3",
        "when_seed_pos4_in": "seed_pos4",
        "when_seed_pos4_not_in": "seed_pos4",
    }

    for cond_col, feat_col in numeric_exact_map.items():
        if cond_col in rule.conditions:
            vals = parse_digit_set(rule.conditions[cond_col])
            rv = to_int_or_none(g(feat_col))
            if rv is None:
                return False
            if cond_col.endswith("_not_in"):
                if vals and rv in vals:
                    return False
            else:
                if vals and rv not in vals:
                    return False

    numeric_range_map = {
        "when_seed_sum_min": ("seed_sum", "min"),
        "when_seed_sum_max": ("seed_sum", "max"),
        "when_seed_spread_min": ("seed_spread", "min"),
        "when_seed_spread_max": ("seed_spread", "max"),
        "when_seed_high_min": ("seed_high_cnt", "min"),
        "when_seed_high_max": ("seed_high_cnt", "max"),
        "when_seed_low_min": ("seed_low_cnt", "min"),
        "when_seed_low_max": ("seed_low_cnt", "max"),
        "when_seed_vtrac_groups_min": ("seed_vtrac_groups", "min"),
        "when_seed_vtrac_groups_max": ("seed_vtrac_groups", "max"),
        "when_seed_count_digits_min": ("seed_unique_digits", "min"),
        "when_seed_count_digits_max": ("seed_unique_digits", "max"),
        "when_seed_first_last_sum_min": ("seed_first_last_sum", "min"),
        "when_seed_first_last_sum_max": ("seed_first_last_sum", "max"),
        "when_seed_middle_sum_min": ("seed_middle_sum", "min"),
        "when_seed_middle_sum_max": ("seed_middle_sum", "max"),
        "when_seed_pairwise_absdiff_sum_min": ("seed_pairwise_absdiff_sum", "min"),
        "when_seed_pairwise_absdiff_sum_max": ("seed_pairwise_absdiff_sum", "max"),
        "when_seed_adj_absdiff_sum_min": ("seed_adj_absdiff_sum", "min"),
        "when_seed_adj_absdiff_sum_max": ("seed_adj_absdiff_sum", "max"),
        "when_seed_adj_absdiff_min_min": ("seed_adj_absdiff_min", "min"),
        "when_seed_adj_absdiff_min_max": ("seed_adj_absdiff_min", "max"),
    }
    for cond_col, (feat_col, kind) in numeric_range_map.items():
        if cond_col in rule.conditions:
            rv = to_int_or_none(g(feat_col))
            cv = to_int_or_none(rule.conditions[cond_col])
            if rv is None or cv is None:
                return False
            if kind == "min" and rv < cv:
                return False
            if kind == "max" and rv > cv:
                return False

    # Digit count set logic for actual digit-pool counts
    if "when_seed_count_digits_set" in rule.conditions:
        digit_set = parse_digit_set(rule.conditions["when_seed_count_digits_set"])
        if not digit_set:
            return False
        cnt = 0
        for d in digit_set:
            col = f"seed_cnt{d}"
            cnt += int(to_int_or_none(g(col)) or 0)
        mn = to_int_or_none(rule.conditions.get("when_seed_count_digits_min", None))
        mx = to_int_or_none(rule.conditions.get("when_seed_count_digits_max", None))
        if mn is not None and cnt < mn:
            return False
        if mx is not None and cnt > mx:
            return False

    # contains any/all/none
    seed_text = clean_seed_text(g("feat_seed", ""))
    if "when_seed_contains_any" in rule.conditions:
        vals = parse_digit_set(rule.conditions["when_seed_contains_any"])
        if vals and not any(str(v) in seed_text for v in vals):
            return False
    if "when_seed_contains_all" in rule.conditions:
        vals = parse_digit_set(rule.conditions["when_seed_contains_all"])
        if vals and not all(str(v) in seed_text for v in vals):
            return False
    if "when_seed_contains_none" in rule.conditions:
        vals = parse_digit_set(rule.conditions["when_seed_contains_none"])
        if vals and any(str(v) in seed_text for v in vals):
            return False

    # text pattern
    if "when_seed_highlow_pattern" in rule.conditions:
        allowed = parse_text_set(rule.conditions["when_seed_highlow_pattern"])
        if allowed and str(g("seed_highlow_pattern", "")).upper() not in allowed:
            return False

    return True


def apply_separator_rules(df: pd.DataFrame, rules: List[OverlayRule]) -> pd.DataFrame:
    out = df.copy()
    for m in MEMBERS:
        out[MEMBER_COLS[m]] = 0.0
    fired_ids = []
    fired_counts = []
    fired_notes = []

    rule_ids_per_row = []
    notes_per_row = []

    for _, row in out.iterrows():
        row_rule_ids = []
        row_notes = []
        for rule in rules:
            if match_rule_to_row(rule, row):
                for m in MEMBERS:
                    out.at[row.name, MEMBER_COLS[m]] += float(rule.deltas.get(m, 0.0))
                row_rule_ids.append(rule.rule_id)
                if rule.note:
                    row_notes.append(rule.note)
        rule_ids_per_row.append("|".join(row_rule_ids))
        notes_per_row.append(" || ".join(row_notes))

    out["FiredRuleIDs"] = rule_ids_per_row
    out["FiredRuleNotes"] = notes_per_row

    # Derive rankings per row
    top1s, top2s, top3s = [], [], []
    top1scores, margins = [], []
    for _, row in out.iterrows():
        scores = [(m, float(row[MEMBER_COLS[m]])) for m in MEMBERS]
        scores.sort(key=lambda x: (x[1], x[0]), reverse=True)
        ranked_members = [m for m, _ in scores]
        ranked_scores = [s for _, s in scores]
        top1s.append(ranked_members[0])
        top2s.append(ranked_members[1])
        top3s.append(ranked_members[2])
        top1scores.append(ranked_scores[0])
        margins.append(ranked_scores[0] - ranked_scores[1])

    out["PredictedMember"] = top1s
    out["Top1_pred"] = top1s
    out["Top2_pred"] = top2s
    out["Top3_pred"] = top3s
    out["Top1Score"] = top1scores
    out["Top1Margin"] = margins
    out["AnyRuleFired"] = out["FiredRuleIDs"].ne("")

    return out


def rank_within_date(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        out["date"] = ""
    out["StreamScore"] = out["Top1Score"] * 1000 + out["Top1Margin"] * 100 + out["score_0025"] + out["score_0225"] + out["score_0255"]
    out = out.sort_values(["date", "StreamScore", "Top1Margin", "stream"], ascending=[True, False, False, True]).copy()
    out["Rank"] = out.groupby("date").cumcount() + 1
    out["Selected50"] = (out["Rank"] <= top_n).astype(int)
    out["CorrectMember"] = ((out["Selected50"] == 1) & (out["PredictedMember"] == out["true_member"])).astype(int)
    out["Top2Needed"] = ((out["Selected50"] == 1) & (out["PredictedMember"] != out["true_member"]) & (out["Top2_pred"] == out["true_member"])).astype(int)
    out["Top3Only"] = ((out["Selected50"] == 1) & (out["PredictedMember"] != out["true_member"]) & (out["Top2_pred"] != out["true_member"]) & (out["Top3_pred"] == out["true_member"])).astype(int)
    out["CapturedMember"] = ((out["Selected50"] == 1) & (out["PredictedMember"] == out["true_member"])).astype(int)
    out["PrevSeedHas9"] = out["feat_seed"].fillna("").astype(str).str.contains("9").astype(int)
    return out


def summarize_results(df: pd.DataFrame) -> Dict[str, int]:
    selected = df[df["Selected50"] == 1]
    return {
        "Selected50": int(selected.shape[0]),
        "Correct-member": int(selected["CorrectMember"].sum() + selected["Top2Needed"].sum()),
        "Top1": int(selected["CorrectMember"].sum()),
        "Top2-needed": int(selected["Top2Needed"].sum()),
        "Top3-only": int(selected["Top3Only"].sum()),
    }


def summarize_split(df: pd.DataFrame, has9_value: int) -> Dict[str, int]:
    sub = df[(df["Selected50"] == 1) & (df["PrevSeedHas9"] == has9_value)]
    return {
        "rows": int(sub.shape[0]),
        "Correct-member": int(sub["CorrectMember"].sum() + sub["Top2Needed"].sum()),
        "Top1": int(sub["CorrectMember"].sum()),
        "Top2-needed": int(sub["Top2Needed"].sum()),
        "Top3-only": int(sub["Top3Only"].sum()),
    }


def build_date_export(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("date", dropna=False).agg(
        total_rows=("stream", "size"),
        selected50=("Selected50", "sum"),
        correct_member=("CorrectMember", "sum"),
        top2_needed=("Top2Needed", "sum"),
        top3_only=("Top3Only", "sum"),
    ).reset_index()
    grp["top1"] = grp["correct_member"]
    return grp


def build_stream_export(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("stream", dropna=False).agg(
        rows=("date", "size"),
        selected50=("Selected50", "sum"),
        correct_member=("CorrectMember", "sum"),
        top2_needed=("Top2Needed", "sum"),
        top3_only=("Top3Only", "sum"),
        avg_rank=("Rank", "mean"),
        avg_streamscore=("StreamScore", "mean"),
    ).reset_index()
    grp["top1"] = grp["correct_member"]
    return grp.sort_values(["correct_member", "avg_streamscore"], ascending=[False, False])


def bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Core 025 — Clean-Room Separator App", layout="wide")
st.title("Core 025 — Clean-Room Separator App")
st.caption(f"BUILD: {APP_VERSION_STR}")

st.warning(
    "This is a true clean-room separator app. It does NOT use embedded legacy weights, rulepacks, downranks, rescues, row-memory, pairwise, or gate logic. "
    "Ranking is driven only by the uploaded member score overlay."
)

with st.sidebar:
    st.header("Inputs")
    feature_upload = st.file_uploader(
        "Upload feature table CSV",
        type=["csv"],
        help="Use a feature table that already includes seed feature columns (like core025_feature_table...).",
        key="cleanroom_feature_upload",
    )
    overlay_upload = st.file_uploader(
        "Member score adjustments overlay CSV or TXT (required)",
        type=["csv", "txt"],
        help="This is the ONLY scoring layer in this app.",
        key="cleanroom_overlay_upload",
    )
    top_n = st.number_input("Top-N cutoff per date", min_value=1, max_value=200, value=50, step=1)
    mode = st.radio("Mode", ["Backtest", "Live-style latest-row ranking"], index=0)
    if st.button("Clear cached uploads"):
        for k in [
            "cleanroom_feature_upload__bytes", "cleanroom_feature_upload__name",
            "cleanroom_overlay_upload__bytes", "cleanroom_overlay_upload__name",
        ]:
            st.session_state.pop(k, None)
        st.rerun()

feature_blob = persist_upload_bytes("cleanroom_feature_upload", feature_upload)
overlay_blob = persist_upload_bytes("cleanroom_overlay_upload", overlay_upload)

if not feature_blob["present"]:
    st.info("Upload the feature table CSV to begin.")
    st.stop()

features_raw = pd.read_csv(io.BytesIO(feature_blob["bytes"]))
features = ensure_feature_columns(features_raw)

rules, overlay_meta = parse_overlay_file(overlay_blob["bytes"], overlay_blob["name"])

# Loaded files display
loaded_rows = [
    {"pack": "feature_table", "source": "upload", "filename": feature_blob.get("name", ""), "sha256": sha256_of_bytes(feature_blob["bytes"]) if feature_blob.get("bytes") is not None else "", "rows": int(features.shape[0])},
    {"pack": "member_score_overlay", "source": overlay_meta.get("source", "none"), "filename": overlay_meta.get("filename", ""), "sha256": overlay_meta.get("sha256", ""), "rows": overlay_meta.get("rows", 0)},
    {"pack": "weights", "source": "disabled", "filename": "disabled:clean_room", "sha256": "", "rows": 0},
    {"pack": "tie_pack", "source": "disabled", "filename": "disabled:clean_room", "sha256": "", "rows": 0},
    {"pack": "rulepack", "source": "disabled", "filename": "disabled:clean_room", "sha256": "", "rows": 0},
    {"pack": "downranks", "source": "disabled", "filename": "disabled:clean_room", "sha256": "", "rows": 0},
    {"pack": "rescues", "source": "disabled", "filename": "disabled:clean_room", "sha256": "", "rows": 0},
]
with st.expander("Loaded Files (source / filename / sha256 / row count)", expanded=True):
    st.dataframe(pd.DataFrame(loaded_rows), use_container_width=True, hide_index=True)
st.caption("Uploads are cached in session state in this version so download clicks should not clear the loaded feature table or overlay.")

if not overlay_blob["present"]:
    st.error("Upload the member score overlay. This clean-room app will not score without it.")
    st.stop()

# Prepare run frame
run_df = features.copy()
# initialize base top placeholders for overlay conditions that reference base top
for c in ["Top1_pred", "Top2_pred", "Top3_pred"]:
    if c not in run_df.columns:
        run_df[c] = ""

scored = apply_separator_rules(run_df, rules)

if mode == "Backtest":
    ranked = rank_within_date(scored, top_n=int(top_n))
    overall = summarize_results(ranked)
    has9 = summarize_split(ranked, 1)
    no9 = summarize_split(ranked, 0)

    st.subheader("Ranking performance within playable universe")
    playable = ranked[ranked["Selected50"] == 1]
    if playable.shape[0]:
        top1 = overall["Top1"]
        top2n = overall["Top2-needed"]
        top3o = overall["Top3-only"]
        st.write(
            f"Playable winner events: {playable.shape[0]} | "
            f"Top1: {top1}/{playable.shape[0]} = {top1/playable.shape[0]:.2%} | "
            f"Top2: {top2n}/{playable.shape[0]} = {top2n/playable.shape[0]:.2%} | "
            f"Top3: {top3o}/{playable.shape[0]} = {top3o/playable.shape[0]:.2%}"
        )
        st.write(
            f"Rank stats (playable winners): avg {playable['Rank'].mean():.2f}, "
            f"median {playable['Rank'].median():.0f}, "
            f"90th pct {playable['Rank'].quantile(0.9):.0f}"
        )

    st.subheader("Quick test summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.code(textwrap.dedent(f"""\
        Selected50: {overall['Selected50']}
        Correct-member: {overall['Correct-member']}
        Top1: {overall['Top1']}
        Top2-needed: {overall['Top2-needed']}
        Top3-only: {overall['Top3-only']}
        """))
    with c2:
        st.code(textwrap.dedent(f"""\
        Has-9 rows: {has9['rows']}
        Correct-member: {has9['Correct-member']}
        Top1: {has9['Top1']}
        Top2-needed: {has9['Top2-needed']}
        Top3-only: {has9['Top3-only']}
        """))
    with c3:
        st.code(textwrap.dedent(f"""\
        No-9 rows: {no9['rows']}
        Correct-member: {no9['Correct-member']}
        Top1: {no9['Top1']}
        Top2-needed: {no9['Top2-needed']}
        Top3-only: {no9['Top3-only']}
        """))

    # Rule impact on screen
    st.subheader("Rule impact on screen")
    impact_rows = []
    for r in rules:
        mask = ranked.apply(lambda row: match_rule_to_row(r, row), axis=1)
        sub = ranked[mask]
        impact_rows.append({
            "rule_id": r.rule_id,
            "rows_matched": int(sub.shape[0]),
            "selected50_rows": int(sub["Selected50"].sum()) if sub.shape[0] else 0,
            "true_0025": int((sub["true_member"] == "0025").sum()) if sub.shape[0] else 0,
            "true_0225": int((sub["true_member"] == "0225").sum()) if sub.shape[0] else 0,
            "true_0255": int((sub["true_member"] == "0255").sum()) if sub.shape[0] else 0,
            "top1_correct_rows": int(sub["CorrectMember"].sum()) if sub.shape[0] else 0,
            "top2_needed_rows": int(sub["Top2Needed"].sum()) if sub.shape[0] else 0,
        })
    impact_df = pd.DataFrame(impact_rows)
    st.dataframe(impact_df, use_container_width=True, hide_index=True)

    per_event_export = ranked[[
        "date", "stream", "feat_seed", "true_member", "PredictedMember", "Top1_pred", "Top2_pred", "Top3_pred",
        "score_0025", "score_0225", "score_0255", "Top1Score", "Top1Margin", "StreamScore", "Rank", "Selected50",
        "CorrectMember", "Top2Needed", "Top3Only", "PrevSeedHas9", "FiredRuleIDs", "FiredRuleNotes"
    ]].rename(columns={
        "date": "PlayDate",
        "stream": "StreamKey",
        "feat_seed": "PrevSeed_text",
        "true_member": "WinningMember_text",
        "PredictedMember": "PredictedMember",
        "Top1_pred": "Top1",
        "Top2_pred": "Top2",
        "Top3_pred": "Top3",
    })

    per_date_export = build_date_export(ranked)
    per_stream_export = build_stream_export(ranked)

    st.download_button("Download per-event CSV", data=bytes_csv(per_event_export),
                       file_name="core025_cleanroom_per_event.csv", mime="text/csv")
    st.download_button("Download per-date CSV", data=bytes_csv(per_date_export),
                       file_name="core025_cleanroom_per_date.csv", mime="text/csv")
    st.download_button("Download per-stream CSV", data=bytes_csv(per_stream_export),
                       file_name="core025_cleanroom_per_stream.csv", mime="text/csv")

    with st.expander("Per-event preview", expanded=False):
        st.dataframe(per_event_export.head(200), use_container_width=True)
else:
    live = scored.sort_values(["date"]).groupby("stream", as_index=False).tail(1).copy()
    live = live.sort_values(["Top1Score", "Top1Margin", "stream"], ascending=[False, False, True]).copy()
    live["Rank"] = np.arange(1, len(live) + 1)
    live["Selected50"] = (live["Rank"] <= int(top_n)).astype(int)
    live_export = live[[
        "date", "stream", "feat_seed", "PredictedMember", "Top1_pred", "Top2_pred", "Top3_pred",
        "score_0025", "score_0225", "score_0255", "Top1Score", "Top1Margin", "Rank", "Selected50",
        "FiredRuleIDs", "FiredRuleNotes"
    ]].rename(columns={
        "date": "AsOfDate",
        "stream": "StreamKey",
        "feat_seed": "PrevSeed_text",
        "Top1_pred": "Top1",
        "Top2_pred": "Top2",
        "Top3_pred": "Top3",
    })
    st.subheader("Live-style latest-row ranking")
    st.dataframe(live_export.head(int(top_n)), use_container_width=True, hide_index=True)
    st.download_button("Download live-style ranking CSV", data=bytes_csv(live_export),
                       file_name="core025_cleanroom_live_style_ranking.csv", mime="text/csv")
