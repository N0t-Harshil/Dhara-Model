from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SHARD_SCHEMA_VERSION = 2


def shard_progress_key(repo: str, name: Optional[str], split: str, fingerprint: str) -> str:
    """Deterministic short filename key (repo/name/split hashed) — keeps the
    on-disk filename short enough for Windows long-path limits while the
    record body stores the readable identity."""
    raw = f"{repo}|{name or ''}|{split}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{digest}__{fingerprint[:16]}"


class ShardProgressStore:
    """Persistent per-shard streaming progress.

    Keyed by (repo, name, split, fingerprint) so any change to the dataset
    revision/files/preprocessing signature starts a fresh progress record.
    Stored as JSON per dataset under <shard_progress_dir>/<key>.json with an
    atomic replace (uuid tmp file + os.replace — Windows-safe).

    Records:
      last_shard: index of the last sequentially completed shard (-1 = none)
      last_offset: raw rows already pulled from shard (last_shard + 1)
      resume: {"shard": idx, "offset": n} — the exact next position to read
              (authoritative when the plan was non-sequential)
      done: True when every shard was processed
      stats: {shard_idx: {streamed, accepted, rejected_*, complete, raw}}
             complete=1 marks shards fully processed (yield-order safe)
      fingerprint: cache fingerprint this record belongs to
      updated_at
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def load(self, repo: str, name: Optional[str], split: str, fingerprint: str) -> Dict[str, Any]:
        key = shard_progress_key(repo, name, split, fingerprint)
        path = self._path(key)
        try:
            if path.exists():
                rec = json.loads(path.read_text(encoding="utf-8"))
                if rec.get("schema_version") == SHARD_SCHEMA_VERSION:
                    return rec
        except Exception as e:
            logger.warning("Shard progress read failed for %s (%s) — starting fresh", key, e)
        return {
            "schema_version": SHARD_SCHEMA_VERSION,
            "key": key,
            "repo": repo,
            "name": name,
            "split": split,
            "fingerprint": fingerprint,
            "last_shard": -1,
            "last_offset": 0,
            "done": False,
            "stats": {},
            "updated_at": 0.0,
        }

    def save(self, rec: Dict[str, Any]) -> None:
        import uuid

        rec["updated_at"] = time.time()
        key = rec.get("key")
        if not key:
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / f".{key}.{uuid.uuid4().hex}.tmp"
            tmp.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, self._path(key))
        except Exception as e:
            logger.warning("Shard progress save failed for %s (%s)", key, e)

    def delete(self, repo: str, name: Optional[str], split: str, fingerprint: str) -> None:
        """Remove a progress record (used when a dataset is fully streamed:
        next run starts from shard 0 — the packed cache covers reuse)."""
        try:
            path = self._path(shard_progress_key(repo, name, split, fingerprint))
            if path.exists():
                path.unlink()
        except Exception as e:
            logger.warning("Shard progress delete failed for %s (%s)",
                           f"{repo}/{name or 'default'}/{split}", e)

    def shard_stats(self, rec: Dict[str, Any], shard_idx: int) -> Dict[str, Any]:
        return rec.setdefault("stats", {}).setdefault(str(shard_idx), {
            "streamed": 0, "accepted": 0,
            "rejected_quality": 0, "rejected_ast": 0, "rejected_dup": 0,
            "rejected_boilerplate": 0, "rejected_short": 0, "rejected_empty": 0,
            "length_sum": 0, "quality_sum": 0.0,
        })

    def update_shard_stats(self, rec: Dict[str, Any], shard_idx: int, delta: Dict[str, Any]) -> None:
        st = self.shard_stats(rec, shard_idx)
        for k, v in delta.items():
            st[k] = st.get(k, 0) + v

    def yield_order(self, rec: Dict[str, Any], n_shards: int) -> List[int]:
        """Remaining shard indices ordered by measured acceptance yield
        (accepted/streamed), most productive first. Completed shards excluded."""
        stats = rec.get("stats", {})
        scored = []
        for i in range(n_shards):
            st = stats.get(str(i))
            if st is None or st.get("streamed", 0) == 0:
                continue  # never touched → leave in natural order at the end
            yield_pct = st.get("accepted", 0) / st.get("streamed", 1)
            scored.append((i, yield_pct))
        scored.sort(key=lambda x: -x[1])
        return [i for i, _ in scored]


def build_shard_plan(
    n_shards: int,
    rec: Dict[str, Any],
    order: str = "sequential",
) -> List[Tuple[int, int]]:
    """Return [(shard_idx, resume_offset), ...] for the shards to read now.

    Shards are excluded when they are marked complete (per-shard ``complete``
    flags — safe under yield ordering) or when a legacy sequential
    ``last_shard`` implies they were finished. The exact resume point
    (``rec["resume"]``) carries the in-flight shard + consumed raw offset.

    With order="yield", remaining shards are prioritized by measured
    acceptance (accepted/streamed); untouched shards stay in natural order
    after the scored ones. Sequential (default) preserves dataset stream
    order exactly (identity-preserving).
    """
    if rec.get("done"):
        return []
    if n_shards <= 0:
        return []
    stats = rec.get("stats", {})
    complete = {int(i) for i, st in stats.items() if st.get("complete")}
    last = int(rec.get("last_shard", -1))
    if last >= 0:
        complete |= set(range(0, last + 1))
    resume = rec.get("resume") or {}
    resume_shard = resume.get("shard")
    resume_offset = int(resume.get("offset", 0) or 0)
    if resume_shard is not None and int(resume_shard) in complete:
        resume_shard, resume_offset = None, 0
    remaining = [i for i in range(n_shards) if i not in complete]
    if resume_shard is not None and int(resume_shard) not in remaining:
        remaining.insert(0, int(resume_shard))
    if order == "yield" and remaining:
        scored = rec_yield_shards(rec, remaining)
        remaining = scored + [i for i in remaining if i not in scored]
    plan = []
    for i, idx in enumerate(remaining):
        offset = resume_offset if i == 0 and idx == int(resume_shard or -1) else 0
        plan.append((idx, offset))
    return plan


def rec_yield_shards(rec: Dict[str, Any], candidates: List[int]) -> List[int]:
    stats = rec.get("stats", {})
    scored = []
    untouched = []
    for i in candidates:
        st = stats.get(str(i))
        if st is None or st.get("streamed", 0) == 0:
            untouched.append(i)
        else:
            scored.append((i, st.get("accepted", 0) / st.get("streamed", 1)))
    scored.sort(key=lambda x: -x[1])
    return [i for i, _ in scored] + untouched
