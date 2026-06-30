"""Save and restore Redis snapshots for notebook-driven GigaEvolve runs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Snapshot GigaEvolve Redis keys.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for sub in (subparsers.add_parser("save"), subparsers.add_parser("restore"), subparsers.add_parser("info")):
        sub.add_argument("--file", required=True)
        sub.add_argument("--redis-host", default="localhost")
        sub.add_argument("--redis-port", type=int, default=6379)
        sub.add_argument("--redis-db", type=int, default=0)

    subparsers.choices["save"].add_argument("--prefix", required=True)
    subparsers.choices["restore"].add_argument("--flush-prefix", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "save":
        payload = save_snapshot(
            file=Path(args.file),
            host=args.redis_host,
            port=args.redis_port,
            db=args.redis_db,
            prefix=args.prefix,
        )
    elif args.command == "restore":
        payload = restore_snapshot(
            file=Path(args.file),
            host=args.redis_host,
            port=args.redis_port,
            db=args.redis_db,
            flush_prefix=bool(args.flush_prefix),
        )
    else:
        payload = snapshot_info(Path(args.file))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def save_snapshot(*, file: Path, host: str, port: int, db: int, prefix: str) -> dict[str, Any]:
    redis = _redis_client(host=host, port=port, db=db)
    pattern = f"{prefix}:*"
    keys = sorted(redis.scan_iter(match=pattern, count=1000))
    items = []
    for key in keys:
        key_s = _decode(key)
        key_type = _decode(redis.type(key))
        item = {
            "key": key_s,
            "type": key_type,
            "ttl_ms": int(redis.pttl(key)),
            "value": _dump_value(redis, key, key_type),
        }
        items.append(item)

    payload = {
        "format": "evohash.redis_snapshot.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "redis": {"host": host, "port": port, "db": db},
        "prefix": prefix,
        "n_keys": len(items),
        "items": items,
    }
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "file": str(file), "prefix": prefix, "n_keys": len(items)}


def restore_snapshot(*, file: Path, host: str, port: int, db: int, flush_prefix: bool) -> dict[str, Any]:
    payload = json.loads(file.read_text(encoding="utf-8"))
    if payload.get("format") != "evohash.redis_snapshot.v1":
        raise ValueError(f"Unsupported snapshot format: {payload.get('format')!r}")
    redis = _redis_client(host=host, port=port, db=db)
    prefix = payload["prefix"]
    if flush_prefix:
        keys = list(redis.scan_iter(match=f"{prefix}:*", count=1000))
        if keys:
            redis.delete(*keys)

    restored = 0
    for item in payload.get("items", []):
        _restore_value(redis, item["key"], item["type"], item["value"])
        ttl_ms = int(item.get("ttl_ms", -1))
        if ttl_ms > 0:
            redis.pexpire(item["key"], ttl_ms)
        restored += 1
    return {"ok": True, "file": str(file), "prefix": prefix, "n_restored": restored}


def snapshot_info(file: Path) -> dict[str, Any]:
    payload = json.loads(file.read_text(encoding="utf-8"))
    by_type: dict[str, int] = {}
    for item in payload.get("items", []):
        by_type[item["type"]] = by_type.get(item["type"], 0) + 1
    return {
        "ok": True,
        "file": str(file),
        "format": payload.get("format"),
        "created_at": payload.get("created_at"),
        "prefix": payload.get("prefix"),
        "n_keys": payload.get("n_keys", 0),
        "by_type": by_type,
    }


def _redis_client(*, host: str, port: int, db: int):
    try:
        import redis
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("The redis package is required for snapshots.") from exc
    client = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=3)
    client.ping()
    return client


def _dump_value(redis, key: bytes, key_type: str) -> Any:
    if key_type == "string":
        return _decode(redis.get(key))
    if key_type == "hash":
        return {_decode(k): _decode(v) for k, v in redis.hgetall(key).items()}
    if key_type == "set":
        return sorted(_decode(v) for v in redis.smembers(key))
    if key_type == "zset":
        return [[_decode(v), float(score)] for v, score in redis.zrange(key, 0, -1, withscores=True)]
    if key_type == "list":
        return [_decode(v) for v in redis.lrange(key, 0, -1)]
    if key_type == "stream":
        return [
            [_decode(entry_id), {_decode(k): _decode(v) for k, v in fields.items()}]
            for entry_id, fields in redis.xrange(key, min="-", max="+")
        ]
    raise ValueError(f"Unsupported Redis type for snapshot: {key_type!r}")


def _restore_value(redis, key: str, key_type: str, value: Any) -> None:
    if key_type == "string":
        redis.set(key, value)
    elif key_type == "hash":
        if value:
            redis.hset(key, mapping=value)
    elif key_type == "set":
        if value:
            redis.sadd(key, *value)
    elif key_type == "zset":
        if value:
            redis.zadd(key, {member: float(score) for member, score in value})
    elif key_type == "list":
        if value:
            redis.rpush(key, *value)
    elif key_type == "stream":
        for entry_id, fields in value:
            redis.xadd(key, fields, id=entry_id)
    else:
        raise ValueError(f"Unsupported Redis type for restore: {key_type!r}")


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        raise
