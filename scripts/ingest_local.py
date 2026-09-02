"""Push a local file through the operational ingestion path (spec 7.4 → 7.6).

The chat MCP surface can only search; putting a document into the corpus goes through
the ops identity: prepare-upload → PUT to the presigned S3 URL → start ingestion → poll.

    uv run python scripts/ingest_local.py ./handbook.md
    uv run python scripts/ingest_local.py ./handbook.md --ops-url http://127.0.0.1:8090 \
        --token ops-dev --content-type text/markdown
"""

from __future__ import annotations

import argparse
import hashlib
import mimetypes
import pathlib
import sys
import time

import httpx

_TIMEOUT = 30.0


def _post(client: httpx.Client, ops_url: str, path: str, token: str, payload: dict) -> dict:
    resp = client.post(
        f"{ops_url.rstrip('/')}{path}",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=_TIMEOUT,
    )
    if resp.status_code >= 400:
        raise SystemExit(f"{path} -> {resp.status_code} {resp.text}")
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=pathlib.Path)
    parser.add_argument("--ops-url", default="http://127.0.0.1:8090")
    parser.add_argument("--token", default="ops-dev", help="ops service token (not the chat one)")
    parser.add_argument("--content-type", default=None)
    # documents.created_by is a UUID column: a free-form name fails with a 500.
    parser.add_argument("--created-by", default="00000000-0000-4000-8000-000000000001")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()

    data = args.file.read_bytes()
    checksum = hashlib.sha256(data).hexdigest()
    content_type = (
        args.content_type
        or ("text/markdown" if args.file.suffix.lower() in (".md", ".markdown") else None)
        or mimetypes.guess_type(args.file.name)[0]
        or "text/plain"
    )

    with httpx.Client() as client:
        prepared = _post(
            client,
            args.ops_url,
            "/internal/documents/prepare-upload",
            args.token,
            {
                "filename": args.file.name,
                "content_type": content_type,
                "size": len(data),
                "checksum": checksum,
                "created_by": args.created_by,
            },
        )
        print(f"document_id = {prepared['document_id']}")

        upload = client.put(
            prepared["upload_url"],
            content=data,
            headers=prepared.get("upload_headers") or {},
            timeout=_TIMEOUT,
        )
        if upload.status_code >= 400:
            raise SystemExit(f"upload -> {upload.status_code} {upload.text}")

        started = _post(
            client,
            args.ops_url,
            "/internal/ingestion/start",
            args.token,
            {
                "document_id": prepared["document_id"],
                "checksum": checksum,
                # Re-running the same file is one effect, not two (invariant 6).
                "idempotency_key": f"ingest:{prepared['document_id']}:{checksum}",
            },
        )
        job_id = started["job_id"]
        print(f"job_id = {job_id} ({started['status']})")

        deadline = time.monotonic() + args.timeout_seconds
        status = {}
        while time.monotonic() < deadline:
            status = _post(
                client,
                args.ops_url,
                "/internal/ingestion/status",
                args.token,
                {"job_id": job_id},
            )
            state = status.get("status")
            print(f"  {state} stage={status.get('stage')} progress={status.get('progress')}")
            if state in ("COMPLETED", "FAILED", "CANCELLED", "DEAD_LETTER"):
                break
            time.sleep(2)

        if status.get("status") == "COMPLETED":
            return 0
        print(f"ingestion did not finish: {status}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
