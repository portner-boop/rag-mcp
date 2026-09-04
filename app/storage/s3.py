from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.shared.errors import NotFoundError, UpstreamError


@dataclass(frozen=True)
class ObjectMeta:
    key: str
    size: int
    content_type: str | None
    etag: str | None


class S3ObjectStore:
    def __init__(
        self,
        *,
        endpoint: str,
        region: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        use_path_style: bool = True,
    ) -> None:
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path" if use_path_style else "auto"},
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    @property
    def bucket(self) -> str:
        return self._bucket

    async def presign_put(self, key: str, *, content_type: str, expires_in: int) -> str:
        return await asyncio.to_thread(
            self._client.generate_presigned_url,
            "put_object",
            Params={"Bucket": self._bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=expires_in,
        )

    async def presign_get(
        self, key: str, *, expires_in: int, download_filename: str | None = None
    ) -> str:
        params: dict[str, Any] = {"Bucket": self._bucket, "Key": key}
        if download_filename:
            # RFC 5987: non-ASCII filenames go into filename*; plain ASCII fallback kept.
            ascii_fallback = (
                download_filename.encode("ascii", "ignore").decode("ascii").replace('"', "").strip()
            )
            if not ascii_fallback or ascii_fallback.startswith("."):
                ext = download_filename.rsplit(".", 1)[-1] if "." in download_filename else ""
                ascii_fallback = f"file.{ext}" if ext else "file"
            encoded = quote(download_filename, safe="")
            params["ResponseContentDisposition"] = (
                f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{encoded}"
            )
        return await asyncio.to_thread(
            self._client.generate_presigned_url,
            "get_object",
            Params=params,
            ExpiresIn=expires_in,
        )

    async def head(self, key: str) -> ObjectMeta:
        def _head() -> dict[str, Any]:
            return self._client.head_object(Bucket=self._bucket, Key=key)

        try:
            resp = await asyncio.to_thread(_head)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                raise NotFoundError("Object not found", details={"key": key}) from exc
            raise UpstreamError("S3 head failed") from exc
        except BotoCoreError as exc:
            raise UpstreamError("S3 head failed") from exc
        return ObjectMeta(
            key=key,
            size=int(resp.get("ContentLength", 0)),
            content_type=resp.get("ContentType"),
            etag=resp.get("ETag"),
        )

    async def exists(self, key: str) -> bool:
        try:
            await self.head(key)
            return True
        except NotFoundError:
            return False

    async def get_bytes(self, key: str) -> bytes:
        def _get() -> bytes:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            return resp["Body"].read()

        try:
            return await asyncio.to_thread(_get)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                raise NotFoundError("Object not found", details={"key": key}) from exc
            raise UpstreamError("S3 get failed") from exc
        except BotoCoreError as exc:
            raise UpstreamError("S3 get failed") from exc

    async def put_bytes(self, key: str, data: bytes, *, content_type: str) -> None:
        def _put() -> None:
            self._client.put_object(
                Bucket=self._bucket, Key=key, Body=data, ContentType=content_type
            )

        try:
            await asyncio.to_thread(_put)
        except (ClientError, BotoCoreError) as exc:
            raise UpstreamError("S3 put failed") from exc

    async def delete(self, key: str) -> None:
        def _delete() -> None:
            self._client.delete_object(Bucket=self._bucket, Key=key)

        try:
            await asyncio.to_thread(_delete)
        except (ClientError, BotoCoreError) as exc:
            raise UpstreamError("S3 delete failed") from exc

    async def ensure_bucket(self) -> None:

        def _ensure() -> None:
            try:
                self._client.head_bucket(Bucket=self._bucket)
            except ClientError:
                self._client.create_bucket(Bucket=self._bucket)

        try:
            await asyncio.to_thread(_ensure)
        except (ClientError, BotoCoreError) as exc:
            raise UpstreamError("S3 ensure bucket failed") from exc
