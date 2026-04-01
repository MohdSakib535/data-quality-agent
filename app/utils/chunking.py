"""
Chunking utilities for streaming CSV/Parquet processing.
"""
import io
from typing import Iterator, Any

import polars as pl

from app.core.config import settings


def iter_csv_chunks(
    data: bytes | io.BytesIO,
    chunk_size: int | None = None,
) -> Iterator[pl.DataFrame]:
    """
    Yield CSV data in chunks of chunk_size rows as Polars DataFrames.
    Useful for processing large CSVs without loading entirely into memory.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE

    if isinstance(data, bytes):
        data = io.BytesIO(data)

    reader = pl.read_csv_batched(
        data,
        batch_size=chunk_size,
        try_parse_dates=True,
        ignore_errors=True,
        truncate_ragged_lines=True,
    )

    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
        yield batches[0]


def iter_parquet_chunks(
    data: bytes | io.BytesIO,
    chunk_size: int | None = None,
) -> Iterator[pl.DataFrame]:
    """
    Yield Parquet data in chunks of chunk_size rows as Polars DataFrames.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE

    if isinstance(data, bytes):
        data = io.BytesIO(data)

    # Read full Parquet and slice (Parquet doesn't have native row-level streaming
    # like CSV, but we slice to limit memory per processing batch)
    full_df = pl.read_parquet(data)
    total = len(full_df)

    for start in range(0, total, chunk_size):
        yield full_df.slice(start, chunk_size)
