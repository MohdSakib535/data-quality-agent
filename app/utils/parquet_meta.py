"""
Parquet metadata utilities via PyArrow.
Read/write custom metadata into Parquet files.
"""
import io
import json
from typing import Any, BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet_with_metadata(
    table: pa.Table,
    output: BinaryIO,
    custom_metadata: dict[str, str],
    compression: str = "snappy",
) -> None:
    """
    Write an Arrow Table to Parquet with custom key-value metadata embedded.
    All metadata values must be strings.
    """
    # Merge with any existing schema metadata
    existing_meta = table.schema.metadata or {}
    merged = {
        **{k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
           for k, v in existing_meta.items()},
        **custom_metadata,
    }
    # PyArrow requires bytes for metadata
    byte_meta = {k.encode(): v.encode() for k, v in merged.items()}
    new_schema = table.schema.with_metadata(byte_meta)
    table = table.cast(new_schema)

    pq.write_table(table, output, compression=compression)


def read_parquet_metadata(source: BinaryIO | str) -> dict[str, str]:
    """
    Read custom metadata from a Parquet file.
    Returns decoded key-value dict.
    """
    parquet_file = pq.ParquetFile(source)
    schema_meta = parquet_file.schema_arrow.metadata or {}
    return {
        k.decode(): v.decode()
        for k, v in schema_meta.items()
    }


def read_parquet_schema(source: BinaryIO | str) -> dict[str, str]:
    """
    Read column names and types from a Parquet file schema.
    Returns {column_name: arrow_type_string}.
    """
    parquet_file = pq.ParquetFile(source)
    schema = parquet_file.schema_arrow
    return {
        field.name: str(field.type)
        for field in schema
    }


def get_parquet_row_count(source: BinaryIO | str) -> int:
    """Get total row count from Parquet metadata without reading all data."""
    parquet_file = pq.ParquetFile(source)
    return parquet_file.metadata.num_rows
