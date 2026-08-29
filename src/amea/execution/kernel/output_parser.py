"""Parses Jupyter IOPub messages, rich representations, DataFrames, images, and errors."""

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from amea.execution.kernel.execution_result import (
    CellOutput,
    CellOutputType,
    DataFramePreview,
)


class OutputParser:
    """Extracts structured, machine-readable cell outputs from Jupyter kernel messages."""

    @classmethod
    def parse_iopub_message(
        cls,
        msg: Dict[str, Any],
        session_artifact_dir: Optional[Path] = None,
    ) -> Optional[CellOutput]:
        """Convert a raw Jupyter IOPub message dict into a typed CellOutput."""
        msg_type = msg.get("msg_type") or msg.get("header", {}).get("msg_type")
        content = msg.get("content", {})

        if msg_type == "stream":
            text = content.get("text", "")
            stream_name = content.get("name", "stdout")
            return CellOutput(
                output_type=CellOutputType.STREAM,
                text=text,
                stream_name=stream_name,
            )

        elif msg_type in ("execute_result", "display_data"):
            data = content.get("data", {})
            return cls.parse_mime_bundle(data, session_artifact_dir)

        elif msg_type == "error":
            ename = content.get("ename", "Exception")
            evalue = content.get("evalue", "")
            tb = content.get("traceback", [])
            # Strip ANSI color codes from traceback
            clean_tb = [re.sub(r"\x1b\[[0-9;]*[mGKF]", "", line) for line in tb]
            return CellOutput(
                output_type=CellOutputType.ERROR,
                error_name=ename,
                error_value=evalue,
                traceback=clean_tb,
                text=f"{ename}: {evalue}",
            )

        return None

    @classmethod
    def parse_mime_bundle(
        cls,
        data: Dict[str, Any],
        session_artifact_dir: Optional[Path] = None,
    ) -> CellOutput:
        """Parse rich display MIME bundle (HTML, Image, Plain text, JSON)."""
        # 1. Images (PNG / JPEG)
        if "image/png" in data:
            img_b64 = data["image/png"]
            art_path = None
            if session_artifact_dir:
                session_artifact_dir.mkdir(parents=True, exist_ok=True)
                art_file = session_artifact_dir / f"plot_{uuid4().hex[:8]}.png"
                art_file.write_bytes(base64.b64decode(img_b64))
                art_path = str(art_file.resolve())

            return CellOutput(
                output_type=CellOutputType.IMAGE,
                image_base64=img_b64,
                image_artifact_path=art_path,
            )

        if "image/jpeg" in data:
            img_b64 = data["image/jpeg"]
            return CellOutput(
                output_type=CellOutputType.IMAGE,
                image_base64=img_b64,
            )

        # 2. HTML (often DataFrame HTML tables or interactive widgets)
        if "text/html" in data:
            html = data["text/html"]
            # Check if this is a DataFrame table
            df_preview = cls.extract_dataframe_from_html(html)
            if df_preview:
                return CellOutput(
                    output_type=CellOutputType.DATAFRAME,
                    dataframe=df_preview,
                    text=data.get("text/plain"),
                )
            return CellOutput(
                output_type=CellOutputType.HTML,
                text=html,
            )

        # 3. JSON
        if "application/json" in data:
            return CellOutput(
                output_type=CellOutputType.JSON,
                data=data["application/json"],
                text=data.get("text/plain"),
            )

        # 4. Plain Text / Scalar
        text_plain = data.get("text/plain", "")
        # Check if scalar (int, float, bool)
        scalar_val = cls.try_parse_scalar(text_plain)
        if scalar_val is not None:
            return CellOutput(
                output_type=CellOutputType.SCALAR,
                scalar_value=scalar_val,
                text=text_plain,
            )

        return CellOutput(
            output_type=CellOutputType.TEXT,
            text=text_plain,
        )

    @classmethod
    def extract_dataframe_from_html(cls, html: str) -> Optional[DataFramePreview]:
        """Inspect HTML table string to extract structured DataFrame headers and rows."""
        if "<table" not in html.lower() or "dataframe" not in html.lower():
            return None

        try:
            # Extract column headers from <thead>
            thead_match = re.search(r"<thead>(.*?)</thead>", html, re.DOTALL | re.IGNORECASE)
            headers: List[str] = []
            if thead_match:
                headers = [
                    re.sub(r"<[^>]+>", "", th).strip()
                    for th in re.findall(r"<th[^>]*>(.*?)</th>", thead_match.group(1), re.DOTALL | re.IGNORECASE)
                    if re.sub(r"<[^>]+>", "", th).strip()
                ]

            # Extract table body rows from <tbody>
            tbody_match = re.search(r"<tbody>(.*?)</tbody>", html, re.DOTALL | re.IGNORECASE)
            tbody_html = tbody_match.group(1) if tbody_match else html
            tr_matches = re.findall(r"<tr>(.*?)</tr>", tbody_html, re.DOTALL | re.IGNORECASE)
            rows: List[Dict[str, Any]] = []

            for tr in tr_matches:
                td_matches = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL | re.IGNORECASE)
                if td_matches:
                    row_dict = {}
                    for i, td in enumerate(td_matches):
                        val_str = re.sub(r"<[^>]+>", "", td).strip()
                        col_name = headers[i] if i < len(headers) else f"col_{i}"
                        try:
                            if "." in val_str:
                                row_dict[col_name] = float(val_str)
                            else:
                                row_dict[col_name] = int(val_str)
                        except ValueError:
                            row_dict[col_name] = val_str
                    rows.append(row_dict)

            if rows or headers:
                return DataFramePreview(
                    columns=headers,
                    dtypes={h: "object" for h in headers},
                    rows_preview_count=len(rows),
                    total_rows=len(rows),
                    total_columns=len(headers),
                    data=rows,
                    is_truncated=False,
                )
        except Exception:
            pass

        return None

    @classmethod
    def try_parse_scalar(cls, text: str) -> Optional[Any]:
        """Check if string is a numeric or boolean scalar."""
        clean = text.strip()
        if clean in ("True", "False"):
            return clean == "True"
        try:
            if "." in clean or "e" in clean.lower():
                return float(clean)
            return int(clean)
        except ValueError:
            return None
