"""Multi-format document parsers for structured and unstructured ingestion."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from amea.rag.models import DocumentType


class DocumentParser:
    """Parses heterogeneous files (TXT, MD, CSV, JSON, PDF/DOCX text) into structured content."""

    @staticmethod
    def detect_type(file_path: str) -> DocumentType:
        ext = Path(file_path).suffix.lower()
        if ext in [".txt"]:
            return DocumentType.TEXT
        elif ext in [".md", ".markdown"]:
            return DocumentType.MARKDOWN
        elif ext in [".csv"]:
            return DocumentType.CSV
        elif ext in [".json"]:
            return DocumentType.JSON
        elif ext in [".pdf"]:
            return DocumentType.PDF
        elif ext in [".docx"]:
            return DocumentType.DOCX
        elif ext in [".py", ".sh", ".sql"]:
            return DocumentType.SOURCE_CODE
        return DocumentType.UNSPECIFIED

    @classmethod
    def parse_file(cls, file_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Parses file into (raw_text, structured_sections, metadata).
        structured_sections: list of {'heading': str, 'content': str, 'page': int, 'section_index': int}
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_type = cls.detect_type(file_path)
        metadata: Dict[str, Any] = {
            "filename": path.name,
            "document_type": doc_type.value,
            "file_size_bytes": path.stat().st_size,
        }

        if doc_type == DocumentType.CSV:
            return cls._parse_csv(path, metadata)
        elif doc_type == DocumentType.JSON:
            return cls._parse_json(path, metadata)
        elif doc_type == DocumentType.MARKDOWN:
            return cls._parse_markdown(path, metadata)
        else:
            return cls._parse_plain_text(path, metadata)

    @classmethod
    def _parse_plain_text(cls, path: Path, metadata: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        sections = []
        for i, p in enumerate(paragraphs):
            sections.append({
                "heading": f"Paragraph {i + 1}",
                "content": p,
                "page": 1,
                "section_index": i,
            })
        return text, sections, metadata

    @classmethod
    def _parse_markdown(cls, path: Path, metadata: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        sections = []
        current_heading = "Introduction"
        current_lines = []
        section_idx = 0

        for line in lines:
            if line.startswith("#"):
                if current_lines:
                    sections.append({
                        "heading": current_heading,
                        "content": "\n".join(current_lines).strip(),
                        "page": 1,
                        "section_index": section_idx,
                    })
                    section_idx += 1
                    current_lines = []
                current_heading = line.lstrip("#").strip()
            else:
                current_lines.append(line)

        if current_lines:
            sections.append({
                "heading": current_heading,
                "content": "\n".join(current_lines).strip(),
                "page": 1,
                "section_index": section_idx,
            })

        return text, sections, metadata

    @classmethod
    def _parse_csv(cls, path: Path, metadata: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        with open(path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            headers = reader.fieldnames or []

        metadata["total_rows"] = len(rows)
        metadata["columns"] = headers

        # Create structured text representation
        lines = [f"CSV Table: {path.name} with columns: {', '.join(headers)}"]
        sections = []
        # Chunk rows into groups of 10 for structured retrieval
        batch_size = 10
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            batch_text = "\n".join([json.dumps(r) for r in batch])
            sections.append({
                "heading": f"Rows {i+1} to {min(i+batch_size, len(rows))}",
                "content": batch_text,
                "page": 1,
                "section_index": i // batch_size,
            })
            lines.append(batch_text)

        full_text = "\n".join(lines)
        return full_text, sections, metadata

    @classmethod
    def _parse_json(cls, path: Path, metadata: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        content = path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(content)
        sections = []

        if isinstance(data, dict):
            for i, (k, v) in enumerate(data.items()):
                sections.append({
                    "heading": f"Key: {k}",
                    "content": f"{k}: {json.dumps(v, indent=2)}",
                    "page": 1,
                    "section_index": i,
                })
        elif isinstance(data, list):
            for i, item in enumerate(data):
                sections.append({
                    "heading": f"Item {i+1}",
                    "content": json.dumps(item, indent=2),
                    "page": 1,
                    "section_index": i,
                })

        return content, sections, metadata
