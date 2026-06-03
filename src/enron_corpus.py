"""Load and manage the parsed Enron email corpus."""

from pathlib import Path
import json
import subprocess
import tempfile
import zipfile

from src.enron_parse import parse_edrm_xml_file, EnronEmail


def _extract_with_zipfile(zip_path: Path, extract_dir: Path) -> None:
    """Extract ZIP contents using Python's zipfile module."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)


def _extract_with_7z(zip_path: Path, extract_dir: Path) -> None:
    """Extract ZIP contents using 7z as a fallback."""
    result = subprocess.run(
        ["7z", "x", str(zip_path), f"-o{extract_dir}", "-y"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"7z extraction failed (rc={result.returncode}): {result.stderr}"
        )


def _find_xml_file(extract_dir: Path) -> Path:
    """Find the single XML metadata file in the extracted directory."""
    xml_files = list(extract_dir.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML file found in {extract_dir}")
    if len(xml_files) > 1:
        # Prefer files matching the EDRM naming convention zl_*_000.xml
        edrm = [f for f in xml_files if f.name.startswith("zl_")]
        if edrm:
            return edrm[0]
    return xml_files[0]


def _load_text_files(extract_dir: Path) -> dict[str, str]:
    """Load all text files from text_*/ directories into a dict."""
    text_files: dict[str, str] = {}
    for txt_dir in extract_dir.rglob("text_*"):
        if not txt_dir.is_dir():
            continue
        for txt_file in txt_dir.glob("*.txt"):
            # Store under both bare filename and path-relative key
            content = txt_file.read_text(errors="replace")
            text_files[txt_file.name] = content
            # Also store with parent dir name as prefix (e.g. "text_000/doc.txt")
            rel_key = f"{txt_dir.name}/{txt_file.name}"
            text_files[rel_key] = content
    return text_files


def parse_custodian_zip(
    zip_path: Path, output_dir: Path | None = None
) -> list[EnronEmail]:
    """Parse an EDRM Enron v2 ZIP file into a list of EnronEmail objects.

    Extracts the XML metadata file and all text files, then parses them.
    Uses 7z as fallback if Python's zipfile can't handle the archive.

    Args:
        zip_path: Path to the custodian ZIP file.
        output_dir: Optional directory for extracted files. If None, uses a temp dir.

    Returns:
        List of EnronEmail objects.
    """
    use_temp = output_dir is None

    if use_temp:
        tmp = tempfile.mkdtemp(prefix="enron_parse_")
        extract_dir = Path(tmp)
    else:
        extract_dir = output_dir
        extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try Python zipfile first, fall back to 7z
        try:
            _extract_with_zipfile(zip_path, extract_dir)
        except (zipfile.BadZipFile, Exception) as e:
            print(f"  Python zipfile failed ({e}), trying 7z...")
            _extract_with_7z(zip_path, extract_dir)

        xml_path = _find_xml_file(extract_dir)
        xml_content = xml_path.read_bytes()

        text_files = _load_text_files(extract_dir)

        emails = parse_edrm_xml_file(xml_content, text_files)
        return emails

    finally:
        # Clean up temp directory if we created one
        if use_temp:
            import shutil

            shutil.rmtree(extract_dir, ignore_errors=True)


def save_parsed_corpus(emails: list[EnronEmail], output_path: Path) -> None:
    """Save parsed emails to a JSON file."""
    data = []
    for e in emails:
        data.append(
            {
                "doc_id": e.doc_id,
                "from_addr": e.from_addr,
                "to_addr": e.to_addr,
                "subject": e.subject,
                "body": e.body,
                "date_sent": e.date_sent,
                "custodian": e.custodian,
                "cc": e.cc,
                "bcc": e.bcc,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)


def load_parsed_corpus(parsed_dir: Path) -> dict[str, EnronEmail]:
    """Load all parsed JSON files into a corpus dict."""
    corpus = {}
    for f in parsed_dir.glob("*.json"):
        with open(f) as fh:
            for e in json.load(fh):
                corpus[e["doc_id"]] = EnronEmail(**e)
    return corpus
