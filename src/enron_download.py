"""Download EDRM Enron v2 XML from Internet Archive.

The full dataset is ~73GB across 159 zip files, organized by custodian.
Zip files are named: edrm-enron-v2_{custodian}_xml.zip (e.g., edrm-enron-v2_allen-p_xml.zip)
This script downloads selectively — only custodians needed for selected TREC topics.
"""

import subprocess
import sys
from pathlib import Path

ARCHIVE_BASE = "https://archive.org/download/edrm.enron.email.data.set.v2.xml"
RAW_DIR = Path("corpus/enron/raw")


def custodian_to_zip_name(custodian: str) -> str:
    """Convert a custodian slug (e.g., 'allen-p') to the archive zip filename."""
    return f"edrm-enron-v2_{custodian}_xml.zip"


def download_custodian(custodian: str, output_dir: Path = RAW_DIR) -> Path:
    """Download a single custodian's zip file from Internet Archive.

    Args:
        custodian: Custodian slug like 'allen-p' or full zip name like 'edrm-enron-v2_allen-p_xml.zip'
        output_dir: Where to save the downloaded file.

    Returns:
        Path to the downloaded zip file.
    """
    if custodian.endswith(".zip"):
        zip_name = custodian
    else:
        zip_name = custodian_to_zip_name(custodian)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / zip_name

    if output_path.exists():
        print(f"Already downloaded: {output_path}")
        return output_path

    url = f"{ARCHIVE_BASE}/{zip_name}"
    print(f"Downloading {url} ...")
    subprocess.run(
        ["curl", "-L", "--retry", "3", "-o", str(output_path), url],
        check=True,
    )
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.enron_download <custodian> [custodian ...]")
        print("Example: python -m src.enron_download allen-p arnold-j")
        sys.exit(1)

    for name in sys.argv[1:]:
        download_custodian(name)
