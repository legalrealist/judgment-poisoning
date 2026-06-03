"""Download EDRM Enron v2 XML from Internet Archive.

The full dataset is 73GB across 159 zip files, organized by custodian.
This script downloads selectively — only custodians needed for selected TREC topics.
"""

import subprocess
import sys
from pathlib import Path

ARCHIVE_BASE = "https://archive.org/download/edrm.enron.email.data.set.v2.xml"
RAW_DIR = Path("corpus/enron/raw")


def list_available_custodians() -> list[str]:
    """Fetch the file listing from Internet Archive and extract custodian zip names."""
    raise NotImplementedError(
        "Implement after inspecting the actual Internet Archive listing. "
        "Use: curl -s 'https://archive.org/download/edrm.enron.email.data.set.v2.xml/' "
        "to see the file listing."
    )


def download_custodian(custodian_zip: str, output_dir: Path = RAW_DIR) -> Path:
    """Download a single custodian's zip file from Internet Archive.

    Args:
        custodian_zip: Filename like 'allen-p.zip'
        output_dir: Where to save the downloaded file.

    Returns:
        Path to the downloaded zip file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / custodian_zip

    if output_path.exists():
        print(f"Already downloaded: {output_path}")
        return output_path

    url = f"{ARCHIVE_BASE}/{custodian_zip}"
    print(f"Downloading {url} ...")
    subprocess.run(
        ["curl", "-L", "-o", str(output_path), url],
        check=True,
    )
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.enron_download <custodian_zip> [custodian_zip ...]")
        print("Example: python -m src.enron_download allen-p.zip arnold-j.zip")
        sys.exit(1)

    for name in sys.argv[1:]:
        download_custodian(name)
