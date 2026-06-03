"""Parse EDRM Enron v2 XML files into structured email objects."""

import re
from dataclasses import dataclass, field
from typing import Optional
from lxml import etree


@dataclass
class EnronEmail:
    doc_id: str
    from_addr: str
    to_addr: str
    subject: str
    body: str
    date_sent: str
    custodian: Optional[str] = None
    cc: str = ""
    bcc: str = ""

    def to_text(self) -> str:
        """Render as plain text for embedding.

        If the body already contains email headers (from EDRM text extraction),
        return it directly to avoid duplication. Otherwise, prepend structured
        metadata.
        """
        if self.body and self.body.lstrip().startswith(("Date:", "From:", "To:", "Subject:")):
            return self.body
        parts = []
        if self.subject:
            parts.append(f"Subject: {self.subject}")
        if self.from_addr:
            parts.append(f"From: {self.from_addr}")
        if self.to_addr:
            parts.append(f"To: {self.to_addr}")
        if self.date_sent:
            parts.append(f"Date: {self.date_sent}")
        if self.body:
            parts.append("")
            parts.append(self.body)
        return "\n".join(parts)


def parse_edrm_xml_file(
    xml_content: str | bytes, text_files: dict[str, str]
) -> list[EnronEmail]:
    """Parse an EDRM XML file and return a list of EnronEmail objects.

    Args:
        xml_content: Raw XML string or bytes.
        text_files: Mapping of filename -> text content for referenced text files.

    Returns:
        List of EnronEmail objects (messages only, attachments skipped).
    """
    if isinstance(xml_content, str):
        xml_content = xml_content.encode("utf-8")

    root = etree.fromstring(xml_content)
    emails = []

    for doc in root.iter("Document"):
        doc_type = doc.get("DocType", "")
        if doc_type != "Message":
            continue

        doc_id = doc.get("DocID", "")
        tags = {}
        for tag in doc.iter("Tag"):
            tags[tag.get("TagName", "")] = tag.get("TagValue", "")

        # Extract body from Text file type only
        body = ""
        for file_el in doc.iter("File"):
            if file_el.get("FileType", "") != "Text":
                continue
            for ext_file in file_el.iter("ExternalFile"):
                fname = ext_file.get("FileName", "")
                fpath = ext_file.get("FilePath", "")
                # Try filename alone, then path/filename
                if fname in text_files:
                    body = text_files[fname]
                    break
                combined = f"{fpath}/{fname}" if fpath else fname
                if combined in text_files:
                    body = text_files[combined]
                    break
            if body:
                break

        # Extract custodian from Locations
        custodian: Optional[str] = None
        custodian_el = doc.find(".//Locations/Location/Custodian")
        if custodian_el is not None and custodian_el.text:
            raw = custodian_el.text
            # Extract email from angle brackets if present
            m = re.search(r"<([^>]+)>", raw)
            if m:
                custodian = m.group(1)
            else:
                custodian = raw.strip()

        email = EnronEmail(
            doc_id=doc_id,
            from_addr=tags.get("#From", ""),
            to_addr=tags.get("#To", ""),
            subject=tags.get("#Subject", ""),
            body=body,
            date_sent=tags.get("#DateSent", ""),
            custodian=custodian,
            cc=tags.get("#CC", ""),
            bcc=tags.get("#BCC", ""),
        )
        emails.append(email)

    return emails
