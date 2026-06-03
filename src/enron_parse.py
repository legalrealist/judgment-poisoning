"""Parse EDRM Enron v2 XML files into structured email objects."""

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
        """Render as plain text for embedding."""
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

        body = ""
        for ext_file in doc.iter("ExternalFile"):
            fname = ext_file.get("FileName", "")
            if fname in text_files:
                body = text_files[fname]
                break

        email = EnronEmail(
            doc_id=doc_id,
            from_addr=tags.get("#From", ""),
            to_addr=tags.get("#To", ""),
            subject=tags.get("#Subject", ""),
            body=body,
            date_sent=tags.get("#DateSent", ""),
            cc=tags.get("#CC", ""),
            bcc=tags.get("#BCC", ""),
        )
        emails.append(email)

    return emails
