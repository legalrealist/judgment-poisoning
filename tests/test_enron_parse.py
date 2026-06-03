import pytest
from src.enron_parse import parse_edrm_xml_file, EnronEmail


def test_parse_edrm_xml_extracts_fields():
    """Parse a single EDRM XML file and extract email fields."""
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
    <Root>
        <Batch>
            <Documents>
                <Document DocID="doc_001" DocType="Message" MimeType="message/rfc822">
                    <Tags>
                        <Tag TagName="#From" TagValue="smith@enron.com" />
                        <Tag TagName="#To" TagValue="jones@enron.com" />
                        <Tag TagName="#Subject" TagValue="Re: California pricing" />
                        <Tag TagName="#DateSent" TagValue="2001-08-15T10:30:00Z" />
                    </Tags>
                    <Files>
                        <File FileType="Text">
                            <ExternalFile FileName="doc_001.txt" />
                        </File>
                    </Files>
                </Document>
            </Documents>
        </Batch>
    </Root>"""
    text_files = {"doc_001.txt": "Let's discuss the California pricing strategy for next quarter."}

    emails = parse_edrm_xml_file(xml_content, text_files)

    assert len(emails) == 1
    email = emails[0]
    assert isinstance(email, EnronEmail)
    assert email.doc_id == "doc_001"
    assert email.from_addr == "smith@enron.com"
    assert email.to_addr == "jones@enron.com"
    assert email.subject == "Re: California pricing"
    assert "California pricing strategy" in email.body
    assert email.custodian is None  # set later from directory structure


def test_parse_edrm_xml_skips_non_message():
    """Non-message documents (attachments) are skipped."""
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
    <Root>
        <Batch>
            <Documents>
                <Document DocID="attach_001" DocType="File" MimeType="application/pdf">
                    <Tags>
                        <Tag TagName="#FileName" TagValue="report.pdf" />
                    </Tags>
                </Document>
            </Documents>
        </Batch>
    </Root>"""

    emails = parse_edrm_xml_file(xml_content, {})
    assert len(emails) == 0
