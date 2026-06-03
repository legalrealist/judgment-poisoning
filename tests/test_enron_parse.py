import pytest
from src.enron_parse import parse_edrm_xml_file, EnronEmail


def test_parse_edrm_xml_extracts_fields():
    """Parse a single EDRM XML file and extract email fields."""
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
    <Root MajorVersion="1" MinorVersion="0" CaseId="test_case" Description="CaseUtil">
        <Batch name=" Batch 1">
            <Documents>
                <Document DocID="doc_001" DocType="Message" MimeType="message/rfc822">
                    <Tags>
                        <Tag TagName="#From" TagDataType="Text" TagValue="smith@enron.com" />
                        <Tag TagName="#To" TagDataType="Text" TagValue="jones@enron.com" />
                        <Tag TagName="#Subject" TagDataType="Text" TagValue="Re: California pricing" />
                        <Tag TagName="#DateSent" TagDataType="DateTime" TagValue="2001-08-15T10:30:00Z" />
                    </Tags>
                    <Files>
                        <File FileType="Native">
                            <ExternalFile FilePath="native_000" FileName="doc_001.eml" FileSize="5000" Hash="abc" />
                        </File>
                        <File FileType="Text">
                            <ExternalFile FilePath="text_000" FileName="doc_001.txt" FileSize="1234" Hash="def" />
                        </File>
                    </Files>
                    <Locations>
                        <Location>
                            <Custodian>John Smith&lt;smith@enron.com&gt;</Custodian>
                            <LocationURI>smith-j\\'Sent Mail</LocationURI>
                        </Location>
                    </Locations>
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
    assert email.custodian == "smith@enron.com"


def test_parse_edrm_xml_skips_non_message():
    """Non-message documents (attachments) are skipped."""
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
    <Root MajorVersion="1" MinorVersion="0">
        <Batch name=" Batch 1">
            <Documents>
                <Document DocID="attach_001" DocType="File" MimeType="application/pdf">
                    <Tags>
                        <Tag TagName="#FileName" TagDataType="Text" TagValue="report.pdf" />
                    </Tags>
                </Document>
            </Documents>
        </Batch>
    </Root>"""

    emails = parse_edrm_xml_file(xml_content, {})
    assert len(emails) == 0


def test_parse_edrm_xml_custodian_extraction():
    """Custodian email is extracted from Locations element."""
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
    <Root>
        <Batch name=" Batch 1">
            <Documents>
                <Document DocID="doc_002" DocType="Message" MimeType="message/rfc822">
                    <Tags>
                        <Tag TagName="#From" TagDataType="Text" TagValue="Phillip K Allen" />
                        <Tag TagName="#Subject" TagDataType="Text" TagValue="Test" />
                        <Tag TagName="#DateSent" TagDataType="DateTime" TagValue="2000-09-26T16:26:00Z" />
                    </Tags>
                    <Files>
                        <File FileType="Text">
                            <ExternalFile FilePath="text_000" FileName="doc_002.txt" />
                        </File>
                    </Files>
                    <Locations>
                        <Location>
                            <Custodian>Phillip K Allen&lt;phillip.k.allen@enron.com&gt;</Custodian>
                        </Location>
                    </Locations>
                </Document>
            </Documents>
        </Batch>
    </Root>"""
    text_files = {"doc_002.txt": "Hello"}

    emails = parse_edrm_xml_file(xml_content, text_files)
    assert len(emails) == 1
    assert emails[0].custodian == "phillip.k.allen@enron.com"


def test_parse_edrm_xml_custodian_no_email():
    """Custodian falls back to full name if no angle-bracket email."""
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
    <Root>
        <Batch name=" Batch 1">
            <Documents>
                <Document DocID="doc_003" DocType="Message" MimeType="message/rfc822">
                    <Tags>
                        <Tag TagName="#From" TagDataType="Text" TagValue="Someone" />
                        <Tag TagName="#Subject" TagDataType="Text" TagValue="Test" />
                        <Tag TagName="#DateSent" TagDataType="DateTime" TagValue="2001-01-01T00:00:00Z" />
                    </Tags>
                    <Locations>
                        <Location>
                            <Custodian>John Doe</Custodian>
                        </Location>
                    </Locations>
                </Document>
            </Documents>
        </Batch>
    </Root>"""

    emails = parse_edrm_xml_file(xml_content, {})
    assert len(emails) == 1
    assert emails[0].custodian == "John Doe"


def test_parse_edrm_xml_text_file_with_filepath_prefix():
    """Text file lookup works when keys use FilePath/FileName format."""
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
    <Root>
        <Batch name=" Batch 1">
            <Documents>
                <Document DocID="doc_004" DocType="Message" MimeType="message/rfc822">
                    <Tags>
                        <Tag TagName="#From" TagDataType="Text" TagValue="test@enron.com" />
                        <Tag TagName="#Subject" TagDataType="Text" TagValue="Hello" />
                        <Tag TagName="#DateSent" TagDataType="DateTime" TagValue="2001-05-01T12:00:00Z" />
                    </Tags>
                    <Files>
                        <File FileType="Text">
                            <ExternalFile FilePath="text_000" FileName="doc_004.txt" FileSize="500" />
                        </File>
                    </Files>
                </Document>
            </Documents>
        </Batch>
    </Root>"""
    # Keys use path/filename format (as extracted from zip)
    text_files = {"text_000/doc_004.txt": "Body content from path-prefixed key."}

    emails = parse_edrm_xml_file(xml_content, text_files)
    assert len(emails) == 1
    assert "Body content from path-prefixed key" in emails[0].body


def test_parse_edrm_xml_ignores_native_files():
    """Only Text FileType is used for body, not Native."""
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
    <Root>
        <Batch name=" Batch 1">
            <Documents>
                <Document DocID="doc_005" DocType="Message" MimeType="message/rfc822">
                    <Tags>
                        <Tag TagName="#From" TagDataType="Text" TagValue="x@enron.com" />
                        <Tag TagName="#Subject" TagDataType="Text" TagValue="Test" />
                        <Tag TagName="#DateSent" TagDataType="DateTime" TagValue="2001-01-01T00:00:00Z" />
                    </Tags>
                    <Files>
                        <File FileType="Native">
                            <ExternalFile FilePath="native_000" FileName="doc_005.eml" />
                        </File>
                    </Files>
                </Document>
            </Documents>
        </Batch>
    </Root>"""
    # Even if the native filename is in text_files, it should not be used
    text_files = {"doc_005.eml": "This should NOT be the body."}

    emails = parse_edrm_xml_file(xml_content, text_files)
    assert len(emails) == 1
    assert emails[0].body == ""
