from pathlib import Path

from aidial_adapter_anthropic.dial.resource import Resource

BLUE_PNG_PICTURE = Resource.from_base64(
    type="image/png",
    data_base64="iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAIAAADZSiLoAAAAF0lEQVR4nGNkYPjPwMDAwMDAxAADCBYAG10BBdmz9y8AAAAASUVORK5CYII=",
)

DOG_PICTURE = Resource(
    type="image/png", data=Path("tests/assets/image1.png").read_bytes()
)

DOG_PICTURE_CONTENT = ["dog"]

PDF_DOCUMENT_RESOURCE = Resource(
    type="application/pdf", data=Path("tests/assets/doc.pdf").read_bytes()
)

EXCEL_DOCUMENT_RESOURCE = Resource(
    type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    data=Path("tests/assets/table.xlsx").read_bytes(),
)

SAMPLE_DOCUMENT_RESOURCE = Resource.from_base64(
    type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    data_base64="iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAIAAADZSiLoAAAAF0lEQVR4nGNkYPjPwMDAwMDAxAADCBYAG10BBdmz9y8AAAAASUVORK5CYII=",
)
