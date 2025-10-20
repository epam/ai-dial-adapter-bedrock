from pathlib import Path

from aidial_adapter_bedrock.utils.resource import Resource

BLUE_PNG_PICTURE = Resource.from_base64(
    type="image/png",
    data_base64="iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAIAAADZSiLoAAAAF0lEQVR4nGNkYPjPwMDAwMDAxAADCBYAG10BBdmz9y8AAAAASUVORK5CYII=",
)

DOG_PICTURE = Resource(
    type="image/png", data=Path("tests/assets/image1.png").read_bytes()
)

DOG_PICTURE_CONTENT = ["dog"]

SAMPLE_DOCUMENT_RESOURCE = Resource.from_base64(
    type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    data_base64="iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAIAAADZSiLoAAAAF0lEQVR4nGNkYPjPwMDAwMDAxAADCBYAG10BBdmz9y8AAAAASUVORK5CYII=",
)
