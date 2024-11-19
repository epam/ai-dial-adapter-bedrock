from pathlib import Path

from aidial_adapter_bedrock.utils.resource import Resource

BLUE_PNG_PICTURE = Resource.from_base64(
    type="image/png",
    data_base64="iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAIAAADZSiLoAAAAF0lEQVR4nGNkYPjPwMDAwMDAxAADCBYAG10BBdmz9y8AAAAASUVORK5CYII=",
)
CURRENT_DIR = Path(__file__).parent
SAMPLE_DOG_IMAGE_PATH = CURRENT_DIR / "images" / "dog-sample-image.png"
SAMPLE_DOG_RESOURCE = Resource(
    type="image/png",
    data=SAMPLE_DOG_IMAGE_PATH.read_bytes(),
)
