
from .auto_caption import Joy_Model_load, Auto_Caption
from .auto_caption import LoadManyImages

NODE_CLASS_MAPPINGS = {
    "Joy Model load":Joy_Model_load,
    "Auto Caption":Auto_Caption,
    "LoadManyImages":LoadManyImages
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Joy Model load":"Joy Model load",
    "Auto Caption":"Auto Caption",
    "LoadManyImages":"Load Many Images"
}