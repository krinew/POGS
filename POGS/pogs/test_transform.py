import torch
print("Script running directly")
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
config = NerfstudioDataParserConfig(data=Path("data/pogs_scenes/open_drawer/shared"))
parser = config.setup()
outputs = parser.get_dataparser_outputs(split="train")
print(outputs.dataparser_transform)
print(outputs.dataparser_scale)
