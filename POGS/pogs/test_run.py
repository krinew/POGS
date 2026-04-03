from pathlib import Path
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
import torch
import json
config = NerfstudioDataParserConfig(data=Path("../data/pogs_scenes/open_drawer/shared"))
parser = config.setup()
outputs = parser.get_dataparser_outputs(split="train")
transform = outputs.dataparser_transform
scale = outputs.dataparser_scale

with open("../data/pogs_scenes/open_drawer/shared/table_bounding_cube.json", 'r') as f:
    bounding_box_dict = json.load(f)
table_z_val = bounding_box_dict['table_height'] + 0.015

table_pt = torch.tensor([0.0, 0.0, table_z_val, 1.0]).float()
table_pt_model = (transform @ table_pt) * scale

print(f"Original table_z_val: {table_z_val}")
print(f"Transformed table_pt_model: {table_pt_model}")
