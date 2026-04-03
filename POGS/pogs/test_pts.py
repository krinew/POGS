from pathlib import Path
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
config = NerfstudioDataParserConfig(data=Path("../data/pogs_scenes/open_drawer/shared"))
parser = config.setup()
outputs = parser.get_dataparser_outputs(split="train")
print(outputs.metadata.keys())
print("Metadata has points3D_xyz?", "points3D_xyz" in outputs.metadata)
if "points3D_xyz" in outputs.metadata:
    print(outputs.metadata["points3D_xyz"][:5])
