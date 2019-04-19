"""
Compile the shaders found in the asset shaders directory using glslangValidator. 
glslangValidator must be in your PATH.

Usage:
1 - Make sure that `glslangValidator` is in your path
2 - `python ./tools/compile_shaders.py`
"""

from pathlib import Path
from io import StringIO
import sys, subprocess

argv = sys.argv

# path_index = argv.index("--path")
# path = Path(argv[path_index+1])
path = Path(".")

# input_index = argv.index("--input")
# input_shader = argv[input_index+1]
# pattern = f"**/{input_shader}.*"
pattern = "tetris*.*"

outputs = {}


for file in path.glob(pattern):
    if file.suffix not in ('.frag', '.vert', '.comp'):
        continue

    file_in = str(file)
    file_out = str(file_in) + '.spv'

    p = subprocess.Popen(["glslangValidator", "-V", file_in, "-o", file_out], stdout=subprocess.PIPE)
    outputs[file.name] = p

for name, process in outputs.items():
    outputs[name] = process.communicate()[0]

if len(outputs) == 0:
    print("Warning! No shader file selected.")

print()
for name, output in outputs.items():
    print(name)
    print(output.decode('utf8'))
