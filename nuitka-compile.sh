python -m nuitka tetris.py\
  --include-plugin-directory ./libs/system\
  --include-plugin-directory ./libs/vulkan\
  --follow-imports\
  --follow-stdlib\
  --standalone\
  --python-flag=-OO\
  --python-flag=no_site\
  -j 2\
  --show-progress --show-modules --show-scons
