# Asteroids Shader

A very simple copy of the arcade game "asteroids" that runs entirely inside shaders (after the usual meaty vulkan setup).

The program was tested on AMD and Nvdia hardware. Sadly, my Intel igpu do not support the required extension.

For more detail, see my blog post here: <https://gabdube.github.io/python/vulkan/2019/01/10/shader-games.html>

![Image](/img/img.png "Image")  

## Requirements

* Python 3.7
* A recent Vulkan driver
  * Extensions `VK_KHR_draw_indirect_count` or `VK_AMD_draw_indirect_count` must be supported
* `glslangValidator` (to compile the shaders)
* (Optional) Vulkan validation layers

## Run

### From the windows binary

Download the binary from `https://github.com/gabdube/asteroids-shader/releases/tag/1.0.0` and run `asteroids.exe`

### From the source

```sh
git clone git@github.com:gabdube/asteroids-shader.git
cd asteroids-shader
python tools/compile_shaders.py
python asteroids.py
```

## Commands

Stuff that I am too lazy to type manually

```sh
source ./env/Scripts/activate
source ~/Documents/vulkan_sdk/1.1.106.0/setup-env.sh
```
