#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (constant_id = 1) const uint MAX_OBJECT_COUNT = 64;

layout (location = 0) in vec4 inPos;

struct GameObject {
    mat4 matrix;
    
    vec2 position;
    float angle;
    float velocity;
    
    float displayAngle;
    float displayAngleUpdate;

    float lifetime;

    uint commandIndex;
};

layout (std430, set=0, binding=0) readonly buffer GameData {
    GameObject objects[MAX_OBJECT_COUNT];
} game;

void main() 
{
	gl_Position = game.objects[gl_InstanceIndex].matrix * inPos;
}
