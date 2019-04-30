#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (constant_id = 1) const uint MAX_OBJECT_COUNT = 64;

layout (location = 0) in vec4 inPos;

const uint MESH_COUNT = 21;
const uint MAX_ASTEROID = 20;
const uint MAX_SHOT = 20;

struct VkDrawIndexedIndirectCommand {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    uint vertexOffset;
    uint firstInstance;
};

struct GameObject {
    VkDrawIndexedIndirectCommand command;

    uint status;
    
    vec2 position;
    float angle;
    float velocity;
    
    float displayAngle;
    float displayAngleUpdate;

    mat4 matrix;
};

struct Mesh {
    uint indicesOffset;
    uint indicesCount;
    uint vertexOffset;
    uint vertexCount;
};

struct Asteroid {
    uint objectIndex;
    uint life;
    float radius;
};

struct AsteroidArray {
    uint count;
    Asteroid asteroids[MAX_ASTEROID];
};

struct Shot {
    uint objectIndex;
    float lifetime;
};

struct ShotArray {
    uint count;
    Shot shots[MAX_SHOT];
};

layout (std430, set=0, binding=0) readonly buffer GameData {
    uint drawCount; 

    uint currentLevel;
    uint over;
    uint score;

    uint asteroidMeshIndex;
    uint asteroidMeshCount;

    GameObject objects[MAX_OBJECT_COUNT];
    Mesh meshes[MESH_COUNT];
    AsteroidArray asteroids;
    ShotArray shots;
} game;

void main() 
{
	gl_Position = game.objects[gl_InstanceIndex].matrix * inPos;
}
