#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 normal;

out vec3 FragNormal;
out vec3 FragPos;

uniform mat4 projection;
uniform mat4 model;
uniform mat4 view;

void main(){
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    FragNormal = mat3(transpose(inverse(model))) * normal;
    FragPos = vec3(model * vec4(aPos, 1.0));
}