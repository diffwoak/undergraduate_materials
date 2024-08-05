#ifndef UTILS_H
#define UTILS_H

#include <GL/glew.h>
#include <assert.h>
#include <cmath>
#include <glm/glm.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <GL/glew.h>

#define MAX_VERTICES_COT 8000
#define MAX_TRIANGLE_COT 8000

using namespace glm;

struct FragmentAttr {
    int x;
    int y;
    float z;
    int edgeID;
    vec3 color = vec3(1.0f, 1.0f, 1.0f);
    vec3 normal;
    vec3 pos_mv;
    FragmentAttr(){}
    FragmentAttr(int xx, int yy, float zz, int in_edgeID) :x(xx), y(yy), z(zz), edgeID(in_edgeID) {
    }
};


struct Triangle {// 3顶点 3法向量
    vec3 triangleVertices[3];
    vec3 triangleNormals[3];
};

struct Model {

    int triangleCount;
    int verticesCount;
    int normalCot;
    vec3 centralPoint;
    vec3 vertices_data[MAX_VERTICES_COT];   // Ver_id x vec3
    int** triangles;                        //triangle_id x 3_Ver_id 通过三角形id号找到3个点的id号
    int** triangle_normals;                 //triangle_id x 3_nor_id 通过三角形id号找到3个一样的法向量id号
    vec3 normals_data[MAX_VERTICES_COT];    // Nor_id x vec3
    std::string loadedPath = "";

    Model() {
        triangles = new int* [MAX_TRIANGLE_COT];
        for (int i = 0; i < MAX_TRIANGLE_COT; ++i) {
            triangles[i] = new int[3];
        }
        triangle_normals = new int* [MAX_TRIANGLE_COT];
        for (int i = 0; i < MAX_TRIANGLE_COT; ++i) {
            triangle_normals[i] = new int[3];
        }
    }

    Triangle getTriangleByID(int id) {
        assert(id < triangleCount);
        int* nowTirVerIDs = triangles[id];
        int* nowTriNormIDs = triangle_normals[id];
        /*vec3 nowTriangleVertices[3];
        vec3 nowTriNorms[3];
        nowTriangleVertices[0] = vertices_data[nowTirVerIDs[0]];
        nowTriangleVertices[1] = vertices_data[nowTirVerIDs[1]];
        nowTriangleVertices[2] = vertices_data[nowTirVerIDs[2]];
        nowTriNorms[0] = normals_data[nowTriNormIDs[0]];
        nowTriNorms[1] = normals_data[nowTriNormIDs[1]];
        nowTriNorms[2] = normals_data[nowTriNormIDs[2]];*/
        Triangle nowTriangle;
        nowTriangle.triangleVertices[0] = vertices_data[nowTirVerIDs[0]];
        nowTriangle.triangleVertices[1] = vertices_data[nowTirVerIDs[1]];
        nowTriangle.triangleVertices[2] = vertices_data[nowTirVerIDs[2]];
        nowTriangle.triangleNormals[0] = normals_data[nowTriNormIDs[0]];
        nowTriangle.triangleNormals[1] = normals_data[nowTriNormIDs[1]];
        nowTriangle.triangleNormals[2] = normals_data[nowTriNormIDs[2]];

        return nowTriangle;
    }

    void loadModel(const std::string& path) {
        if (path == loadedPath)
            return;
        loadedPath = path;
        std::ifstream file(path);
        std::string line;
        this->triangleCount = 0;
        this->normalCot = 1;
        this->verticesCount = 1; //obj 文件顶点从1开始计数
        this->centralPoint = vec3(0, 0, 0);
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string type;
            iss >> type;
            if (type == "v") {  //顶点
                if (verticesCount >= MAX_VERTICES_COT)
                    continue;
                vec3 vertex;
                iss >> vertex.x >> vertex.y >> vertex.z;
                vertices_data[verticesCount] = vertex;
                centralPoint += vertex;     //起点平移
                verticesCount++;
            }
            else if (type == "vn") {    //法向量
                if (normalCot >= MAX_VERTICES_COT)
                    continue;
                vec3 vertex;
                iss >> vertex.x >> vertex.y >> vertex.z;
                normals_data[normalCot] = vertex;
                normalCot++;
            }
            else if (type == "f") {     //三角形
                if (triangleCount >= MAX_TRIANGLE_COT)
                    continue;
                for (int i = 0; i < 3; ++i) {
                    char slash;
                    int vertexIndex, normalIndex, textIndex;
                    // 读取顶点索引、忽略纹理坐标、读取法线索引
                    iss >> vertexIndex >> slash >> textIndex >> slash >> normalIndex;
                    triangles[triangleCount][i] = vertexIndex;
                    triangle_normals[triangleCount][i] = normalIndex;
                }
                triangleCount++;
            }
        }
        centralPoint /= verticesCount;
        std::cout << "read " << triangleCount << " triangles" << std::endl;
    }
};

FragmentAttr getLinearInterpolation(const FragmentAttr& a, FragmentAttr& b, int x_position,int y_position);
void renderWithTexture(vec3* render_buffer,int w,int h);
#endif // UTILS.H