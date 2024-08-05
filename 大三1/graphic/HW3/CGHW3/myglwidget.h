#ifndef MYGLWIDGET_H
#define MYGLWIDGET_H

#ifdef MAC_OS
#include <QtOpenGL/QtOpenGL>
#else
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#endif
#include <QtGui>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLFunctions_3_3_Core>
#include "utils.h"

#define MAX_Z_BUFFER 99999999.0f
#define MIN_FLOAT 1e-10f

using namespace glm;

class MyGLWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
    Q_OBJECT

public:
    MyGLWidget(QWidget *parent = nullptr);
    ~MyGLWidget();

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int width, int height) override;

private:
    
    void Sphere(GLfloat* vertices);
    void Sphere_index(GLfloat* vertices);
    void usemodel(GLfloat* vertice);
    void initShader(const char* vertexPath, const char* fragmentPath, unsigned int* ID);
    void initVboVao();
    void initVboVao_index();
    void draw();
    void gldraw();
    void draw_index();

    unsigned int program_id;
    GLuint vertexShader;
    GLuint fragmentShader;
    GLuint vboId;
    GLuint vaoId;
    GLuint iboId;
    int triangle_num;
    vec3 centerposition;
    clock_t start, end;
    GLfloat offset_z;
    std::string type = "VBO";//"VBO" / "glvertex" / "index"

    int WindowSizeH = 0;
    int WindowSizeW = 0;
    int scene_id;
    int degree = 0;


    Model objModel;


};

#endif // MYGLWIDGET_H
