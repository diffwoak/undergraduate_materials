#include "myglwidget.h"
#include <GL/glew.h>
#include <algorithm>
#include <time.h>
#include <map>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <QtCore/qdir.h>
#include <qopenglextrafunctions.h>

using namespace std;
#define PI acos(-1)
#define lats 512
#define lons 512

GLfloat vertices[6 * 3 * lats * lons * 2];
GLfloat vertices_index[3 * lats * lons * 2];
GLuint index[6* lats * lons];//三角形点索引
MyGLWidget::MyGLWidget(QWidget* parent)
	:QOpenGLWidget(parent)
{
}
MyGLWidget::~MyGLWidget()
{
	glDetachShader(program_id, vertexShader);
	glDetachShader(program_id, fragmentShader);
	glDeleteProgram(program_id);
}

vec3 Spherepoint(float u, float v) {
	vec3 point;
	point.x = sin(2 * PI * u) * cos(2 * PI * v);
	point.y = sin(2 * PI * u) * sin(2 * PI * v);
	point.z = cos(2 * PI * u);
	return point;
}
void MyGLWidget::Sphere(GLfloat* vertice) {
	GLfloat lon_step = 1.0f / lons;		
	GLfloat lat_step = 1.0f / lats;			
	GLuint offset = 0;
	vec3 t1, t2, t3, t4;
	vec3 n1, n2, n3, n4;
	triangle_num = 2 * lons * lats;
	offset_z = 10.0f;
	centerposition = vec3(0, 0, 0.0f);
	//将所有的点按顺序一次存储
	for (int lat = 0; lat < lats; lat++) {
		for (int lon = 0; lon < lons; lon++) {
			// index array 四个点生成两个三角形
			t1 = Spherepoint(lat * lat_step, lon * lon_step);
			t2 = Spherepoint((lat + 1) * lat_step, lon * lon_step);
			t3 = Spherepoint((lat + 1) * lat_step, (lon + 1) * lon_step);
			t4 = Spherepoint(lat * lat_step, (lon + 1) * lon_step);
			n1 = normalize(t1); n2 = normalize(t2); n3 = normalize(t3); n4 = normalize(t4);
			/*计算整个面的法向量
			vec3 r1, r2, rn;
			r1 = t1 - t4;
			r2 = t1 - t3;
			rn.x = r1.y * r2.z - r1.z * r2.y;
			rn.y = r2.x * r1.z - r2.z * r1.x;
			rn.z = r2.y * r1.x - r2.x * r1.y;
			rn = normalize(rn);
			*/
			memcpy(vertice + offset, &t1, sizeof(vec3)); offset += 3;
			memcpy(vertice + offset, &n1, sizeof(vec3)); offset += 3;
			memcpy(vertice + offset, &t4, sizeof(vec3)); offset += 3;
			memcpy(vertice + offset, &n4, sizeof(vec3)); offset += 3;
			memcpy(vertice + offset, &t3, sizeof(vec3)); offset += 3;
			memcpy(vertice + offset, &n3, sizeof(vec3)); offset += 3;
			/*计算面法向量
			r1 = t1 - t3;
			r2 = t1 - t2;
			rn.x = r1.y * r2.z - r1.z * r2.y;
			rn.y = r2.x * r1.z - r2.z * r1.x;
			rn.z = r2.y * r1.x - r2.x * r1.y;
			rn = normalize(rn);*/
			memcpy(vertice + offset, &t1, sizeof(vec3)); offset += 3;
			memcpy(vertice + offset, &n1, sizeof(vec3)); offset += 3;
			memcpy(vertice + offset, &t3, sizeof(vec3)); offset += 3;
			memcpy(vertice + offset, &n3, sizeof(vec3)); offset += 3;
			memcpy(vertice + offset, &t2, sizeof(vec3)); offset += 3;
			memcpy(vertice + offset, &n2, sizeof(vec3)); offset += 3;
		}
	}
}
void MyGLWidget::usemodel(GLfloat* vertice) {
	//objModel.loadModel("./objs/teapot_600.obj");
	objModel.loadModel("./objs/teapot_8000.obj");
	//objModel.loadModel("./objs/rock.obj");
	//objModel.loadModel("./objs/cube.obj");
	//objModel.loadModel("./objs/singleTriangle.obj");
	GLuint offset = 0;
	triangle_num = objModel.triangleCount;
	offset_z = 200.0f;
	centerposition = -objModel.centralPoint;
	for (int i = 0; i < objModel.triangleCount; i++) {
		Triangle nowTriangle = objModel.getTriangleByID(i);
		for (int i = 0; i < 3; i++) {
			memcpy(vertice + offset, &(nowTriangle.triangleVertices[i]), sizeof(vec3)); offset += 3;
			memcpy(vertice + offset, &normalize(nowTriangle.triangleNormals[i]), sizeof(vec3)); offset += 3;
		}
	}
}
void MyGLWidget::Sphere_index(GLfloat* vertice) {
	GLfloat lon_step = 1.0f / lons;
	GLfloat lat_step = 1.0f / lats;
	GLuint offset = 0;
	GLuint index_offset = 0;
	vec3 t;
	GLuint t2, t3, t4;
	vec3 n;
	triangle_num = 2 * lons * lats;
	offset_z = 10.0f;
	centerposition = vec3(0, 0, 0.0f);
	for (int lat = 0; lat < lats; lat++) {
		for (int lon = 0; lon < lons; lon++) {
			t = Spherepoint(lat * lat_step, lon * lon_step); n = normalize(t);
			memcpy(vertice + offset, &t, sizeof(vec3)); offset += 3;
			memcpy(vertice + offset, &n, sizeof(vec3)); offset += 3;
			t2 = lats * lat + lon + 1; t3 = lats * (lat + 1) + lon + 1; t4 = lats * (lat + 1) + lon;
			if (lat == lats - 1) {
				t4 -= lats * (lat + 1); t3 -= lats * (lat + 1);
			}
			if (lon == lons - 1) {
				t3 -= (lon + 1); t2 -= (lon + 1);
			}
			index[index_offset] = lats * lat + lon;
			index_offset++;
			index[index_offset] = t2;
			index_offset++;
			index[index_offset] = t3;
			index_offset++;
			index[index_offset] = lats * lat + lon;
			index_offset++;
			index[index_offset] = t4;
			index_offset++;
			index[index_offset] = t3;
			index_offset++;
		}
	}
}

void MyGLWidget::initShader(const char* vertexPath, const char* fragmentPath, unsigned int* ID)
{
	string Vcode;
	string Fcode;
	ifstream Vfile;
	ifstream Ffile;
	stringstream Vstream;
	stringstream Fstream;
	const char* vertexShaderSource;
	const char* fragmentShaderSource;
	// 绝对路径
	string vertexDir = QDir::currentPath().toStdString() + "/" + vertexPath;
	string fragmentDir = QDir::currentPath().toStdString() + "/" + fragmentPath;
	//加载代码
	Vfile.open(vertexDir);
	Vstream << Vfile.rdbuf();
	Vfile.close();
	Ffile.open(fragmentDir);
	Fstream << Ffile.rdbuf();
	Ffile.close();
	//字符转换
	Vcode = Vstream.str();
	Fcode = Fstream.str();
	vertexShaderSource = Vcode.c_str();
	fragmentShaderSource = Fcode.c_str();
	//编译顶点着色器
	int success;
	char infoLog[512];
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		cout << "error in vertexshader: compilation failed\n" << infoLog << endl;
	}
	else
		cout << "vertshader compiled successfully" << endl;
	//编译片元着色器
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		cout << "error in fragmentshader: compilation failed\n" << infoLog << endl;
	}
	else
		cout << "fragmentshader compiled successfully" << endl;
	//绑定并链接着色器
	*ID = glCreateProgram();
	glAttachShader(*ID, vertexShader);
	glAttachShader(*ID, fragmentShader);
	glLinkProgram(*ID);
	glGetProgramiv(*ID, GL_LINK_STATUS, &success);
	if (!success)
	{
		glGetProgramInfoLog(*ID, 512, NULL, infoLog);
		cout << "error: link failed\n" << infoLog << endl;
	}
	else cout << "link successfully" << endl;
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}

void MyGLWidget::initVboVao() {
	// 创建物体的 VAO,VBO
	glGenVertexArrays(1, &vaoId);
	glGenBuffers(1, &vboId);
	//绑定数组指针
	glBindVertexArray(vaoId);
	glBindBuffer(GL_ARRAY_BUFFER, vboId);
	//写入数据
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	//设置顶点属性指针
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	//设置法向量属性
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GL_FLOAT)));
	glEnableVertexAttribArray(1);
	//解绑VAO
	glBindVertexArray(0);
}
void MyGLWidget::initVboVao_index() {
	// 创建物体的 VAO,VBO
	glGenVertexArrays(1, &vaoId);
	glGenBuffers(1, &vboId);
	glGenBuffers(1, &iboId);
	// 绑定数组指针
	glBindVertexArray(vaoId);
	// 绑定并写入位置和法向量数据
	glBindBuffer(GL_ARRAY_BUFFER, vboId);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_index), vertices_index, GL_STATIC_DRAW);
	// 设置顶点属性指针（位置和法向量）
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0); // 位置
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GL_FLOAT))); // 法向量
	glEnableVertexAttribArray(1);
	// 绑定并写入索引数据
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboId);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(index), index, GL_STATIC_DRAW);
	// 解绑VAO
	glBindVertexArray(0);
}

void MyGLWidget::draw() {
	glUseProgram(program_id);
	float mat[16];
	//设置shader参数
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f,0.0f, -offset_z);
	glRotatef(30.0f, 1, 0, 0);
	glRotatef(60.0f, 0, 1, 0);
	glTranslatef(centerposition.x, centerposition.y, centerposition.z);//平移到原点
	glGetFloatv(GL_MODELVIEW_MATRIX, mat);
	glUniformMatrix4fv(glGetUniformLocation(program_id, "model"), 1, GL_FALSE, mat);
	glLoadIdentity();      
	gluLookAt(0, 0, 1, 0, 0, 0, 0, 1, 0);
	glGetFloatv(GL_MODELVIEW_MATRIX, mat);
	glUniformMatrix4fv(glGetUniformLocation(program_id, "view"), 1, GL_FALSE, mat);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(30.0f, 1.0f, 0.1f, 1000.0f);	//透视投影
	glGetFloatv(GL_PROJECTION_MATRIX, mat);
	glUniformMatrix4fv(glGetUniformLocation(program_id, "projection"), 1, GL_FALSE, mat);
	// 绘制图像
	glPushMatrix();
	glBindVertexArray(vaoId);
	glDrawArrays(GL_TRIANGLES, 0, 3 * triangle_num);
	glBindVertexArray(0);
	glPopMatrix();
	end = clock();
	//cout << double(end - start) / CLOCKS_PER_SEC << endl;
}
void MyGLWidget::gldraw() {
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	GLfloat light_position[] = { 30.0, 100.0, 100.0, 1.0 };  // 光源位置，最后一个值为0表示无穷远处的光源
	GLfloat light_ambient[] = { 0.2, 0.2, 0.2, 1.0 };    // 环境光分量
	GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 1.0 };    // 散射光分量
	GLfloat light_specular[] = { 1.0, 1.0, 1.0, 1.0 };   // 镜面反射光分量
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	GLfloat material_diffuse[] = { 0.6f, 0.8f, 0.8f, 1.0f };  // 漫反射颜色
	GLfloat material_ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f };  // 环境光反射颜色
	GLfloat material_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f }; // 镜面反射颜色
	GLfloat material_shininess[] = { 50.0f }; // 镜面反射高光指数
	glMaterialfv(GL_FRONT, GL_DIFFUSE, material_diffuse);
	glMaterialfv(GL_FRONT, GL_AMBIENT, material_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, material_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, material_shininess);
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glPushMatrix();
	gluPerspective(30.0f, 1.0f, 0.1f, 1000.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0, 0, 1, 0, 0, 0, 0, 1, 0);
	glPushMatrix();
	glTranslatef(0.0f, 0.0f, -offset_z);
	//gluLookAt(0, 0, 0, centerposition.x, centerposition.y, centerposition.z, 0, 1, 0);
	glRotatef(30.0f, 1, 0, 0);
	glRotatef(60.0f, 0, 1, 0);
	glTranslatef(centerposition.x, centerposition.y, centerposition.z);
	glBegin(GL_TRIANGLES);
	for (int i = 0; i < triangle_num; i++) {
		for (int j = 0; j < 3; j++) {
			glNormal3f(vertices[18 * i + 6 * j + 3], vertices[18 * i + 6 * j + 4], vertices[18 * i + 6 * j + 5]);
			glVertex3f(vertices[18 * i + 6 * j], vertices[18 * i + 6 * j + 1], vertices[18 * i + 6 * j + 2]);
		}
	}
	glEnd();
	glPopMatrix();
}
void MyGLWidget::draw_index() {
	glUseProgram(program_id);
	float mat[16];
	//设置shader参数
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -offset_z);
	glRotatef(30.0f, 1, 0, 0);
	glRotatef(60.0f, 0, 1, 0);
	glTranslatef(centerposition.x, centerposition.y, centerposition.z);//平移到原点
	glGetFloatv(GL_MODELVIEW_MATRIX, mat);
	glUniformMatrix4fv(glGetUniformLocation(program_id, "model"), 1, GL_FALSE, mat);
	glLoadIdentity();
	gluLookAt(0, 0, 1, 0, 0, 0, 0, 1, 0);
	glGetFloatv(GL_MODELVIEW_MATRIX, mat);
	glUniformMatrix4fv(glGetUniformLocation(program_id, "view"), 1, GL_FALSE, mat);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(30.0f, 1.0f, 0.1f, 1000.0f);	//透视投影
	glGetFloatv(GL_PROJECTION_MATRIX, mat);
	glUniformMatrix4fv(glGetUniformLocation(program_id, "projection"), 1, GL_FALSE, mat);

	glPushMatrix();
	glBindVertexArray(vaoId);
	glDrawElements(GL_TRIANGLES, 6 * lats * lons, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
	glPopMatrix();
	end = clock();
	//cout << double(end - start) / CLOCKS_PER_SEC<<endl;
}

void MyGLWidget::initializeGL()
{	
	initializeOpenGLFunctions();
	WindowSizeW = width();
	WindowSizeH = height();
	glViewport(0, 0, WindowSizeW, WindowSizeH);
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);
	type = "VBO";			//使用fragment shader及VBO实现phong shading
	//type = "glvertex";	//使用opengl自带的smoothing shading
	//type = "index";		//VBO使用 index array
	start = clock();
	if (type == "VBO"||type=="glvertex") {
		Sphere(vertices);
		//usemodel(vertices);	//加载模型
	}
	else {
		Sphere_index(vertices_index);
	}
	initShader("VertexShader.vert", "FragmentShader.frag", &program_id);
	if(type =="VBO")initVboVao();
	else if (type=="index")initVboVao_index();
}
void MyGLWidget::paintGL()
{	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	if (type == "VBO") {
		draw();
	}
	else if(type == "glvertex"){
		gldraw();
	}
	else if(type == "index"){
		draw_index();
	}
}
void MyGLWidget::resizeGL(int w, int h)
{
	glViewport(0, 0, w, h);
	update();
}
	