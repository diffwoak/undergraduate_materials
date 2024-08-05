#include "myglwidget.h"
#include <GL/glew.h>
#include <algorithm>
#include <time.h>
#include <map>
#include <algorithm>
#include <cmath>
using namespace std;

MyGLWidget::MyGLWidget(QWidget *parent)
	:QOpenGLWidget(parent)
{
}

MyGLWidget::~MyGLWidget()
{
	delete[] render_buffer;
	delete[] temp_render_buffer;
	delete[] temp_z_buffer;
	delete[] z_buffer;
}

void MyGLWidget::resizeBuffer(int newW, int newH) {
	delete[] render_buffer;
	delete[] temp_render_buffer;
	delete[] temp_z_buffer;
	delete[] z_buffer;
	WindowSizeW = newW;
	WindowSizeH = newH;
	render_buffer = new vec3[WindowSizeH*WindowSizeW];
	temp_render_buffer = new vec3[WindowSizeH*WindowSizeW];
	temp_z_buffer = new float[WindowSizeH*WindowSizeW];
	z_buffer = new float[WindowSizeH*WindowSizeW];
}

void MyGLWidget::initializeGL()
{
	WindowSizeW = width();
	WindowSizeH = height();
	glViewport(0, 0, WindowSizeW, WindowSizeH);
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glDisable(GL_DEPTH_TEST);
	offset = vec2(WindowSizeW / 2, WindowSizeH / 2);
	// 对定义的数组初始化
	render_buffer = new vec3[WindowSizeH*WindowSizeW];
	temp_render_buffer = new vec3[WindowSizeH*WindowSizeW];
	temp_z_buffer = new float[WindowSizeH*WindowSizeW];
	z_buffer = new float[WindowSizeH*WindowSizeW];
	for (int i = 0; i < WindowSizeH*WindowSizeW; i++) {
		render_buffer[i] = vec3(0, 0, 0);
		temp_render_buffer[i] = vec3(0, 0, 0);
		temp_z_buffer[i] = MAX_Z_BUFFER;			
		z_buffer[i] = MAX_Z_BUFFER;
	}
}

void MyGLWidget::keyPressEvent(QKeyEvent *e) {
	
	switch (e->key()) {
		case Qt::Key_0: scene_id = 0;update(); break;
		case Qt::Key_1: scene_id = 1;update(); break;
		case Qt::Key_9: degree += 15;update(); break;
		case Qt::Key_Up: offset.y -= 5; update(); break;
		case Qt::Key_Down: offset.y += 5; update(); break;
		case Qt::Key_Left: offset.x -= 5; update(); break;
		case Qt::Key_Right: offset.x += 5; update(); break;
	}
}

void MyGLWidget::paintGL()
{
	switch (scene_id) {
		case 0:scene_0(); break;
		case 1:scene_1(); break;
	}
}
void MyGLWidget::clearBuffer(vec3* now_buffer) {
	for (int i = 0; i < WindowSizeH*WindowSizeW; i++) {
		now_buffer[i] = vec3(0,0,0);
	}
}

void MyGLWidget::clearBuffer(int* now_buffer) {
	memset(now_buffer, 0, WindowSizeW * WindowSizeH * sizeof(int));
}


void MyGLWidget::clearZBuffer(float* now_buffer) {
	std::fill(now_buffer,now_buffer+WindowSizeW * WindowSizeH, MAX_Z_BUFFER);
}


// 窗口大小变动后，需要重新生成render_buffer等数组
void MyGLWidget::resizeGL(int w, int h)
{
	resizeBuffer(w, h);
	offset = vec2(WindowSizeW / 2,WindowSizeH / 2 );
	clearBuffer(render_buffer);
}

void MyGLWidget::scene_0()
{
	// 选择要加载的model
	objModel.loadModel("./objs/singleTriangle.obj");

	// 自主设置变换矩阵
	camPosition = vec3(100 * sin(degree * 3.14 / 180.0) + objModel.centralPoint.y, 100 * cos(degree * 3.14 / 180.0) + objModel.centralPoint.x, 10+ objModel.centralPoint.z);
	camLookAt = objModel.centralPoint;     // 例如，看向物体中心
	camUp = vec3(0, 1, 0);         // 上方向向量
	projMatrix = glm::perspective(radians(20.0f), 1.0f, 0.1f, 2000.0f);

	// 单一点光源，可以改为数组实现多光源
	lightPosition = objModel.centralPoint + vec3(0,1000,1000);
	clearBuffer(render_buffer);
	clearZBuffer(z_buffer);
	for (int i = 0; i < objModel.triangleCount; i++) {
		Triangle nowTriangle = objModel.getTriangleByID(i);
		drawTriangle(nowTriangle);
	}
	glClear(GL_COLOR_BUFFER_BIT);
	renderWithTexture(render_buffer,WindowSizeH,WindowSizeW);
}


void MyGLWidget::scene_1()
{	
	clock_t start, end;
	start = clock();
	// 选择要加载的model
	//objModel.loadModel("./objs/teapot_600.obj");
	objModel.loadModel("./objs/teapot_8000.obj");
	//objModel.loadModel("./objs/rock.obj");
	//objModel.loadModel("./objs/cube.obj");
	//objModel.loadModel("./objs/singleTriangle.obj");
	// 自主设置变换矩阵
	camPosition = vec3(100 * sin(degree * 3.14 / 180.0) + objModel.centralPoint.x, 100 * cos(degree * 3.14 / 180.0) + objModel.centralPoint.y, 10+ objModel.centralPoint.z);
	camLookAt = objModel.centralPoint;     // 例如，看向物体中心
	camUp = vec3(0, 1, 0);         // 上方向向量
	projMatrix = glm::perspective(radians(30.0f), 1.0f, 0.1f, 2000.0f);

	// 单一点光源，可以改为数组实现多光源
	lightPosition = objModel.centralPoint + vec3(0,100,100);
	clearBuffer(render_buffer);
	clearZBuffer(z_buffer);
	for (int i = 0; i < objModel.triangleCount; i++) {
		Triangle nowTriangle = objModel.getTriangleByID(i);
		drawTriangle(nowTriangle);
	}
	glClear(GL_COLOR_BUFFER_BIT);
	renderWithTexture(render_buffer, WindowSizeH, WindowSizeW);
	end = clock();
	std::cout << double(end - start) / CLOCKS_PER_SEC << "+";
}

void MyGLWidget::drawTriangle(Triangle triangle) {
	Shadingtype = "Phong";// Gouraud / Phong / blinnphong
	// 三维顶点映射到二维平面
	vec3* vertices = triangle.triangleVertices;
	vec3* normals = triangle.triangleNormals;
	FragmentAttr transformedVertices[3];
	clearBuffer(this->temp_render_buffer);
	clearZBuffer(this->temp_z_buffer);
	mat4 viewMatrix = glm::lookAt(camPosition, camLookAt, camUp);
    for (int i = 0; i < 3; i++) { 
		vec4 ver_mv = viewMatrix * vec4(vertices[i], 1.0f);
		//float nowz = glm::length(camPosition - vec3(ver_mv));
		float nowz = glm::length(vec3(ver_mv));
		vec4 ver_proj = projMatrix * ver_mv;//与
		transformedVertices[i].x = ver_proj.x + offset.x;
		transformedVertices[i].y = ver_proj.y + offset.y;
		transformedVertices[i].z = nowz;
		mat3 normalMatrix = mat3(viewMatrix);
		vec3 normal_mv = normalMatrix * normals[i];
		transformedVertices[i].normal = normal_mv;
		transformedVertices[i].pos_mv= vec3(ver_mv);
		transformedVertices[i].color = PhoneShading(transformedVertices[i]);
    }
	map<int, map<int, FragmentAttr>> line_l;
	map<int, map<int, FragmentAttr>> line_r;
	/*DDA(transformedVertices[0], transformedVertices[1], 2, line_l, line_r);
	DDA(transformedVertices[1], transformedVertices[2], 0, line_l, line_r);
	DDA(transformedVertices[2], transformedVertices[0], 1, line_l, line_r);  */
	
	bresenham(transformedVertices[0], transformedVertices[1], 2, line_l, line_r);
    bresenham(transformedVertices[1], transformedVertices[2], 0, line_l, line_r);
	bresenham(transformedVertices[2], transformedVertices[0], 1, line_l, line_r);
    // HomeWork: 2: 用edge-walking填充三角形内部到temp_buffer中
    int firstChangeLine = edge_walking(transformedVertices, line_l, line_r);
	//int firstChangeLine = edge_walking();
	//int firstChangeLine = 0;
	for(int h = firstChangeLine; h < WindowSizeH ; h++){
		auto render_row = &render_buffer[h * WindowSizeW];
		auto temp_render_row = &temp_render_buffer[h * WindowSizeW];
		auto z_buffer_row = &z_buffer[h*WindowSizeW];
		auto temp_z_buffer_row = &temp_z_buffer[h*WindowSizeW];
		for (int i = 0 ; i < WindowSizeW ; i++){
			if (z_buffer_row[i] < temp_z_buffer_row[i])
				continue;
			else
			{
				z_buffer_row[i] = temp_z_buffer_row[i];
				render_row[i] = temp_render_row[i];
			}
		}
	}
}
// 遍历edge_recorder在不同高度的起点、终点，用shading model计算内部每个像素的颜色
int MyGLWidget::edge_walking(FragmentAttr* v, std::map<int, std::map<int, FragmentAttr>> l, std::map<int, std::map<int, FragmentAttr>> r) {
	//line_id: v0-v1=2 ,v1-v2=0,v0-v2=1
	int max_id = ((v[0].y >= v[1].y) && (v[0].y >= v[2].y)) ? 0 : ((v[1].y >= v[2].y) ? 1 : 2);
	int mid_id = (max_id != 0 && (v[0].y- v[1].y)*(v[0].y - v[2].y)<=0) ? 0 : ((max_id != 1 &&((v[1].y - v[0].y) * (v[1].y - v[2].y) <= 0)) ? 1 : 2);//高度位于中间的点id
	int min_id = 3 - max_id - mid_id;
	vec3 avg_normal = normalize(normalize(v[0].normal) + normalize(v[1].normal) + normalize(v[2].normal)); 
	if (v[mid_id].x > r[mid_id][v[mid_id].y].x) {//向右扩展
		if (l[min_id][v[mid_id].y].x > l[max_id][v[mid_id].y].x)l[min_id][v[mid_id].y] = l[max_id][v[mid_id].y];//对中间点拐角处做处理
		for (int h = v[min_id].y +1; h < v[mid_id].y; h++) {//最低点到中间点的高度扫描，另一边line_id为max_id
			if(h <= WindowSizeH && h > 0)
			for (int i = r[mid_id][h].x + 1; i < WindowSizeW && i<l[max_id][h].x; i++) {
				if (i < 0)continue;
				FragmentAttr px = getLinearInterpolation(r[mid_id][h], l[max_id][h], i, h); if (Shadingtype != "Gouraud") { px.normal = avg_normal; px.color = PhoneShading(px); }
				temp_render_buffer[(WindowSizeH - h) * WindowSizeW + i] =  px.color;//FragmentAttr插值或暂定为绿色
				temp_z_buffer[(WindowSizeH - h) * WindowSizeW + i] = px.z; //r[mid_id][h].z+(i- r[mid_id][h].x)*(l[max_id][h].z -r[mid_id][h].z)/(float)(l[max_id][h].x -r[mid_id][h].x);//FragmentAttr插值或暂定手动计算
				//z_l + (i - l) * (z_r - z_l) / (float)(r - l);
			}
		}
		for (int h = v[max_id].y - 1; h >= v[mid_id].y; h--) {//中间点到最高点的高度扫描，另一边line_id为min_id
			if (h <= WindowSizeH && h > 0)
			for (int i = r[mid_id][h].x + 1; i < WindowSizeW && i < l[min_id][h].x; i++) {
				if (i < 0)continue;
				FragmentAttr px = getLinearInterpolation(r[mid_id][h], l[min_id][h], i, h); if (Shadingtype != "Gouraud") { px.normal = avg_normal; px.color = PhoneShading(px); }
				temp_render_buffer[(WindowSizeH - h) * WindowSizeW + i] = px.color;//FragmentAttr插值或暂定为绿色
				temp_z_buffer[(WindowSizeH - h) * WindowSizeW + i] = px.z;
					//r[mid_id][h].z + (i - r[mid_id][h].x) * (l[min_id][h].z - r[mid_id][h].z) / (float)(l[min_id][h].x - r[mid_id][h].x);//FragmentAttr插值或暂定手动计算
			}
		}
	}
	else if(v[mid_id].x < l[mid_id][v[mid_id].y].x){//向左扩展
		if (r[min_id][v[mid_id].y].x < r[max_id][v[mid_id].y].x)r[min_id][v[mid_id].y] = r[max_id][v[mid_id].y];//对中间点拐角处做处理
		for (int h = v[min_id].y + 1; h < v[mid_id].y; h++) {//最小点到中间点的高度扫描，另一边line_id为max_id
			if (h <= WindowSizeH && h > 0)
				for (int i = l[mid_id][h].x - 1; i >= 0 && i > r[max_id][h].x; i--) {
					if (i >= WindowSizeW)continue;
					FragmentAttr px = getLinearInterpolation(l[mid_id][h], r[max_id][h], i, h); if (Shadingtype != "Gouraud") { px.normal = avg_normal; px.color = PhoneShading(px); }
					temp_render_buffer[(WindowSizeH - h) * WindowSizeW + i] = px.color;//FragmentAttr插值或暂定为绿色
					temp_z_buffer[(WindowSizeH - h) * WindowSizeW + i] = px.z; //r[max_id][h].z + (i - r[max_id][h].x) * (l[mid_id][h].z - r[max_id][h].z) / (float)(l[mid_id][h].x - r[max_id][h].x);//FragmentAttr插值或暂定手动计算
				}
		}
		for (int h = v[max_id].y - 1; h >= v[mid_id].y; h--) {//最大点到中间点的高度扫描，另一边line_id为min_id
			if (h <= WindowSizeH && h > 0)
				for (int i = l[mid_id][h].x - 1; i >= 0 && i > r[min_id][h].x; i--) {
					if (i >= WindowSizeW)continue;
					FragmentAttr px = getLinearInterpolation(l[mid_id][h], r[min_id][h], i, h); if (Shadingtype != "Gouraud") { px.normal = avg_normal; px.color = PhoneShading(px); }
					temp_render_buffer[(WindowSizeH - h) * WindowSizeW + i] = px.color;//FragmentAttr插值或暂定为绿色
					temp_z_buffer[(WindowSizeH - h) * WindowSizeW + i] = px.z; //r[min_id][h].z + (i - r[min_id][h].x) * (l[mid_id][h].z - r[min_id][h].z) / (float)(l[mid_id][h].x - r[min_id][h].x);//FragmentAttr插值或暂定手动计算
				}
		}
	}
	return (WindowSizeH - v[max_id].y) > 0 ? (WindowSizeH - v[max_id].y) : 0;
}
int MyGLWidget::edge_walking() {
	int firstChangeLine = WindowSizeH;
	bool nochange = true;
	for (int h = 0; h < WindowSizeH; h++) {
		auto temp_render_row = &temp_render_buffer[h * WindowSizeW];
		auto temp_z_buffer_row = &temp_z_buffer[h * WindowSizeW];
		int l = 0, r = WindowSizeW - 1;
		bool lp = true, rp = true;
		while (l < r && (lp || rp)) {
			if (lp && temp_render_row[l] == vec3(0, 0, 0))l++;
			else lp = false;
			if (rp && temp_render_row[r] == vec3(0, 0, 0))r--;
			else rp = false;
		}
		float z_l = temp_z_buffer_row[l], z_r = temp_z_buffer_row[r];
		for (int i = l + 1; i < r; i++) {
			temp_render_row[i] = vec3(0, 1, 0);
			temp_z_buffer_row[i] = z_l + (i - l) * (z_r - z_l) / (float)(r - l);
		}
		if (nochange && !(lp && rp)) { firstChangeLine = h; nochange = false; }
	}
	return firstChangeLine;
}

vec3 MyGLWidget::PhoneShading(FragmentAttr& nowPixelResult) {
	mat4 viewMatrix = glm::lookAt(camPosition, camLookAt, camUp);
	vec3 pos_l = vec3(viewMatrix * vec4(lightPosition, 1.0f));//光源在相机坐标位置
	vec3 L = normalize(pos_l - nowPixelResult.pos_mv);//光线单位向量
	vec3 V = normalize(-nowPixelResult.pos_mv);//视线单位向量
	vec3 N = normalize(nowPixelResult.normal);//单位法向量
	vec3 La = vec3(0.6f, 0.6f, 0.6f), Ld = vec3(1.0f, 1.0f, 1.0f), Ls = vec3(1.0f, 1.0f, 1.0f);//光照强度
	vec3 Ka = vec3(0.3f, 0.3f, 0.3f), Kd = vec3(0.7f, 0.7f, 0.6f), Ks = vec3(0.9f, 0.9f, 0.9f);//光照系数
	int alpha = 8;
	vec3 Ia, Id, Is;
	Ia.x = Ka.x * La.x; Ia.y = Ka.y * La.y; Ia.z = Ka.z * La.z;
	float dot_nl = dot(N, L);
	if (dot_nl > 0) {
		Id.x = Kd.x * Ld.x * dot_nl; Id.y = Kd.y * Ld.y * dot_nl; Id.z = Kd.z * Ld.z * dot_nl;
		if (Shadingtype == "phong" || Shadingtype == "Gouraud") {
			vec3 vec_2d = vec3(N.x * dot_nl * 2, N.y * dot_nl * 2, N.z * dot_nl * 2);
			vec3 R = vec_2d - L;//镜面反射单位向量
			float dot_rv = dot(R, V);
			if (dot_rv > 0) {
				Is.x = Ks.x * Ls.x * pow(dot_rv, alpha); Is.y = Ks.y * Ls.y * pow(dot_rv, alpha); Is.z = Ks.z * Ls.z * pow(dot_rv, alpha);
			}else Is = vec3(0, 0, 0);
		}else if(Shadingtype == "blinnphong"){
			vec3 H = normalize(L + V);//半程向量
			float dot_nh = dot(N, H);
			if (dot_nh > 0) {
				Is.x = Ks.x * Ls.x * pow(dot_nh, alpha); Is.y = Ks.y * Ls.y * pow(dot_nh, alpha); Is.z = Ks.z * Ls.z * pow(dot_nh, alpha);
			}else Is = vec3(0, 0, 0);
		}else Is = vec3(0, 0, 0);
	}
	else {Id = vec3(0, 0, 0);Is = vec3(0, 0, 0);}
	vec3 color = Ia + Id + Is;
	return color;
}
void MyGLWidget::DDA(FragmentAttr& start, FragmentAttr& end, int id,map<int,map<int, FragmentAttr>>& line_l,map<int,map<int, FragmentAttr>>& line_r) {
	int dx = end.x - start.x, dy = end.y - start.y, steps;
	double delta_x, delta_y, x = start.x, y = start.y;
	steps = (abs(dx) > abs(dy)) ? (abs(dx)) : (abs(dy));
	delta_x = dx / (float)steps;delta_y = dy / (float)steps;
	std::map<int, FragmentAttr> lside, rside;
	for (int i = 0; i <= steps; i++) {
		int intx = round(x), inty = round(y);
		FragmentAttr px = getLinearInterpolation(start, end, intx, inty);
		if (abs(delta_y) == 1) {lside[inty] = px;rside[inty] = px;}//对rside、lside进行赋值
		else {
			if (delta_x > 0) {//向右迭代，起点或左边无点则为最左，终点或右边无点则为最右
				if (i == 0 || round(y - delta_y) != inty)lside[inty] = px;
				if (i == steps || round(y + delta_y) != inty)rside[inty] = px;
			}
			else {//lside rside反过来
				if (i == 0 || round(y - delta_y) != inty)rside[inty] = px;
				if (i == steps || round(y + delta_y) != inty)lside[inty] = px;
			}
		}
		if (inty <= WindowSizeH && inty > 0 && intx < WindowSizeW && intx > 0) {//在窗口访问内进行temp_render_buffer和temp_z_buffer赋值
			temp_render_buffer[(WindowSizeH - inty) * WindowSizeW + intx] =  vec3(0, 0, 0);
			temp_z_buffer[(WindowSizeH - inty) * WindowSizeW + intx] = px.z;
		}
		x += delta_x;
		y += delta_y;
	}
	line_l[id] = lside;
	line_r[id] = rside;
}
void MyGLWidget::bresenham(FragmentAttr& start, FragmentAttr& end, int id,map<int,map<int, FragmentAttr>>& line_l,map<int,map<int, FragmentAttr>>& line_r) {
	int dx = abs(end.x - start.x), dy = abs(end.y - start.y), x, y, Iy, steps;
	double delta_x, delta_y, p;
	map<int, FragmentAttr> lside, rside;
	steps = (dx > dy) ? dx : dy;
	if (dx >= dy) {//斜率小于1
		x = (start.x < end.x) ? start.x : end.x;
		y = (start.x < end.x) ? start.y : end.y;
		delta_x = dx / (float)steps;delta_y = dy / (float)steps;
		p = 2 * delta_y - delta_x;
		if ((end.x - start.x) * (end.y - start.y) >= 0)Iy = 1;//判断y是+还是-
		else Iy = -1;
		for (int i = 0; i <= steps; i++) {
			FragmentAttr px = getLinearInterpolation(start, end, x, y); 
			if (i == 0)lside[y] = px;
			if (i == steps)rside[y] = px;
			if (y <= WindowSizeH && y > 0 && x < WindowSizeW && x > 0) {//temp_render_buffer和temp_z_buffer赋值
				temp_render_buffer[(WindowSizeH - y) * WindowSizeW + x] = vec3(0, 0, 0);
				temp_z_buffer[(WindowSizeH - y) * WindowSizeW + x] = px.z;
			}
			if (p <= 0) p += 2 * delta_y;
			else if (i < steps) {
				rside[y] = px;
				y += Iy;
				lside[y] = getLinearInterpolation(start, end, x+1, y);
				p += 2 * (delta_y - delta_x);
			}
			x += 1;
		}
	}
	else {//斜率大于1
		x = (start.y < end.y) ? start.y : end.y;
		y = (start.y < end.y) ? start.x : end.x;
		delta_x = dy / (float)steps; delta_y = dx / (float)steps;
		p = 2 * delta_y - delta_x;
		if ((end.x - start.x) * (end.y - start.y) >= 0)Iy = 1;
		else Iy = -1;
		for (int i = 0; i <= steps; i++) {
			FragmentAttr px = getLinearInterpolation(start, end, y, x);
			rside[x] = lside[x] = px;
			if (x <= WindowSizeH && x > 0 && y < WindowSizeW && y > 0) {//temp_render_buffer和temp_z_buffer赋值
				temp_render_buffer[(WindowSizeH - x) * WindowSizeW + y] = vec3(0,0,0);
				temp_z_buffer[(WindowSizeH - x) * WindowSizeW + y] = px.z;
			}
			if (p <= 0) p += 2 * delta_y;
			else {
				y += Iy;
				p += 2 * (delta_y - delta_x);
			}
			x += 1;
		}
	}
	line_l[id] = lside;
	line_r[id] = rside;
}

