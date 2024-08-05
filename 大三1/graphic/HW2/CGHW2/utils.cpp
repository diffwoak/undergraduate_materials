#include "utils.h"
FragmentAttr getLinearInterpolation(const FragmentAttr& a, FragmentAttr& b, int x_position, int y_position) {
    FragmentAttr result;
    result.x = x_position;
	result.y = y_position;
	int mol,den;
	if (abs(a.x - b.x) >= abs(a.y - b.y)) {
		mol = x_position - a.x; den =b.x - a.x;
	}
	else {
		mol = y_position - a.y; den = b.y - a.y;
		result.z = ((b.y - a.y)*a.z + (y_position - a.y) * (b.z - a.z))/ float(b.y - a.y);
	}
    //result.z = a.z + t * (b.z - a.z);
	//插值
	result.color.r = (den * a.color.r + mol * (b.color.r - a.color.r)) / float(den);
	result.color.g = (den * a.color.g + mol * (b.color.g - a.color.g)) / float(den);
	result.color.b = (den * a.color.b + mol * (b.color.b - a.color.b)) / float(den);

    result.normal.x = (den * a.normal.x + mol * (b.normal.x - a.normal.x)) / float(den);
    result.normal.y = (den * a.normal.y + mol * (b.normal.y - a.normal.y)) / float(den);
    result.normal.z = (den * a.normal.z + mol * (b.normal.z - a.normal.z)) / float(den);

	result.pos_mv.x = (den * a.pos_mv.x + mol * (b.pos_mv.x - a.pos_mv.x)) / float(den);
	result.pos_mv.y = (den * a.pos_mv.y + mol * (b.pos_mv.y - a.pos_mv.y)) / float(den);
	result.pos_mv.z = (den * a.pos_mv.z + mol * (b.pos_mv.z - a.pos_mv.z)) / float(den);

	result.z = glm::length(result.pos_mv);
    return result;
}


void renderWithTexture(vec3* render_buffer,int h, int w) {
	// 保存当前矩阵模式和矩阵状态
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glEnable(GL_TEXTURE_2D);	// 启用2D纹理映射
	// 生成纹理ID并绑定纹理
	GLuint texID;
    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_2D, texID);
	// 设置纹理参数
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	// 将渲染缓冲区的数据传递给纹理
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_FLOAT, render_buffer);
	// 启用2D纹理映射并绑定纹理
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texID);
	// 绘制一个矩形（四边形），并映射纹理坐标
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(-1.0f, -1.0f);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(-1.0f, 1.0f);
	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(1.0f, 1.0f);
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(1.0f, -1.0f);
	glEnd();
	// 禁用2D纹理映射
	glDisable(GL_TEXTURE_2D);
	// 恢复矩阵模式和状态
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	// 恢复OpenGL属性
	glPopAttrib();
}