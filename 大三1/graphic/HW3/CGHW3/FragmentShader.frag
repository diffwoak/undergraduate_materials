#version 410 core

in vec3 FragPos;
in vec3 FragNormal;
out vec4 FragColor;
vec3 lightColor = vec3(1.0f, 1.0f ,1.0f);           // 光的强度
vec3 lightPos =  vec3(30.0f, 10.0f, 10.0f);   // 光源位置
vec3 viewPos = vec3(0.0f,0.0f,1.0f);                // 观察位置
void main(){
	vec3 Ia, Id, Is;
	vec3 Ka = vec3(0.2f, 0.2f, 0.2f);	//环境光分量
	vec3 Kd = vec3(1.0f, 1.0f, 1.0f);	//漫反射分量
	vec3 Ks = vec3(1.0f, 1.0f, 1.0f);	//镜面反射分量
	vec3 La = vec3(0.2f, 0.2f, 0.2f);	//环境光颜色
	vec3 Ld = vec3(0.6f, 0.8f, 0.8f);	//漫反射颜色
	vec3 Ls = vec3(1.0f, 1.0f, 1.0f);	//镜面反射颜色
	float alpha = 50.0f;	//镜面反射高光系数
	Ia.x = Ka.x * La.x; Ia.y = Ka.y * La.y; Ia.z = Ka.z * La.z;
	vec3 lightDir = normalize(lightPos - FragPos);	//光线单位向量
	vec3 normal = normalize(FragNormal);			//单位法向量
	vec3 viewDir = normalize(viewPos - FragPos);	//视线单位向量
	float dot_nl = dot(normal, lightDir);
	if(dot_nl > 0){
		Id.x = Kd.x * Ld.x * dot_nl; Id.y = Kd.y * Ld.y * dot_nl; Id.z = Kd.z * Ld.z * dot_nl;
		vec3 reflectDir = normalize(reflect(-lightDir, normal));	//镜面反射单位向量
		float dot_rv = dot(reflectDir, viewDir);
		if(dot_rv > 0){
			Is.x = Ks.x * Ls.x * pow(dot_rv, alpha); Is.y = Ks.y * Ls.y * pow(dot_rv, alpha); Is.z = Ks.z * Ls.z * pow(dot_rv, alpha);
		}else Is = vec3(0, 0, 0);
	}else {Id = vec3(0, 0, 0);Is = vec3(0, 0, 0);}
	vec3 result = Ia + Id +Is;
	FragColor = vec4(result,1.0f);
}

