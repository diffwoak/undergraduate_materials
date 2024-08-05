#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<stdlib.h>
#include<winsock.h>
#include<mysql.h>
MYSQL mysql;
int create_sc_table();
int insert_rows_into_sc_table();
int main(int argc, char** argv, char** envp) {
	char fu[2];
	mysql_init(&mysql);
	//登录数据库，进入数据库xxgl
	if (mysql_real_connect(&mysql, "localhost", "root", "root", "xxgl", 3306, 0, 0)) {
		printf("1--创建sc表   2--插入sc表   0--退出\n");
		printf("\n");
		fu[0] = '0';
		scanf("%s",&fu);
		if (fu[0] == '0')exit(0);
		if (fu[0] == '1')create_sc_table();
		if (fu[0] == '2')insert_rows_into_sc_table();
	}
	else {
		printf("数据库不存在！");
	}
	return 0;
}
int create_sc_table() {
	char yn[2];
	//判断有无sc表
	if (mysql_query(&mysql,"select * from sc;")==0) {
		if (mysql_store_result(&mysql)) {
			printf("The sc table already exists, Do you want to delete it?\n");
			printf("Delete the table? (y--yes,n--no):");
			scanf("%s", &yn);
			if (yn[0] == 'y' || yn[0] == 'Y') {
				if (!mysql_query(&mysql, "drop table sc;")) {
					printf("Drop table sc successfully!%d\n\n");
				}
				else { printf("ERROR: drop table sc%d\n\n"); }
			}
		}
	}
	//创建表sc
	if (mysql_query(&mysql, "create table sc(sno CHAR(8) NOT NULL,cno CHAR(3) NOT NULL,grade INT,PRIMARY KEY(cno, sno))engine=innodb;") == 0) {
		printf("create table sc successfully!%d\n\n");
	}else { printf("ERROR: create table sc%d\n\n"); }
	//插入初始数据
	if (mysql_query(&mysql,"insert into sc values('2005001','1',85),('2005002','2',90);") == 0) {
		printf("Success to insert rows to sc table!%d\n\n");
	}else { printf("ERROR:insert row%d\n\n"); }
	return 0;
}
int insert_rows_into_sc_table() {
	//创建sc的插入数据变量
	char isno[] = "2005001";
	char icno[] = "00000";
	char igrade[] = "100";
	char strquery[100] = "insert into sc(sno,cno,grade) value('";
	char yn[2];
	//循环输入插入数据，连接成insert语句
	while (1) {
		printf("Please input sno(eg:2005001):"); scanf("%s", isno); strcat(strquery, isno);
		strcat(strquery, "','");
		printf("Please input cno(eg:1):"); scanf("%s", icno); strcat(strquery, icno);
		strcat(strquery, "',");
		printf("Please input grade(eg:80):"); scanf("%s", igrade); strcat(strquery, igrade);
		strcat(strquery, ");");
		if (mysql_query(&mysql, strquery) == 0) { printf("execute successfully!%d\n\n"); }
		else { printf("ERROR:execute%d\n"); }
		printf("Insert again? (y--yes,n--no)");
		scanf("%s", &yn);
		if (yn[0] == 'y' || yn[0] == 'Y') { continue; }
		else break;
	}
	return 0;
}