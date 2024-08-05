#include<iostream>
using namespace std;
#ifndef STONE_H
#define STONE_H
class Stone{
public:
enum TYPE{jiang,che,pao,ma,bing,shi,xiang};
unsigned short _row, _col;
short _id;
bool _dead;
bool _red;
int _type;
int scoreboard[10][9];
TCHAR* getText();
void init(int id,bool first);
void sb1();
void sb2();
void sb3();
void sb4();
void sb5();
void sb6();
void sb7();
};
#endif