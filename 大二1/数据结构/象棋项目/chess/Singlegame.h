#ifndef SINGLEGAME_H
#define SINGLEGAME_H
#include "Board.h"
class Singlegame : public Board{
public:
Singlegame(){_level=4;}
unsigned short _level;
void click(short id,unsigned short row,unsigned short col);//虚函数
Step* getbestmove();
void getallmove(vector<Step*>& steps);
void fakemove(Step* step);//移动 杀死
void unfakemove(Step* step);//移回 复活
int calcscore();//评价分数
int getminscore(unsigned short level,int curmax);//用于min-max算法
int getmaxscore(unsigned short level,int curmin);
};
#endif