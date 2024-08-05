#include<iostream>
#include<vector>
#include"Stone.h"
#include"Step.h"
using namespace std;
#ifndef BOARD_H
#define BOARD_H
const unsigned short d=40;
const unsigned short width = 10*d;
const unsigned short height = 11*d;
extern bool first;
class Board{
public:
    Board(){
        _selectid=-1;
        _redturn=true;
        for(int i=0;i<32;i++)_s[i].init(i,first);
    }
    Stone _s[32];//32子
    short _selectid;//已选中点击
    bool _redturn;
    void paint();
    void mouseevent();
    void click(unsigned short x,unsigned short y);
    virtual void click(short id,unsigned short row,unsigned short col);
    unsigned short center_x(short id);
    unsigned short center_y(short id);
    void drawstone(short id);
    bool getrowcol(unsigned short x,unsigned short y,unsigned short &row,unsigned short &col);
    bool canmove(short movid,unsigned short row,unsigned short col,short killid);
    bool canmove1(short movid,unsigned short row,unsigned short col,short killid);
    bool canmove2(short movid,unsigned short row,unsigned short col,short killid);
    bool canmove3(short movid,unsigned short row,unsigned short col,short killid);
    bool canmove4(short movid,unsigned short row,unsigned short col,short killid);
    bool canmove5(short movid,unsigned short row,unsigned short col,short killid);
    bool canmove6(short movid,unsigned short row,unsigned short col,short killid);
    bool canmove7(short movid,unsigned short row,unsigned short col,short killid);
    short getstoneid(unsigned short row,unsigned short col);
    unsigned short countline(unsigned short row1,unsigned short col1,unsigned short row2,unsigned short col2);

    //more for step
    void movestone(short movid, unsigned short row, unsigned short col);
    void movestone(short movid,short killed,unsigned short row,unsigned short col);
    void killstone(short id);
    void relivestone(short id);
    void savestep(short movid, short killid, unsigned short row, unsigned short col, vector<Step*>& steps);
};

#endif // BOARD_H