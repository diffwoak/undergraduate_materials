# pragma warning (disable:4819)
#include "Board.h"
#include"Stone.h"
#include"Step.h"
#include<iostream>
#include<stdio.h>
#include<easyx.h>
#include<graphics.h>
extern bool first;
void Board::paint(){
    cleardevice();
    //画棋盘
    setfillcolor(RGB(255,236,139));
    fillrectangle(0,0,width,height);
    setlinecolor(BLACK);
    for(unsigned short i=1;i<11;i++)line(d,i*d,9*d,i*d);
    line(d,d,d,10*d);line(9*d,d,9*d,10*d);
    for(unsigned short i=2;i<9;i++){
        line(i*d,d,i*d,5*d);line(i*d,6*d,i*d,10*d);
    }
    line(4*d,d,6*d,3*d);line(4*d,3*d,6*d,d);
    line(4*d,8*d,6*d,10*d);line(4*d,10*d,6*d,8*d);
    //添加外框
    line(d-d/10,d-d/10,d-d/10,10*d+d/10);
    line(d-d/10,d-d/10,9*d+d/10,d-d/10);
    line(9*d+d/10,d-d/10,9*d+d/10,10*d+d/10);
    line(d-d/10,10*d+d/10,9*d+d/10,10*d+d/10);
    //炮兵位置
    for(short i=1;i<10;i++){
        if(i%2){
            if(i<9){
                line(i*d+d/10,4*d-d/3,i*d+d/10,4*d-d/10);line(i*d+d/10,4*d+d/3,i*d+d/10,4*d+d/10);
                line(i*d+d/10,7*d-d/3,i*d+d/10,7*d-d/10);line(i*d+d/10,7*d+d/3,i*d+d/10,7*d+d/10);
                line(i*d+d/10,4*d-d/10,i*d+d/3,4*d-d/10);line(i*d+d/10,7*d+d/10,i*d+d/3,7*d+d/10);
                line(i*d+d/10,4*d+d/10,i*d+d/3,4*d+d/10);line(i*d+d/10,7*d-d/10,i*d+d/3,7*d-d/10);
            }
            if(i>1){
                line(i*d-d/10,4*d-d/3,i*d-d/10,4*d-d/10);line(i*d-d/10,4*d+d/3,i*d-d/10,4*d+d/10);
                line(i*d-d/10,7*d-d/3,i*d-d/10,7*d-d/10);line(i*d-d/10,7*d+d/3,i*d-d/10,7*d+d/10);
                line(i*d-d/10,4*d-d/10,i*d-d/3,4*d-d/10);line(i*d-d/10,7*d-d/10,i*d-d/3,7*d-d/10);
                line(i*d-d/10,7*d+d/10,i*d-d/3,7*d+d/10);line(i*d-d/10,4*d+d/10,i*d-d/3,4*d+d/10);
            } 
        }
        else{
            if(i==2||i==8){
                line(i*d+d/10,3*d-d/3,i*d+d/10,3*d-d/10);line(i*d+d/10,3*d+d/3,i*d+d/10,3*d+d/10);
                line(i*d+d/10,8*d-d/3,i*d+d/10,8*d-d/10);line(i*d+d/10,8*d+d/3,i*d+d/10,8*d+d/10);
                line(i*d+d/10,3*d-d/10,i*d+d/3,3*d-d/10);line(i*d+d/10,8*d+d/10,i*d+d/3,8*d+d/10);
                line(i*d+d/10,3*d+d/10,i*d+d/3,3*d+d/10);line(i*d+d/10,8*d-d/10,i*d+d/3,8*d-d/10);

                line(i*d-d/10,3*d-d/3,i*d-d/10,3*d-d/10);line(i*d-d/10,3*d+d/3,i*d-d/10,3*d+d/10);
                line(i*d-d/10,8*d-d/3,i*d-d/10,8*d-d/10);line(i*d-d/10,8*d+d/3,i*d-d/10,8*d+d/10);
                line(i*d-d/10,3*d-d/10,i*d-d/3,3*d-d/10);line(i*d-d/10,8*d-d/10,i*d-d/3,8*d-d/10);
                line(i*d-d/10,8*d+d/10,i*d-d/3,8*d+d/10);line(i*d-d/10,3*d+d/10,i*d-d/3,3*d+d/10);
            }
                
        } 
    }
    //加楚河汉界
    setbkcolor(RGB(255,236,139));
    settextstyle(4*d/5,0, _T("华文隶书"));
    TCHAR s1[20] = _T("楚河");
    TCHAR s2[20] = _T("汉界");
    outtextxy(width/2-3*d,height/2-2*d/5,s1);
    outtextxy(width/2+3*d-8*d/5,height/2-2*d/5,s2);
    //画棋子
    for(int i=0;i<32;i++){
        drawstone(i);
    }
}
void Board::drawstone(short id){
    if(_s[id]._dead)return;
    if(id==_selectid){
        setfillcolor(LIGHTGRAY);setbkcolor(LIGHTGRAY);
    }else{
        setfillcolor(YELLOW);setbkcolor(YELLOW);
    }
    
    fillcircle(center_x(id),center_y(id),d*2/5);
    if(!_s[id]._red)settextcolor(BLACK);
    else settextcolor(RED);
    
    settextstyle(0, 0, _T("楷体"));
    LOGFONT f;
    gettextstyle(&f);
    f.lfQuality = ANTIALIASED_QUALITY;
    outtextxy(center_x(id)-d/4,center_y(id)-d/4,_s[id].getText());
}
unsigned short Board::center_x(short id){
    return (_s[id]._col+1)*d;
}
unsigned short Board::center_y(short id){
    return (_s[id]._row+1)*d;
}
bool Board::getrowcol(unsigned short x,unsigned short y,unsigned short &row,unsigned short &col){
    if(x<d/2||x>(10*d-d/2)||y<d/2||y>(11*d-d/2))return false;
    row=(y-d/2)/d;
    col=(x-d/2)/d;
    return true;
}
short Board::getstoneid(unsigned short row,unsigned short col){
    for(short i=0;i<32;i++){
        if(_s[i]._row==row&&_s[i]._col==col&&!_s[i]._dead)return i;
    }
    return -1;
}
unsigned short Board::countline(unsigned short row1,unsigned short col1,unsigned short row2,unsigned short col2){
    unsigned short num=0;
    if(row1!=row2&&col1!=col2)return -1;
    if(row1==row2&&col1==col2)return -1;
    if(row1==row2){
        unsigned short min=col1<col2?col1:col2;
        unsigned short max=col1<col2?col2:col1;
        for(short i=min+1;i<max;i++)if(getstoneid(row1,i)!=-1)++num;
    }
    else{
        unsigned short min=row1<row2?row1:row2;
        unsigned short max=row1<row2?row2:row1;
        for(short i=min+1;i<max;i++)if(getstoneid(i,col1)!=-1)++num;
    }
    return num;
}
bool Board::canmove1(short movid,unsigned short row,unsigned short col,short killid){
    if(killid != -1 && _s[killid]._type == Stone::jiang)
    return canmove4(movid,row,col,killid);
    if(_s[movid]._red&&!first||!_s[movid]._red&&first){if(row>=3)return false;}
    else if(row<7)return false;
    if(col<3||col>5)return false;
    short dr=_s[movid]._row-row;
    short dc=_s[movid]._col-col;
    unsigned short d=abs(dr*10)+abs(dc);
    if(d==1||d==10)return true;
    return false;
}
bool Board::canmove2(short movid,unsigned short row,unsigned short col,short killid){
    if(_s[movid]._red&&!first||!_s[movid]._red&&first){if(row>=3)return false;}
    else if(row<7)return false;
    if(col<3||col>5)return false;
    short dr=_s[movid]._row-row;
    short dc=_s[movid]._col-col;
    unsigned short d=abs(dr*10)+abs(dc);
    if(d==11)return true;
    return false;
    }
bool Board::canmove3(short movid,unsigned short row,unsigned short col,short killid){
    short dr=_s[movid]._row-row;
    short dc=_s[movid]._col-col;
    unsigned short d=abs(dr*10)+abs(dc);
    if(d!=22)return false;
    unsigned short r=_s[movid]._row+row;r/=2;
    unsigned short c=_s[movid]._col+col;c/=2;
    if(getstoneid(r,c)!=-1)return false;//阻挡
    if(!_s[movid]._red&&!first||_s[movid]._red&&first){if(row<4)return false;}
    else{if(row>5)return false;}
    return true;
}
bool Board::canmove4(short movid,unsigned short row,unsigned short col,short killid){
    unsigned short num=countline(_s[movid]._row,_s[movid]._col,row,col);
    if(num==0)return true;
    return false;
}
bool Board::canmove5(short movid,unsigned short row,unsigned short col,short killid){
    short dr=_s[movid]._row-row;
    short dc=_s[movid]._col-col;
    unsigned short d=abs(dr*10)+abs(dc);
    if(d!=12&&d!=21)return false;
    //马脚
    if(d==12){if(getstoneid(_s[movid]._row,(_s[movid]._col+col)/2)!=-1)return false;}
    else{if(getstoneid((_s[movid]._row+row)/2,_s[movid]._col)!=-1)return false;}
    return true;}
bool Board::canmove6(short movid,unsigned short row,unsigned short col,short killid){
    unsigned short num=countline(_s[movid]._row,_s[movid]._col,row,col);
    if(killid!=-1){if(num==1)return true;}
    else{if(num==0)return true;}
    return false;
}
bool Board::canmove7(short movid,unsigned short row,unsigned short col,short killid){
    short dr=_s[movid]._row-row;
    short dc=_s[movid]._col-col;
    unsigned short d=abs(dr*10)+abs(dc);
    if(d!=1&&d!=10)return false;
    if(!_s[movid]._red&&!first||_s[movid]._red&&first){
        if(row>_s[movid]._row)return false;
        if(_s[movid]._row>=5&&row==_s[movid]._row)return false;
    }
    else{
        if(row<_s[movid]._row)return false;
        if(_s[movid]._row<=4&&row==_s[movid]._row)return false;
    }
    return true;}
bool Board::canmove(short movid,unsigned short row,unsigned short col,short killid){
    if(killid!=-1)if(_s[movid]._red==_s[killid]._red){
        return false;
    }
    switch(_s[movid]._type)
    {
    case Stone::jiang:
        return canmove1(movid,row,col,killid);
    case Stone::shi:
        return canmove2(movid,row,col,killid);
    case Stone::xiang:
        return canmove3(movid,row,col,killid);
    case Stone::che:
        return canmove4(movid,row,col,killid);
    case Stone::ma:
        return canmove5(movid,row,col,killid);
    case Stone::pao:
        return canmove6(movid,row,col,killid);
    case Stone::bing:
        return canmove7(movid,row,col,killid);
    }
    return true;
}
void Board::mouseevent(){
    ExMessage msg;
    if(peekmessage(&msg,EM_MOUSE)){    

        if(msg.message==WM_LBUTTONDOWN){
            unsigned short x=msg.x,y=msg.y;
            click(x,y);
        }        
        else if(first&&!_redturn||!first&&_redturn){
            click(msg.x,msg.y);return;
        }
    }
}
void Board::click(unsigned short x,unsigned short y){//转化成棋盘中的行和列
    unsigned short row,col;
    bool bret=getrowcol(x,y,row,col);
    if(!bret)return;
    short id=getstoneid(row,col);
    click(id,row,col);
}
void Board::click(short id,unsigned short row,unsigned short col){
    if(first) _redturn=!_redturn;
    short i;
    short clickid=-1;//现在点击
    for(i=0;i<32;i++){
        if(_s[i]._row==row&&_s[i]._col==col&&_s[i]._dead==false)break;
    }
    if(i<32){clickid=i;}//点到目标存在
    if(_selectid==-1){//第一次点击
        if(clickid!=-1)if(_redturn==_s[clickid]._red)_selectid=clickid;
    }
    else{//走棋     第二次点击
        if(canmove(_selectid,row,col,clickid)){
            _s[_selectid]._row=row;
            _s[_selectid]._col=col;
            if(clickid!=-1){_s[clickid]._dead=true;}
            _selectid=-1;
            _redturn=!_redturn;
        }
        //同色 -> 第二次置为第一次
        else if(clickid!=-1&&_s[_selectid]._red==_s[clickid]._red)_selectid=clickid;
    }
    if(first) _redturn=!_redturn;
}

void Board::movestone(short movid, unsigned short row, unsigned short col){
    _s[movid]._row = row;
    _s[movid]._col = col;
    _redturn = !_redturn;
}
void Board::movestone(short movid,short killid,unsigned short row,unsigned short col){
    //savestep(movid,killid,row,col,_steps);
    killstone(killid);
    movestone(movid,row,col);
}
void Board::killstone(short id){
    if(id==-1) return;
    _s[id]._dead = true;
}
void Board::relivestone(short id){
    if(id==-1) return;
    _s[id]._dead = false;
}
void Board::savestep(short movid, short killid, unsigned short row, unsigned short col, vector<Step*>& steps){
    Step* step = new Step;
    step->_colFrom = _s[movid]._col;
    step->_colTo = col;
    step->_rowFrom = _s[movid]._row;
    step->_rowTo = row;
    step->_movid = movid;
    step->_killid = killid;
    steps.push_back(step);
}