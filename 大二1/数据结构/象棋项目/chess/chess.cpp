#include<iostream>
#include<stdio.h>
#include<easyx.h>           //针对 C++ 的图形库,帮助绘制自己棋盘棋子
#include<graphics.h>
#include"Board.h"
#include"Board.cpp"
#include"Stone.h"
#include"Stone.cpp"
#include"Singlegame.h"
#include"Singlegame.cpp"
#include"Step.h"
#include"Step.cpp"
using namespace std;
bool first;
bool pabegin(int x,int y){//绘制选择键，选择“先手”或“后手”
    bool bom=false;
    //setfillcolor(RGB(255,236,139));
    //fillrectangle(0,0,width,height);
    unsigned short bx1,by1,bx2,by2;
    bx1=width/2-30;bx2=width/2+30;
    by1=height/2-15-15;by2=height/2+15-15;
    if(x>=bx1&&x<=bx2&&y>=by1&&y<=by2){setfillcolor(BLACK);setbkcolor(BLACK);bom=true;first=true;}
    else {setfillcolor(RGB(139,137,137));setbkcolor(RGB(139,137,137));}    
    //字
    settextcolor(WHITE);
    TCHAR t1[20] = _T("先手");
	TCHAR t2[20] = _T("后手");
    settextstyle((by2-by1)*4/5, 0, _T("楷体"));
    LOGFONT f;
    gettextstyle(&f);
    f.lfQuality = ANTIALIASED_QUALITY;
    fillroundrect(bx1,by1,bx2,by2,3,3);///
    outtextxy((bx1+bx2)/2-(bx2-bx1)*2/5,(by1+by2)/2-(by2-by1)*2/5,t1);///
    by1=by1+35;by2=by2+35;
    if(x>=bx1&&x<=bx2&&y>=by1&&y<=by2){setfillcolor(BLACK);setbkcolor(BLACK);bom=true;first=false;}
    else {setfillcolor(RGB(139,137,137));setbkcolor(RGB(139,137,137));}   
    fillroundrect(bx1,by1,bx2,by2,3,3);///
    outtextxy((bx1+bx2)/2-(bx2-bx1)*2/5,(by1+by2)/2-(by2-by1)*2/5,t2);///
    return bom;
}
bool select(){//开始界面读取鼠标操作
    MOUSEMSG m; 
    m = GetMouseMsg();
    if(m.uMsg == WM_MOUSEMOVE){
        bool bom=pabegin(m.x,m.y);
    }
    if(m.uMsg == WM_LBUTTONDOWN){
        bool bom=pabegin(m.x,m.y);
        if(bom)return true;
    }
    return false;
}
bool drawend(bool blk,unsigned short x,unsigned short y){
    bool bom=false;
    unsigned short bx1,by1,bx2,by2;
    bx1=width/2-50;bx2=width/2+50;
    by1=height/2-15-15;by2=height/2+15-15;
    if(x>=bx1&&x<=bx2&&y>=by1&&y<=by2){setfillcolor(BLACK);setbkcolor(BLACK);bom=true;first=true;}
    else {setfillcolor(RGB(139,137,137));setbkcolor(RGB(139,137,137));}    
    //字
    settextcolor(WHITE);
    TCHAR t1[20] = _T("红方胜");
	TCHAR t2[20] = _T("黑方胜");
    settextstyle((by2-by1)*4/5, 0, _T("楷体"));
    LOGFONT f;
    gettextstyle(&f);
    f.lfQuality = ANTIALIASED_QUALITY;
    fillroundrect(bx1,by1,bx2,by2,3,3);///
    if(!blk)outtextxy((bx1+bx2)/2-(bx2-bx1)*2/5.5,(by1+by2)/2-(by2-by1)*2/5,t1);///
    else outtextxy((bx1+bx2)/2-(bx2-bx1)*2/5.5,(by1+by2)/2-(by2-by1)*2/5,t2);
    //by1=by1+35;by2=by2+35;
    //if(x>=bx1&&x<=bx2&&y>=by1&&y<=by2){setfillcolor(BLACK);setbkcolor(BLACK);bom=true;first=false;}
    //else {setfillcolor(RGB(139,137,137));setbkcolor(RGB(139,137,137));}   
    //fillroundrect(bx1,by1,bx2,by2,3,3);///
    //outtextxy((bx1+bx2)/2-(bx2-bx1)*2/5,(by1+by2)/2-(by2-by1)*2/5,t2);///
    return bom;
}
bool select1(bool blk){
    MOUSEMSG m; 
    m = GetMouseMsg();
    if(m.uMsg == WM_MOUSEMOVE){
        bool bom=drawend(blk,m.x,m.y);
    }
    if(m.uMsg == WM_LBUTTONDOWN){
        bool bom=drawend(blk,m.x,m.y);
        if(bom)return true;
    }
    return false;
}
int main(){
    //Board board;
    initgraph(width,height);    //初始化图框
    IMAGE back;
    loadimage(&back,_T("./chess-img/back1.png"),width,1125*width/750,true);
    putimage(0,-80,&back);    //开始界面
    
    BeginBatchDraw();           //批量绘制图形的函数
    while(1){
        if(select())break;      //读取鼠标操作
        FlushBatchDraw();       //
    }FlushBatchDraw();
    EndBatchDraw();             //
    Singlegame board;
    BeginBatchDraw();
    while(1){                   //过程更新界面
        board.paint();              //每一步重新画棋盘
        if(board._s[4]._dead==true||board._s[20]._dead ==true){break;}//判断游戏结束
        board.mouseevent();         //鼠标操作
        FlushBatchDraw();
    }FlushBatchDraw();
    EndBatchDraw();
    BeginBatchDraw();           //绘制结束界面
    while(1){
        if(select1(board._redturn))break;   //鼠标操作判定退出界面
        FlushBatchDraw();
    }FlushBatchDraw();
    EndBatchDraw();
    return 0;
}
/*
1、建立类Board、stone 其中有初始化stone位置、绘制棋盘、实现棋子移动等功能
2、在实现棋盘双方能够按规则自由下棋后，建立类Singlegame（继承board），step记录怎么走，实现对方自动下象棋。
3、完善功能，如选择先手后手，提高自动下棋速度等
*/