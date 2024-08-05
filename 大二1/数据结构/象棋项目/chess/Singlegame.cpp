#include"Singlegame.h"
#include"Step.h"
#include<windows.h>
extern bool first;
void Singlegame::click(short id,unsigned short row,unsigned short col){    
    if(first) _redturn=!_redturn;
    if(!this->_redturn)
    {Board::click(id,row,col);}
    else if(this->_redturn){
        Step* step=getbestmove();
        movestone(step->_movid,step->_killid,step->_rowTo,step->_colTo);
        delete step;
    }
    if(first) _redturn=!_redturn;
    //else Board::click(id,row,col);
}
Step* Singlegame::getbestmove(){
    vector<Step*> steps;
    if(first) _redturn=!_redturn;
    getallmove(steps);//取得所有可能方案
    int maxscore=-1000000;
    Step* ret=NULL;//选择出最佳方案return
    int score;
    while(steps.size()){// min max
        Step* step=steps.back();
        steps.pop_back();
        fakemove(step);     //执行
        if(_s[4]._dead==true||_s[20]._dead==true){//特殊情况：将军若被吃掉则无需计算下一步分数，并直接跳出
            if(ret) delete ret;
            ret=step;
            unfakemove(step);
            while(steps.size()){
                Step* stepp=steps.back();
                steps.pop_back();
                delete stepp;
            }
            break;
        }
        else {score=getminscore(_level-1,maxscore);}//最小最大算法
        unfakemove(step);   //撤回
        if(score>maxscore){
            maxscore=score;
            if(ret) delete ret;//比较分数，选择最优方案
            ret=step;
        }
        else{delete step;}
    }
    if(first) _redturn=!_redturn;
    return ret;
}
void Singlegame::fakemove(Step* step){
    killstone(step->_killid);
    movestone(step->_movid,step->_rowTo,step->_colTo);
}
void Singlegame::unfakemove(Step* step){
    relivestone(step->_killid);
    movestone(step->_movid,step->_rowFrom,step->_colFrom);
}
int Singlegame::calcscore(){
    //enum TYPE{jiang,che,pao,ma,bing,shi,xiang};
    static int chessscore[]={50000,100,50,45,10,20,20};
    //黑棋-红棋
    int redscore=0,blackscore=0;
    for(short i=0;i<16;i++){
        if(_s[i]._dead)continue;
        redscore+=(chessscore[_s[i]._type]+_s[i].scoreboard[_s[i]._col][_s[i]._row]);
    }
    for(short i=16;i<32;i++){
        if(_s[i]._dead)continue;
        blackscore+=(chessscore[_s[i]._type]+_s[i].scoreboard[_s[i]._col][_s[i]._row]);
    }
    return redscore-blackscore;//改进：棋子本身分值+棋子对应位置分值
}
void Singlegame::getallmove(vector<Step*>& steps){
    short min=16,max=32;
    if(this->_redturn&&!first||!this->_redturn&&first){
        min=0;max=16;
    }//选择哪些棋子作为这轮自动下的棋子
    for(short i=min;i<max;i++){
        if(_s[i]._dead)continue;
        for(unsigned short row=0;row<=9;++row){
            for(unsigned short col=0;col<=8;++col){
                short killid=this->getstoneid(row,col);
                //if(_s[killid]._red==_s[i]._red)continue;
                if(canmove(i,row,col,killid)){
                    savestep(i,killid,row,col,steps);//遍历所有能走到方案，记录在steps中
                }
            }
        }
    }
}
int Singlegame::getminscore(unsigned short level,int curmax){//找到让分数最小方案（对面最佳方案）
    if(level==0)return calcscore();
    vector<Step*> steps;
    int minscore=10000000;int score;
    getallmove(steps);
    while(steps.size()){
        Step* step=steps.back();steps.pop_back();
        fakemove(step);     //if(_s[4]._dead==true||_s[20]._dead==true){_level=1;}
        if(_s[4]._dead==true||_s[20]._dead==true){
            int sco=calcscore();
            unfakemove(step);
            while(steps.size()){
                Step* step=steps.back();
                steps.pop_back();
                delete step;
            }
            return sco;
        }
        else {score= getmaxscore(level-1,minscore);}//不同
        unfakemove(step);
        delete step;
        if(score<=curmax){//剪枝
            while(steps.size()){
                Step* step=steps.back();
                steps.pop_back();
                delete step;
            }
            return score;
        }
        if(score<minscore){minscore=score;}///
    }
    return minscore;
}
int Singlegame::getmaxscore(unsigned short level,int curmin){
    if(level==0)return calcscore();
    vector<Step*> steps;
    int maxscore=-10000000;int score;
    getallmove(steps);
    while(steps.size()){
        Step* step=steps.back();
        steps.pop_back();
        fakemove(step);     //if(_s[4]._dead==true||_s[20]._dead==true){_level=1;}
        if(_s[4]._dead==true||_s[20]._dead==true){
            int sco=calcscore();
            unfakemove(step);
            while(steps.size()){
                Step* stepp=steps.back();
                steps.pop_back();
                delete stepp;
            }
            return sco;
        }
        else {score= getminscore(level-1,maxscore);}
        unfakemove(step);
        delete step;
        if(score>=curmin){
            while(steps.size()){
                Step* stepp=steps.back();
                steps.pop_back();
                delete stepp;
            }
            return score;}
        if(score>maxscore){maxscore=score;}
    }
    return maxscore;
}