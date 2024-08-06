@echo off
cd bin
java -classpath .;..\lib\jflex-full-1.8.2.jar LexicalAnalysis ..\testcases\factorial.obr
pause
@echo on