@echo off
@echo Running Testcase 004: IllegalOctalException
@echo ==============================================
cd bin
java -classpath .;..\lib\jflex-full-1.8.2.jar LexicalAnalysis ..\testcases\factorial.004
cd ..
@echo ==============================================
pause
@echo on