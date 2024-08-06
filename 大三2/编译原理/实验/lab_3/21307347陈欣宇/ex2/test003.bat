@echo off
@echo Running Testcase 003: IllegalIntegerRangeException
@echo ==============================================
cd bin
java -classpath .;..\lib\jflex-full-1.8.2.jar LexicalAnalysis ..\testcases\factorial.003
cd ..
@echo ==============================================
pause
@echo on