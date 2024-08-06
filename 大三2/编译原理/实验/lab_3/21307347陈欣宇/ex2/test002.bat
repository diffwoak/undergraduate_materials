@echo off
@echo Running Testcase 002: IllegalIntegerException
@echo ==============================================
cd bin
java -classpath .;..\lib\jflex-full-1.8.2.jar LexicalAnalysis ..\testcases\factorial.002
cd ..
@echo ==============================================
pause
@echo on