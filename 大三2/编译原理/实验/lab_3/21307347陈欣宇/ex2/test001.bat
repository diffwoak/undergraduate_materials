@echo off
@echo Running Testcase 001: IllegalSymbolException
@echo ==============================================
cd bin
java -classpath .;..\lib\jflex-full-1.8.2.jar LexicalAnalysis ..\testcases\factorial.001
cd ..
@echo ==============================================
pause
@echo on