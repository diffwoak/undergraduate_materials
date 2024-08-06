@echo off
@echo Running Testcase 005: IllegalIdentifierLengthException
@echo ==============================================
cd bin
java -classpath .;..\lib\jflex-full-1.8.2.jar LexicalAnalysis ..\testcases\factorial.005
cd ..
@echo ==============================================
pause
@echo on