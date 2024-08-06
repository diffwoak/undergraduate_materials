@echo off
@echo Running Testcase 001: MismatchedCommentException
@echo ==============================================
cd bin
java -classpath .;..\lib\jflex-full-1.8.2.jar LexicalAnalysis ..\testcases\factorial.006
cd ..
@echo ==============================================
pause
@echo on