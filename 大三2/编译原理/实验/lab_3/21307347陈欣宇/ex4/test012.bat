@echo off
@echo Running Testcase 012: TypeMismatchedException
@echo ==============================================
cd bin
java -cp .;..\lib\flowchart.jar;..\lib\jgraph.jar;..\lib\callgraph.jar;..\lib\java-cup-11b-runtime.jar;..\lib\jflex-full-1.8.2.jar. Main ..\testcases\factorial.012
cd ..
@echo ==============================================
pause
@echo on