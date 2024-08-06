@echo off
@echo Running Testcase 010: MissingOperandException
@echo ==============================================
cd bin
java -cp .;..\lib\jgraph.jar;..\lib\callgraph.jar;..\lib\java-cup-11b-runtime.jar;..\lib\jflex-full-1.8.2.jar. Main ..\testcases\factorial.010
cd ..
@echo ==============================================
pause
@echo on