@echo off
cd bin
java -cp .;..\lib\flowchart.jar;..\lib\jgraph.jar;..\lib\callgraph.jar;..\lib\java-cup-11b-runtime.jar;..\lib\jflex-full-1.8.2.jar. Main ..\testcases\factorial.obr
cd ..
pause