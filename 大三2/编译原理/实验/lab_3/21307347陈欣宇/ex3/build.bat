@echo off
cd src
javac -d ..\bin -cp .;..\lib\callgraph.jar;..\lib\jgraph.jar;..\lib\java-cup-11b.jar;..\lib\jflex-full-1.8.2.jar *.java
cd ..
pause