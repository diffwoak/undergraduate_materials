@echo off
cd src
javadoc -private -author -version -d ..\doc -classpath .;..\lib\callgraph.jar;..\lib\jgraph.jar;..\lib\java-cup-11b.jar;..\lib\jflex-full-1.8.2.jar *.java
@REM javadoc -private -author -version -d ..\doc -classpath .;..\lib\callgraph.jar;..\lib\jgraph.jar;..\lib\java-cup-11b.jar;..\lib\jflex-full-1.8.2.jar exceptions\*.java
cd ..
pause
@echo on