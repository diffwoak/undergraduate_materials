@echo off
cd src
javac -d ..\bin -classpath .;..\lib\jflex-full-1.8.2.jar *.java
javac -d ..\bin -classpath .;..\lib\jflex-full-1.8.2.jar exceptions\*.java
pause
@echo on