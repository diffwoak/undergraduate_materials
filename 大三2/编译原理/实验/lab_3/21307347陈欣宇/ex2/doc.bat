@echo off
cd src
javadoc -private -author -version -d ..\doc -classpath .;..\lib\jflex-full-1.8.2.jar *.java
REM javadoc -private -author -version -d ..\doc -classpath .;..\lib\jflex-full-1.8.2.jar exceptions\*.java
cd ..
pause
@echo on