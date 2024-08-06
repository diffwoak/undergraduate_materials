@echo off
cd src
java -jar ..\lib\jflex-full-1.8.2.jar oberon.flex
java -jar ..\lib\java-cup-11b.jar -interface -parser Parser -symbols Symbol oberon.cup
cd ..
pause