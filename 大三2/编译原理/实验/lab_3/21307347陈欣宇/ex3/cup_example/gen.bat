@echo off
cd src
java -jar ../lib/java-cup-11b.jar -interface -parser Parser -symbols sym calc.cup
cd ..
pause