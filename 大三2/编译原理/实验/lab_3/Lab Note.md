## Lab Note

```oberon
MODULE FactorialExample;

CONST
  maxNumber = 10;

TYPE
  intArray = ARRAY maxNumber OF INTEGER;
  bool = BOOLEAN;

VAR
  n, r, factorialResult, combinationResult: INTEGER;
  factorials: intArray;

(* 计算整数 n 的阶乘 *)
PROCEDURE Factorial(n: INTEGER): INTEGER;
VAR
  result, i: INTEGER;
BEGIN
  result := 1;
  i := 1;
  WHILE i <= n DO
    result := result * i;
    i := i + 1
  END;
  RETURN result
END Factorial;

(* 计算组合数 C(n, r) = n! / (r! * (n - r)!) *)
PROCEDURE Combination(n, r: INTEGER): INTEGER;
VAR
  numerator, denominator: INTEGER;
BEGIN
  numerator := Factorial(n);
  denominator := Factorial(r) * Factorial(n - r);
  RETURN numerator DIV denominator
END Combination;

(* 打印数组内容 *)
PROCEDURE PrintArray(arr: intArray; size: INTEGER);
VAR
  i: INTEGER;
BEGIN
  FOR i := 0 TO size - 1 DO
    Write(arr[i]);
    Write(" ")
  END;
  WriteLn
END PrintArray;

BEGIN
  (* 输入 n 和 r *)
  Write("Enter value for n: "); Read(n);
  Write("Enter value for r: "); Read(r);


  (* 计算并输出组合数 C(n, r) *)
  combinationResult := Combination(n, r);
  WriteString("Combination C(");
  Write(n);
  Write(", ");
  Write(r);
  Write(") is ");
  Write(combinationResult);
  WriteLn;

  (* 计算并存储从 0 到 maxNumber-1 的阶乘 *)
  FOR n := 0 TO maxNumber - 1 DO
    factorials[n] := Factorial(n)
  END;

  (* 输出阶乘数组 *)
  Write("Factorials from 0 to ");
  Write(maxNumber - 1);
  Write(": ");
  PrintArray(factorials, maxNumber)
END FactorialExample.

```





``test001.bat``:java -classpath .;..\lib\complexity.jar;..\lib\exceptions.jar;..\lib\java-cup-11b.jar;..\lib\jflex-full-1.8.2.jar LexicalMain ..\testcases\gcd.001

``build.bat``:javac -d ..\bin -classpath .;..\lib\complexity.jar;..\lib\exceptions.jar;..\lib\java-cup-11b.jar;..\lib\jflex-full-1.8.2.jar *.java

``run.bat``:java -classpath .;..\lib\complexity.jar;..\lib\exceptions.jar;..\lib\java-cup-11b.jar;..\lib\jflex-full-1.8.2.jar LexicalAnalysis ..\src\factorial.obr









