import java.io.*;
import exceptions.*;

/**
 * 词法分析器
 */
public class LexicalAnalysis {
    /**
     * 词法分析器的入口
     * @param args 读入oberon-0程序
     * @throws Exception 抛出异常
     */
    public static void main(String[] args) throws Exception {
        OberonScanner scanner = new OberonScanner(new java.io.FileReader(args[0]));
        int flag = 0;
        while (!scanner.yyatEOF()) {
            try {
                String lex = scanner.yylex();
                if(lex != null){
                    String text = String.format("%-25s",scanner.yytext());
                    System.out.println(text + lex);
                }
            } catch (LexicalException e) {
                System.out.print("########## Error happen ##########\n");
                System.out.print(scanner.yytext() + " : " + e + "\n");
                flag += 1;
            }
        }
        System.out.println("Lexical analysis done. With " + flag +" lexical error");
    }
}
