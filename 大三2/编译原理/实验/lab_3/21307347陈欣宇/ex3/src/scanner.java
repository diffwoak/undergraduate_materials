import callgraph.CallGraph;
import java_cup.runtime.*;
import java.util.*;
import exceptions.*;

/**
 * 编译器的入口
 */
class Main {
	/**
	 * 编译器的入口
	 * @param argv 编译文件路径
	 * @throws Exception exception
	 */
	public static void main(String[] argv) throws Exception {
		OberonScanner scanner = new OberonScanner(new java.io.FileReader(argv[0]));
		scanner myScanner = new scanner(scanner);
		CallGraph graph = new CallGraph();
		Parser parser = new Parser(myScanner,graph);
		try {
			parser.parse();
			System.out.println("Parser analysis done. With 0 lexical error");
		} catch (Exception e) {
			System.out.println("Error happen at line " + parser.getLine() + ", column " + parser.getColumn() + ".");
			e.printStackTrace();
		}
	}
}


/**
 * 自定义扫描类
 */
public class scanner {
	/** lookahead token */
	protected static String lookahead_token;
	/** lookahead value */
	protected static String lookahead_value;
	/** tokens 列表*/
	protected static ArrayList <String> tokens;
	/** values 列表 */
	protected  static ArrayList <String> values;
	/** token 行 */
	protected static ArrayList <Integer> lines;
	/** token列 */
	protected static ArrayList <Integer> columns;
	/** tokens 索引 */
	protected static int index;
	/** SymbolFactory:用于生成输入语法分析器的Symbol */
	private SymbolFactory sf = new DefaultSymbolFactory();

	/**
	 * 构造函数
	 * @param scanner JFelx生成的词法分析器
	 * @throws Exception Exception
	 */
	public scanner(OberonScanner scanner) throws Exception {
		tokens = new ArrayList<>();
		values = new ArrayList<>();
		lines = new ArrayList<>();
		columns = new ArrayList<>();
		index = 0;
		lookahead_token = "";
		lookahead_value = "";
		int flag = 0;
		while (!scanner.yyatEOF()) {
			try{
				String lex = scanner.yylex();
				if(lex != null){
					switch (lex) {
						case "Integer":
							tokens.add("NUMBER");
							values.add(scanner.yytext());
							break;
						case "Comment":
							break;
						case "Identifier":
							tokens.add("IDENTIFIER");
							values.add(scanner.yytext());
							break;
						default:
							tokens.add(scanner.yytext());
							values.add(scanner.yytext());
					}
					lines.add(scanner.getLine());
					columns.add(scanner.getColumn());
				}
			}catch (LexicalException e) {
				System.out.print("########## Error happen ##########\n");
				System.out.print(scanner.yytext() + " : " + e + "\n");
				flag += 1;
			}
		}
		System.out.println("Lexical analysis done. With " + flag +" lexical error");
		tokens.add("_");
		values.add("EOF");
		lines.add(scanner.getLine());
		columns.add(scanner.getColumn());
	}

	/**
	 * 返回当前token行
	 * @return int
	 */
	public static int getLine() {
		return lines.get(index-1);
	}

	/**
	 * 返回当前token列
	 * @return int
	 */
	public static int getColumn() {
		return columns.get(index-1);
	}

	/**
	 * 初始化函数
	 */
	public static void init() {
		lookahead_token = tokens.get(index);
		lookahead_value = values.get(index);
		index++;
	}

	/**
	 * 返回下一个token
	 * @return Symbol
	 */
	public java_cup.runtime.Symbol next_token(){
		String value = lookahead_value;
		String token = lookahead_token;
		lookahead_token = tokens.get(index);
		lookahead_value = values.get(index);
		index++;
		switch (token.toUpperCase()) {
			case "MODULE" -> {
				return sf.newSymbol("MODULE", Symbol.MODULE, value);
			}
			case "BEGIN" -> {
				return sf.newSymbol("BEGIN", Symbol.BEGIN, value);
			}
			case "END" -> {
				return sf.newSymbol("END", Symbol.END, value);
			}
			case "CONST" -> {
				return sf.newSymbol("CONST", Symbol.CONST, value);
			}
			case "TYPE" -> {
				return sf.newSymbol("TYPE", Symbol.TYPE, value);
			}
			case "VAR" -> {
				return sf.newSymbol("VAR", Symbol.VAR, value);
			}
			case "PROCEDURE" -> {
				return sf.newSymbol("PROCEDURE", Symbol.PROCEDURE, value);
			}
			case "RECORD" -> {
				return sf.newSymbol("RECORD", Symbol.RECORD, value);
			}
			case "ARRAY" -> {
				return sf.newSymbol("ARRAY", Symbol.ARRAY, value);
			}
			case "OF" -> {
				return sf.newSymbol("OF", Symbol.OF, value);
			}
			case "WHILE" -> {
				return sf.newSymbol("WHILE", Symbol.WHILE, value);
			}
			case "DO" -> {
				return sf.newSymbol("DO", Symbol.DO, value);
			}
			case "IF" -> {
				return sf.newSymbol("IF", Symbol.IF, value);
			}
			case "THEN" -> {
				return sf.newSymbol("THEN", Symbol.THEN, value);
			}
			case "ELSIF" -> {
				return sf.newSymbol("ELSIF", Symbol.ELSIF, value);
			}
			case "ELSE" -> {
				return sf.newSymbol("ELSE", Symbol.ELSE, value);
			}
			case ";" -> {
				return sf.newSymbol("SEMI", Symbol.SEMI, value);
			}
			case "." -> {
				return sf.newSymbol("DOT", Symbol.DOT, value);
			}
			case "=" -> {
				return sf.newSymbol("EQUAL", Symbol.EQUAL, value);
			}
			case ":" -> {
				return sf.newSymbol("COLON", Symbol.COLON, value);
			}
			case "(" -> {
				return sf.newSymbol("LEFTPAR", Symbol.LEFTPAR, value);
			}
			case ")" -> {
				return sf.newSymbol("RIGHTPAR", Symbol.RIGHTPAR, value);
			}
			case "[" -> {
				return sf.newSymbol("LEFTMIDPAR", Symbol.LEFTMIDPAR, value);
			}
			case "]" -> {
				return sf.newSymbol("RIGHTMIDPAR", Symbol.RIGHTMIDPAR, value);
			}
			case "," -> {
				return sf.newSymbol("COMMA", Symbol.COMMA, value);
			}
			case ":=" -> {
				return sf.newSymbol("COLONEQ", Symbol.COLONEQ, value);
			}
			case "#" -> {
				return sf.newSymbol("NOTEQUAL", Symbol.NOTEQUAL, value);
			}
			case "<" -> {
				return sf.newSymbol("LESS", Symbol.LESS, value);
			}
			case "<=" -> {
				return sf.newSymbol("LEQ", Symbol.LEQ, value);
			}
			case ">" -> {
				return sf.newSymbol("GREAT", Symbol.GREAT, value);
			}
			case ">=" -> {
				return sf.newSymbol("GEQ", Symbol.GEQ, value);
			}
			case "+" -> {
				return sf.newSymbol("ADD", Symbol.ADD, value);
			}
			case "-" -> {
				return sf.newSymbol("MINUS", Symbol.MINUS, value);
			}
			case "OR" -> {
				return sf.newSymbol("OR", Symbol.OR, value);
			}
			case "*" -> {
				return sf.newSymbol("MUL", Symbol.MUL, value);
			}
			case "DIV" -> {
				return sf.newSymbol("DIV", Symbol.DIV, value);
			}
			case "MOD" -> {
				return sf.newSymbol("MOD", Symbol.MOD, value);
			}
			case "&" -> {
				return sf.newSymbol("AND", Symbol.AND, value);
			}
			case "~" -> {
				return sf.newSymbol("NOT", Symbol.NOT, value);
			}
			case "IDENTIFIER" -> {
				return sf.newSymbol("IDENTIFIER", Symbol.IDENTIFIER, value);
			}
			case "INTEGER" -> {
				return sf.newSymbol("INT", Symbol.INT, value);
			}
			case "BOOLEAN" -> {
				return sf.newSymbol("BOOL", Symbol.BOOL, value);
			}
			case "READ" -> {
				return sf.newSymbol("READ", Symbol.READ, value);
			}
			case "WRITE" -> {
				return sf.newSymbol("WRITE", Symbol.WRITE, value);
			}
			case "WRITELN" -> {
				return sf.newSymbol("WRITELN", Symbol.WRITELN, value);
			}
			case "NUMBER" -> {
				return sf.newSymbol("NUMBER", Symbol.NUMBER, value);
			}
			case "_" -> {
				return sf.newSymbol("EOF", Symbol.EOF, value);
			}
		}
		index --;
		return sf.newSymbol("EOF", Symbol.EOF, value);
	}
};