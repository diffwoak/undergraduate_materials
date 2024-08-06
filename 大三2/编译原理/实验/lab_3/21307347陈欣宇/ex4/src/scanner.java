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
		OberonParser oberonParser = new OberonParser(myScanner);
		try {
			oberonParser.parse();
		} catch (Exception e) {
			System.out.println("Error happen at line " + oberonParser.getLine() + ", column " + oberonParser.getColumn() + ".");
			e.printStackTrace();
		}
	}
}


/**
 * 自定义扫描类
 */
public class scanner {
	/** 词法扫描器*/
	private OberonScanner Oscanner;

	/**
	 * 构造函数
	 * @param _scanner JFelx生成的词法分析器
	 * @throws Exception Exception
	 */
	public scanner(OberonScanner _scanner){
		Oscanner = _scanner;
	}
	/**
	 * 返回下一个token
	 * @return Symbol
	 */
	public Symbol next_token() throws Exception  {
		while (!Oscanner.yyatEOF()) {
			try {
				String lex = Oscanner.yylex();
				int line = Oscanner.getLine();
				int column = Oscanner.getColumn();
				String value = Oscanner.yytext();
				switch (lex) {
					case "Integer":
						return new Symbol("NUMBER", value, line, column);
					case "Comment":
						break;
					case "Identifier":
						return new Symbol("IDENTIFIER", value, line, column);
					default:
						return new Symbol(value.toUpperCase(), value, line, column);
				}
			}catch (LexicalException e) {
				System.out.print("########## Error happen ##########\n");
				System.out.print(Oscanner.yytext() + " : " + e + "\n");
			}
		}
		return new Symbol("_","EOF", Oscanner.getLine(), Oscanner.getColumn());
	}
};