/**
 * token类
 */
public class Symbol {
  /** token的类别 */
  public String token;
  /** token值 */
  public String value;
  /** 所在行号 */
  private final int line;
  /** 所在列号 */
  private final int column;

  /**
   * 构造函数
   * @param token token类别
   * @param line 行号
   * @param column 列号
   */
  public Symbol(String token, int line, int column) {
    this.token = token;
    this.line = line;
    this.column = column;
  }

  /**
   * 数值和变量构造函数
   * @param token token类别
   * @param line 行号
   * @param column 列号
   * @param value token内容
   */
  public Symbol(String token, String value, int line, int column) {
    this.token = token;
    this.value = value;
    this.line = line;
    this.column = column;
  }
  public Symbol(String token, String value) {
    this.token = token;
    this.line = 0;
    this.column = 0;
    this.value = value;
  }
  /**
   * 获取token行号
   * @return 行号
   */
  public int getLine() {
    return line;
  }

  /**
   * 获取token列号
   * @return 列号
   */
  public int getColumn() {
    return column;
  }
}