package exceptions;

public class LexicalException extends ExpressionException {
    public LexicalException() {
        this("Lexical error.");
    }

    public LexicalException(String var1) {
        super(var1);
    }
}
