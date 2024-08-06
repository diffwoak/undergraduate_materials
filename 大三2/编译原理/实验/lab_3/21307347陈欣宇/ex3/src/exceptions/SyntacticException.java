package exceptions;

public class SyntacticException extends ExpressionException {
    public SyntacticException() {
        this("Syntactic error.");
    }

    public SyntacticException(String var1) {
        super(var1);
    }
}
