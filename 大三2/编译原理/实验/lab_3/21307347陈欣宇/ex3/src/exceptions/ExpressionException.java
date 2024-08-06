package exceptions;

public class ExpressionException extends Exception {
    public ExpressionException() {
        this("Error found in the expression.");
    }

    public ExpressionException(String var1) {
        super(var1);
    }
}