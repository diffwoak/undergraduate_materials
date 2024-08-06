package exceptions;

public class SemanticException extends ExpressionException {
    public SemanticException() {
        this("Semantic error.");
    }

    public SemanticException(String var1) {
        super(var1);
    }
}