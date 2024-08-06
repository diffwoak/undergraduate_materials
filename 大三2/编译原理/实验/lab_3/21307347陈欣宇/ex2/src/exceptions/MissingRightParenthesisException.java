package exceptions;

public class MissingRightParenthesisException extends SyntacticException {
    public MissingRightParenthesisException() {
        this("Right parenthesis ')' is expected.");
    }

    public MissingRightParenthesisException(String var1) {
        super(var1);
    }
}
