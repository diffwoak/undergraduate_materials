package exceptions;

public class MissingLeftParenthesisException extends SyntacticException {
    public MissingLeftParenthesisException() {
        this("Left parenthesis '(' is expected.");
    }

    public MissingLeftParenthesisException(String var1) {
        super(var1);
    }
}