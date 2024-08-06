package exceptions;

public class MissingOperatorException extends SyntacticException {
    public MissingOperatorException() {
        this("An operator is expected.");
    }

    public MissingOperatorException(String var1) {
        super(var1);
    }
}