package exceptions;

public class MissingOperandException extends SyntacticException {
    public MissingOperandException() {
        this("An operand is expected.");
    }

    public MissingOperandException(String var1) {
        super(var1);
    }
}
