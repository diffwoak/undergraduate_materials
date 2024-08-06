package exceptions;

public class TypeMismatchedException extends SemanticException {
    public TypeMismatchedException() {
        this("Type mismatched.");
    }

    public TypeMismatchedException(String var1) {
        super(var1);
    }
}