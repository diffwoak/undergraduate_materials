package exceptions;

public class IllegalSymbolException extends LexicalException {
    public IllegalSymbolException() {
        this("Unknown character.");
    }

    public IllegalSymbolException(String var1) {
        super(var1);
    }
}