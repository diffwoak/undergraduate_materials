package exceptions;

public class IllegalOctalException extends LexicalException {
    public IllegalOctalException() {
        this("Illegal Octal number.");
    }

    public IllegalOctalException(String s) {
        super(s);
    }
}
