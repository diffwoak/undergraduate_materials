package exceptions;

public class IllegalIntegerRangeException extends LexicalException{
    public IllegalIntegerRangeException() {
        this("Illegal IntegerRange: more than 12.");
    }

    public IllegalIntegerRangeException(String s) {
        super(s);
    }
}
