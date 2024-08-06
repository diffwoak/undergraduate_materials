import java.io.*;
import exceptions.*;

%%

%public
%class OberonScanner
%ignorecase
%unicode
%type String
%line
%column
%yylexthrow LexicalException
%eofval{
	return "_";
%eofval}
%{
	int getLine() {
		return yyline + 1;
	}
	int getColumn() {
		return yycolumn + 1;
	}
%}

ReservedWord = "module"|"begin"|"end"|"const"|"type"|"var"|"procedure"|"array"|"of"|"record"|"if"|"then"|"else"|"elsif"|"while"|"do"|"return"|"or"|"div"|"mod"
Keyword = "integer"|"boolean"|"read"|"write"|"writeln"
Operator = "="|"#"|"<"|"<="|">"|">="|"*"|"div"|"mod"|"+"|"-"|"&"|"or"|"~"|":="|":"|"("|")"|"["|"]"
Delimiter = ";"|","|"."
Comment = "(*"~"*)"
Identifier = [a-zA-Z_][a-zA-Z0-9_]*
Integer = 0[0-7]* | [1-9]+[0-9]*

IllegalOctal = 0[0-7]*[8-9]+[0-9]*
IllegalInteger 	= {Integer}+{Identifier}+
MismatchedComment= "(*" ( [^\*] | "*"+[^\)] )* | ( [^\(] | "("+[^\*] )* "*)"
WhiteSpace 	= " "|\r|\n|\r\n|\t

%%

<YYINITIAL>
{
	{ReservedWord}  {return "ReservedWord";}
	{Keyword}       {return "Keyword";}
	{Operator}      {return "Operator";}
	{Delimiter}     {return "Delimiter";}
	{Comment}       {return "Comment";}
	{Identifier}    {
                        if (yylength() > 24)
                            throw new IllegalIdentifierLengthException();
                        else
                            return "Identifier";
					}
	{Integer}		{
                        if (yylength() > 12)
                            throw new IllegalIntegerRangeException();
                        else
                            return "Integer";
					}
	{IllegalOctal}  {throw new IllegalOctalException();}
	{IllegalInteger}    {throw new IllegalIntegerException();}
	{MismatchedComment} {throw new MismatchedCommentException();}
	{WhiteSpace}        {}
     .				    {throw new IllegalSymbolException();}
}