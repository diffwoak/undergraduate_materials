import exceptions.*;
import java.util.*;
import flowchart.*;


/**
 * Parser类
 * 进行语法分析和语义分析。
 * 画出流程图
 */
public class OberonParser
{
    /**
     * 存储抽象statement的栈,用于存储IfStatement/WhileStatement等不同类型statement
     */
    private final Stack<StatementSequence> signStack;
    /** 存储函数声明*/
    public Vector<Type> procedures;
    /** 存储变量类型*/
    public Vector<Type> typeList;
    /** 自定义的scanner类*/
    public scanner scanner;
    /** 当前读取符号*/
    public Symbol lookahead;
    /**
     * 用于绘制流程图的基础module
     */
    flowchart.Module myModule;
    /**
     * 用于绘制流程图,表示当前所处procedure
     */
    Procedure proc;
    /**
     * 用于绘制流程图,表示当前所处 WHILE statement
     */
    WhileStatement whileStmt;
    /**
     * 用于绘制流程图,表示当前所处IF statement
     */
    IfStatement ifStmt;
    /**
     * 构造函数
     * @param scanner 接受自定义词法分析器
     *
     */
    public OberonParser(scanner scanner) throws Exception {
        signStack = new Stack<>();
        typeList = new Vector<>();
        procedures = new Vector<>();
        proc = null;
        whileStmt = null;
        ifStmt = null;
        this.scanner = scanner;
        lookahead = next_token();
    }
    /**
     * 获取当前lookahead的行号
     * @return 行号
     */
    public int getLine() {
        return lookahead.getLine();
    }

    /**
     * 获取当前lookahead的列号
     * @return 列号
     */
    public int getColumn() {
        return lookahead.getColumn();
    }
    /**
     * 判断lookahead是不是token
     * @param token 待判断的字符串
     * @return 是否匹配成功
     */
    private boolean is(String token) {
        return lookahead.token.equals(token);
    }
    /**
     * 获取下一个token。
     * @return Symbol Scanner词法分析后获得的Token
     */
    public Symbol next_token() throws Exception {
        return scanner.next_token();
    }
    /**
     * 进行语法分析和语义分析，并返回分析后的结果。
     */
    public void parse()throws Exception {
        /* 进行module板块的语法分析 */
        String name1, name2;
        if(!is("MODULE"))
            throw new SyntacticException();
        lookahead = next_token();
        name1 = lookahead.value;
        myModule = new flowchart.Module(name1);
        lookahead = next_token();
        if(!is(";"))
            throw new SyntacticException();
        lookahead = next_token();
        declaration();
        if (is("BEGIN")){
            beginStatementSequence();
        }
        if(!is("END"))
            throw new SyntacticException();
        lookahead = next_token();
        name2 = lookahead.value;
        if (!name1.equals(name2))
            throw new SyntacticException();
        lookahead = next_token();
        if(!is("."))
            throw new SyntacticException();
        System.out.println("Parse Finished ...\nshow flowchart ...");
        myModule.show();
    }


    /**
     * 处理 module begin。
     */
    public void beginStatementSequence() throws Exception {
        if(!is("BEGIN"))
            throw new SyntacticException();
        proc = myModule.add("Main");
        statement();
    }

    /**
     * 处理 declaration。
     */
    public void declaration() throws Exception {
        if(!is("CONST") && !is("TYPE") && !is("VAR") &&
                !is("PROCEDURE") && !is("BEGIN") && !is("END"))
            throw new SyntacticException();
        if (is("CONST"))
            constDeclare();
        else if (is("TYPE"))
            typeDeclare();
        else if (is("VAR"))
            varDeclare();
        else if (is("PROCEDURE"))
            procedureDeclare();
        if (!is("BEGIN") && !is("END")) declaration();
    }

    /**
     * 处理 procedure declaration。
     */
    public void procedureDeclare() throws Exception {
        String name1, name2;
        name1 = procedureHeading();
        if(!is(";"))
            throw new SyntacticException();
        lookahead = next_token();
        name2 = procedureBody();
        if (!name1.equals(name2)){
            throw new SyntacticException();}
        lookahead = next_token();
        if(!is(";"))
            throw new SyntacticException();
        lookahead = next_token();
    }

    /**
     * 处理 procedure body。
     * @return 返回过程名
     */
    public String procedureBody() throws Exception {
        declaration();
        if (is("BEGIN"))
            procedureBegin();

        if(!is("END"))
            throw new SyntacticException();

        lookahead = next_token();
        return lookahead.value;
    }

    /**
     * 处理 procedure begin。
     */
    public void procedureBegin() throws Exception {
        if(!is("BEGIN"))
            throw new SyntacticException();
        statement();
    }

    /**
     * 处理 procedure heading。
     * @return 返回过程名
     */
    public String procedureHeading() throws Exception {
        Type func_p;
        String name;
        if(!is("PROCEDURE"))
            throw new SyntacticException();
        lookahead = next_token();
        name = lookahead.value;
        lookahead = next_token();
        if (is("(")) {
            func_p = formalParameters(name);
            lookahead = next_token();
        } else {
            if(!is(";"))
                throw new MissingLeftParenthesisException();
            func_p = new Type();
        }
        func_p.name = name;
        procedures.addElement(func_p);
        proc = myModule.add(name);
        return name;
    }

    /**
     * 处理 formal_parameters
     * @param name: 传入的过程名
     * @return 该过程的参数列表
     */
    public Type formalParameters(String name) throws Exception {
        Type func_p = new Type("PROCEDURE", name, new Vector<>());
        lookahead = next_token();
        if (is(")"))
            return func_p;
        fpSection(func_p);
        if(!is(")"))
            throw new SyntacticException();
        return func_p;
    }

    /**
     * 处理fp_section
     * @param func_p 调用的参数
     */
    public void fpSection(Type func_p) throws Exception{
        Type fp_t;
        Vector<String> fp_name = new Vector<>();
        if (is("VAR"))
            lookahead = next_token();
        identifierList(fp_name);
        if(!is(":"))
            throw new SyntacticException();
        fp_t =  typeKind();
        for (int i=0; i<fp_name.size(); i++)
        {
            func_p.recordTypes.addElement(new Type(fp_t));
            typeList.addElement(new Type(fp_t.token,fp_name.elementAt(i),fp_t.arrayType,fp_t.recordTypes));
        }
        lookahead = next_token();
        if (is(";"))
        {
            lookahead = next_token();
            fpSection(func_p);
        }
    }
    /**
     * 处理 const_declare。
     */
    public void constDeclare()throws Exception {
        Type tmp = new Type();
        Symbol t;
        lookahead = next_token();
        if (!is("IDENTIFIER"))
            return ;
        tmp.name = lookahead.value;
        lookahead = next_token();
        if(!is("="))
            throw new SyntacticException();
        t = expression();
        tmp.token = t.token;
        typeList.addElement(tmp);
        if(!is(";"))
            throw new SyntacticException();
        constDeclare();
    }
    /**
     * 处理 var declaration。
     */
    public void varDeclare() throws Exception {
        Type tmp;
        Vector<String> id_v = new Vector<>();
        lookahead = next_token();
        if (!is("IDENTIFIER"))
            return;
        identifierList(id_v);
        if(!is(":"))
            throw new SyntacticException();
        tmp = typeKind();
        for (int i=0; i<id_v.size(); i++) {
            typeList.addElement(new Type(tmp.token,id_v.elementAt(i),tmp.arrayType,tmp.recordTypes));
        }
        lookahead = next_token();
        if(!is(";"))
            throw new SyntacticException();
        varDeclare();
    }

    /**
     * 处理 type declaration。
     */
    public void typeDeclare()throws Exception {
        lookahead = next_token();
        if (!is("IDENTIFIER"))
            return ;
        String name = lookahead.value;
        lookahead = next_token();
        if(!is("="))
            throw new SyntacticException();
        Type tmp = typeKind();
        tmp.name = name;
        typeList.addElement(tmp);
        lookahead = next_token();
        if(!is(";"))
            throw new SyntacticException();
        typeDeclare();
    }

    /**
     * 处理 type kind。
     * @return Type类型的INTEGER/BOOLEAN/RECORD/ARRAY。
     */
    public Type typeKind() throws Exception {
        lookahead = next_token();
        if (is("INTEGER"))
            return new Type("NUMBER");
        else if (is("BOOLEAN"))
            return new Type("BOOLEAN");
        else if (is("IDENTIFIER")) {
            for (int i = 0; i< typeList.size(); i++) {
                if (lookahead.value.equals(typeList.elementAt(i).name))
                    return typeList.elementAt(i);
            }
            throw new SemanticException("Not declared");
        }
        else if (is("ARRAY"))
            return arrayType();
        else if (is("RECORD"))
            return recordType();
        else
            throw new SyntacticException();
    }

    /**
     * 处理 array type。
     * @return arr 处理好的array类型
     */
    public Type arrayType() throws Exception {
        if(!is("ARRAY"))
            throw new SyntacticException();
        Type arr = new Type("ARRAY");
        Symbol tmp = expression();
        if (!Objects.equals(tmp.token, "NUMBER"))
            throw new TypeMismatchedException();
        if(!is("OF"))
            throw new SyntacticException();
        arr.arrayType = new Type(typeKind());
        return arr;
    }
    /**
     * 处理 record type。
     * @return 处理好的record类型
     */
    public Type recordType() throws Exception {
        if(!is("RECORD"))
            throw new SyntacticException();
        Type tmp = new Type("RECORD");
        tmp.recordTypes = new Vector<>();
        lookahead = next_token();
        if (is("END"))
            return tmp;
        else
            return fieldList(tmp);
    }

    /**
     * 处理 filed list。
     * @param filed: record类型的Type
     * @return filed经过处理后的返回值
     */
    public Type fieldList(Type filed) throws Exception {
        Vector<String> tmp = new Vector<>();
        identifierList(tmp);
        if (is("END"))
            return filed;
        else if (is(";")) {
            lookahead = next_token();
            return fieldList(filed);

        }
        if (!tmp.isEmpty()) {
            if(!is(":"))
                throw new SyntacticException();
            Type t = typeKind();
            for (int i=0; i<tmp.size(); i++) {
                filed.recordTypes.addElement(new Type(t.token,tmp.elementAt(i),t.arrayType,t.recordTypes));
            }
        }
        lookahead = next_token();
        return fieldList(filed);
    }
    /**
     * 处理 identifier list。
     * @param id_v:存储identifier名的列表
     */
    public void identifierList(Vector<String> id_v) throws Exception {
        if (!is("IDENTIFIER"))
            return;
        id_v.addElement(lookahead.value);
        lookahead = next_token();
        if (is(",")) {
            lookahead = next_token();
            identifierList(id_v);
        }
    }
    public void check_procedure_call(Type type_of_ap)throws Exception{
        int name_match = 0;
        for (int j = 0; j< procedures.size(); j++) {
            if (type_of_ap.name.equals(procedures.elementAt(j).name)) {
                if(type_of_ap.recordTypes.size()!= procedures.elementAt(j).recordTypes.size())
                    throw new ParameterMismatchedException();
                for (int k = 0; k<type_of_ap.recordTypes.size(); k++)
                    if (!Objects.equals(type_of_ap.recordTypes.elementAt(k).token, procedures.elementAt(j).recordTypes.elementAt(k).token))
                        throw new TypeMismatchedException();
                name_match = 1;
            }
        }
        if (name_match == 0)
            throw new SemanticException("Not declared");
    }
    /**
     * 处理 statement
     */
    public void statement() throws Exception {
        Type type_of_ap;
        String name;
        String ap = "";
        if (is("END"))
            return;
        lookahead = next_token();
        if (is("WHILE"))
            whileStatement();
        else if (is("IF"))
            ifStatement();
        else if (is("READ") ||
                is("WRITE") || is("WRITELN"))
            rwStatement(lookahead.token);
        else if (is("IDENTIFIER")) {
            name = lookahead.value;
            lookahead = next_token();
            if (is("(") || is(";")) {
                type_of_ap = new Type("PROCEDURE", name, new Vector<>());
                if (is("(")) {
                    ap = actualParameter(type_of_ap);
                    lookahead = next_token();
                }

                check_procedure_call(type_of_ap);

                String t = name + "( " + ap + " )";
                addStatement(new PrimitiveStatement(t));
            }
            else assign(name);
        }
        if (is("ELSE") || is("ELSIF"))
            return ;
        if(is(";"))
            statement();
    }


    /**
     * 处理 assign
     * @param  name 赋值的变量名
     */
    public void assign(String name) throws Exception {
        Symbol l, r;
        Type id_t;
        l = new Symbol("_", "");
        for (int i = 0; i< typeList.size(); i++) {
            if (typeList.elementAt(i).name.equals(name)) {
                id_t = typeList.elementAt(i);
                selector(id_t, l);
                break;
            }
        }
        if (l.token.equals("_"))
            throw new SemanticException("Not declared");
        else {
            if (!is(":="))
                throw new SyntacticException();
            r = expression();
            if (!l.token.equals(r.token))
                throw new TypeMismatchedException();
            addStatement(new PrimitiveStatement(name + " := " + r.value));
        }
    }

    /**
     * 处理 actual parameter
     * @param ap_type 参数类型
     * @return parameter_information 处理后参数的信息
     */
    public String actualParameter(Type ap_type) throws Exception{
        Symbol expr = expression();
        ap_type.recordTypes.addElement(new Type(expr.token));
        if (is(")"))
            return expr.value;
        else if (is(",")) {
            expr.value += ", ";
            expr.value += actualParameter(ap_type);
        }
        else throw new SyntacticException();
        return expr.value;
    }


    /**
     * 处理 write/read/writeln statement
     * @param token: rw的类型,例如WRITELN、READ、WRITE
     */
    public void rwStatement(String token) throws Exception {
        String t;
        switch (token) {
            case "WRITELN" -> {
                lookahead = next_token();
                if (is("(")) {
                    Symbol expr = expression();
                    if (!is(")"))
                        throw new MissingRightParenthesisException();
                    lookahead = next_token();
                    t = "Write(" + expr.value + ")";
                } else
                    t = "Writeln";
            }
            case "READ" -> {
                lookahead = next_token();
                if (!is("("))
                    throw new MissingLeftParenthesisException();
                Symbol expr = expression();
                if (!is(")"))
                    throw new MissingRightParenthesisException();
                t = "Read( " + expr.value + " )";
                lookahead = next_token();
            }
            case "WRITE" -> {
                lookahead = next_token();
                if (!is("("))
                    throw new MissingLeftParenthesisException();
                Symbol expr = expression();
                if (!is(")"))
                    throw new MissingRightParenthesisException();
                t = "Write( " + expr.value + " )";
                lookahead = next_token();
            }
            default -> throw new SyntacticException();
        }
        // 流程图
        addStatement(new PrimitiveStatement(t));
    }

    /**
     * 处理 while statement
     */
    public void whileStatement() throws Exception {
        Symbol expr;
        if (!is("WHILE"))
            throw new SyntacticException();
        expr = expression();
        if (!is("DO"))
            throw new SyntacticException();
        whileStmt = new WhileStatement(expr.value);  // 流程图
        addStatement(whileStmt);
        signStack.push(whileStmt.getLoopBody());
        statement();
        signStack.pop();
        if (!is("END"))
            throw new SyntacticException();
        lookahead = next_token();
    }

    /**
     * 处理 if statement
     */
    public void ifStatement() throws Exception {
        if (!is("IF"))
            throw new MissingLeftParenthesisException();
        Symbol expr = expression();
        if (!expr.token.equals("BOOLEAN"))
            throw new TypeMismatchedException();
        if (!is("THEN"))
            throw new MissingLeftParenthesisException();
        ifStmt = new IfStatement(expr.value);   //流程图
        addStatement(ifStmt);
        signStack.push(ifStmt.getFalseBody());
        signStack.push(ifStmt.getTrueBody());
        statement();
        signStack.pop();
        if (is("ELSIF"))
            elsifStatement();
        else if (is("ELSE"))
            elseStatement();
        if (is("END"))
            lookahead = next_token();
        signStack.pop();
    }


    /**
     * 处理 elsif statement。
     */
    public void elsifStatement() throws Exception {
        if (!is("ELSIF"))
            throw new MissingLeftParenthesisException();
        Symbol expr = expression();
        if (!expr.token.equals("BOOLEAN"))
            throw new TypeMismatchedException();
        IfStatement elsif = new IfStatement(expr.value);
        addStatement(elsif);
        signStack.pop();
        signStack.push(elsif.getFalseBody());
        signStack.push(elsif.getTrueBody());
        if (!is("THEN"))
            throw new MissingLeftParenthesisException();
        statement();
        signStack.pop();
        if (!is("ELSIF"))
            elsifStatement();
    }

    /**
     * 处理 else statement。
     */
    public void elseStatement() throws Exception {
        if (!is("ELSE"))
            throw new MissingLeftParenthesisException();
        statement();
        if (!is("END"))
            throw new MissingLeftParenthesisException();
    }

    /**
     * 处理 expression
     * @return Symbol
     */
    public Symbol expression() throws Exception {
        Symbol expr = simpleExpression();
        if (is("=") || is("#") || is("<") ||
                is("<=") || is(">") || is(">=") ) {
            if (!expr.token.equals("NUMBER"))
                throw new TypeMismatchedException();
            if (is("="))	expr.value += " = ";
            if (is("#")) expr.value += " # ";
            if (is("<"))	expr.value += " &lt ";
            if (is("<="))	expr.value += " &lt = ";
            if (is(">"))	expr.value += " &gt ";
            if (is(">="))	expr.value += " &gt = ";
            Symbol expr1 = simpleExpression();
            if (!expr1.token.equals("NUMBER"))
                throw new TypeMismatchedException();
            expr.value = expr.value + expr1.value;
            expr.token = "BOOLEAN";
        }
        return expr;
    }

    /**
     * 处理 simple expression
     * @return Symbol
     */
    public Symbol simpleExpression() throws Exception {
        Symbol expr = term();
        if (is("+") || is("-") || is("OR")) {
            if (!expr.token.equals("BOOLEAN") && is("OR"))
                throw new TypeMismatchedException();
            if (!expr.token.equals("NUMBER") && (is("+") || is("-")))
                throw new TypeMismatchedException();
            if (is("+"))	expr.value += "+";
            if (is("-"))	expr.value += "-";
            if (is("OR")) {
                expr.token = "BOOLEAN";
                expr.value += "OR";
            }
            Symbol expr1 = simpleExpression();
            expr.value += expr1.value;
        }
        return expr;
    }

    /**
     * 处理 term
     * @return Symbol
     */
    public Symbol term() throws Exception {
        Symbol expr = factor();
        if (is("*") || is("DIV") ||is("MOD") || is("&")) {
            if (!expr.token.equals("BOOLEAN") && is("&"))
                throw new TypeMismatchedException();
            if (!expr.token.equals("NUMBER") && (is("*") || is("DIV") || is("MOD")))
                throw new TypeMismatchedException();
            if (is("*"))    expr.value += " * ";
            if (is("DIV"))  expr.value += " DIV ";
            if (is("MOD"))  expr.value += " MOD ";
            if (is("&")) {
                expr.value += " & ";
                expr.token = "BOOLEAN";
            }
            Symbol term1 = term();
            if (!expr.token.equals(term1.token))
                throw new TypeMismatchedException();
            expr.value += term1.value;
        } else if(is("IDENTIFIER") || is("NUMBER") || is("BOOLEAN"))
            throw new MissingOperatorException();
        return expr;
    }


    /**
     * 处理 factor
     * @return Symbol
     */
    public Symbol factor() throws Exception {
        // 初始化Symbol
        Symbol sym = new Symbol("_","");
        lookahead = next_token();
        int neg = 0;
        if (is("+") || is("-")){  //termHead
            if(is("-")) {
                neg=1;
                sym = new Symbol("_", "-");
            }
            lookahead = next_token();
        }
        if (is("NUMBER")) {
            if(neg == 0)
                sym = new Symbol("NUMBER", lookahead.value);
            else sym = new Symbol("NUMBER", "-" + lookahead.value);
            lookahead = next_token();
        } else if (is("~")) {
            if(neg == 1) throw new TypeMismatchedException();
            sym = factor();
            lookahead = next_token();
        } else if (is("IDENTIFIER")) {
            int i;
            for (i=0; i< typeList.size(); i++) {
                if (lookahead.value.equals(typeList.elementAt(i).name)) {
                    selector(typeList.elementAt(i), sym);
                    break;
                }
            }
            if (i >= typeList.size())
                throw new SemanticException("Not declared");
            if (sym.token.equals("_")) {
                sym = new Symbol(typeList.elementAt(i).token, typeList.elementAt(i).name);
                lookahead = next_token();
            }
        } else if (is("(")) {
            sym = expression();
            if (!is(")"))
                throw new MissingRightParenthesisException();
            if(neg == 0)
                sym.value = "( " + sym.value + " )";
            else sym.value = "-( " + sym.value + " )";
            lookahead = next_token();
        } else if (is("+") || is("-") || is("*") || is("DIV") || is("&") || is("OR")
                || is("<") || is("<=") ||is(">") || is(">=") || is("#") || is("="))
            throw new MissingOperandException();
        if (sym.token.equals("_")) throw new MissingOperandException();
        return sym;
    }
    /**
     * 处理 selector
     * @param t record或者array类型的Type
     * @param sym record或array类型的Symbol
     */
    public void selector(Type t, Symbol sym) throws Exception {
        if (t.name != null)
            sym.value = sym.value + t.name;
        if (is("IDENTIFIER"))
            lookahead = next_token();
        if (is(".")) {
            if (!t.token.equals("RECORD"))
                throw new  TypeMismatchedException();
            int match = 0;
            lookahead = next_token();
            for (int i = 0; i<t.recordTypes.size(); i++) {
                t = t.recordTypes.elementAt(i);
                if (t.name.equals(lookahead.value)) {
                    match = 1;
                    sym.value = sym.value + ("."+lookahead.value);
                    selector(t, sym);
                }
            }
            if (match == 0)
                throw new SemanticException("Not declared");
        } else if (is("[")) {
            if (!t.token.equals("ARRAY"))
                throw new TypeMismatchedException();
            Symbol expr = expression();
            if (!expr.token.equals("NUMBER"))
                throw new TypeMismatchedException();
            sym.value = sym.value + ("[" + expr.value + "]");
            t = t.arrayType;
            selector(t, sym);
            return ;
        } else
            sym.token = t.token;
        if (is("]"))
            lookahead = next_token();
    }


    /**
     * 添加statement到流程图中。
     * @param st: AbstractStatement
     */
    public void addStatement(AbstractStatement st) {
        StatementSequence stmt;
        if (signStack.isEmpty())
            proc.add(st);
        else {
            stmt = signStack.peek();
            stmt.add(st);
        }
    }
}