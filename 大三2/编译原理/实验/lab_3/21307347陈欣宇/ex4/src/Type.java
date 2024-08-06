import java.util.*;
/**
 * Type类记录非终结符的信息
 * 主要用于ARRAY和RECORD类的处理
 */
public class Type
{
    /** Type类型,用于判断ARRAY或RECORD*/
    public String token;
    /** ARRAY或RECORD名称 */
    public String name;
    /** 记录ARRAY内元素的类型*/
    public Type arrayType;
    /** 记录RECORD内元素的类型*/
    public Vector<Type> recordTypes;

    public Type(String token, String name, Type arrayType, Vector<Type> recordTypes)
    {
        this.token = token;
        this.name = name;
        this.arrayType = arrayType;
        this.recordTypes = recordTypes;
    }
    public Type()
    {
        this.token = null;
        this.name = null;
        this.arrayType = null;
        this.recordTypes = null;
    }

    public Type(String token)
    {
        this.token = token;
        this.name = null;
        this.arrayType = null;
        this.recordTypes = null;
    }
    public Type(String token, String name, Vector<Type> recordTypes)
    {
        this.token = token;
        this.name = name;
        this.arrayType = null;
        this.recordTypes = recordTypes;
    }
    public Type(Type t)
    {
        this.token = t.token;
        this.name = t.name;
        this.recordTypes = t.recordTypes;
        this.arrayType = t.arrayType;
    }
}