
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

import java.awt.*;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.lang.reflect.Method;

/**
 * 类 {@code PersonalTaxCalcTest} 类PersonalTaxCalc的测试类
 *
 * <p> 主要功能:
 * <p> 1. 测试计算个人所得税的功能
 * <p> 2. 测试调整个人所得税起征点的功能
 * <p> 3. 测试调整个人所得税各级税率的功能
 * <p> 4. 测试自定义输入方法的功能
 *
 * @author chen xinyu
 * @since 2024/03/09
 */
public class PersonalTaxCalcTest {

    /**
     * 计算器重置
     *
     * <p> 在执行每个测试函数前调用，用于重置个人所得税计算器，防止因为函数调用顺序不同导致测试失败
     */
    @Before
    public void before(){
        PersonalTaxCalc.threshold = 5000; // 重置起征点
        PersonalTaxCalc.taxRates = new double[]{0.03, 0.1, 0.2, 0.25, 0.3, 0.35, 0.45}; // 重置税率
    }

    /**
     * 测试计算个人所得税的功能
     *
     * <p> 输入多个不同测试用例，包括：
     * <p>1. 输入无效字符串，期望返回无效并提醒重新输入
     * <p>2. 输入负数，期望返回无效并提醒重新输入
     * <p>3. 输入起征点以下的值，期望返回与正常计算相同的值
     * <p>4. 输入输入各级税率之间的正常值，期望返回与正常计算相同的值
     * <p>5. 输入输入超出最高税率级别的值，期望返回与正常计算相同的值
     */
    @Test
    public void testCalculateTax() throws IOException {
        Path filePath = Paths.get("test/resources/testcases_calculateTax.txt"); // 测试用例文件路径
        List<String> testCases = Files.readAllLines(filePath, Charset.defaultCharset());
        for (String testCase : testCases) {
            String[] testData = testCase.split(","); // 测试数据以逗号分隔
            Scanner scanner = readIntoScanner(testData,1);
            double expectedTaxes = Double.parseDouble(testData[testData.length-1]);
            double actualTax = PersonalTaxCalc.calculateTax(scanner);
            assertEquals(expectedTaxes, actualTax, 1e-6); // 断言实际起征点与期望值相等
        }
    }

    /**
     * 测试调整个人所得税起征点的功能
     *
     * <p> 输入多个不同测试用例，包括：
     * <p>1. 输入无效字符串，期望返回无效并提醒重新输入
     * <p>2. 输入负数，期望返回无效并提醒重新输入
     * <p>3. 输入正常的起征点值，期望起征点被正常调整
     */
    @Test
    public void testAdjustThreshold() throws IOException {
        Path filePath = Paths.get("test/resources/testcases_adjustThreshold.txt"); // 测试用例文件路径
        List<String> testCases = Files.readAllLines(filePath, Charset.defaultCharset());
        for (String testCase : testCases) {
            String[] testData = testCase.split(","); // 测试数据以逗号分隔
            Scanner scanner = readIntoScanner(testData,1);
            double expectedThreshold = Double.parseDouble(testData[testData.length-1]);
            PersonalTaxCalc.adjustThreshold(scanner);
            assertEquals(expectedThreshold, PersonalTaxCalc.threshold, 1e-6); // 断言实际起征点与期望值相等
        }
    }

    /**
     * 测试调整个人所得税各级税率的功能
     *
     * <p> 输入多个不同测试用例，包括：
     * <p>1. 输入无效字符串，期望返回无效并提醒重新输入
     * <p>2. 输入负数，期望返回无效并提醒重新输入
     * <p>3. 输入非整数的税率级数，期望返回无效并提醒重新输入
     * <p>4. 输入正常的税率级数和待调整税率值，期望对应级数的税率被正常调整
     */
    @Test
    public void testAdjustTaxRates() throws IOException {
        Path filePath = Paths.get("test/resources/testcases_adjustTaxRates.txt"); // 测试用例文件路径
        List<String> testCases = Files.readAllLines(filePath, Charset.defaultCharset());
        for (String testCase : testCases) {
            String[] testData = testCase.split(","); // 测试数据以逗号分隔
            Scanner scanner = readIntoScanner(testData,2);
            double expectedRate = Double.parseDouble(testData[testData.length-1]);
            int expectedId = Integer.parseInt(testData[testData.length-2]);
            PersonalTaxCalc.adjustTaxRates(scanner);
            assertEquals(expectedRate, PersonalTaxCalc.taxRates[expectedId-1], 1e-6); // 断言实际起征点与期望值相等
        }

    }

    /**
     * 测试自定义输入方法的功能
     *
     * <p> 输入多个不同测试用例，包括：
     * <p>1. 输入无效字符串，期望返回无效并提醒重新输入
     * <p>2. 输入负数，期望返回无效并提醒重新输入
     * <p>3. 对于isRate=true的情况，输入非整数和超出范围的数，期望返回无效并提醒重新输入
     * <p>4. 输入正常的数值，期望返回与输入值相同的结果
     *
     * <p>使用反射机制测试私有方法
     */
    @Test
    public void testgetInputValue() throws Exception{
        Method getInputValue = PersonalTaxCalc.class.getDeclaredMethod("getInputValue",
                Scanner.class, String.class);
        Method getInputValueRate = PersonalTaxCalc.class.getDeclaredMethod("getInputValue",
                Scanner.class, String.class, boolean.class);
        getInputValue.setAccessible(true); // 设置为可访问
        getInputValueRate.setAccessible(true);

        String[] inputs = {"-10\n5000\n","5000\n","ab\n5000\n"}; // 模拟多个用户输入
        double[] expectedValue = {5000.0,5000.0,5000.0}; // 期望的税额
        for (int i = 0; i < inputs.length; i++) {
            System.setIn(new ByteArrayInputStream(inputs[i].getBytes()));
            Scanner scanner = new Scanner(System.in);
            double actualValue =  (double)getInputValue.invoke(null,scanner,"输入文本:"); // 调用私有方法
//            double actualValue = PersonalTaxCalc.getInputValue(scanner, "输入文本:");
            assertEquals(expectedValue[i], actualValue,1e-6); // 断言实际输入值与期望值相等
        }
        String[] inputsRate = {"-10\n3\n","3\n","11\n6\n"}; // 模拟多个用户输入
        double[] expectedValueRate = {3.0,3.0,6.0}; // 期望的税额
        for (int i = 0; i < inputsRate.length; i++) {
            System.setIn(new ByteArrayInputStream(inputsRate[i].getBytes()));
            Scanner scanner = new Scanner(System.in);
            double actualValue =  (double)getInputValueRate.invoke(null,scanner,"输入文本:",true); // 调用私有方法
//            double actualValue = PersonalTaxCalc.getInputValue(scanner, "输入文本:", true);
            assertEquals(expectedValueRate[i], actualValue,1e-6); // 断言实际输入值与期望值相等
        }
    }

    /**
     * 读取字符串列转为scanner
     *
     * <p>读取转化为用于scanner的字符串
     * @param testData 读取到文件的字符串数组
     * @param expectedNumber 用于测试结果是否相同占用的字符串个数
     * @return 返回模拟用户输入的Scanner对象
     */
    public Scanner readIntoScanner(String[] testData,int expectedNumber){
        String inputValue = testData[0] + "\n";
        for (int i = 1; i < testData.length - expectedNumber; i++) {
            inputValue = inputValue + testData[i] + "\n";
        }
        System.setIn(new ByteArrayInputStream((inputValue).getBytes()));
        Scanner scanner = new Scanner(System.in);
        return scanner;
    }
}