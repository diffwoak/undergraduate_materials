import java.util.Scanner;
import java.util.InputMismatchException;
/**
 * 类 {@code PersonalTaxCalc} 个人所得税计算器
 *
 * <p> 主要功能:
 * <p> 1. 计算个人所得税
 * <p> 2. 调整个人所得税起征点
 * <p> 3. 调整个人所得税各级税率
 * <p> 4. 查看个人所得税各级税率
 *
 * @author chen xinyu
 * @since 2024/03/09
 */
public class PersonalTaxCalc {

    /** 起征点 */
    public static double threshold = 5000; // 起征点
    /** 各级税率 */
    public static double[] taxRates = {0.03, 0.1, 0.2, 0.25, 0.3, 0.35, 0.45}; // 税率
    /** 税阶段起征点 {@value} */
    private static final int[] taxThresholds = {3000, 12000, 25000, 35000, 55000, 80000}; // 税阶段起征点

    /**
     * 主方法，程序入口。
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.println("请选择以下功能进行操作：");
            System.out.println("1. 计算个人所得税额");
            System.out.println("2. 调整个人所得税起征点");
            System.out.println("3. 调整个人所得税各级税率");
            System.out.println("4. 查看个人所得税各级税率");
            System.out.println("5. 退出");
            int tmp = (int)getInputValue(scanner,"",true);
            switch (tmp) {
                case 1: calculateTax(scanner);break;
                case 2: adjustThreshold(scanner);break;
                case 3: adjustTaxRates(scanner);break;
                case 4: showTaxRates();break;
                case 5: {
                    System.out.println("退出成功！");
                    return;
                }
                default: System.out.println("输入无效，请输入一个有效的数值!");
            }
        }
    }

    /**
     * 计算个人所得税。
     *
     * <p> 根据用户输入工资薪金总额计算个人所得税额
     *
     * @param scanner 用于接收用户输入的Scanner对象
     * @return 返回个人所得税额
     */
    public static double calculateTax(Scanner scanner) {
        double tax = 0.0;
        double income = getInputValue(scanner,"请输入本月工资薪金总额：");
        double incomeForTax = Math.max(income- threshold, 0);
        for(int i = taxThresholds.length -1; i >= 0; i--){
            if(incomeForTax > taxThresholds[i]){
                tax += (incomeForTax - taxThresholds[i]) * taxRates[i+1];
                incomeForTax = taxThresholds[i];
            }
        }
        tax += incomeForTax * taxRates[0];
        System.out.println("本月应缴纳个人所得税额为：" + tax + " 元");
        return tax;
    }

    /**
     * 调整个人所得税起征点。
     *
     * <p> 输出当前个人所得税起征点，并根据用户输入修改个人所得税起征点
     * @param scanner 用于接收用户输入的Scanner对象
     */
    public static void adjustThreshold(Scanner scanner) {
        System.out.println("当前个人所得税起征点为"+threshold+" 元");
        threshold = getInputValue(scanner,"请输入新的个人所得税起征点:");
        System.out.println("修改成功，当前个人所得税起征点已调整为"+threshold+" 元");
    }

    /**
     * 调整个人所得税各级税率。
     * <p> 根据用户输入的税率级数以及新的税率修改该级数的税率值
     * @param scanner 用于接收用户输入的Scanner对象
     */
    public static void adjustTaxRates(Scanner scanner) {
        double rate = getInputValue(scanner,"请输入调整税率的级数:",true);
        taxRates[(int)rate-1] = getInputValue(scanner,"请输入新的税率:");
        System.out.println("调整税率成功!");
    }

    /**
     * 查看个人所得税各级税率
     *
     * <p> 展示个人所得税各级税率表
     */
    public static void showTaxRates(){
        System.out.println("当前个人所得税各级税率为:");
        System.out.println("级数\t应纳税所得额\t税率");
        System.out.println("1\t0-"+taxThresholds[0]+"  \t"+taxRates[0]);
        for(int i = 1; i < taxRates.length-1; i++){
            System.out.println((i+1)+"\t"+taxThresholds[i-1]+"-"+taxThresholds[i]+"\t"+taxRates[i]);
        }
        System.out.println("7\t超出"+taxThresholds[5]+" \t"+taxRates[6]);
    }

    /**
     * 自定义输入方法
     *
     * <p> 打印提示用户输入的内容，并判定输入内容是否有效，有效则返回输入值，无效则提醒用户重新输入
     * <p> 该函数是一个重载函数，相当于设置默认参数 isRate=false
     *
     *
     * @param scanner 用于接收用户输入的Scanner对象
     * @param promptMessage 用于提醒用户输入的文本
     * @return 返回用户输入的有效值
     */
    private static double getInputValue(Scanner scanner, String promptMessage){
        return getInputValue(scanner,promptMessage,false);
    }
    /**
     * 自定义输入方法
     *
     * <p> 打印提示用户输入的内容，并判定输入内容是否有效，有效则返回输入值，无效则提醒用户重新输入
     * <p> 输入不合法进行异常处理，输入小于0也提醒输入无效
     * <p> 同时根据 isRate，如果输入内容是税率级数，则需要额外判定级数是否为整数以及是否在1-7范围内
     *
     * @param scanner 用于接收用户输入的Scanner对象
     * @param promptMessage 用于提醒用户输入的文本
     * @param isRate 对于输入级数的情况，进行额外的有效值判定
     * @return 返回用户输入的有效值
     */
    private static double getInputValue(Scanner scanner, String promptMessage, boolean isRate) {
        double value = 0.0;
        boolean isValidInput = false;
        while (!isValidInput) {
            System.out.println(promptMessage);
            try {
                value = scanner.nextDouble();
                if(isRate){
                    if(value - (int)value == 0 && value>=1 && value <= 7)isValidInput = true;
                    else System.out.println("输入无效，请输入一个有效的数值!");
                }
                else if(value >=0 )isValidInput = true;
                else System.out.println("输入无效，请输入一个有效的数值!");
            } catch (InputMismatchException e) {
                System.out.println("输入无效，请输入一个有效的数值!");
                scanner.next();
            }
        }
        return value;
    }
}
