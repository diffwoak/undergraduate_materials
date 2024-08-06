
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class NaturalJoin {
    public static class TupleMapper extends Mapper<Object, Text, Text, Text> {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // 获取当前的表名:Table_1.txt / Table_2.txt
            org.apache.hadoop.mapreduce.InputSplit inputSplit = context.getInputSplit();
            String fileName = ((org.apache.hadoop.mapreduce.lib.input.FileSplit) inputSplit).getPath().getName();
            // 将输入Text转为Sting类型,并根据"\n"分行
            String line = new String(value.getBytes(), StandardCharsets.UTF_8);
            StringTokenizer tokenArticle = new StringTokenizer(line,"\n");
            // 遍历每行,根据表名判断b的位置
            // 记录record="Table_1.txt a值"或"Table_2.txt c值"
            // 记录name = "b值"
            while(tokenArticle.hasMoreElements()) {
                String record = fileName;
                String[] tokenLine = tokenArticle.nextToken().split(" ");
                String name;
                if(Objects.equals(fileName, "Table_1.txt")){
                    record = record + " "+ tokenLine[0];
                    name = tokenLine[1];
                }else {
                    record = record + " "+ tokenLine[1];
                    name = tokenLine[0];
                }
                context.write(new Text(name), new Text(record));
            }
        }
    }

    public static class MatchReduce extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            // 对于一个键值b,left记录对应a值,right记录对应c值
            StringBuilder left = new StringBuilder();
            List<String> right = new ArrayList<>();
            for (Text value : values) {
                StringTokenizer itr = new StringTokenizer(value.toString());
                if (Objects.equals(itr.nextToken(), "Table_1.txt")) {
                    left.append(itr.nextToken());
                } else {
                    right.add(itr.nextToken());
                }
            }
            // 对a、c值组合，其中a值唯一,遍历c值即可
            for (String tmp : right) {
                context.write(key, new Text(left+ " "+tmp));
            }
        }
    }

    public static void main(String[] args)
            throws IOException, ClassNotFoundException, InterruptedException {

        String[] otherArgs = new String[]{"input/Table_1.txt","input/Table_2.txt","output"};
//        String[] otherArgs = new String[]{"hdfs://localhost:8020/worddir/Table_1.txt","hdfs://localhost:8020/worddir/Table_2.txt","hdfs://localhost:8020/worddir/output"};
        deleteDir("output");    // 自动清除output


        Configuration conf = new Configuration();
//        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        Job job = Job.getInstance(conf, "NaturalJoin");       // 创建空任务
        job.setJarByClass(NaturalJoin.class);                       // 设置任务的启动类
        job.setMapperClass(NaturalJoin.TupleMapper.class);          // 设置Mapper任务类
        job.setReducerClass(NaturalJoin.MatchReduce.class);         // 设置Reduce任务类
        job.setOutputKeyClass(Text.class);                          // 设置输出Key的类型
        job.setOutputValueClass(Text.class);                        // 设置输出Value的类型
        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));  // 设置输入第一个数据源
        FileInputFormat.addInputPath(job, new Path(otherArgs[1]));  // 设置输入第二个数据源
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[2]));// 设置输出数据目标
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

    /**
     * 删除文件夹及其文件
     * @param path 文件夹在项目根目录下相对路径
     */
    public static void deleteDir(String path) {
        File dir = new File(path);
        if (dir.exists()) {
            for (File f : dir.listFiles()) {
                if (f.isDirectory()) {
                    deleteDir(f.getName());
                } else {
                    f.delete();
                }
            }
            dir.delete();
        }
    }

}
