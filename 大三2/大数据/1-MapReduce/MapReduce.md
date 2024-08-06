## MapReduce学习实践

### 描述

**MapReduce：**

分布式、并行处理、计算框架

**相关**：

HDFS：存储非常大的文件，采用流式数据访问方式，运行于商业硬件，详见1. [深入理解Hadoop HDFS【一篇就够】-CSDN博客](https://blog.csdn.net/sjmz30071360/article/details/79877846)2. [Hadoop 入门教程（超详细）_hadoop教程-CSDN博客](https://blog.csdn.net/weixin_42837961/article/details/105493561)

#### 解决思路

1. 在IDEA编写好代码文件
2. 配置虚拟机+安装好jdk、hadoop
3. 打包好jar包传送到虚拟机

#### 开启hadoop集群

```
%HADOOP_HOME%/sbin/start-all.cmd
```

#### 遇到问题

- hadoop需要解压在没有带空格的文件路径下，如``program file``目录下是不允许的
- 需要在https://github.com/steveloughran/winutils下载winutils.exe以及hadoop.dll，注意版本不要太高，因为我用的hadoop是2.10.2的老古董，免得不兼容
- 问题：sbin下start-yarn.cmd修改，无法解决

```
原内容
@rem start resourceManager
start "Apache Hadoop Distribution" yarn resourcemanager
@rem start nodeManager
start "Apache Hadoop Distribution" yarn nodemanager
@rem start proxyserver
@rem start "Apache Hadoop Distribution" yarn proxyserver
修改后
@rem start resourceManager
start "Apache Hadoop Distribution" %HADOOP_BIN_PATH%\yarn resourcemanager
@rem start nodeManager
start "Apache Hadoop Distribution" %HADOOP_BIN_PATH%\yarn nodemanager
@rem start proxyserver
@rem start "Apache Hadoop Distribution" %HADOOP_BIN_PATH%\yarn
```

#### 碎碎念

Hadoop：基本就用到mapreduce，大部分都spark化了，面试还是会问，大学没必要浪费时间在搭建部署环境上。

### 参考链接

- [（超详细）MapReduce工作原理及基础编程_mapreduce编程-CSDN博客](https://blog.csdn.net/JunLeon/article/details/121051075)
- [Hadoop 伪分布式环境搭建_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1mL4y1T7em/?spm_id_from=333.337.search-card.all.click&vd_source=6b153427393db22a222ff8aec7bb5efb)
- [CentOS7下安装Hadoop伪分布式教程_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1i5411d7aG/?spm_id_from=333.337.search-card.all.click&vd_source=6b153427393db22a222ff8aec7bb5efb)
- [【Hadoop】：Windows下使用IDEA搭建Hadoop开发环境-CSDN博客](https://blog.csdn.net/Geeksongs/article/details/111940739)
- [MapReduce编程(一) Intellij Idea配置MapReduce编程环境_在idea中搭建mapreduce环境-CSDN博客](https://blog.csdn.net/napoay/article/details/68491469)
- [windows下hadoop安装时出现error Couldn‘t find a package.json file in “D:\\hadoop\hadoop-2.7.7\\sbin“问题_yarn run v1.22.19 error couldn't find a package.js-CSDN博客](https://blog.csdn.net/m_phappy/article/details/110856485)
- [IDEA-Maven项目中：java:程序包org.apache.hadoop.conf.fs等众多Hadoop包不存在的问题_java: 程序包org.apache.hadoop.conf不存在-CSDN博客](https://blog.csdn.net/qq_46092061/article/details/120127385)
- [Spark No FileSystem for scheme file 解决方法-腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1454044)





### 示例

chatgpt

```java
import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount {

  public static class TokenizerMapper
       extends Mapper<Object, Text, Text, IntWritable>{

    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();

    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {
        word.set(itr.nextToken());
        context.write(word, one);
      }
    }
  }

  public static class IntSumReducer
       extends Reducer<Text,IntWritable,Text,IntWritable> {
    private IntWritable result = new IntWritable();

    public void reduce(Text key, Iterable<IntWritable> values,
                       Context context
                       ) throws IOException, InterruptedException {
      int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
      result.set(sum);
      context.write(key, result);
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "word count");
    job.setJarByClass(WordCount.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(IntSumReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}

```

csdn

```java

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.Iterator;
import java.util.StringTokenizer;

/**
 * Created by bee on 3/25/17.
 */
public class WordCount {


    public static class TokenizerMapper extends
            Mapper<Object, Text, Text, IntWritable> {


        public static final IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                this.word.set(itr.nextToken());
                context.write(this.word, one);
            }
        }

    }

    public static class IntSumReduce extends
            Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            IntWritable val;
            for (Iterator i = values.iterator(); i.hasNext(); sum += val.get()) {
                val = (IntWritable) i.next();
            }
            this.result.set(sum);
            context.write(key, this.result);
        }
    }

    public static void main(String[] args)
            throws IOException, ClassNotFoundException, InterruptedException {

//        FileUtil.deleteDir("output");
        Configuration conf = new Configuration();

        String[] otherArgs = new String[]{"input/dream.txt","output"};
        if (otherArgs.length != 2) {
            System.err.println("Usage:Merge and duplicate removal <in> <out>");
            System.exit(2);
        }

        Job job = Job.getInstance(conf, "WordCount");
        job.setJarByClass(WordCount.class);
        job.setMapperClass(WordCount.TokenizerMapper.class);
        job.setReducerClass(WordCount.IntSumReduce.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
```

