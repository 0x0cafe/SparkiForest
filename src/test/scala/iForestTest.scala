import java.util.Date
import iForest._
import org.apache.spark.{SparkConf, SparkContext}
import scala.util.Random

object iForestTest {
    def main(args: Array[String]): Unit = {

        val conf = new SparkConf().setAppName("iForest").setMaster("local[*]")
        val sc = new SparkContext(conf)

        Random.setSeed(2018)
        val lines = sc.textFile("data/data.csv")
        val data = lines.map(line => line.split(",")).map(s => s.slice(1,s.length))
        val header = data.first()
        val rows = data.filter(line => line(0) != header(0)).map(s => s.map(_.toDouble))
        val start = new Date().getTime
        val forest = buildForest(rows, numTrees=10)
        saveModel(sc,forest,"model/forest")
        val saveforest = loadModel(sc,"model/forest")
        val end = new Date().getTime

        val result_rdd = rows.map(row => row ++ Array(forest.predict(row))).cache()
        val result_rdd_save = rows.map(row => row ++ Array(saveforest.predict(row))).cache()

        println(s"NJULOG--->Finished Isolation use ${end-start}ms")
        result_rdd.map(lines => lines.mkString(",")).repartition(1).saveAsTextFile("result/test_predict_labels")
        result_rdd_save.map(lines => lines.mkString(",")).repartition(1).saveAsTextFile("result/test_predict_labels_save")
    }
}
