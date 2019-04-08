# SparkiForest
```
//加载数据
val lines = sc.textFile("hdfs://.../data/data.csv")
val data = lines.map(line => line.split(",")).map(s => s.slice(1,s.length))
val header = data.first()
val rows = data.filter(line => line(0) != header(0)).map(s => s.map(_.toDouble))

//训练模型
val forest = buildForest(rows, numTrees=10)
//保存模型
saveModel(sc,forest,"model/forest")
//加载模型
val saveforest = loadModel(sc,"model/forest")
//使用训练后的模型直接检测
val result_rdd = rows.map(row => row ++ Array(forest.predict(row))).cache()
//使用保存的模型来检测
val result_rdd_save = rows.map(row => row ++ Array(saveforest.predict(row))).cache()
//保存模型
result_rdd.map(lines => lines.mkString(",")).repartition(1).saveAsTextFile("result/test_predict_labels")
result_rdd_save.map(lines => lines.mkString(",")).repartition(1).saveAsTextFile("result/test_predict_labels_save")
```
