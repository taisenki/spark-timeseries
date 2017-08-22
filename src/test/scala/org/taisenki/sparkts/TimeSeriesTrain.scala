/**
  * Copyright (c) 2015, Cloudera, Inc. All Rights Reserved.
  *
  * Cloudera, Inc. licenses this file to you under the Apache License,
  * Version 2.0 (the "License"). You may not use this file except in
  * compliance with the License. You may obtain a copy of the License at
  *
  *     http://www.apache.org/licenses/LICENSE-2.0
  *
  * This software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
  * CONDITIONS OF ANY KIND, either express or implied. See the License for
  * the specific language governing permissions and limitations under the
  * License.
  */

package org.taisenki.sparkts

import java.sql.Timestamp
import java.time.{ZoneId, ZonedDateTime}

import com.cloudera.sparkts._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * 时间序列模型time-series的建立
 * Created by llq on 2017/4/17.
 */
object TimeSeriesTrain {

  /**
   * 把数据中的“time”列转换成固定时间格式：ZonedDateTime（such as 2007-12-03T10:15:30+01:00 Europe/Paris.）
   * @param timeDataKeyDf
   * @param sqlContext
   * @param hiveColumnName
   * @return zonedDateDataDf
   */
  def timeChangeToDate(timeDataKeyDf:DataFrame,sqlContext: SQLContext,
                       hiveColumnName:List[String],startTime:String,sc:SparkContext): DataFrame ={
    var rowRDD:RDD[Row]=sc.parallelize(Seq(Row(""),Row("")))
    //具体到月份
    if(startTime.length==6){
      rowRDD=timeDataKeyDf.rdd.map{row=>
        row match{
          case Row(time,key,data)=>{
            val dt = ZonedDateTime.of(time.toString.substring(0,4).toInt,
              time.toString.substring(4).toInt,1,0,0,0,0,ZoneId.systemDefault())
            Row(Timestamp.from(dt.toInstant), key.toString, data.toString.toDouble)
          }
        }
      }
    }else if(startTime.length==8){
      //具体到日
      rowRDD=timeDataKeyDf.rdd.map{row=>
        row match{
          case Row(time,key,data)=>{
            val dt = ZonedDateTime.of(time.toString.substring(0,4).toInt,
              time.toString.substring(4,6).toInt,
              time.toString.substring(6).toInt,0,0,0,0,ZoneId.systemDefault())
            Row(Timestamp.from(dt.toInstant), key.toString, data.toString.toDouble)
          }
        }
      }
    }
    //根据模式字符串生成模式，转化成dataframe格式
    val field=Seq(
      StructField(hiveColumnName(0), TimestampType, true),
      StructField(hiveColumnName(1), StringType, true),
      StructField(hiveColumnName(2), DoubleType, true)
    )
    val schema=StructType(field)
    val zonedDateDataDf=sqlContext.createDataFrame(rowRDD,schema)
    return zonedDateDataDf
  }


  /**
   * 总方法调用
   * @param args
   */
  def main(args: Array[String]):Unit = {
    /*****环境设置*****/
    //shield the unnecessary log in terminal
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    //set the environment
    System.setProperty("hadoop.home.dir", "D:\\data\\hadoop\\")
    val conf = new SparkConf().setAppName("timeSeries.local.TimeSeriesTrain").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val sqlContext=new SQLContext(sc)

    /*****参数设置*****/
    //hive中的数据库名字.数据表名
//    val databaseTableName="time_series.jxt_electric_month"
    //选择模型(holtwinters或者是arima)
    val modelName="holtwinters"
    //选择要hive的数据表中要处理的time和data列名（输入表中用于训练的列名,必须前面是时间，后面是data）
    val hiveColumnName=List("time","org_no","data")
    //日期的开始和结束，格式为“yyyyMM”或者为“yyyyMMdd”
    val startTime="201601"
    val endTime="201704"
    //预测后面N个值
    val predictedN=3
    //存放的表名字
    val outputTableName="timeseries_outputdate"

    //只有holtWinters才有的参数
    //季节性参数（12或者4）
    val period=6
    //holtWinters选择模型：additive（加法模型）、Multiplicative（乘法模型）
    val holtWintersModelType="Multiplicative"

    /*****读取数据和创建训练数据*****/
    //    //read the data form the hive
    //    val hiveDataDf=hiveContext.sql("select * from "+databaseTableName)
    //      .select(hiveColumnName.head,hiveColumnName.tail:_*)
//    val hiveDataDf=sqlContext.load("com.databricks.spark.csv",
    // Map("path" -> "src/test/resources/201601-201706.csv", "header" -> "true"))
      /*.select(hiveColumnName.head,hiveColumnName.tail:_*)*/

    val dataFile = getClass.getClassLoader.getResourceAsStream("201601-201706-dd.csv")
    val rawData = scala.io.Source.fromInputStream(dataFile).getLines().toArray
    val head = rawData.apply(0).split(",").drop(1).map(_.trim)
    val rawData2 = rawData.drop(1).flatMap(s=>{
      val ss = s.split(",").map(_.trim)
      ss.drop(1).zip(head).map(r=>Row(r._2,ss(0).trim,r._1))
    })

//    rawData2.foreach(println(_))

    //把dateData转换成dataframe
    val schema=StructType(hiveColumnName
      .map(fieldName=>StructField(fieldName,StringType,true)))
    val timeDataKeyDf = sqlContext.createDataFrame(sc.parallelize(rawData2),schema)

    //In hiveDataDF:increase a new column.This column's name is hiveColumnName(0)+"Key",it's value is 0.
    //The reason is:The string column labeling which string key the observation belongs to.
//    val timeDataKeyDf=hiveDataDf/*.withColumn(hiveColumnName(0)+"Key",hiveDataDf(hiveColumnName(2))*0)*/
//      .select(hiveColumnName(0),hiveColumnName(1),hiveColumnName(2))
    val zonedDateDataDf=timeChangeToDate(timeDataKeyDf,sqlContext,hiveColumnName,startTime,sc)

    /**
     * 创建数据中时间的跨度（Create an daily DateTimeIndex）:开始日期+结束日期+递增数
     * 日期的格式要与数据库中time数据的格式一样
     */
    //参数初始化
    val zone = ZoneId.systemDefault()
    var dtIndex:UniformDateTimeIndex=DateTimeIndex.uniformFromInterval(
      ZonedDateTime.of(2016, 1, 1, 0, 0, 0, 0, zone),
      ZonedDateTime.of(2017, 7, 1, 0, 0, 0, 0, zone),
      new MonthFrequency(1))

    //具体到月份
    if(startTime.length==6) {
      dtIndex = DateTimeIndex.uniformFromInterval(
        ZonedDateTime.of(startTime.substring(0, 4).toInt, startTime.substring(4).toInt, 1, 0, 0, 0, 0, zone),
        ZonedDateTime.of(endTime.substring(0, 4).toInt, endTime.substring(4).toInt, 1, 0, 0, 0, 0, zone),
        new MonthFrequency(1))
    }else if(startTime.length==8){
      //具体到日,则把dtIndex覆盖了
      dtIndex = DateTimeIndex.uniformFromInterval(
        ZonedDateTime.of(startTime.substring(0,4).toInt,startTime.substring(4,6).toInt,startTime.substring(6).toInt,0,0,0,0,zone),
        ZonedDateTime.of(endTime.substring(0,4).toInt,endTime.substring(4,6).toInt,endTime.substring(6).toInt,0,0,0,0,zone),
        new DayFrequency(1))
    }

    //创建训练数据TimeSeriesRDD(key,DenseVector(series))
    val trainTsrdd = TimeSeriesRDD.timeSeriesRDDFromObservations(dtIndex, zonedDateDataDf,
      hiveColumnName(0), hiveColumnName(1), hiveColumnName(2))

    /*****建立Modle对象*****/
    val timeSeriesModel=new TimeSeriesModel(predictedN,outputTableName)
    var forecastValue:RDD[String]=sc.parallelize(Seq(""))
    //选择模型
    modelName match{
      case "arima"=>{
        //创建和训练arima模型
        forecastValue=timeSeriesModel.arimaModelTrain(trainTsrdd)
      }
      case "holtwinters"=>{
        //创建和训练HoltWinters模型(季节性模型)
        forecastValue=timeSeriesModel.holtWintersModelTrain(trainTsrdd,period,holtWintersModelType)
      }
      case "ARGARCH"=>{
        forecastValue=timeSeriesModel.argarchModelTrain(trainTsrdd)
      }
      case _=>throw new UnsupportedOperationException("Currently only supports 'ariam' and 'holtwinters")
    }

    //合并实际值和预测值，并加上日期,形成dataframe(Date,Data)，并保存
//    timeSeriesModel.actualForcastDateSaveInText(trainTsrdd,forecastValue,modelName,predictedN,startTime,endTime,sc,hiveColumnName,sqlContext)
  }
}
