package org.taisenki.sparkts

import java.lang.Math.toRadians
import java.util.ArrayList
import java.util.Collection

import org.apache.commons.math3.fitting.{PolynomialCurveFitter, WeightedObservedPoint}


/**
  * 曲线拟合demo
  */
object TestInstance {

  def main(args: Array[String]): Unit = {
    val doubles = trainPolyFit(3, 10000)
    //下面是测试
    System.out.println("a1: " + doubles(0))
    System.out.println("a2: " + doubles(1))
    System.out.println("a3: " + doubles(2))
    System.out.println("a4: " + doubles(3))
    System.out.println("double的个数" + doubles.length)
    val hehe = Math.cos(toRadians(40))
    val fo = doubles(3) * 40 * 40 * 40 + doubles(2) * 40 * 40 + doubles(1) * 40 + doubles(0)
    System.out.println("hehe: " + hehe)
    System.out.println("fo: " + fo)
    val sub = hehe - fo
    System.out.println(sub * 300000)
  }

  /**
    *
    * @param degree 代表你用几阶去拟合
    * @param Length 把10 --60 分成多少个点去拟合，越大应该越精确
    * @return
    */
  def trainPolyFit(degree: Int, Length: Int): Array[Double] = {
    val polynomialCurveFitter = PolynomialCurveFitter.create(degree)
    val minLat = 10.0
    //中国最低纬度
    val maxLat = 60.0
    //中国最高纬度
    val interv = (maxLat - minLat) / Length.toDouble
    val weightedObservedPoints = new ArrayList[WeightedObservedPoint]
    var i = 0
    while (i < Length) {
      val weightedObservedPoint = new WeightedObservedPoint(1, minLat + i.toDouble * interv, Math.cos(toRadians(minLat + i.toDouble * interv)))
      weightedObservedPoints.add(weightedObservedPoint)
      i += 1
    }
    polynomialCurveFitter.fit(weightedObservedPoints)
  }

  def distanceSimplifyMore(lat1: Double, lng1: Double, lat2: Double, lng2: Double, a: Array[Double]): Double = {
    //1) 计算三个参数
    val dx = lng1 - lng2
    // 经度差值
    val dy = lat1 - lat2
    // 纬度差值
    val b = (lat1 + lat2) / 2.0
    // 平均纬度
    //2) 计算东西方向距离和南北方向距离(单位：米)，东西距离采用三阶多项式
    val Lx = (a(3) * b * b * b + a(2) * b * b + a(1) * b + a(0)) * toRadians(dx) * 6367000.0
    // 东西距离
    val Ly = 6367000.0 * toRadians(dy) // 南北距离
    //3) 用平面的矩形对角距离公式计算总距离
    Math.sqrt(Lx * Lx + Ly * Ly)
  }
}
