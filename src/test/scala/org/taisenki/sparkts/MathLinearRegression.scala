package org.taisenki.sparkts

import org.apache.commons.math3.stat.regression.{OLSMultipleLinearRegression, SimpleRegression}

object MathLinearRegression {

  def main(args: Array[String]): Unit = {
//    simpleRegression()
    multipleRegression()
  }

  private def multipleRegression(): Unit = {
    System.out.println("multipleRegression")
    val regression2 = new OLSMultipleLinearRegression
    val y = Array(3d, 5, 7, 9, 11)
    val x2 = Array(Array(1d), Array(2d), Array(3d), Array(4d), Array(5d))
    regression2.newSampleData(y, x2)
    val beta = regression2.estimateRegressionParameters
    for (d <- beta) {
      System.out.println("D: " + d)
    }
    println(s"Residuals is ${regression2.estimateResiduals().mkString(",")}")
    System.out.println("prediction for 1.5 = " + predict(Array[Double](1, 1.5), beta))
  }

  private def predict(data: Array[Double], beta: Array[Double]) = {
    var result = 0d
    var i = 0
    while (i < data.length) {
      result += data(i) * beta(i)
      i += 1 ;
    }
    result
  }

  private def simpleRegression(): Unit = {
    System.out.println("simpleRegression")
    // creating regression object, passing true to have intercept term
    val simpleRegression = new SimpleRegression(true)
    // passing data to the model
    // model will be fitted automatically by the class
    simpleRegression.addData(Array[Array[Double]](Array(1, 2), Array(2, 3), Array(3, 4), Array(4, 5), Array(5, 6)))
    // querying for model parameters
    System.out.println("slope = " + simpleRegression.getSlope)
    System.out.println("intercept = " + simpleRegression.getIntercept)
    // trying to run model for unknown data
    System.out.println("prediction for 1.5 = " + simpleRegression.predict(1.5))
  }
}
