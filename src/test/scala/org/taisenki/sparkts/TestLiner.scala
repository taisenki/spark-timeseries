package org.taisenki.sparkts

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression

object TestLiner {

  def main(args:Array[String]): Unit ={
    val regression = new OLSMultipleLinearRegression
    val y = Array[Double](11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
    val x = new Array[Array[Double]](6)
    x(0) = Array[Double](0, 0, 0, 0, 0)
    x(1) = Array[Double](2.0, 0, 0, 0, 0)
    x(2) = Array[Double](0, 3.0, 0, 0, 0)
    x(3) = Array[Double](0, 0, 4.0, 0, 0)
    x(4) = Array[Double](0, 0, 0, 5.0, 0)
    x(5) = Array[Double](0, 0, 0, 0, 6.0)
    regression.newSampleData(y, x)
    val betaHat = regression.estimateRegressionParameters
    System.out.println("Estimates the regression parameters b:")
    println(betaHat.mkString(","))
    val residuals = regression.estimateResiduals
    System.out.println("Estimates the residuals, ie u = y - X*b:")
    println(residuals.mkString(","))
    val vary = regression.estimateRegressandVariance
    System.out.println("Returns the variance of the regressand Var(y):")
    System.out.println(vary)
    val erros = regression.estimateRegressionParametersStandardErrors
    System.out.println("Returns the standard errors of the regression parameters:")
    println(erros.mkString(","))
    val varb = regression.estimateRegressionParametersVariance
    regression
  }
}
