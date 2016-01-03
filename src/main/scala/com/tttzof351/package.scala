package com

import java.io.{DataInputStream, FileInputStream}
import java.lang.Math._

import breeze.linalg.{sum, DenseMatrix, DenseVector}
import breeze.plot._

import scala.util.Random

package object tttzof351 {
  val ceilLength = 28
  val ceilSize = ceilLength * ceilLength
  val singleLimit = 500
  val path = "./mnist/data" // http://cis.jhu.edu/~sachin/digit/digit.html

  def loadRandMnist(): IndexedSeq[(DenseVector[Double], DenseVector[Double])] = {
    val res =
      (0 to 9).flatMap { i =>
        val inputStream = new FileInputStream(path + i)
        val dataStream = new DataInputStream(inputStream)
        val inAndOut =
          (0 until singleLimit).map { _ =>
            val array = (0 until ceilSize).map(_ => (dataStream.readByte & 0xff) / 255d).toArray
            DenseVector(array) -> vecFromVal(i)
          }

        inputStream.close()
        dataStream.close()
        inAndOut
      }

    Random.shuffle(res)
  }

  def showMatrix(ceil: DenseMatrix[Double]): Unit = {
    val f2 = Figure()
    f2.subplot(0) += image(ceil)
    f2.saveas("image.png")
  }

  def vecFromVal(value: Int): DenseVector[Double] = {
    require(value >= 0 && value <= 9)
    val vec = DenseVector.fill[Double](10)(0d)
    vec(value) = 1d
    vec
  }

  def valFromVec(vec: DenseVector[Double]): Int = {
    require(vec.length == 10)
    val s = vec.toArray.toSeq
    val el = s.max
    val index = s.indexOf(el)
    index
  }

  def sigma(z: Double): Double = pow(exp(-z) + 1, -1)

  def toMatrix(vectors: Seq[DenseVector[Double]]): DenseMatrix[Double] = {
    val length = vectors.head.length
    val m = vectors.size

    val arrays =
      vectors.foldLeft(Array.empty[Double]) { case (acc, vec) =>
        require(vec.length == length)
        acc ++ vec.toArray
      }

    new DenseMatrix[Double](m, length, arrays, 0, length, isTranspose = true)
  }

  def toSize(mtx: DenseMatrix[Double]): String = s"(${mtx.rows}, ${mtx.cols})"

  def energy(netRes: DenseMatrix[Double],
             reqRes: DenseMatrix[Double]): Double = {
    require(netRes.rows == reqRes.rows && netRes.cols == reqRes.cols)
    val m = netRes.rows.toDouble
    val leftPart = -reqRes :* netRes.map(math.log)
    val rightPart = -(1d - reqRes) :* netRes.map(v => math.log(1d - v))
    val eng = leftPart + rightPart
    sum(eng) / m.toDouble
  }

  def addColumn(column: DenseVector[Double], matrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    require(column.length == matrix.rows)
    DenseMatrix.tabulate(matrix.rows, matrix.cols + 1) { (i,j) =>
      if (j == 0) column(j)
      else matrix(i, j-1)
    }
  }

  def removeRow(matrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    matrix(1 to matrix.rows - 1, 0 to matrix.cols - 1)
  }
}
