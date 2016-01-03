package com.tttzof351

import breeze.linalg.DenseVector._
import breeze.linalg._
import breeze.optimize.{LBFGS, DiffFunction}

import scala.util.Random

object Perseptron {
  def main(args: Array[String]): Unit = {
    Network.mnistLearn()
  }
}

object Network {

  def mnistLearn(): Unit = {
    val inputsAndOutputs = loadRandMnist
    val size = inputsAndOutputs.size
    val (learnSet, testSet) = inputsAndOutputs.splitAt((size*0.8).toInt)

    val inputs = toMatrix(learnSet.map(_._1))
    val outputs = toMatrix(learnSet.map(_._2))

    val inLength = inputs.cols
    val outLength = outputs.cols

    val network = Network(inLength, Seq(30), outLength)
    val weights = Array.fill[Double](network.length)(Random.nextDouble - 0.5d)

    def showTest(ws: Array[Double]): Unit = {
      val dataSet = testSet
      val testPercent =
        dataSet.map { case (in, out) =>
        val res = network.result(in.toDenseMatrix, ws).toDenseVector
        val resVl = valFromVec(res)
        val outVl = valFromVec(out)
        if (resVl == outVl) 1 else 0
      }.sum * 100d / dataSet.length

      println(s"test: $testPercent %")
    }

    def showWeights(ws: Array[Double]): Unit =
      Option(network.slices.head).map { case (offset, (rows, cols)) =>
        val w  = new DenseMatrix[Double](rows, cols, ws, offset)
        val array = w(0 to 3, 1 to w.cols - 1).toArray
        new DenseMatrix[Double](4*ceilLength, ceilLength, array)
      }.foreach(showMatrix)

    showWeights(weights)
    showTest(weights)
    val functOpt = new DiffFunction[DenseVector[Double]] {
      override def calculate(ws: DenseVector[Double]) = {
        val (res, gradient) = network.backPropagation(inputs, outputs, ws.toArray)
        val eng = energy(res, outputs)
        println("eng: " + eng)
        (eng, DenseVector(gradient))
      }
    }

    val lbfgs = new LBFGS[DenseVector[Double]](maxIter=200, m=3)
    val optimum = lbfgs.minimize(functOpt, DenseVector(weights)).toArray

    showTest(optimum)
    showWeights(optimum)
  }

  def checkConstGradient(): Unit = {
    val network = Network(3, Seq(5), 3)
    println("Length: " + network.length)
    val inputs = DenseMatrix(( 0.084147d, -0.027942d,  -0.099999d),
                             ( 0.090930d,  0.065699d,  -0.053657d),
                             ( 0.014112d,  0.098936d,   0.042017),
                             (-0.075680d,  0.041212d,   0.099061d),
                             (-0.095892d, -0.054402d,   0.065029d))

    val outputs = DenseMatrix((0d, 1d, 0d),
                              (0d, 0d, 1d),
                              (1d, 0d, 0d),
                              (0d, 1d, 0d),
                              (0d, 0d, 1d))

    val weights = Array( 0.084147d,  -0.027942d,  -0.099999d,  -0.028790d,
                         0.090930d,   0.065699d,  -0.053657d,  -0.096140d,
                         0.014112d,   0.098936d,   0.042017d,  -0.075099d,
                        -0.075680d,   0.041212d,   0.099061d,   0.014988d,
                        -0.095892d,  -0.054402d,   0.065029d,   0.091295d) ++
                  Array( 0.084147d,  -0.075680d,   0.065699d,  -0.054402d, 0.042017d,  -0.028790d,
                         0.090930d,  -0.095892d,   0.098936d,  -0.099999d, 0.099061d,  -0.096140d,
                         0.014112d,  -0.027942d,   0.041212d,  -0.053657d, 0.065029d,  -0.075099d)

    val numGradient = network.numericalGradient(inputs, outputs, weights, 0.0001d)
    val (res, gradient) = network.backPropagation(inputs, outputs, weights)
    val eng = energy(res, outputs)
    println("eng: " + eng + "; should be ~2.100")

    weights.indices.foreach { i =>
      val g = gradient(i)
      val b = numGradient(i)
      println(s"${i+1}) $g $b")
    }
  }

  def checkRandGradient(): Unit = {
    val network = Network(5, Seq(4,9), 4)
    val m = 100
    val inputs = DenseMatrix.rand[Double](m, network.inLength)
    val outputs = DenseMatrix.rand[Double](m, network.outLength)
    val weights = Array.fill[Double](network.length)(Random.nextDouble)

    val numGradient = network.numericalGradient(inputs, outputs, weights, 0.0001d)
    val (res, gradient) = network.backPropagation(inputs, outputs, weights)
    val diff = numGradient zip gradient
    println("sum diff: " + diff.map { case (g, b) => math.abs(g - b) }.max )
  }
}

case class Network(inLength: Int, hiden: Seq[Int], outLength: Int) {
  val wsizes = (hiden :+ outLength) zip (inLength +: hiden :+ outLength).map(_ + 1)
  val length: Int = wsizes.map { case (l, r) => l * r }.sum
  val slices = wsizes.scanLeft(0) { case (offset, (l, r)) =>
    offset + l * r
  } zip wsizes

  def forwardPropagation(input: DenseMatrix[Double],
                         weights: Array[Double]): Seq[DenseMatrix[Double]] = {
    require(input.cols == inLength)
    slices.scanLeft(input) { case (in, (offset, (rows, cols))) =>
      val w = new DenseMatrix[Double](rows, cols, weights, offset, cols, isTranspose = true)
      val inBias = addColumn(ones(in.rows), in)
      val z = (inBias * w.t).asInstanceOf[DenseMatrix[Double]]
      z.map(sigma)
    }
  }

  def backPropagation(inputs: DenseMatrix[Double],
                      requireRes: DenseMatrix[Double],
                      weights: Array[Double]): (DenseMatrix[Double], Array[Double]) = {
    require(requireRes.cols == outLength)
    require(inputs.rows == requireRes.rows)

    val outputs = forwardPropagation(inputs, weights)
    val netRes = outputs.last

    val firstDelta = netRes - requireRes
    val layers = (slices zip outputs.dropRight(1)).reverse

    val gradient = Array.fill[Double](length)(0d)
    layers.foldLeft(firstDelta) { case (delta, ((offset, (rows, cols)), out)) =>
      val outWithBias = addColumn(ones(out.rows), out)
      val w = new DenseMatrix[Double](rows, cols, weights, offset, cols, isTranspose = true)
      val nextDeltaWithBias = (outWithBias :* (1d - outWithBias)) :* (delta * w)
      val nextDelta = removeRow(nextDeltaWithBias.t).t
      val dw = (outWithBias.t * delta).asInstanceOf[DenseMatrix[Double]].toArray

      System.arraycopy(dw, 0, gradient, offset, dw.length)
      nextDelta
    }

    (netRes, gradient)
  }

  def result(input: DenseMatrix[Double],
             weights: Array[Double]): DenseMatrix[Double] =
    forwardPropagation(input, weights).last

  def numericalGradient(input: DenseMatrix[Double],
                        requireRes: DenseMatrix[Double],
                        weights: Array[Double], delta: Double): Array[Double] = {
    require(requireRes.cols == outLength)
    require(requireRes.rows == input.rows)

    val currentWeights = weights.clone
    currentWeights.indices.map { i =>
      val wi = currentWeights(i)

      currentWeights.update(i, wi + delta)
      val deltaPlus = result(input, currentWeights)
      val engPlus = energy(deltaPlus, requireRes)
      currentWeights.update(i, wi)

      currentWeights.update(i, wi - delta)
      val deltaMinus = result(input, currentWeights)
      val engMinus = energy(deltaMinus, requireRes)
      currentWeights.update(i, wi)

      (engPlus - engMinus) / (2*delta)
    }.toArray
  }
}
