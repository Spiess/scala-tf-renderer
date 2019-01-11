package meshrenderer

import java.io.File
import java.net.URI
import java.nio.{ByteBuffer, ByteOrder}

import breeze.linalg.{DenseMatrix, DenseVector}
import org.platanios.tensorflow.api
import org.platanios.tensorflow.api.ops.variables.ZerosInitializer
import org.platanios.tensorflow.api.{tf, _}
import scalismo.faces.io.MoMoIO
import scalismo.faces.momo.{MoMoBasic, MoMoExpress}
import scalismo.faces.parameters.MoMoInstance

/**
  * Created by andreas on 8/18/18.
  */

case class TFMoMoBasic(shape: Tensor[Float], shapeStddev: Tensor[Float], color: Tensor[Float], colorStddev: Tensor[Float])
case class TFMoMoExpress(shape: Tensor[Float], shapeStddev: Tensor[Float], color: Tensor[Float], colorStddev: Tensor[Float], expression: Tensor[Float], expressionStddev: Tensor[Float])


object TFMoMo {
  def apply(model: MoMoBasic): TFMoMoBasic = {
    TFMoMoBasic(
      TFMoMoConversions.toTensor(model.shape.basisMatrix), TFMoMoConversions.toTensor(model.shape.variance.map(math.sqrt)).transpose(),
      TFMoMoConversions.toTensor(model.color.basisMatrix), TFMoMoConversions.toTensor(model.color.variance.map(math.sqrt)).transpose()
    )
  }

  def apply(model: MoMoExpress): TFMoMoExpress = {
    TFMoMoExpress(
      TFMoMoConversions.toTensor(model.shape.basisMatrix), TFMoMoConversions.toTensor(model.shape.variance.map(math.sqrt)).transpose(),
      TFMoMoConversions.toTensor(model.color.basisMatrix), TFMoMoConversions.toTensor(model.color.variance.map(math.sqrt)).transpose(),
      TFMoMoConversions.toTensor(model.expression.basisMatrix), TFMoMoConversions.toTensor(model.expression.variance.map(math.sqrt)).transpose()
    )
  }

}

case class TFMoMoBasicParameterModel(model: TFMoMoBasic, mean: TFMesh, initPose: TFPoseTensor, initCamera: TFCameraTensor, initLight: Tensor[Float])
  extends OffsetFromInitializationModel(mean.pts, mean.colors, initPose, initCamera, initLight) {

  override lazy val ptsVar: api.tf.Variable[Float] = tf.variable[Float]("shapeCoefficients", Shape(model.shape.shape(1),1), ZerosInitializer)
  override lazy val colorsVar: api.tf.Variable[Float] = tf.variable[Float]("colorCoefficients", Shape(model.color.shape(1),1), ZerosInitializer)

  println("ptsVar", ptsVar.shape)
  println("model.shape", model.shape)
  println("model.shapeVariance", model.shapeStddev)
  println("ptsVar*model.shapeVariance", ptsVar*model.shapeStddev)

  override lazy val ptsOffset: Output[Float] = tf.matmul(model.shape, ptsVar*model.shapeStddev).reshape(Shape(model.shape.shape(0)/3, 3)).transpose()
  override lazy val colorsOffset: Output[Float] = tf.matmul(model.color, colorsVar*model.colorStddev).reshape(Shape(model.shape.shape(0)/3, 3))
}

/** Expression model concatenated to shape model. */
case class TFMoMoExpressParameterModel(model: TFMoMoExpress, mean: TFMesh, initPose: TFPoseTensor, initCamera: TFCameraTensor, initLight: Tensor[Float])
  extends OffsetFromInitializationModel(mean.pts, mean.colors, initPose, initCamera, initLight) {

  override lazy val ptsVar: api.tf.Variable[Float] = tf.variable[Float]("shapeCoefficients", Shape(model.shape.shape(1) + model.expression.shape(1),1), ZerosInitializer)
  override lazy val colorsVar: api.tf.Variable[Float] = tf.variable[Float]("colorCoefficients", Shape(model.color.shape(1),1), ZerosInitializer)

  println("ptsVar", ptsVar.shape)
  println("model.shape", model.shape)
  println("model.shapeVariance", model.shapeStddev)

  val combinedShape: Output[Float] = tf.concatenate(Seq(model.shape, model.expression), 1)
  val combinedShapeStddev: Output[Float] = tf.concatenate(Seq(model.shapeStddev, model.expressionStddev), 0)

  println("combinedShape", combinedShape)
  println("combinedShapeStddev", combinedShapeStddev)
  println("ptsVar*model.shapeVariance", ptsVar*combinedShapeStddev)


  override lazy val ptsOffset: Output[Float] = tf.matmul(combinedShape, ptsVar*combinedShapeStddev).reshape(Shape(model.shape.shape(0)/3, 3)).transpose()
  override lazy val colorsOffset: Output[Float] = tf.matmul(model.color, colorsVar*model.colorStddev).reshape(Shape(model.shape.shape(0)/3, 3))
}

object TFMoMoConversions {

  def toTensor(momo: MoMoBasic): Unit = {
    val ev = momo.shape.basisMatrix
    val variance = momo.shape.variance
  }

  @deprecated
  def toTensor(mat: DenseMatrix[Double]): Tensor[Float] = {
    toTensorNotTransposed(mat).transpose()
  }

  def toTensorNotTransposed(mat: DenseMatrix[Double]): Tensor[Float] = {
    val bufferCapacity = mat.size * 4
    val buffer = ByteBuffer.allocateDirect(bufferCapacity).order(ByteOrder.nativeOrder())
    mat.toArray.map(_.toFloat).foreach(buffer.putFloat)
    buffer.flip()
    Tensor.fromBuffer[Float](Shape(mat.cols, mat.rows), bufferCapacity, buffer)
  }

  def toTensor(vec: DenseVector[Double]): Tensor[Float] = {
    val bufferCapacity = vec.size * 4
    val buffer = ByteBuffer.allocateDirect(bufferCapacity).order(ByteOrder.nativeOrder())
    vec.toArray.map(_.toFloat).foreach(buffer.putFloat)
    buffer.flip()
    Tensor.fromBuffer[Float](Shape(1, vec.length), bufferCapacity, buffer)
  }

  def main(args: Array[String]): Unit = {

    val ten = Tensor(1,2,3,4,5,6,7,8,9,10,11,12).reshape(Shape(1,12))
    println("tem", ten)
    println(ten.reshape(Shape(4,3)).summarize())
    //sys.exit()

    scalismo.initialize()

    /*val momo = MoMoIO.read(new File("/home/andreas/export/model2017-1_bfm_nomouth.h5")).get
    val model = momo.neutralModel.truncate(5,5)
    MoMoIO.write(model, new File("tmp.h5")).get*/
    val model = MoMoIO.read(new File("tmp.h5")).get.neutralModel
    //val mat = model.shape.basisMatrix
    //println("mat", mat)
    println("loaded data.")

    //val basisMatrix = toTensor(mat)
    //println("ten", basisMatrix.summarize(maxEntries = 30))

    val inst = MoMoInstance(IndexedSeq(1f,0.1f, 0.2f, 0.3f, 0.4f), IndexedSeq(0f,0f, 0f, 0f, 0f), IndexedSeq.fill(5)(0.0f), new URI(""))

    val param = Tensor(inst.coefficients.shape.map(_.toFloat).toArray.toSeq).transpose()
    println("param", param)

    val tfModel = TFMoMo(model)

    println(tfModel.shape.summarize(maxEntries = 50))

    println("tfModel", tfModel.shape)

    //println("", tfModel.shape.transpose().reshape(Shape(tfModel.shape.shape(0)/3, 3)).summarize())

    val mult = tf.matmul(tfModel.shape, param * tfModel.shapeStddev)
    println("mult", mult)

    val tfMean = TFMesh(model.mean)
    val pts = tfMean.pts.transpose()
    println("pts", pts)

    val lin = tf.reshape(mult, Shape(53149, 3))
    println("lin", lin)

    val res =  lin + pts

    val sess = Session()

    val ret = sess.run(fetches = Seq(res)).toTensor

    println("ret", ret(0).summarize())
    //println("mult", ret(1).summarize())



    val test = model.instance(inst.coefficients)

    for(n <- Seq(100,1000,10000)) {
      println(s"vtx $n", ret(0, n, ::).summarize())

      println(s"vtxgt $n", test.shape.position.pointData(n))
    }

    //println("res", res.summarize())
  }
}


