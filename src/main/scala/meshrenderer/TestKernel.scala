package meshrenderer

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.client.FeedMap
import org.platanios.tensorflow.api.{Op, Output, tf, _}
import org.platanios.tensorflow.api.ops.Gradients
import org.platanios.tensorflow.api.tensors.Tensor

object TestKernel {

  def main(args: Array[String]): Unit = {
    org.platanios.tensorflow.jni.TensorFlow.loadOpLibrary("lib/rasterize_triangles_kernel.so")

    val image_width = 227
    val image_height = 227

//    val vertices = tf.placeholder[Float](Shape(-1, 3))
//    val triangles = tf.placeholder[Int](Shape(-1, 3))
    val vertices = Tensor(0 until 30).toFloat.reshape(Shape(10, 3))
    val triangles = Tensor(Seq(0 until 15)).reshape(Shape(5, 3))

    val gradientFn: Gradients.GradientFn[Seq[Output[Any]], Seq[Output[Float]], Seq[Output[Any]], Seq[Output[Float]]] = rasterizeTrianglesGrad

    val inputs: Seq[Output[Any]] = Seq(vertices.toOutput, triangles.toOutput)

    val outs: Op[Seq[Output[Any]], Seq[Output[Float]]] = Op.Builder[Seq[Output[Any]], Seq[Output[Float]]](opType = "RasterizeTriangles", "rasterize_triangles", inputs, addAsIndividualInputs = true)
      .setAttribute("image_width", image_width)
      .setAttribute("image_height", image_height)
      .setGradientFn(gradientFn)
      .build()

    println(outs.outputsSeq)

    println(triangles.summarize())

    using(Session())(session => {
//      val feeds: FeedMap = Seq(FeedMap(Map(vertices -> v)), FeedMap(Map(triangles -> t)))

      val results = session.run(fetches = outs.outputsSeq(1))

      println(results.summarize())
    })
  }

  def rasterizeTrianglesGrad(op: Op[Seq[Output[Any]], Seq[Output[Float]]], outputGradients: Seq[Output[Float]]): Seq[Output[Float]] = {
    println("outputGradients", outputGradients.length)
    println("outputGradients", outputGradients.head)
    println("outputGradients", outputGradients(1))
    println("outputGradients", outputGradients(2))
    println("op.outputs", op.outputsSeq.head)
    println("op.outputs", op.outputsSeq(1))
    println("op.inputs", op.inputsSeq.length)
    println("op.inputs", op.inputsSeq.length)
    //outputGradients: dfdBarycentriCoordinates: Output, df_didsIgnored: Output, df_dzIgnored: Output
    // TODO: Find out how Op registration works in TF Scala 0.4
    val outGrad = Op.Builder[Seq[Output[Float]], Seq[Output[Float]]](opType = "RasterizeTrianglesGrad", "rasterizeTrianglesGrad",
      input = Seq(op.inputsSeq.head.toFloat, op.inputsSeq(1).toFloat, op.outputsSeq.head.toFloat, op.outputsSeq(1).toFloat, outputGradients.head))
      .setAttribute("image_width", op.longAttribute("image_width"))
      .setAttribute("image_height", op.longAttribute("image_height"))
      .build()

    println("outGrad", outGrad.outputsSeq.length, outGrad)
    Seq(outGrad.outputsSeq.head.toFloat, tf.identity(outGrad.outputsSeq.head.toFloat)) //zBuffer gradients missing but we need to supply something!
  }
}
