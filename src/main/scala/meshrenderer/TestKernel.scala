package meshrenderer

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.client.FeedMap
import org.platanios.tensorflow.api.{Op, Output, tf, _}
import org.platanios.tensorflow.api.ops.Gradients
import org.platanios.tensorflow.api.tensors.Tensor

object TestKernel {

  def main(args: Array[String]): Unit = {
    org.platanios.tensorflow.jni.TensorFlow.loadOpLibrary("lib/rasterize_triangles_kernel.so")

    // Why is this not accurate? Currently claims to be 1.12.0-rc0 when it is definitely 1.13.0-dev20181121 or similar.
    println("TensorFlow Version: " + org.platanios.tensorflow.jni.TensorFlow.version)

    val image_width = 227
    val image_height = 227

//    val vertices = tf.placeholder[Float](Shape(-1, 3))
//    val triangles = tf.placeholder[Int](Shape(-1, 3))
    val vertices = Tensor(Tensor(-1, -1, 1), Tensor(-1, -1, -1), Tensor(-1, 1, -1), Tensor(-1, 1, 1), Tensor(1, -1, 1), Tensor(1, -1, -1), Tensor(1, 1, -1), Tensor(1, 1, 1)).toFloat
    val triangles = Tensor(Tensor(0, 1, 2), Tensor(2, 3, 0), Tensor(3, 2, 6), Tensor(6, 7, 3), Tensor(7, 6, 5), Tensor(5, 4, 7), Tensor(4, 5, 1), Tensor(1, 0, 4), Tensor(5, 6, 2), Tensor(2, 1, 5), Tensor(7, 4, 0), Tensor(0, 3, 7))

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
