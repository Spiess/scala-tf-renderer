package meshrenderer

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.Gradients
import org.platanios.tensorflow.api.{Op, Output, tf, _}

/**
  * Test object to check that the rasterize triangles kernel is working correctly.
  * Tests both rasterization and gradient calculation.
  */
object TestKernel {

  def main(args: Array[String]): Unit = {
    // Why is this not accurate? Currently claims to be 1.12.0-rc0 when it is definitely 1.13.0-dev20181121 or similar.
    println("TensorFlow Version: " + org.platanios.tensorflow.jni.TensorFlow.version)

    val image_width = 227
    val image_height = 227

    val vertices: Output[Float] = tf.variable[Float]("vertices", Shape(20, 3))
    val triangles: Output[Int] = tf.variable[Int]("triangles", Shape(30, 3))

    val outs = Rasterizer.rasterize_triangles(vertices, triangles, image_width, image_height)

    println(outs)

    println("Has gradient: " + outs.barycetricImage.hasGradient)

    val xs = Seq(vertices)
    val ys: Seq[Output[Float]] = Seq(outs.barycetricImage)

    // Check that gradients can be collected
    val grad: Seq[OutputLike[Float]] = Gradients.gradients(ys, xs, Float)

    println(s"grad: $grad")

    using(Session())(session => {
      session.run(targets = tf.globalVariablesInitializer())

      val results = session.run(fetches = Seq(outs.barycetricImage, grad.head.toOutput))

      println(results.head.summarize())
      println(results(1).summarize())
    })
  }
}
