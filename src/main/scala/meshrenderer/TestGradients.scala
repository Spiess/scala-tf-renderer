package meshrenderer

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.Gradients
import org.platanios.tensorflow.api.{Output, tf, _}

/**
  * Object to test the gradient function and how it behaves with variables.
  */
object TestGradients {
  def main(args: Array[String]): Unit = {
    val aVar = tf.variable[Float]("a", Shape(20, 3))
    val bVar = tf.variable[Float]("b", Shape(20, 3))

    val a: Output[Float] = aVar
    val b: Output[Float] = bVar

    val xs: Seq[Output[Float]] = Seq(a, b)
    val ys: Seq[Output[Float]] = Seq(a - b)

    val gradientsAB = Gradients.gradients(ys, xs, Float)

    val c = tf.variable[Float]("c", Shape(20, 3))
    val d = tf.variable[Float]("d", Shape(20, 3))

    val nxs: Seq[Output[Float]] = Seq(c, d)
    val nys: Seq[Output[Float]] = Seq(c - d)

    val gradientsCD = Gradients.gradients(nys, nxs, Float)

    println(gradientsAB)
    println(gradientsCD)

    using(Session())(session => {
      session.run(targets = tf.globalVariablesInitializer())

      {
        val res = session.run(fetches = Seq(a, b))
        println(s"a: ${res.head.summarize()}, b: ${res(1).summarize()}")
      }

      val assignmentOp = aVar.assign(tf.fill(Shape(20, 3))(5f))
      session.run(targets = assignmentOp)

      {
        val res = session.run(fetches = Seq(a, b))
        println(s"a: ${res.head.summarize()}, b: ${res(1).summarize()}")
      }
    })
  }
}
