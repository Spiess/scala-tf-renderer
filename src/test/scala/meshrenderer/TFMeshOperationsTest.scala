package meshrenderer

import org.platanios.tensorflow.api._
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class TFMeshOperationsTest extends FlatSpec with Matchers {
  private val random = new Random(5234)

  private val numberOfPoints = 10
  private val batchSize = 2

  private val testPoints = Tensor((0 until batchSize * numberOfPoints * 3).map(_ => random.nextFloat())).reshape(Shape(batchSize, numberOfPoints, 3))
  private val testTriangles = Tensor((0 until 15).map(_ => random.nextInt(10))).reshape(Shape(5, 3))

  "The batchTriangleNormals method" should "calculate triangle normals" in {
    val nonBatchOutput = (0 until batchSize).map(i => TFMeshOperations.triangleNormals(testPoints(i, ::, ::), testTriangles))

    val batchOutput = tf.stack((0 until batchSize).map(i => TFMeshOperations.triangleNormals(testPoints(i, ::, ::), testTriangles)))

    assert(batchOutput.shape(0) == batchSize)

    val results = using(Session())(_.run(fetches = Seq(nonBatchOutput(0), batchOutput)))
    assert(results(1)(0, ::, ::) == results.head)
  }
}
