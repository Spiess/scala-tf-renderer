package meshrenderer

import java.nio.{ByteBuffer, ByteOrder}

import org.platanios.tensorflow.api.{Shape, Tensor, tf, _}
import scalismo.common.Vectorizer
import scalismo.faces.color.RGB
import scalismo.faces.image.{ImageBuffer, PixelImage, PixelImageDomain}
import scalismo.faces.parameters.{Camera, Pose}
import scalismo.geometry.{Point, Vector, _3D}

/**
  * Created by andreas on 8/15/18.
  *
  * Convert between tensorflow and scalismo-faces data structures.
  */
object TFConversions {

  // TODO: Update with better shape (3) or (1, 3)
  def pt2Output(pt: Point[_3D]): Tensor[Float] = {
    Tensor(pt.x.toFloat, pt.y.toFloat, pt.z.toFloat).reshape(Shape(3, 1))
  }

  def vec2Output(pt: Vector[_3D]): Output[Float] = {
    Tensor(pt.x.toFloat, pt.y.toFloat, pt.z.toFloat).reshape(Shape(3, 1))
  }

  def pt2Tensor(pt: Point[_3D]): Tensor[Float] = {
    Tensor(pt.x.toFloat, pt.y.toFloat, pt.z.toFloat).reshape(Shape(3, 1))
  }

  def vec2Tensor(pt: Vector[_3D]): Tensor[Float] = {
    Tensor(Seq(pt.x.toFloat, pt.y.toFloat, pt.z.toFloat)).reshape(Shape(3, 1))
  }

  @deprecated("Uses deprecated point ordering (dimensions, numPoints).", "0.1-SNAPSHOT")
  def pointsToTensorTransposed[A](points: IndexedSeq[A])(implicit vectorizer: Vectorizer[A]): Tensor[Float] = {
    pointsToTensor(points).transpose()
  }

  def pointsToTensor[A](points: IndexedSeq[A])(implicit vectorizer: Vectorizer[A]): Tensor[Float] = {
    val d = vectorizer.dim
    val bufferCapacity = points.length * d * 4
    val buffer = ByteBuffer.allocateDirect(bufferCapacity).order(ByteOrder.nativeOrder())

    for (i <- points.indices) {
      val vec = vectorizer.vectorize(points(i))
      for (j <- 0 until d) {
        buffer.putFloat(vec(j).toFloat)
      }
    }

    buffer.flip()
    Tensor.fromBuffer[Float](Shape(points.length, 3), bufferCapacity, buffer)
  }

  // TODO: Change image conversion from (height, width, dimensions) to (width, height, dimensions)
  def image3dToTensor(image: PixelImage[RGB]): Tensor[Float] = {
    val bufferCapacity = image.height * image.width * 3 * 4
    val buffer = ByteBuffer.allocateDirect(bufferCapacity).order(ByteOrder.nativeOrder())
    for (
      y <- 0 until image.height;
      x <- 0 until image.width
    ) {
      val rgb = image.valueAt(x, y)
      buffer.putFloat(rgb.r.toFloat)
      buffer.putFloat(rgb.g.toFloat)
      buffer.putFloat(rgb.b.toFloat)
    }
    buffer.flip()
    val ret = Tensor.fromBuffer[Float](Shape(image.height, image.width, 3), bufferCapacity, buffer)

    ret
  }

  def tensorImage3dToPixelImage(dataRaw: Tensor[Float], domain: PixelImageDomain): PixelImage[RGB] = {
    val data: Seq[Double] = dataRaw.entriesIterator.toIndexedSeq.map {
      _.toDouble
    }
    sequenceImage3dToPixelImage(data, domain)
  }

  /**
    * Converts a tensor to a pixel image.
    *
    * @param dataRaw Image value tensor with shape (width, height, 3)
    */
  def oneToOneTensorImage3dToPixelImage(dataRaw: Tensor[Float]): PixelImage[RGB] = {
    val entries = dataRaw.entriesIterator.toArray
    val data = (0 until dataRaw.size.toInt by 3).map(i => RGB(entries(i), entries(i + 1), entries(i + 2)))
    PixelImage(PixelImageDomain(dataRaw.shape(0), dataRaw.shape(1)), data)
  }

  def tensorImage3dIntToPixelImage(dataRaw: Tensor[Int], domain: PixelImageDomain): PixelImage[RGB] = {
    val data: Seq[Double] = dataRaw.entriesIterator.toIndexedSeq.map {
      _.toDouble
    }
    sequenceImage3dToPixelImage(data, domain)
  }

  def tensorIntImage3dToPixelImage(dataRaw: Tensor[Int], domain: PixelImageDomain): PixelImage[RGB] = {
    val data: Seq[Double] = dataRaw.entriesIterator.toIndexedSeq.map {
      _.toDouble
    }
    sequenceImage3dToPixelImage(data, domain)
  }

  def sequenceImage3dToPixelImage(data: Seq[Double], domain: PixelImageDomain): PixelImage[RGB] = {
    require(data.size == domain.width * domain.height * 3)
    PixelImage(domain.width, domain.height, (x, y) => {
      val first = domain.index(x, y) * 3
      val second = domain.index(x, y) * 3 + 1
      val third = domain.index(x, y) * 3 + 2
      RGB(data(first), data(second), data(third))
    })
  }

  def tensorImage1dToPixelImage(dataRaw: Tensor[Float], domain: PixelImageDomain): PixelImage[Double] = {
    val data: Seq[Double] = dataRaw.entriesIterator.toIndexedSeq.map {
      _.toDouble
    }
    sequenceImage1dToPixelImage(data, domain)
  }

  def sequenceImage1dToPixelImage(data: Seq[Double], domain: PixelImageDomain): PixelImage[Double] = {
    require(data.size == domain.width * domain.height)
    PixelImage(domain.width, domain.height, (x, y) => {
      data(domain.index(x, y))
    })
  }
}

// TODO: Remove the need for TF*Tensor (just use tensor version because tensors are output too)

//row vectors
case class TFPose(rotation: Output[Float], translation: Output[Float]) {
  val yaw: Output[Float] = rotation(0, 0)
  val pitch: Output[Float] = rotation(0, 1)
  val roll: Output[Float] = rotation(0, 2)
}

case class TFPoseTensor(rotation: Tensor[Float], translation: Tensor[Float]) {
  val yaw: Tensor[Float] = rotation(0, 0)
  val pitch: Tensor[Float] = rotation(0, 1)
  val roll: Tensor[Float] = rotation(0, 2)
}

object TFPose {
  def apply(pose: Pose) = TFPoseTensor(
    Tensor(Seq(pose.yaw.toFloat, pose.pitch.toFloat, pose.roll.toFloat)),
    TFConversions.vec2Tensor(pose.translation)
  )

  def apply(pose: TFPoseTensor): TFPose = TFPose(pose.rotation, pose.translation)
}

case class TFCamera(parameters: Output[Float]) {
  val focalLength: Output[Float] = parameters(0, 0)
  val principalPointX: Output[Float] = parameters(0, 1)
  val principalPointY: Output[Float] = parameters(0, 2)
  val sensorSizeX: Output[Float] = parameters(0, 3)
  val sensorSizeY: Output[Float] = parameters(0, 4)
  val near: Output[Float] = parameters(0, 5)
  val far: Output[Float] = parameters(0, 6)
}

case class TFCameraTensor(parameters: Tensor[Float]) {
  val focalLength: Tensor[Float] = parameters(0, 0)
  val principalPointX: Tensor[Float] = parameters(0, 1)
  val principalPointY: Tensor[Float] = parameters(0, 2)
  val sensorSizeX: Tensor[Float] = parameters(0, 3)
  val sensorSizeY: Tensor[Float] = parameters(0, 4)
  val near: Tensor[Float] = parameters(0, 5)
  val far: Tensor[Float] = parameters(0, 6)
}

object TFCamera {
  def apply(cam: Camera): TFCameraTensor = TFCameraTensor(
    Tensor(Seq(
      cam.focalLength.toFloat,
      cam.principalPoint.x.toFloat, cam.principalPoint.y.toFloat,
      cam.sensorSize.x.toFloat, cam.sensorSize.y.toFloat,
      cam.near.toFloat,
      cam.far.toFloat
    )
    ) //row vector
  )

  def apply(cam: TFCameraTensor): TFCamera = TFCamera(cam.parameters)
}