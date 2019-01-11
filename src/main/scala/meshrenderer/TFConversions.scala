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

  @deprecated
  def pointsToTensor[A](points: IndexedSeq[A])(implicit vectorizer: Vectorizer[A]): Tensor[Float] = {
    pointsToTensorNotTransposed(points).transpose()
  }

  def pointsToTensorNotTransposed[A](points: IndexedSeq[A])(implicit vectorizer: Vectorizer[A]): Tensor[Float] = {
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

  def image3dToTensor(image: PixelImage[RGB]): Tensor[Float] = {
//    val res = Array.fill(image.height, image.width, 3)(0.0f)
//    for (r <- 0 until image.height) {
//      for (c <- 0 until image.width) {
//        val px = image(c, r)
//        res(r)(c)(0) = px.r.toFloat
//        res(r)(c)(1) = px.g.toFloat
//        res(r)(c)(2) = px.b.toFloat
//      }
//    }
//    println(res)
    // TODO: Implement fast version with ByteBuffer
    val bufferCapacity = image.height * image.width * 3 * 4
    val buffer = ByteBuffer.allocateDirect(bufferCapacity).order(ByteOrder.nativeOrder())
    for (
      x <- 0 until image.width;
      y <- 0 until image.height
    ) {
      val rgb = image.valueAt(x, y)
      buffer.putFloat(rgb.r.toFloat)
      buffer.putFloat(rgb.g.toFloat)
      buffer.putFloat(rgb.b.toFloat)
    }
    buffer.flip()
    val ret = Tensor.fromBuffer[Float](Shape(image.width, image.height, 3), bufferCapacity, buffer)
    println(ret.summarize())

    ret(0)
  }

  def tensorImage3dToPixelImage(dataRaw: Tensor[Float], domain: PixelImageDomain): PixelImage[RGB] = {
    val data: Seq[Double] = dataRaw.entriesIterator.toIndexedSeq.map {
      _.toDouble
    }
    sequenceImage3dToPixelImage(data, domain)
  }

  def oneToOneTensorImage3dToPixelImage(dataRaw: Tensor[Float]): PixelImage[RGB] = {
    //require( == domain.width &&  == domain.height)
    val w = dataRaw.shape(1)
    val h = dataRaw.shape(0)
    val buffer = ImageBuffer.makeInitializedBuffer(w, h)(RGB.Black)
    var r = 0
    while (r < h) {
      var c = 0
      while (c < w) {
        buffer(c, r) = RGB(
          dataRaw(r, c, 0).entriesIterator.toIndexedSeq(0)
          ,
          dataRaw(r, c, 1).entriesIterator.toIndexedSeq(0)
          ,
          dataRaw(r, c, 2).entriesIterator.toIndexedSeq(0)
        )
        c += 1
      }
      r += 1
    }
    buffer.toImage
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
    val w = domain.width
    val h = domain.height
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