package meshrenderer

import org.platanios.tensorflow.api.{tf, _}
import scalismo.faces.parameters.{ImageSize, Pose, RenderParameter}
import scalismo.geometry.{Point, Vector, _3D}
import scalismo.mesh.TriangleMesh3D
import scalismo.utils.Random


object Transformations {

  @deprecated("Uses deprecated point ordering (dimensions, numPoints).", "0.1-SNAPSHOT")
  def poseTransform(pt: Output[Float], pitch: Output[Float], yaw: Output[Float], roll: Output[Float], translation: Output[Float]): Output[Float] = {
    poseRotationTransform(pt, pitch, yaw, roll) + translation
  }

  def batchPoseTransform(points: Output[Float], pitch: Output[Float], yaw: Output[Float], roll: Output[Float], translation: Output[Float]): Output[Float] = {
    batchPoseRotationTransform(points, pitch, yaw, roll) + translation
  }

  @deprecated("Uses deprecated point ordering (dimensions, numPoints).", "0.1-SNAPSHOT")
  def poseRotationTransform(pt: Output[Float], pitch: Output[Float], yaw: Output[Float], roll: Output[Float]): Output[Float] = {
    val X = {
      val pitchc = tf.cos(pitch)
      val pitchs = tf.sin(pitch)
      val temp: Seq[Output[Float]] = Seq(1f, 0f, 0f,
        0f, pitchc, -pitchs,
        0f, pitchs, pitchc
      )
      tf.stack(temp).reshape(Shape(3, 3))
    }

    val Y = {
      val yawc = tf.cos(yaw)
      val yaws = tf.sin(yaw)
      val temp: Seq[Output[Float]] = Seq(yawc, 0f, yaws,
        0f, 1f, 0f,
        -yaws, 0f, yawc)
      tf.stack(temp).reshape(Shape(3, 3))
    }

    val Z = {
      val rollc = tf.cos(roll)
      val rolls = tf.sin(roll)
      val temp: Seq[Output[Float]] = Seq(rollc, -rolls, 0f,
        rolls, rollc, 0f,
        0f, 0f, 1f
      )
      tf.stack(temp).reshape(Shape(3, 3))
    }
    val multX = tf.matmul(X, pt)

    val multY = tf.matmul(Y, multX)

    val multZ = tf.matmul(Z, multY)

    multZ
  }

  def batchPoseRotationTransform(points: Output[Float], pitch: Output[Float], yaw: Output[Float], roll: Output[Float]): Output[Float] = {
    // Define batch-length ones and zeros tensors for the rotation matrices
    val ones = Output.onesLike(pitch)
    val zeros = Output.zerosLike(pitch)

    /*
    Each rotation matrix is constructed as (x, y, batch) but should be (batch, x, y). Because of this, each matrix needs
    to be transposed before it is used.
     */
    val X = {
      val pitchc = tf.cos(pitch)
      val pitchs = tf.sin(pitch)
      Output(
        Output(ones, zeros, zeros),
        Output(zeros, pitchc, pitchs),
        Output(zeros, -pitchs, pitchc)
      ).transpose(Tensor(2, 0, 1))
    }

    val Y = {
      val yawc = tf.cos(yaw)
      val yaws = tf.sin(yaw)
      Output(
        Output(yawc, zeros, -yaws),
        Output(zeros, ones, zeros),
        Output(yaws, zeros, yawc)
      ).transpose(Tensor(2, 0, 1))
    }

    val Z = {
      val rollc = tf.cos(roll)
      val rolls = tf.sin(roll)
      Output(
        Output(rollc, rolls, zeros),
        Output(-rolls, rollc, zeros),
        Output(zeros, zeros, ones)
      ).transpose(Tensor(2, 0, 1))
    }

    val multX = tf.matmul(points, X)

    val multY = tf.matmul(multX, Y)

    val multZ = tf.matmul(multY, Z)

    multZ
  }

  @deprecated("Uses deprecated point ordering (dimensions, numPoints).", "0.1-SNAPSHOT")
  def projectiveTransformation(pt: Output[Float],
                               near: Output[Float], far: Output[Float],
                               sensorSizeX: Output[Float], sensorSizeY: Output[Float],
                               focalLength: Output[Float],
                               principalPointX: Output[Float], principalPointY: Output[Float]): Output[Float] = {
    val f = far
    val n = near
    val ppx = principalPointX
    val ppy = principalPointY
    val fl = focalLength
    val ssx = sensorSizeX
    val ssy = sensorSizeY
    val px = pt(0, ::)
    val py = pt(1, ::)
    val pz = pt(2, ::)

    val newpx = ppx - (px * 2f * fl) / (pz * ssx)
    val newpy = ppy - (py * 2f * fl) / (pz * ssy)
    val newpz = (f * n * 2f / pz + n + f) / (f - n)
    tf.stack(Seq(newpx, newpy, newpz), axis = 1).transpose() //.reshape(Shape(3,2))
  }

  def batchProjectiveTransformation(points: Output[Float],
                                    near: Output[Float], far: Output[Float],
                                    sensorSize: Output[Float],
                                    focalLength: Output[Float],
                                    principalPoint: Output[Float]): Output[Float] = {
    val px = points(::, ::, 0)
    val py = points(::, ::, 1)
    val pz = points(::, ::, 2)

    val ppx = principalPoint(::, 0).expandDims(1)
    val ppy = principalPoint(::, 1).expandDims(1)

    val ssx = sensorSize(::, 0).expandDims(1)
    val ssy = sensorSize(::, 1).expandDims(1)

    val newpx = ppx - (px * 2f * focalLength) / (pz * ssx)
    val newpy = ppy - (py * 2f * focalLength) / (pz * ssy)
    val newpz = (far * near * 2f / pz + near + far) / (far - near)
    tf.stack(Seq(newpx, newpy, newpz), axis = 0).transpose(Tensor(1, 2, 0))
  }

  def batchProjectiveTransformation2D(points: Output[Float],
                                    sensorSize: Output[Float],
                                    focalLength: Output[Float],
                                    principalPoint: Output[Float]): Output[Float] = {
    val px = points(::, ::, 0)
    val py = points(::, ::, 1)
    val pz = points(::, ::, 2)

    val ppx = principalPoint(::, 0).expandDims(1)
    val ppy = principalPoint(::, 1).expandDims(1)

    val ssx = sensorSize(::, 0).expandDims(1)
    val ssy = sensorSize(::, 1).expandDims(1)

    val newpx = ppx - (px * 2f * focalLength) / (pz * ssx)
    val newpy = ppy - (py * 2f * focalLength) / (pz * ssy)
    tf.stack(Seq(newpx, newpy), axis = 0).transpose(Tensor(1, 2, 0))
  }

  @deprecated("Uses deprecated point ordering (dimensions, numPoints).", "0.1-SNAPSHOT")
  def screenTransformation(pt: Output[Float], width: Output[Float], height: Output[Float]): Output[Float] = {
    val n = 0f
    val f = 1f
    val px = pt(0, ::)
    val py = pt(1, ::)
    val pz = pt(2, ::)
    val newpx = (px + 1f) * width / 2f
    val newpy = (-py + 1f) * height / 2f
    val newpz = pz * (f - n) / 2f + (f + n) / 2f
    tf.stack(Seq(newpx, newpy, newpz), axis = 1).transpose()
  }

  def batchScreenTransformation(pt: Output[Float], width: Output[Float], height: Output[Float]): Output[Float] = {
    val n = 0f
    val f = 1f
    val px = pt(::, ::, 0)
    val py = pt(::, ::, 1)
    val pz = pt(::, ::, 2)
    val newpx = (px + 1f) * width / 2f
    val newpy = (-py + 1f) * height / 2f
    val newpz = pz * (f - n) / 2f + (f + n) / 2f
    tf.stack(Seq(newpx, newpy, newpz), axis = 0).transpose(Tensor(1, 2, 0))
  }

  def batchScreenTransformation2D(pt: Output[Float], width: Output[Float], height: Output[Float]): Output[Float] = {
    val px = pt(::, ::, 0)
    val py = pt(::, ::, 1)
    val newpx = (px + 1f) * width / 2f
    val newpy = (-py + 1f) * height / 2f
    tf.stack(Seq(newpx, newpy), axis = 0).transpose(Tensor(1, 2, 0))
  }

  // TODO: Change the way points are stored: change (dimension, point) to (point dimension) because this is incredibly unintuitive and often inconvenient

  @deprecated("Uses deprecated point ordering (dimensions, numPoints).", "0.1-SNAPSHOT")
  def objectToNDC(pts: Output[Float], pose: TFPose, camera: TFCamera): Output[Float] = {
    val poseTransformed = poseTransform(pts, pose.pitch, pose.yaw, pose.roll, pose.translation)

    projectiveTransformation(
      poseTransformed,
      camera.near, camera.far,
      camera.sensorSizeX, camera.sensorSizeY,
      camera.focalLength,
      camera.principalPointX, camera.principalPointY
    )
  }

  /**
    * Transforms a batch of point lists to a batch of normalized device coordinates.
    *
    * @param points         mesh points of shape (batchSize, numPoints, pointDimensions [x, y, z])
    * @param roll           roll values of shape (batchSize)
    * @param pitch          pitch values of shape (batchSize)
    * @param yaw            yaw values of shape (batchSize)
    * @param translation    translation of shape (batchSize, 1, pointDimensions [x, y, z])
    * @param cameraNear     values of shape (1)
    * @param cameraFar      values of shape (1)
    * @param sensorSize     values of shape (batchSize, 2 [sensorWidth, sensorHeight])
    * @param focalLength    values of shape (1)
    * @param principalPoint values of shape (batchSize, 2 [principalPointX, principalPointY])
    * @return normalized device coordinates of shape (batchSize, numPoints, pointDimensions [x, y, z])
    */
  def batchPointsToNDC(
                        points: Output[Float],
                        roll: Output[Float], pitch: Output[Float], yaw: Output[Float],
                        translation: Output[Float],
                        cameraNear: Output[Float], cameraFar: Output[Float],
                        sensorSize: Output[Float],
                        focalLength: Output[Float],
                        principalPoint: Output[Float]
                      ): Output[Float] = {
    val poseTransformed = batchPoseTransform(points, pitch, yaw, roll, translation)

    batchProjectiveTransformation(poseTransformed, cameraNear, cameraFar, sensorSize, focalLength, principalPoint)
  }

  /**
    * Transforms a batch of point lists to a batch of 2D normalized device coordinates. By only calculating the 2D
    * landmark positions, this function requires fewer parameters and is more efficient.
    *
    * @param points         mesh points of shape (batchSize, numPoints, pointDimensions [x, y, z])
    * @param roll           roll values of shape (batchSize)
    * @param pitch          pitch values of shape (batchSize)
    * @param yaw            yaw values of shape (batchSize)
    * @param translation    translation of shape (batchSize, 1, pointDimensions [x, y, z])
    * @param sensorSize     values of shape (batchSize, 2 [sensorWidth, sensorHeight])
    * @param focalLength    values of shape (1)
    * @param principalPoint values of shape (batchSize, 2 [principalPointX, principalPointY])
    * @return normalized device coordinates of shape (batchSize, numPoints, pointDimensions [x, y, z])
    */
  def batchPointsToNDC2D(
                        points: Output[Float],
                        roll: Output[Float], pitch: Output[Float], yaw: Output[Float],
                        translation: Output[Float],
                        sensorSize: Output[Float],
                        focalLength: Output[Float],
                        principalPoint: Output[Float]
                      ): Output[Float] = {
    val poseTransformed = batchPoseTransform(points, pitch, yaw, roll, translation)

    batchProjectiveTransformation2D(poseTransformed, sensorSize, focalLength, principalPoint)
  }

  /** normalized device coordinates of rasterizer are different than in the scalismo-faces renderer. */
  @deprecated("Uses deprecated point ordering (dimensions, numPoints).", "0.1-SNAPSHOT")
  def transposedNdcToTFNdc(pts: Output[Float], width: Int, height: Int): Output[Float] = {
    val px = pts(0, ::)
    val py = pts(1, ::)
    val pz = pts(2, ::)
    val yoffset = 1f / height
    //there is a 0.5 pixel diagonal shift between the tf mesh rasterizer and the scalismo-faces renderer,
    val xoffset = 1f / width
    //tf.stack(Seq(-py+offset, px, pz), axis=1).transpose()
    tf.stack(Seq(px + xoffset, -py + yoffset, pz))
  }

  /**
    * Converts normalized device coordinates from the space used in Scalismo to that used by the TensorFlow rasterizer.
    *
    * @param ndc normalized device coordinates in the Scalismo space of shape (numPoints, dimensions)
    * @return normalized device coordinates in the TensorFlow rasterizer space of shape (numPoints, dimensions)
    */
  def ndcToTFNdc(ndc: Output[Float], imageWidth: Int, imageHeight: Int): Output[Float] = {
    //there is a 0.5 pixel diagonal shift between the tf mesh rasterizer and the scalismo-faces renderer,
    val multiplier = Output(1f, -1f, 1f)
    val offset = Output(1f / imageWidth, 1f / imageHeight, 0f)
    ndc * multiplier + offset
  }

  def renderTransform(pts: Output[Float], pose: TFPose, camera: TFCamera, width: Output[Float], height: Output[Float]): Output[Float] = {
    val toNDC = objectToNDC(pts, pose, camera)
    screenTransformation(toNDC, width, height)
  }

  def bccScreenToWorldCorrection(bccScreen: Output[Float], zBuffer: Output[Float], zCoordinatesPerVertex: Output[Float]): Output[Float] = {
    val d = zBuffer
    val z1 = zCoordinatesPerVertex(::, ::, 0)
    val z2 = zCoordinatesPerVertex(::, ::, 1)
    val z3 = zCoordinatesPerVertex(::, ::, 2)
    //val a = bccScreen(::, ::, 0)
    val b = bccScreen(::, ::, 1)
    val c = bccScreen(::, ::, 2)
    //val d = z2*z3 + z3*b * (z1-z2) + z2*c * (z1-z3)
    val dIsZero = tf.where(tf.equal(d, 0f))
    val newBccDIsZero = bccScreen * dIsZero.toFloat
    val newB = z1 * z3 * b / d
    val newC = z1 * z2 * c / d
    val newBCC = tf.stack(Seq(1 - newB - newC, newB, newC))
    newBCC * (1f - dIsZero.toFloat) + newBccDIsZero
  }
}