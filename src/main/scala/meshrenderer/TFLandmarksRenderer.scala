package meshrenderer

import org.platanios.tensorflow.api.{tf, _}
import scalismo.faces.momo.{MoMoBasic, MoMoExpress}

/**
  * Class for determining landmark positions in pixel coordinates on an image.
  *
  * @param basisMatrix  the basis matrix of the chosen model
  * @param parameterStd the parameter standard deviations of the chosen model
  * @param meanMesh     the mean mesh points of the used morphable model
  */
case class TFLandmarksRenderer(basisMatrix: Tensor[Float], parameterStd: Tensor[Float], meanMesh: Tensor[Float]) {

  /**
    * Calculates the point positions for the model instance defined by the given parameters. Only returns the points
    * specified during creation of this TFLandmarksRenderer.
    * <br>
    * Tensor version which immediately calculates the result.
    *
    * @param parameters model parameters
    * @return [[Tensor]] of mesh points
    */
  def getInstance(parameters: Tensor[Float]): Tensor[Float] = {
    val mesh = getInstance(parameters.toOutput)

    using(Session())(_.run(fetches = mesh))
  }

  /**
    * Calculates the point positions for the model instance defined by the given parameters. Only returns the points
    * specified during creation of this TFLandmarksRenderer.
    *
    * @param parameters model parameters
    * @return [[Output]] of mesh points
    */
  def getInstance(parameters: Output[Float]): Output[Float] = {
    val offsets = tf.matmul(basisMatrix, parameters * parameterStd).reshape(Shape(-1, 3)).transpose()
    meanMesh + offsets
  }

  def batchGetInstance(parameters: Tensor[Float]): Tensor[Float] = {
    val mesh = batchGetInstance(parameters.toOutput)

    using(Session())(_.run(fetches = mesh))
  }

  def batchGetInstance(parameters: Output[Float]): Output[Float] = {
    val offsets = tf.matmul(parameters * parameterStd, basisMatrix).reshape(Shape(parameters.shape(0), basisMatrix.shape(1) / 3, 3))
    meanMesh + offsets
  }

  /**
    * Calculates landmark positions in the rendered image for the given parameters.
    * <br>
    * Tensor version which immediately calculates the result.
    *
    * @param parameters  model parameters
    * @param pose        the pose of the model instance
    * @param camera      the camera for which to project the landmarks
    * @param imageWidth  the width of the rendered image
    * @param imageHeight the height of the rendered image
    * @return [[Tensor]] of projected landmark positions
    */
  def calculateLandmarks(parameters: Tensor[Float], pose: TFPose, camera: TFCamera, imageWidth: Int, imageHeight: Int): Tensor[Float] = {
    val landmarks = calculateLandmarks(parameters.toOutput, pose, camera, imageWidth, imageHeight)

    using(Session())(_.run(fetches = landmarks))
  }

  /**
    * Calculates landmark positions in the rendered image for the given parameters.
    *
    * @param parameters  model parameters
    * @param pose        the pose of the model instance
    * @param camera      the camera for which to project the landmarks
    * @param imageWidth  the width of the rendered image
    * @param imageHeight the height of the rendered image
    * @return [[Output]] of projected landmark positions
    */
  def calculateLandmarks(parameters: Output[Float], pose: TFPose, camera: TFCamera, imageWidth: Int, imageHeight: Int): Output[Float] = {
    val points = getInstance(parameters)
    projectPointsOnImage(points, pose, camera, imageWidth, imageHeight)
  }

  def batchCalculateLandmarks(parameters: Tensor[Float], pose: TFPose, camera: TFCamera, imageWidth: Int, imageHeight: Int): Tensor[Float] = {
    val landmarks = batchCalculateLandmarks(parameters.toOutput, pose, camera, imageWidth, imageHeight)

    using(Session())(_.run(fetches = landmarks))
  }

  def batchCalculateLandmarks(parameters: Output[Float], pose: TFPose, camera: TFCamera, imageWidth: Int, imageHeight: Int): Output[Float] = {
    val points = batchGetInstance(parameters)

    batchProjectPointsOnImage(points, pose, camera, imageWidth, imageHeight)
  }

  /**
    * Projects the given points onto an image given the pose, camera and image parameters.
    *
    * @param points      points to be projected onto the image
    * @param pose        the pose of the model instance
    * @param camera      the camera for which to project the landmarks
    * @param imageWidth  the width of the rendered image
    * @param imageHeight the height of the rendered image
    * @return [[Output]] of projected landmark positions
    */
  def projectPointsOnImage(points: Output[Float], pose: TFPose, camera: TFCamera, imageWidth: Int, imageHeight: Int): Output[Float] = {
    val normalizedDeviceCoordinates = Transformations.objectToNDC(points, pose, camera)
    Transformations.screenTransformation(normalizedDeviceCoordinates, imageWidth, imageHeight)
  }

  def batchProjectPointsOnImage(points: Output[Float], pose: TFPose, camera: TFCamera, imageWidth: Int, imageHeight: Int): Output[Float] = {
    val translation = pose.translation.transpose().expandDims(0).tile(Tensor(2, 1, 1))
    val sensorSize = Output(camera.sensorSizeX, camera.sensorSizeY)
    val principalPoint = Output(camera.principalPointX, camera.principalPointY).expandDims(0).tile(Tensor(2, 1))
    val roll = Output(pose.roll, -0.1f)
    val pitch = Output(pose.pitch, 0.4f)
    val yaw = Output(pose.yaw, 0.7f)

    val normalizedDeviceCoordinates = Transformations.pointsToNDCBatch(points, roll, pitch, yaw,
      translation, camera.near, camera.far, sensorSize, camera.focalLength, principalPoint)

    Transformations.batchScreenTransformation(normalizedDeviceCoordinates, imageWidth, imageHeight)
  }
}

object TFLandmarksRenderer {
  /**
    * Creates a TFLandmarksRenderer for basic morphable models.
    * The landmarks renderer can be restricted to a specific set of pointIds for efficiency. In that case the results
    * [[TFLandmarksRenderer.getInstance]] and [[TFLandmarksRenderer.calculateLandmarks]] will be
    * only the specified points in the specified order.
    *
    * @param momo             model to use
    * @param landmarkPointIds point ids of the desired landmarks or null for all points
    */
  def apply(momo: MoMoBasic, landmarkPointIds: IndexedSeq[Int]): TFLandmarksRenderer = {
    val shapeBasis = {
      val fullShapeBasis = TFMoMoConversions.toTensor(momo.shape.basisMatrix)

      if (landmarkPointIds != null) {
        val basisIndices = landmarkPointIds.flatMap(i => Seq(i * 3, i * 3 + 1, i * 3 + 2))
        fullShapeBasis.gather(basisIndices, 0)
      } else {
        fullShapeBasis
      }
    }

    val shapeStd = TFMoMoConversions.toTensor(momo.shape.variance.map(math.sqrt)).transpose()

    val meanPoints = TFConversions.pointsToTensor(landmarkPointIds.collect(momo.mean.shape.position.pointData))

    TFLandmarksRenderer(shapeBasis, shapeStd, meanPoints)
  }

  def notTransposed(momo: MoMoBasic, landmarkPointIds: IndexedSeq[Int]): TFLandmarksRenderer = {
    val shapeBasis = {
      val fullShapeBasis = TFMoMoConversions.toTensorNotTransposed(momo.shape.basisMatrix)

      if (landmarkPointIds != null) {
        val basisIndices = landmarkPointIds.flatMap(i => Seq(i * 3, i * 3 + 1, i * 3 + 2))
        fullShapeBasis.gather(basisIndices, 1)
      } else {
        fullShapeBasis
      }
    }

    val shapeStd = TFMoMoConversions.toTensor(momo.shape.variance.map(math.sqrt))

    val meanPoints = TFConversions.pointsToTensorNotTransposed(landmarkPointIds.collect(momo.mean.shape.position.pointData))

    TFLandmarksRenderer(shapeBasis, shapeStd, meanPoints)
  }

  /**
    * Creates a TFLandmarksRenderer for expression models.
    * The landmarks renderer can be restricted to a specific set of pointIds for efficiency. In that case the results
    * [[TFLandmarksRenderer.getInstance]] and [[TFLandmarksRenderer.calculateLandmarks]] will be
    * only the specified points in the specified order.
    *
    * @param momo             model to use
    * @param landmarkPointIds point ids of the desired landmarks or null for all points
    */
  def apply(momo: MoMoExpress, landmarkPointIds: IndexedSeq[Int]): TFLandmarksRenderer = {
    val (shapeBasis, expressionBasis) = {
      val fullShapeBasis = TFMoMoConversions.toTensor(momo.shape.basisMatrix)
      val fullExpressionBasis = TFMoMoConversions.toTensor(momo.expression.basisMatrix)

      if (landmarkPointIds != null) {
        val basisIndices = landmarkPointIds.flatMap(i => Seq(i * 3, i * 3 + 1, i * 3 + 2))
        (fullShapeBasis.gather(basisIndices, 0), fullExpressionBasis.gather(basisIndices, 0))
      } else {
        (fullShapeBasis, fullExpressionBasis)
      }
    }

    val shapeStd = TFMoMoConversions.toTensor(momo.shape.variance.map(math.sqrt)).transpose()
    val expressionStd = TFMoMoConversions.toTensor(momo.expression.variance.map(math.sqrt)).transpose()

    val (combinedBasis, combinedStd) = using(Session())(session => {
      val bases = Seq(shapeBasis.toOutput, expressionBasis.toOutput)
      val standardDevs = Seq(shapeStd.toOutput, expressionStd.toOutput)

      val results = session.run(fetches = Seq(tf.concatenate(bases, axis = 1), tf.concatenate(standardDevs, axis = 0)))
      (results.head, results.last)
    })

    val meanPoints = TFConversions.pointsToTensor(landmarkPointIds.collect(momo.mean.shape.position.pointData))

    TFLandmarksRenderer(combinedBasis, combinedStd, meanPoints)
  }

  def notTransposed(momo: MoMoExpress, landmarkPointIds: IndexedSeq[Int]): TFLandmarksRenderer = {
    val (shapeBasis, expressionBasis) = {
      val fullShapeBasis = TFMoMoConversions.toTensorNotTransposed(momo.shape.basisMatrix)
      val fullExpressionBasis = TFMoMoConversions.toTensorNotTransposed(momo.expression.basisMatrix)

      if (landmarkPointIds != null) {
        val basisIndices = landmarkPointIds.flatMap(i => Seq(i * 3, i * 3 + 1, i * 3 + 2))
        (fullShapeBasis.gather(basisIndices, 1), fullExpressionBasis.gather(basisIndices, 1))
      } else {
        (fullShapeBasis, fullExpressionBasis)
      }
    }

    val shapeStd = TFMoMoConversions.toTensor(momo.shape.variance.map(math.sqrt))
    val expressionStd = TFMoMoConversions.toTensor(momo.expression.variance.map(math.sqrt))

    val (combinedBasis, combinedStd) = using(Session())(session => {
      val bases = Seq(shapeBasis.toOutput, expressionBasis.toOutput)
      val standardDevs = Seq(shapeStd.toOutput, expressionStd.toOutput)

      val results = session.run(fetches = Seq(tf.concatenate(bases, axis = 0), tf.concatenate(standardDevs, axis = 1)))
      (results.head, results.last)
    })

    val meanPoints = TFConversions.pointsToTensorNotTransposed(landmarkPointIds.collect(momo.mean.shape.position.pointData))

    TFLandmarksRenderer(combinedBasis, combinedStd, meanPoints)
  }
}