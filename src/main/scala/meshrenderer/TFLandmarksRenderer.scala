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
  @deprecated("Uses deprecated point ordering (dimensions, numPoints).", "0.1-SNAPSHOT")
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
  @deprecated("Uses deprecated point ordering (dimensions, numPoints).", "0.1-SNAPSHOT")
  def getInstance(parameters: Output[Float]): Output[Float] = {
    val offsets = tf.matmul(basisMatrix, parameters * parameterStd).reshape(Shape(-1, 3)).transpose()
    meanMesh + offsets
  }

  /**
    * Calculates the point positions for the model instance defined by the given parameters. Only returns the points
    * specified during creation of this TFLandmarksRenderer. Version for batches of parameters.
    * <br>
    * Tensor version which immediately calculates the result.
    *
    * @param parameters batch of model parameters with shape (batchSize, numParameters)
    * @return [[Tensor]] of mesh points with shape (batchSize, numPoints, pointDimensions [x, y, z])
    */
  def batchGetInstance(parameters: Tensor[Float]): Tensor[Float] = {
    val mesh = batchGetInstance(parameters.toOutput)

    using(Session())(_.run(fetches = mesh))
  }

  /**
    * Calculates the point positions for the model instance defined by the given parameters. Only returns the points
    * specified during creation of this TFLandmarksRenderer. Version for batches of parameters.
    *
    * @param parameters batch of model parameters with shape (batchSize, numParameters)
    * @return [[Output]] of mesh points with shape (batchSize, numPoints, pointDimensions [x, y, z])
    */
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
  @deprecated("Uses deprecated point ordering (dimensions, numPoints).", "0.1-SNAPSHOT")
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
  @deprecated("Uses deprecated point ordering (dimensions, numPoints).", "0.1-SNAPSHOT")
  def calculateLandmarks(parameters: Output[Float], pose: TFPose, camera: TFCamera, imageWidth: Int, imageHeight: Int): Output[Float] = {
    val points = getInstance(parameters)
    projectPointsOnImage(points, pose, camera, imageWidth, imageHeight)
  }

  /**
    * Calculates landmark positions in the rendered image for the given parameters. Version for batches of parameters.
    * <br>
    * Tensor version which immediately calculates the result.
    *
    * @param parameters     batch of model parameters with shape (batchSize, numParameters)
    * @param roll           roll values of shape (batchSize)
    * @param pitch          pitch values of shape (batchSize)
    * @param yaw            yaw values of shape (batchSize)
    * @param translation    translation of shape (batchSize, 1, pointDimensions [x, y, z])
    * @param cameraNear     values of shape (1)
    * @param cameraFar      values of shape (1)
    * @param sensorSize     values of shape (batchSize, 2 [sensorWidth, sensorHeight])
    * @param focalLength    values of shape (1)
    * @param principalPoint values of shape (batchSize, 2 [principalPointX, principalPointY])
    * @return [[Tensor]] of projected landmark positions with shape (batchSize, numPoints, pointDimensions [x, y, z])
    */
  def batchCalculateLandmarks(parameters: Tensor[Float], roll: Output[Float], pitch: Output[Float], yaw: Output[Float],
                              translation: Output[Float], cameraNear: Output[Float], cameraFar: Output[Float],
                              sensorSize: Output[Float], focalLength: Output[Float], principalPoint: Output[Float],
                              imageWidth: Int, imageHeight: Int): Tensor[Float] = {
    val landmarks = batchCalculateLandmarks(parameters.toOutput, roll, pitch, yaw, translation, cameraNear, cameraFar,
      sensorSize, focalLength, principalPoint, imageWidth, imageHeight)

    using(Session())(_.run(fetches = landmarks))
  }

  /**
    * Calculates landmark positions in the rendered image for the given parameters. Version for batches of parameters.
    * <br>
    * The translation and sensorSize parameters can also be specified as constant for the entire batch by providing an
    * [[Output]] of shape (1, 1, 3) or (1, 2) respectively.
    *
    * @param parameters     batch of model parameters with shape (batchSize, numParameters)
    * @param roll           roll values of shape (batchSize)
    * @param pitch          pitch values of shape (batchSize)
    * @param yaw            yaw values of shape (batchSize)
    * @param translation    translation of shape (batchSize, 1, pointDimensions [x, y, z])
    * @param cameraNear     values of shape (1)
    * @param cameraFar      values of shape (1)
    * @param sensorSize     values of shape (batchSize, 2 [sensorWidth, sensorHeight])
    * @param focalLength    values of shape (1)
    * @param principalPoint values of shape (batchSize, 2 [principalPointX, principalPointY])
    * @return [[Output]] of projected landmark positions with shape (batchSize, numPoints, pointDimensions [x, y, z])
    */
  def batchCalculateLandmarks(parameters: Output[Float], roll: Output[Float], pitch: Output[Float], yaw: Output[Float],
                              translation: Output[Float], cameraNear: Output[Float], cameraFar: Output[Float],
                              sensorSize: Output[Float], focalLength: Output[Float], principalPoint: Output[Float],
                              imageWidth: Int, imageHeight: Int): Output[Float] = {
    val points = batchGetInstance(parameters)

    batchProjectPointsOnImage(points, roll, pitch, yaw, translation, cameraNear, cameraFar, sensorSize, focalLength, principalPoint, imageWidth, imageHeight)
  }

  /**
    * Calculates 2D landmark positions in the rendered image for the given parameters. Version for batches of
    * parameters. By only calculating the 2D landmark positions, this function requires fewer parameters and is more
    * efficient than [[batchCalculateLandmarks]].
    * <br>
    * The translation and sensorSize parameters can also be specified as constant for the entire batch by providing an
    * [[Output]] of shape (1, 1, 3) or (1, 2) respectively.
    *
    * @param parameters     batch of model parameters with shape (batchSize, numParameters)
    * @param roll           roll values of shape (batchSize)
    * @param pitch          pitch values of shape (batchSize)
    * @param yaw            yaw values of shape (batchSize)
    * @param translation    translation of shape (batchSize, 1, pointDimensions [x, y, z])
    * @param sensorSize     values of shape (batchSize, 2 [sensorWidth, sensorHeight])
    * @param focalLength    values of shape (1)
    * @param principalPoint values of shape (batchSize, 2 [principalPointX, principalPointY])
    * @return [[Output]] of projected landmark positions with shape (batchSize, numPoints, pointDimensions [x, y])
    */
  def batchCalculateLandmarks2D(parameters: Output[Float], roll: Output[Float], pitch: Output[Float], yaw: Output[Float],
                                translation: Output[Float],
                                sensorSize: Output[Float], focalLength: Output[Float], principalPoint: Output[Float],
                                imageWidth: Int, imageHeight: Int): Output[Float] = {
    val points = batchGetInstance(parameters)

    batchProjectPointsOnImage2D(points, roll, pitch, yaw, translation, sensorSize, focalLength, principalPoint, imageWidth, imageHeight)
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
  @deprecated("Uses deprecated point ordering (dimensions, numPoints).", "0.1-SNAPSHOT")
  def projectPointsOnImage(points: Output[Float], pose: TFPose, camera: TFCamera, imageWidth: Int, imageHeight: Int): Output[Float] = {
    val normalizedDeviceCoordinates = Transformations.objectToNDC(points, pose, camera)
    Transformations.screenTransformation(normalizedDeviceCoordinates, imageWidth, imageHeight)
  }

  /**
    * Projects the given points onto an image given the pose, camera and image parameters. Version for batches of
    * points.
    *
    * @param points         points to be projected onto the image of shape (batchSize, numPoints, pointDimensions)
    * @param roll           roll values of shape (batchSize)
    * @param pitch          pitch values of shape (batchSize)
    * @param yaw            yaw values of shape (batchSize)
    * @param translation    translation of shape (batchSize, 1, pointDimensions [x, y, z])
    * @param cameraNear     values of shape (1)
    * @param cameraFar      values of shape (1)
    * @param sensorSize     values of shape (batchSize, 2 [sensorWidth, sensorHeight])
    * @param focalLength    values of shape (1)
    * @param principalPoint values of shape (batchSize, 2 [principalPointX, principalPointY])
    * @return [[Output]] of projected landmark positions of shape (batchSize, numLandmarks, landmarkDimensions [x, y, z])
    */
  def batchProjectPointsOnImage(points: Output[Float], roll: Output[Float], pitch: Output[Float], yaw: Output[Float],
                                translation: Output[Float], cameraNear: Output[Float], cameraFar: Output[Float],
                                sensorSize: Output[Float], focalLength: Output[Float], principalPoint: Output[Float],
                                imageWidth: Int, imageHeight: Int): Output[Float] = {

    val normalizedDeviceCoordinates = Transformations.batchPointsToNDC(points, roll, pitch, yaw, translation,
      cameraNear, cameraFar, sensorSize, focalLength, principalPoint)

    Transformations.batchScreenTransformation(normalizedDeviceCoordinates, imageWidth, imageHeight)
  }

  /**
    * Projects the given points onto an image given the pose, camera and image parameters. Version for batches of
    * points. By only calculating the 2D landmark positions, this function requires fewer parameters and is more
    * efficient than [[batchProjectPointsOnImage]].
    *
    * @param points         points to be projected onto the image of shape (batchSize, numPoints, pointDimensions)
    * @param roll           roll values of shape (batchSize)
    * @param pitch          pitch values of shape (batchSize)
    * @param yaw            yaw values of shape (batchSize)
    * @param translation    translation of shape (batchSize, 1, pointDimensions [x, y, z])
    * @param sensorSize     values of shape (batchSize, 2 [sensorWidth, sensorHeight])
    * @param focalLength    values of shape (1)
    * @param principalPoint values of shape (batchSize, 2 [principalPointX, principalPointY])
    * @return [[Output]] of projected landmark positions of shape (batchSize, numLandmarks, landmarkDimensions [x, y])
    */
  def batchProjectPointsOnImage2D(points: Output[Float], roll: Output[Float], pitch: Output[Float], yaw: Output[Float],
                                  translation: Output[Float],
                                  sensorSize: Output[Float], focalLength: Output[Float], principalPoint: Output[Float],
                                  imageWidth: Int, imageHeight: Int): Output[Float] = {

    val normalizedDeviceCoordinates = Transformations.batchPointsToNDC2D(points, roll, pitch, yaw, translation,
      sensorSize, focalLength, principalPoint)

    Transformations.batchScreenTransformation2D(normalizedDeviceCoordinates, imageWidth, imageHeight)
  }
}

object TFLandmarksRenderer {
  /**
    * Creates a TFLandmarksRenderer for basic morphable models.
    * The landmarks renderer can be restricted to a specific set of pointIds for efficiency. In that case the results
    * [[TFLandmarksRenderer.getInstance]] and [[TFLandmarksRenderer.calculateLandmarks]] will be
    * only the specified points in the specified order.
    * <br>
    * Creates a landmarks renderer with transposed standard deviation, basis and mean points for use with deprecated
    * functions still using the points shape (dimensions, numPoints).
    *
    * @param momo             model to use
    * @param landmarkPointIds point ids of the desired landmarks or null for all points
    */
  @deprecated("Uses deprecated point ordering (dimensions, numPoints).", "0.1-SNAPSHOT")
  def transposed(momo: MoMoBasic, landmarkPointIds: IndexedSeq[Int]): TFLandmarksRenderer = {
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

    val meanPoints = {
      val meanPointData = momo.mean.shape.position.pointData
      val collectedMeanPointData = if (landmarkPointIds == null) meanPointData else landmarkPointIds.collect(meanPointData)
      TFConversions.pointsToTensor(collectedMeanPointData)
    }

    TFLandmarksRenderer(shapeBasis, shapeStd, meanPoints)
  }

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
      val fullShapeBasis = TFMoMoConversions.toTensorNotTransposed(momo.shape.basisMatrix)

      if (landmarkPointIds != null) {
        val basisIndices = landmarkPointIds.flatMap(i => Seq(i * 3, i * 3 + 1, i * 3 + 2))
        fullShapeBasis.gather(basisIndices, 1)
      } else {
        fullShapeBasis
      }
    }

    val shapeStd = TFMoMoConversions.toTensor(momo.shape.variance.map(math.sqrt))

    val meanPoints = {
      val meanPointData = momo.mean.shape.position.pointData
      val collectedMeanPointData = if (landmarkPointIds == null) meanPointData else landmarkPointIds.collect(meanPointData)
      TFConversions.pointsToTensorNotTransposed(collectedMeanPointData)
    }

    TFLandmarksRenderer(shapeBasis, shapeStd, meanPoints)
  }

  /**
    * Creates a TFLandmarksRenderer for expression models.
    * The landmarks renderer can be restricted to a specific set of pointIds for efficiency. In that case the results
    * [[TFLandmarksRenderer.getInstance]] and [[TFLandmarksRenderer.calculateLandmarks]] will be
    * only the specified points in the specified order.
    * <br>
    * Creates a landmarks renderer with transposed standard deviation, basis and mean points for use with deprecated
    * functions still using the points shape (dimensions, numPoints).
    *
    * @param momo             model to use
    * @param landmarkPointIds point ids of the desired landmarks or null for all points
    */
  @deprecated("Uses deprecated point ordering (dimensions, numPoints).", "0.1-SNAPSHOT")
  def transposed(momo: MoMoExpress, landmarkPointIds: IndexedSeq[Int]): TFLandmarksRenderer = {
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

    val meanPoints = {
      val meanPointData = momo.mean.shape.position.pointData
      val collectedMeanPointData = if (landmarkPointIds == null) meanPointData else landmarkPointIds.collect(meanPointData)
      TFConversions.pointsToTensor(collectedMeanPointData)
    }

    TFLandmarksRenderer(combinedBasis, combinedStd, meanPoints)
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

    val meanPoints = {
      val meanPointData = momo.mean.shape.position.pointData
      val collectedMeanPointData = if (landmarkPointIds == null) meanPointData else landmarkPointIds.collect(meanPointData)
      TFConversions.pointsToTensorNotTransposed(collectedMeanPointData)
    }

    TFLandmarksRenderer(combinedBasis, combinedStd, meanPoints)
  }
}