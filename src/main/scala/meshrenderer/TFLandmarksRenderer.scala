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

    val session = Session()
    val result = session.run(fetches = mesh)

    session.close()

    result
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

    val session = Session()
    val result = session.run(fetches = landmarks)

    session.close()

    result
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

  /**
    * Projects the given points onto an image given the pose, camera and image parameters.
    *
    * @param points points to be projected onto the image
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
}

object TFLandmarksRenderer {
  def apply(momo: MoMoBasic, landmarkPointIds: IndexedSeq[Int] = null): TFLandmarksRenderer = {
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

  /**
    * Creates a TFLandmarksRenderer for expression models.
    * The landmarks renderer can be restricted to a specific set of pointIds for efficiency. In that case the results
    * [[TFLandmarksRenderer.getInstance]] and [[TFLandmarksRenderer.calculateLandmarks]] will be
    * only the specified points in the specified order.
    *
    * @param momo model to use
    * @param landmarkPointIds point ids of the desired landmarks or null for all points
    */
  def apply(momo: MoMoExpress, landmarkPointIds: IndexedSeq[Int] = null): TFLandmarksRenderer = {
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

    val (combinedBasis, combinedStd) = {
      val bases = Seq(shapeBasis.toOutput, expressionBasis.toOutput)
      val standardDevs = Seq(shapeStd.toOutput, expressionStd.toOutput)

      val session = Session()
      val results = session.run(fetches = Seq(tf.concatenate(bases, axis = 1), tf.concatenate(standardDevs, axis = 0)))
      (results.head, results.last)
    }

    val meanPoints = TFConversions.pointsToTensor(landmarkPointIds.collect(momo.mean.shape.position.pointData))

    TFLandmarksRenderer(combinedBasis, combinedStd, meanPoints)
  }
}