package meshrenderer

import org.platanios.tensorflow.api.{tf, _}
import scalismo.faces.momo.{MoMo, MoMoBasic, MoMoExpress}

/**
  * Class for rendering Morphable Model parameters using differentiable TensorFlow operations.
  *
  * @param landmarksRenderer landmarks renderer used to calculate point positions for the rendered mesh
  * @param colorBasisMatrix  the color basis matrix of the chosen model
  * @param colorParameterStd the color parameter standard deviations of the chosen model
  * @param meanColor         the mean color of the chosen model
  */
case class TFImageRenderer(landmarksRenderer: TFLandmarksRenderer,
                           colorBasisMatrix: Tensor[Float], colorParameterStd: Tensor[Float], meanColor: Tensor[Float],
                           triangles: Tensor[Int], trianglesForPointData: Tensor[Int]) {

  /**
    * Renders a single set of parameters.
    *
    * @param shapeParameters            all shape related parameters with shape (1, number of parameters)
    * @param colorParameters            color parameters with shape (1, number of parameters)
    * @param environmentMapCoefficients environment map coefficients with shape (number of coefficients, 3)
    * @param roll                       shape ()
    * @param pitch                      shape ()
    * @param yaw                        shape ()
    * @param translation                pose translation with shape (3 [x, y, z])
    * @param cameraNear                 shape ()
    * @param cameraFar                  shape ()
    * @param sensorSize                 camera sensor size of shape (2 [width, height])
    * @param focalLength                shape ()
    * @param principalPoint             camera principal point of shape (2 [x, y])
    * @return image output of shape (height, width, 3 [r, g, b])
    */
  def render(
              shapeParameters: Output[Float], colorParameters: Output[Float], environmentMapCoefficients: Output[Float],
              roll: Output[Float], pitch: Output[Float], yaw: Output[Float],
              imageWidth: Int, imageHeight: Int,
              translation: Output[Float], cameraNear: Output[Float], cameraFar: Output[Float],
              sensorSize: Output[Float], focalLength: Output[Float], principalPoint: Output[Float]
            ): Output[Float] = {

    val points = landmarksRenderer.batchGetInstance(shapeParameters).reshape(Shape(-1, 3))

    val color = {
      val colorOffset = tf.matmul(colorParameters * colorParameterStd, colorBasisMatrix)
        .reshape(Shape(colorBasisMatrix.shape(1) / 3, 3))
      meanColor + colorOffset
    }

    val batchRoll = roll.expandDims(0)
    val batchPitch = pitch.expandDims(0)
    val batchYaw = yaw.expandDims(0)

    val normals = TFMeshOperations.vertexNormals(points, triangles, trianglesForPointData)
    val worldNormals = Transformations.batchPoseRotationTransform(normals.expandDims(0), batchPitch, batchYaw, batchRoll).reshape(Shape(-1, 3))

    val ndcPts = {
      val batchPoints = points.expandDims(0)

      val batchTranslation = translation.expandDims(0)
      val batchSensorSize = sensorSize.expandDims(0)
      val batchPrincipalPoint = principalPoint.expandDims(0)

      Transformations.batchPointsToNDC(batchPoints, batchRoll, batchPitch, batchYaw, batchTranslation, cameraNear,
        cameraFar, batchSensorSize, focalLength, batchPrincipalPoint).reshape(Shape(-1, 3))
    }
    val ndcPtsTf = Transformations.ndcToTFNdc(ndcPts, imageWidth, imageHeight)

    val triangleIdsAndBCC = Rasterizer.rasterize_triangles(ndcPtsTf, triangles, imageWidth, imageHeight)
    val vtxIdxPerPixel = tf.gather(triangles, triangleIdsAndBCC.triangleIds.reshape(Shape(-1)))

    val interpolatedAlbedo = Shading.interpolateVertexDataPerspectiveCorrect(triangleIdsAndBCC, vtxIdxPerPixel, color, ndcPtsTf(::, 2))

    val interpolatedNormals = Shading.interpolateVertexDataPerspectiveCorrect(triangleIdsAndBCC, vtxIdxPerPixel, worldNormals, ndcPtsTf(::, 2))

    Shading.sphericalHarmonicsLambertShader(interpolatedAlbedo, interpolatedNormals, environmentMapCoefficients)
  }

  /**
    * Renders a batch of parameters.
    *
    * @param shapeParameters            all shape related parameters with shape (batchSize, number of parameters)
    * @param colorParameters            color parameters with shape (batchSize, number of parameters)
    * @param environmentMapCoefficients environment map coefficients with shape (batchSize, number of coefficients, 3)
    * @param roll                       shape (batchSize)
    * @param pitch                      shape (batchSize)
    * @param yaw                        shape (batchSize)
    * @param translation                pose translation with shape (batchSize, 1, pointDimensions [x, y, z])
    * @param cameraNear                 shape ()
    * @param cameraFar                  shape ()
    * @param sensorSize                 camera sensor size of shape (batchSize, 2 [width, height])
    * @param focalLength                shape ()
    * @param principalPoint             camera principal point of shape (batchSize, 2 [x, y])
    * @return image output of shape (batchSize, height, width, 3 [r, g, b])
    */
  def batchRender(
                   shapeParameters: Output[Float], colorParameters: Output[Float], environmentMapCoefficients: Output[Float],
                   roll: Output[Float], pitch: Output[Float], yaw: Output[Float],
                   imageWidth: Int, imageHeight: Int,
                   translation: Output[Float], cameraNear: Output[Float], cameraFar: Output[Float],
                   sensorSize: Output[Float], focalLength: Output[Float], principalPoint: Output[Float]
                 ): Output[Float] = {
    // TODO: implement batch rendering

    val points = landmarksRenderer.batchGetInstance(shapeParameters).reshape(Shape(-1, 3))

    val color = {
      val colorOffset = tf.matmul(colorParameters * colorParameterStd, colorBasisMatrix)
        .reshape(Shape(colorParameters.shape(0), colorBasisMatrix.shape(1) / 3, 3))
      meanColor + colorOffset
    }

    val normals = TFMeshOperations.vertexNormals(points, triangles, trianglesForPointData)
    val worldNormals = Transformations.batchPoseRotationTransform(normals.expandDims(0), pitch, yaw, roll)

    ???
  }
}

object TFImageRenderer {

  def apply(momo: MoMoBasic): TFImageRenderer = {
    val landmarksRenderer = TFLandmarksRenderer(momo, null)

    val colorBasisMatrix = TFMoMoConversions.toTensor(momo.color.basisMatrix)
    val colorParameterStd = TFMoMoConversions.toTensor(momo.color.variance.map(math.sqrt))
    val meanColor = TFConversions.pointsToTensor(momo.mean.color.pointData.map(_.toRGB))

    apply(momo, landmarksRenderer, colorBasisMatrix, colorParameterStd, meanColor)
  }

  def apply(momo: MoMoExpress): TFImageRenderer = {
    val landmarksRenderer = TFLandmarksRenderer(momo, null)

    val colorBasisMatrix = TFMoMoConversions.toTensor(momo.color.basisMatrix)
    val colorParameterStd = TFMoMoConversions.toTensor(momo.color.variance.map(math.sqrt))
    val meanColor = TFConversions.pointsToTensor(momo.mean.color.pointData.map(_.toRGB))

    apply(momo, landmarksRenderer, colorBasisMatrix, colorParameterStd, meanColor)
  }

  def apply(momo: MoMo, landmarksRenderer: TFLandmarksRenderer, colorBasisMatrix: Tensor[Float],
            colorParameterStd: Tensor[Float], meanColor: Tensor[Float]): TFImageRenderer = {

    val triangles = TFMesh.triangulationAsTensor(momo.referenceMesh.triangulation)
    val trianglesForPointData = {
      val triForPoint = momo.referenceMesh.pointSet.pointIds.toIndexedSeq.map { id =>
        (id, momo.referenceMesh.triangulation.adjacentTrianglesForPoint(id))
      }
      TFMeshOperations.trianglesForPoint(triForPoint)
    }

    TFImageRenderer(landmarksRenderer, colorBasisMatrix, colorParameterStd, meanColor, triangles, trianglesForPointData)
  }
}
