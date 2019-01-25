package meshrenderer

import org.platanios.tensorflow.api.Tensor
import scalismo.faces.momo.{MoMoBasic, MoMoExpress}

/**
  * Class for rendering Morphable Model parameters using differentiable TensorFlow operations.
  *
  * @param landmarksRenderer landmarks renderer used to calculate point positions for the rendered mesh
  * @param colorBasisMatrix  the color basis matrix of the chosen model
  * @param colorParameterStd the color parameter standard deviations of the chosen model
  * @param meanColor         the mean color of the chosen model
  */
case class TFImageRenderer(landmarksRenderer: TFLandmarksRenderer, colorBasisMatrix: Tensor[Float],
                           colorParameterStd: Tensor[Float], meanColor: Tensor[Float]) {
  // TODO: Implement rendering functions
}

object TFImageRenderer {

  def apply(momo: MoMoBasic): TFImageRenderer = {
    val landmarksRenderer = TFLandmarksRenderer(momo, null)

    val colorBasisMatrix = TFMoMoConversions.toTensorNotTransposed(momo.color.basisMatrix)
    val colorParameterStd = TFMoMoConversions.toTensor(momo.color.variance.map(math.sqrt))
    val meanColor = TFConversions.pointsToTensorNotTransposed(momo.mean.color.pointData.map(_.toRGB))

    TFImageRenderer(landmarksRenderer, colorBasisMatrix, colorParameterStd, meanColor)
  }

  def apply(momo: MoMoExpress): TFImageRenderer = {
    val landmarksRenderer = TFLandmarksRenderer(momo, null)

    val colorBasisMatrix = TFMoMoConversions.toTensorNotTransposed(momo.color.basisMatrix)
    val colorParameterStd = TFMoMoConversions.toTensor(momo.color.variance.map(math.sqrt))
    val meanColor = TFConversions.pointsToTensorNotTransposed(momo.mean.color.pointData.map(_.toRGB))

    TFImageRenderer(landmarksRenderer, colorBasisMatrix, colorParameterStd, meanColor)
  }
}
