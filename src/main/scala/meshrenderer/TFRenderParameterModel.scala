package meshrenderer

import org.platanios.tensorflow.api
import org.platanios.tensorflow.api.core.client.FeedMap
import org.platanios.tensorflow.api.{tf, _}

/**
  * Created by andreas on 8/18/18.
  */

trait TFRenderParameterModel {
  val pts : Output[Float]
  val colors: Output[Float]
  val illumination: Output[Float]
  val pose: TFPose
  val camera: TFCamera
}

/* Models all variables as initialization + offset. Initialization is fixed and offset is variable. */
class OffsetFromInitializationModel(initPts: Tensor[Float], initColors: Tensor[Float], initPose: TFPoseTensor, initCamera: TFCameraTensor, initLight: Tensor[Float]) extends TFRenderParameterModel {

  //placeholders and variables
  val initalPoints: Output[Float] = tf.placeholder[Float](initPts.shape, "initPts")
  val initialColors: Output[Float] = tf.placeholder[Float](initColors.shape, "initColors")
  val initialIllumination: Output[Float] = tf.placeholder[Float](Shape(9, 3), "illumination")
  val initialPoseRotation: Output[Float] = tf.placeholder[Float](initPose.rotation.shape, "poseRotation")
  val initialPoseTranslation: Output[Float] = tf.placeholder[Float](initPose.translation.shape, "poseTranslation")
  val initialCamera: Output[Float] = tf.placeholder[Float](initCamera.parameters.shape, "camera")

  lazy val ptsVarVariable: api.tf.Variable[Float] = tf.variable[Float]("pointsOffset", initPts.shape, tf.ZerosInitializer)
  lazy val ptsVar: Output[Float] = ptsVarVariable
  lazy val colorsVarVariable: api.tf.Variable[Float] = tf.variable[Float]("colorsOffset", initColors.shape, tf.ZerosInitializer)
  lazy val colorsVar: Output[Float] = colorsVarVariable
  lazy val illumVarVariable: api.tf.Variable[Float] = tf.variable[Float]("illuminationOffset", initialIllumination.shape, tf.ZerosInitializer)
  lazy val illumVar: Output[Float] = illumVarVariable
  lazy val poseRotVarVariable: api.tf.Variable[Float] = tf.variable[Float]("poseRotationOffset", initialPoseRotation.shape, tf.ZerosInitializer)
  lazy val poseRotVar: Output[Float] = poseRotVarVariable
  lazy val poseTransVar: api.tf.Variable[Float] = tf.variable[Float]("poseTranslationOffset", initialPoseTranslation.shape, tf.ZerosInitializer)
  lazy val cameraVar: api.tf.Variable[Float] = tf.variable[Float]("cameraOffset", initialCamera.shape, tf.ZerosInitializer)

  lazy val illumOffset: Output[Float] = tf.identity(illumVar)
  lazy val colorsOffset: Output[Float] = tf.identity(colorsVar)
  lazy val ptsOffset: Output[Float] = tf.identity(ptsVar)
  lazy val poseRotationOffset: Output[Float] = tf.identity(poseRotVar)
  lazy val poseTranslationOffset: Output[Float] = tf.identity(poseTransVar)
  lazy val cameraOffset: Output[Float] = tf.identity(cameraVar)

  //val colors = tf.variable("color", FLOAT32, tfMesh.colors.shape, tf.RandomUniformInitializer())
  //val colors = tfMesh.colors

  lazy val pts: Output[Float] = initalPoints + ptsOffset
  lazy val colors: Output[Float] = initialColors + colorsOffset
  lazy val illumination: Output[Float] = initialIllumination + illumOffset
  lazy val poseRotation: Output[Float] = initialPoseRotation + poseRotationOffset
  lazy val poseTranslation: Output[Float] = initialPoseTranslation + poseTranslationOffset
  lazy val pose = TFPose(poseRotation, poseTranslation)
  lazy val camera = TFCamera(initialCamera + cameraOffset)

  val feeds: FeedMap = Map(
    initalPoints -> initPts,
    initialColors -> initColors,
    initialIllumination -> initLight,
    initialPoseRotation -> initPose.rotation, initialPoseTranslation -> initPose.translation,
    initialCamera -> initCamera.parameters
  )
}