package meshrenderer

import java.io.File

import breeze.linalg.DenseVector
import org.platanios.tensorflow.api
import org.platanios.tensorflow.api.ops.Gradients
import org.platanios.tensorflow.api.{tf, _}
import scalismo.common.PointId
import scalismo.faces.color.{RGB, RGBA}
import scalismo.faces.image.PixelImage
import scalismo.faces.io.{MeshIO, MoMoIO, PixelImageIO, RenderParameterIO}
import scalismo.faces.mesh.{ColorNormalMesh3D, VertexColorMesh3D}
import scalismo.faces.momo.{MoMo, MoMoCoefficients}
import scalismo.faces.parameters._
import scalismo.faces.sampling.face.MoMoRenderer
import scalismo.geometry.{Point, Vector}
import scalismo.mesh.TriangleMesh3D
import scalismo.utils.Random

/**
  * Object containing some example and some test code.
  */
object Example {

  def main(args: Array[String]): Unit = {

    val runRenderTest = true
    val runLandmarkTest = false
    val runOptimizationExample = false

    val model = time("Initializing and loading model",
      {
        scalismo.initialize()
        val momoFn = new File("../face-autoencoder/model2017-1_bfm_nomouth.h5")
        MoMoIO.read(momoFn).get
      })

    if (runRenderTest) time("Running render test", {renderTest(model)})

    if (runLandmarkTest || runOptimizationExample) {
      // Set up shared resources for optimization and landmark test
      val imageFn = new File("data/JimmyCarterPortrait2.png")
      val image = {
        val img = PixelImageIO.read[RGB](imageFn).get
        img.resample((img.width * 0.1).toInt, (img.height * 0.1).toInt)
      }

      val param = RenderParameterIO.read(new File("data/fit-best.rps")).get.fitToImageSize(image.width, image.height)

      val initPose = time("Transforming TFPose", {TFPose(param.pose)})
      val initCamera = time("Transforming TFCamera", {TFCamera(param.camera)})

      val mesh = model.instance(param.momo.coefficients)

      if (runLandmarkTest) time("Running landmark test", {testLandmarks(param, model, image, initPose, initCamera, mesh)})
      if (runOptimizationExample) time("Running optimization example", {optimizationExample(param, model, image, initPose, initCamera, mesh)})
    }
  }

  /**
    * Helper function to time code execution.
    */
  def time[R](name: String, block: => R): R = {
    println(s"$name...")
    val start = System.nanoTime()
    val result = block
    val end = System.nanoTime()
    println(s"$name complete: ${(end - start) / 1e9}s")

    result
  }

  /**
    * Method to test rendering of parameters using TensorFlow and the differentiable rasterizer.
    */
  def renderTest(model: MoMo): Unit = {
    implicit val rand: Random = Random(160)

    // Get expression model
    val expressionModel = model.expressionModel.get

    // Get random parameters
    val coefficients = model.sampleCoefficients()
    val parameters = {
      val default = RenderParameter.default.withMoMo(MoMoInstance.fromCoefficients(coefficients, new File("../face-autoencoder/model2017-1_bfm_nomouth.h5").toURI))

      default.withPose(Pose(scaling = default.pose.scaling, translation = default.pose.translation, rand.rng.scalaRandom.nextGaussian() * 0.1, rand.rng.scalaRandom.nextGaussian() * 0.5, rand.rng.scalaRandom.nextGaussian() * 0.3))
        .withEnvironmentMap(default.environmentMap.copy(coefficients = default.environmentMap.coefficients.map(_.map(_ + rand.rng.scalaRandom.nextGaussian() * 0.2)))) // Randomize environment map
//        .withCamera(default.camera.copy(sensorSize = Vector(9, 9)))
//        .withImageSize(ImageSize(227, 227))
    }

    val coefficients2 = model.sampleCoefficients()
    val parameters2 = {
      val default = RenderParameter.default.withMoMo(MoMoInstance.fromCoefficients(coefficients2, new File("../face-autoencoder/model2017-1_bfm_nomouth.h5").toURI))

      default.withPose(Pose(scaling = default.pose.scaling, translation = default.pose.translation, rand.rng.scalaRandom.nextGaussian() * 0.1, rand.rng.scalaRandom.nextGaussian() * 0.5, rand.rng.scalaRandom.nextGaussian() * 0.3))
        .withEnvironmentMap(default.environmentMap.copy(coefficients = default.environmentMap.coefficients.map(_.map(_ + rand.rng.scalaRandom.nextGaussian() * 0.2)))) // Randomize environment map
      //        .withCamera(default.camera.copy(sensorSize = Vector(9, 9)))
      //        .withImageSize(ImageSize(227, 227))
    }

    // Convert parameters to outputs
    val colorCoefficients: Output[Float] = TFMoMoConversions.toTensor(coefficients.color)
    val shapeExpressionParams: Output[Float] = TFMoMoConversions.toTensor(DenseVector.vertcat(coefficients.shape, coefficients.expression))

    val roll: Output[Float] = parameters.pose.roll.toFloat
    val pitch: Output[Float] = parameters.pose.pitch.toFloat
    val yaw: Output[Float] = parameters.pose.yaw.toFloat

    val batchRoll = roll.expandDims(0)
    val batchPitch = pitch.expandDims(0)
    val batchYaw = yaw.expandDims(0)

    val imageWidth = parameters.imageSize.width
    val imageHeight = parameters.imageSize.height

    val translation: Tensor[Float] = Tensor(parameters.pose.translation.x.toFloat, parameters.pose.translation.y.toFloat, parameters.pose.translation.z.toFloat)
    val sensorSize: Tensor[Float] = Tensor(parameters.camera.sensorSize.x.toFloat, parameters.camera.sensorSize.y.toFloat)
    val principalPoint: Tensor[Float] = Tensor(parameters.camera.principalPoint.x.toFloat, parameters.camera.principalPoint.y.toFloat)

    val cameraNear = parameters.camera.near.toFloat
    val cameraFar = parameters.camera.far.toFloat
    val focalLength: Float = parameters.camera.focalLength.toFloat

    // Calculate illumination
    val illumination = TFConversions.pointsToTensor(parameters.environmentMap.coefficients)

    println(s"Illumination shape: ${illumination.shape}")

    // TF image renderer
    val imageRenderer = TFImageRenderer(model.expressionModel.get)
    val basicImageRenderer = TFImageRenderer(model.neutralModel)

    val rendererImage = imageRenderer.render(shapeExpressionParams, colorCoefficients, illumination, roll, pitch, yaw, imageWidth, imageHeight, translation, cameraNear, cameraFar, sensorSize, focalLength, principalPoint)

    val (rendererBatchImage, basicBatchImage) = {
      val colorCoefficients2: Output[Float] = TFMoMoConversions.toTensor(coefficients2.color)
      val shapeExpressionParams2: Output[Float] = TFMoMoConversions.toTensor(DenseVector.vertcat(coefficients2.shape, coefficients2.expression))

      val roll2: Output[Float] = parameters2.pose.roll.toFloat
      val pitch2: Output[Float] = parameters2.pose.pitch.toFloat
      val yaw2: Output[Float] = parameters2.pose.yaw.toFloat

      val principalPoint2: Tensor[Float] = Tensor(parameters2.camera.principalPoint.x.toFloat, parameters2.camera.principalPoint.y.toFloat)

      val illumination2 = TFConversions.pointsToTensor(parameters2.environmentMap.coefficients)

      val batchShapeParams = tf.concatenate(Seq(shapeExpressionParams, shapeExpressionParams2))
      val batchColorCoefficients = tf.concatenate(Seq(colorCoefficients, colorCoefficients2))
      val batchEnvironmentMap = tf.stack(Seq(illumination.toOutput, illumination2.toOutput))

      val multiBatchRoll = tf.stack(Seq(roll, roll2))
      val multiBatchPitch = tf.stack(Seq(pitch, pitch2))
      val multiBatchYaw = tf.stack(Seq(yaw, yaw2))

      val batchTranslation = translation.reshape(Shape(1, 1, 3))
      val batchSensorSize = sensorSize.reshape(Shape(1, 2))
      val batchPrincipalPoint = tf.stack(Seq(principalPoint.toOutput, principalPoint2.toOutput))

      val basicBatchShapeParams = tf.concatenate(Seq(TFMoMoConversions.toTensor(coefficients.shape).toOutput, TFMoMoConversions.toTensor(coefficients2.shape).toOutput))

      (
        imageRenderer.batchRender(batchShapeParams, batchColorCoefficients, batchEnvironmentMap, multiBatchRoll,
          multiBatchPitch, multiBatchYaw, imageWidth, imageHeight, batchTranslation, cameraNear, cameraFar,
          batchSensorSize, focalLength, batchPrincipalPoint, 2),
        basicImageRenderer.batchRender(basicBatchShapeParams, batchColorCoefficients, batchEnvironmentMap, multiBatchRoll,
          multiBatchPitch, multiBatchYaw, imageWidth, imageHeight, batchTranslation, cameraNear, cameraFar,
          batchSensorSize, focalLength, batchPrincipalPoint, 2)
      )
    }

    // Calculate color
    val color = {
      val meanColor = TFConversions.pointsToTensor(model.mean.color.pointData.map(_.toRGB))

      println(s"Mean color shape: ${meanColor.shape}")

      val colorBasisMatrix = TFMoMoConversions.toTensor(expressionModel.color.basisMatrix)
      val colorStandardDeviation = TFMoMoConversions.toTensor(expressionModel.color.variance.map(math.sqrt))

      println(s"Color coefficients shape: ${colorCoefficients.shape}")

      println(s"Color basis matrix shape: ${colorBasisMatrix.shape}")
      println(s"Color standard deviation shape: ${colorStandardDeviation.shape}")

      val colorOffset = tf.matmul(colorCoefficients * colorStandardDeviation, colorBasisMatrix).reshape(Shape(colorBasisMatrix.shape(1) / 3, 3))

      println(s"Color offset shape: ${colorOffset.shape}")

      meanColor + colorOffset
    }

    // Create landmarksRenderer for calculating points
    val landmarksRenderer = TFLandmarksRenderer(model.expressionModel.get, null)

    println(s"Batch param shape: ${shapeExpressionParams.shape}")

    // Calculate mesh points
    val batchPoints = landmarksRenderer.batchGetInstance(shapeExpressionParams)
    println(s"Batch points shape: ${batchPoints.shape}")
    val points = batchPoints.reshape(Shape(-1, 3))

    // Retrieve required mesh information
    val triangles = TFMesh.triangulationAsTensor(model.referenceMesh.triangulation)
    val trianglesForPointData = {
      val triForPoint = model.referenceMesh.pointSet.pointIds.toIndexedSeq.map { id =>
        (id, model.referenceMesh.triangulation.adjacentTrianglesForPoint(id))
      }
      TFMeshOperations.trianglesForPoint(triForPoint)
    }


    // TODO: Not batch optimized yet
    val normals: Output[Float] = TFMeshOperations.vertexNormals(points, triangles, trianglesForPointData)

    println(s"Normals shape: ${normals.shape}")

    val worldNormals: Output[Float] = Transformations.batchPoseRotationTransform(normals.expandDims(0), batchPitch, batchYaw, batchRoll).reshape(Shape(-1, 3))

    println(s"World normals shape: ${worldNormals.shape}")

    // TODO: Not batch optimized yet
    val ndcPts = {
      val batchTranslation = translation.expandDims(0)
      val batchSensorSize = sensorSize.expandDims(0)
      val batchPrincipalPoint = principalPoint.expandDims(0)

      Transformations.batchPointsToNDC(batchPoints, batchRoll, batchPitch, batchYaw, batchTranslation, cameraNear,
        cameraFar, batchSensorSize, focalLength, batchPrincipalPoint)
        .reshape(Shape(-1, 3))
    }
    val ndcPtsTf: Output[Float] = Transformations.ndcToTFNdc(ndcPts, imageWidth, imageHeight)

    println(s"NDC shape: ${ndcPts.shape}")
    println(s"NDC TF shape: ${ndcPtsTf.shape}")

    // TODO: Not batch optimized yet
    val triangleIdsAndBCC: Rasterizer.RasterizationOutput = Rasterizer.rasterize_triangles(ndcPtsTf, triangles, imageWidth, imageHeight)
    val vtxIdxPerPixel: Output[Int] = tf.gather(triangles, triangleIdsAndBCC.triangleIds.reshape(Shape(-1)))

    println(s"Triangle Ids and BCC shape: ${triangleIdsAndBCC.barycetricImage.shape}, ${triangleIdsAndBCC.triangleIds.shape}, ${triangleIdsAndBCC.zBufferImage.shape}")
    println(s"vtxIdxPerPixel shape: ${vtxIdxPerPixel.shape}")

    // TODO: Not batch optimized yet
    val vtxIdxPerPixelGath: Output[Int] = tf.gatherND(triangles, tf.expandDims(triangleIdsAndBCC.triangleIds, 2))

    println(s"vtxIdxPerPixelGath shape: ${vtxIdxPerPixelGath.shape}")

    val interpolatedAlbedo: Output[Float] = Shading.interpolateVertexDataPerspectiveCorrect(triangleIdsAndBCC, vtxIdxPerPixel, color, ndcPtsTf(::, 2))

    println(s"Interpolated albedo shape: ${interpolatedAlbedo.shape}")

    val interpolatedNormals: Output[Float] = Shading.interpolateVertexDataPerspectiveCorrect(triangleIdsAndBCC, vtxIdxPerPixel, worldNormals, ndcPtsTf(::, 2))

    println(s"Interpolated normals shape: ${interpolatedNormals.shape}")

    //val lambert = Renderer.lambertShader(interpolatedAlbedo, Tensor(0.5f, 0.5f, 0.5f), Tensor(0.5f, 0.5f, 0.5f), Tensor(Seq(0f,0f,1f)), interpolatedNormals)
    val shShader: Output[Float] = Shading.sphericalHarmonicsLambertShader(interpolatedAlbedo, interpolatedNormals, illumination)

    println(s"Spherical Harmonics Shader shape: ${shShader.shape}")

    // Check results
    using(Session())(session => {
      val result = time("Rendering result with TensorFlow", {session.run(fetches = Seq(shShader, rendererImage))})
      val batchResult = time("Rendering batch result with TensorFlow", {session.run(fetches = rendererBatchImage)})
      val basicBatchResult = time("Rendering basic batch result with TensorFlow", {session.run(fetches = basicBatchImage)})

      val rendering = time("Converting results to Pixel Images", {result.map(r => TFConversions.oneToOneTensorImage3dToPixelImage(r.transpose(permutation = Tensor(1, 0, 2))))})
      time("Writing TF Render to file", {PixelImageIO.write(rendering.head, new File("/tmp/renderTestRendering.png")).get})
      PixelImageIO.write(rendering(1), new File("/tmp/renderTestRendererImage.png")).get

      val batchRendering = TFConversions.oneToOneTensorImage3dToPixelImage(batchResult(0, ::, ::, ::).transpose(permutation = Tensor(1, 0, 2)))
      val batchRendering2 = TFConversions.oneToOneTensorImage3dToPixelImage(batchResult(1, ::, ::, ::).transpose(permutation = Tensor(1, 0, 2)))
      PixelImageIO.write(batchRendering, new File("/tmp/renderTestRendererBatchImage0.png")).get
      PixelImageIO.write(batchRendering2, new File("/tmp/renderTestRendererBatchImage1.png")).get
      val basicBatchRendering = TFConversions.oneToOneTensorImage3dToPixelImage(basicBatchResult(0, ::, ::, ::).transpose(permutation = Tensor(1, 0, 2)))
      val basicBatchRendering2 = TFConversions.oneToOneTensorImage3dToPixelImage(basicBatchResult(1, ::, ::, ::).transpose(permutation = Tensor(1, 0, 2)))
      PixelImageIO.write(basicBatchRendering, new File("/tmp/renderTestRendererBasicBatchImage0.png")).get
      PixelImageIO.write(basicBatchRendering2, new File("/tmp/renderTestRendererBasicBatchImage1.png")).get
    })

    // Render with MoMoRenderer for comparison
    {
      val momoRenderer = MoMoRenderer(model)
      val basicMoMoRenderer = MoMoRenderer(model.neutralModel)

      val renderedImage = momoRenderer.renderImage(parameters)
      val renderedImage2 = momoRenderer.renderImage(parameters2)

      val basicRenderedImage = basicMoMoRenderer.renderImage(parameters)
      val basicRenderedImage2 = basicMoMoRenderer.renderImage(parameters2)

      PixelImageIO.write(renderedImage, new File("/tmp/renderTestGroundTruth.png")).get
      PixelImageIO.write(renderedImage2, new File("/tmp/renderTestRendererBatchImage1GroundTruth.png")).get

      PixelImageIO.write(basicRenderedImage, new File("/tmp/renderTestRendererBasicBatchImage0GroundTruth.png")).get
      PixelImageIO.write(basicRenderedImage2, new File("/tmp/renderTestRendererBasicBatchImage1GroundTruth.png")).get
    }
  }

  /**
    * Method showing an example script on how to directly optimize model parameters using the differentiable renderer.
    */
  def optimizationExample(param: RenderParameter, model: MoMo, image: PixelImage[RGB], initPose: TFPoseTensor, initCamera: TFCameraTensor, mesh: VertexColorMesh3D): Unit = {
    /*val param = RenderParameter.defaultSquare.withPose(Pose(1.0, Vector(0,0,-1000), 0,1.1,0)).fitToImageSize(200,250).withEnvironmentMap(
      SphericalHarmonicsLight(SphericalHarmonicsLight.frontal.coefficients ++ IndexedSeq(Vector(0.5,0.5,0.1), Vector(0.2,0.1,0.7), Vector(0.0,0.2,0.0), Vector(0.2,0.1,-0.1), Vector(-0.1,-0.1,-0.1)))
    )*/
    //
    //    //val image = PixelImageIO.read[RGB](new File("/tmp/tf_rendering_sh_lambert.png")).get
    //

    //    //val mesh = MeshIO.read(new File("/home/andreas/export/mean2012_l7_bfm_pascaltex.msh.gz")).get.colorNormalMesh3D.get

    ////    time("Writing temp model", {
    ////      MeshIO.write(mesh, new File("/tmp/jimmi_fit.ply")).get
    ////    })

    val tfMesh = time("Transforming TFMesh", {
      TFMesh(mesh)
    })

    val initLight = time("Transforming TFLight", {
      TFConversions.pointsToTensorTransposed(param.environmentMap.coefficients).transpose()
    })

    //    //val variableModel = new OffsetFromInitializationModel(tfMesh.pts, tfMesh.colors, initPose, initCamera, initLight)
    val tfModel = time("Creating TFMoMo", {
      TFMoMo(model.expressionModel.get.truncate(80, 40, 5))
    })
    //    /*
    //    ==================================
    //    This part used the incorrect mean!
    //    ==================================
    //     */
    //    //    val tfMean = TFMesh(model.neutralModel.mean)
    val tfMean = time("Creating mean TFMesh", {
      TFMesh(model.mean)
    })

    val variableModel = time("Creating variable model", {
      TFMoMoExpressParameterModel(tfModel, tfMean, initPose, initCamera, initLight)
    })


    val renderer = TFRenderer(tfMesh, variableModel.pts, variableModel.colors, variableModel.pose, variableModel.camera, variableModel.illumination, param.imageSize.width, param.imageSize.height)

    def renderInitialParametersAndCompareToGroundTruth(): Unit = {
      val sess = Session()
      sess.run(targets = tf.globalVariablesInitializer())

      // Test output to check renderer results
      {
        val testResults = sess.run(
          feeds = variableModel.feeds,
          fetches = renderer.ndcPtsTf
        )

        println("ndcPtsTf: ")
        println(testResults.summarize())
      }

      val result = sess.run(
        feeds = variableModel.feeds,
        fetches = Seq(renderer.shShader)
      )

      val tensorImg = result.head.toTensor
      val img = TFConversions.oneToOneTensorImage3dToPixelImage(tensorImg.transpose(permutation = Tensor(1, 0, 2)))
      PixelImageIO.write[RGB](img, new File("/tmp/fit.png")).get

      {
        val img = ParametricRenderer.renderParameterMesh(param, ColorNormalMesh3D(mesh), RGBA.Black)
        PixelImageIO.write[RGBA](img, new File("/tmp/fitgt.png")).get
      }
    }

    val targetImage = TFConversions.image3dToTensor(image)

    val test = TFConversions.oneToOneTensorImage3dToPixelImage(targetImage.transpose(permutation = Tensor(1, 0, 2)))
    PixelImageIO.write(test, new File("/tmp/test.png")).get

    renderInitialParametersAndCompareToGroundTruth()

    //loss
    val target = tf.placeholder[Float](Shape(param.imageSize.height, param.imageSize.width, 3), "target")

    //val reg = TFMeshOperations.vertexToNeighbourDistance(tfMesh.adjacentPoints, renderer.ptsOffset)

    val vtxsPerTriangle = tf.gather(renderer.normals, renderer.mesh.triangles)
    println("vtxsPerTriangle", vtxsPerTriangle)
    //println("validIds", validIds)
    //val regPre = tf.tile(tf.expandDims(validIds, 1), Seq(1, 3, 1)) * vtxsPerTriangle
    //rintln("regPre", regPre)
    //val reg = tf.sum(vtxsPerTriangle)
    //val reg = tf.sum(tf.abs(variableModel.ptsVar))

    val validArea = tf.tile(tf.expandDims(renderer.triangleIdsAndBCC.triangleIds > 0, 2), Seq(1, 1, 3))

    val recPart1 = target * validArea.toFloat
    val recPart2 = renderer.shShader * validArea.toFloat

    val rec = tf.sum(tf.mean(tf.abs(recPart1 - recPart2)))

    val pixelFGLH = {
      val sdev = 0.0043f // sdev corresponding to very good fit so this can be smaller than 0.043
      val normalizer = (-3 / 2 * math.log(2 * math.Pi) - 3 * math.log(sdev)).toFloat
      tf.sum(tf.square(target - renderer.shShader), axes = Seq(2)) * (-0.5f) / sdev / sdev + normalizer
    }

    val pixelBGLH = {
      val sdev = 0.25f
      val normalizer = (-3 / 2 * math.log(2 * math.Pi) - 3 * math.log(sdev)).toFloat
      tf.sum(tf.square(target - renderer.shShader), axes = Seq(2)) * (-0.5f) / sdev / sdev + normalizer
    }

    println("pxielFGLH", pixelFGLH)
    println("pxielBGLH", pixelBGLH)

    val lh = tf.mean(pixelFGLH - pixelBGLH)

    /*
    val dDiff = shapeCoeff.foldLeft(0.0) { (z: Double, v: Double) => z + math.pow(v - mean, 2) / sdev / sdev }
        val dNorm = -0.5 * shapeCoeff.size * math.log(2 * math.Pi) - shapeCoeff.size * math.log(sdev)
        dNorm - 0.5 * dDiff
     */
    val shapePrior = {
      val n = variableModel.ptsVar.shape(0)
      val s = tf.cumsum(tf.square(variableModel.ptsVar), 0)
      (-0.5f * n * math.log(2 * math.Pi)).toFloat - (0.5f * s(n - 1))
    }


    //println("reg", reg)
    println("rec", rec)
    //val loss = rec + reg / 37f / 1000f  //+ reg * 0.1f

    //val shapeLH = -0.5

    val loss = -(lh + shapePrior)

    println("loss", loss)

    val ys = Seq(loss)
    val xs: Seq[Output[Float]] = Seq(
      variableModel.ptsVar, variableModel.colorsVar,
      variableModel.illumVar,
      variableModel.poseRotVar
    )
    val grad: Seq[OutputLike[Float]] = Gradients.gradients(
      ys,
      xs,
      Float
    )

    println(s"val grad: $grad")

    println()

    val optimizer = tf.train.AMSGrad(0.05f, name = "adal")
    // TODO: applyGradients requires Variable[Any] here for some reason, bug maybe?
    val gradients: Seq[(OutputLike[Float], api.tf.Variable[Any])] = Seq(
      (grad(0), variableModel.ptsVarVariable),
      (grad(1), variableModel.colorsVarVariable),
      (grad(2), variableModel.illumVarVariable),
      (grad(3), variableModel.poseRotVarVariable)
    )
    val optFn = optimizer.applyGradients(
      gradients
    )

    //    val optFn = optimizer.minimize(-loss)


    val session = Session()
    session.run(targets = tf.globalVariablesInitializer())

    for (i <- 0 to 180) {
      val result = session.run(
        feeds = Map(target -> targetImage) ++ variableModel.feeds,
        fetches = Seq(renderer.shShader, loss, rec, lh, shapePrior),
        targets = Seq(optFn)
      )

      println(s"iter $i", result(1).toTensor.summarize(), result(2).toTensor.summarize(), result(3).toTensor.summarize(), result(4).toTensor.summarize())

      if (i % 30 == 0) {
        val rendering = TFConversions.oneToOneTensorImage3dToPixelImage(result.head.toTensor.transpose(permutation = Tensor(1, 0, 2)))
        PixelImageIO.write(rendering, new File(s"/tmp/${i}_tf_rendering.png")).get
      }
    }

    val fetch = session.run(
      feeds = Map(target -> targetImage) ++ variableModel.feeds,
      fetches = Seq(renderer.pts, loss)
    )

    val finalMesh = {
      val vtx = {
        val finalPts = fetch.head.toTensor
        val n = finalPts.shape(1)

        for (i <- 0 until n) yield {
          val x = finalPts(0, i).entriesIterator.toIndexedSeq(0).toDouble
          val y = finalPts(1, i).entriesIterator.toIndexedSeq(0).toDouble
          val z = finalPts(2, i).entriesIterator.toIndexedSeq(0).toDouble
          Point(x, y, z)
        }
      }

      TriangleMesh3D(vtx, mesh.shape.triangulation)
    }

    val finalFullMesh = VertexColorMesh3D(finalMesh, mesh.color)
    MeshIO.write(finalFullMesh, new File("/tmp/jimmi.ply")).get
  }

  /**
    * Method to test landmark calculation through several methods: MoMo, deprecated calculation with TF and batch
    * calculation with TF.
    */
  def testLandmarks(param: RenderParameter, model: MoMo, image: PixelImage[RGB], initPose: TFPoseTensor, initCamera: TFCameraTensor, mesh: VertexColorMesh3D): Unit = {
    // Parameter initialization
    val verificationParam = {
      val rps = RenderParameterIO.read(new File("../../Documents/datasets/data_300K_fewParameters50_crop_noSensor_lms/0_0.rps")).get
      rps.copy(momo = rps.momo.copy(expression = rps.momo.expression.take(5)), camera = param.camera.copy(principalPoint = rps.camera.principalPoint, sensorSize = rps.camera.sensorSize), imageSize = param.imageSize)
    }

    def coefficientsToTensor(coefficients: MoMoCoefficients): Tensor[Float] = {
      TFMoMoConversions.toTensor(DenseVector.vertcat(
        coefficients.shape,
        DenseVector.zeros[Double](80 - coefficients.shape.length),
        coefficients.expression,
        DenseVector.zeros[Double](5 - coefficients.expression.length)
      ))
    }

    val paramTensorProto = coefficientsToTensor(param.momo.coefficients)
    val verificationParamTensor = coefficientsToTensor(verificationParam.momo.coefficients)

    val paramTensor = paramTensorProto.transpose()

    val paramTensorStacked = using(Session())(_.run(fetches = tf.concatenate(Seq(paramTensorProto.toOutput, verificationParamTensor.toOutput), axis = 0)))

    // Prepare render parameters
    val translation = Tensor(TFConversions.vec2Tensor(param.pose.translation).transpose(), TFConversions.vec2Tensor(verificationParam.pose.translation).transpose())

    val cameraSensorSize = Output(Output(param.camera.sensorSize.x.toFloat, param.camera.sensorSize.y.toFloat), Output(verificationParam.camera.sensorSize.x.toFloat, verificationParam.camera.sensorSize.y.toFloat))

    val cameraPrincipalPoint = Output(Output(param.camera.principalPoint.x.toFloat, param.camera.principalPoint.y.toFloat),
      Output(verificationParam.camera.principalPoint.x.toFloat, verificationParam.camera.principalPoint.y.toFloat))

    val roll = Output(param.pose.roll.toFloat, verificationParam.pose.roll.toFloat)
    val pitch = Output(param.pose.pitch.toFloat, verificationParam.pose.pitch.toFloat)
    val yaw = Output(param.pose.yaw.toFloat, verificationParam.pose.yaw.toFloat)
    val cameraNear = param.camera.near.toFloat
    val cameraFar = param.camera.far.toFloat
    val focalLength = param.camera.focalLength.toFloat

    //    println(paramTensorStacked.shape)

    //    println(paramTensor.shape)

    //    val results = using(Session())(session => {
    //      session.run(targets = tf.globalVariablesInitializer())
    //      val assignOp = variableModel.ptsVar.assign(paramTensor)
    //      session.run(targets = assignOp)
    //      session.run(feeds = variableModel.feeds, fetches = variableModel.pts)
    //    })

    val landmarkId = "left.eye.corner_outer"

    val landmarkPointId = model.landmarkPointId(landmarkId).get

    val tfLandmarksRenderer = time("Creating landmarksRenderer", {
      TFLandmarksRenderer.transposed(model.expressionModel.get.truncate(80, 40, 5), IndexedSeq(0, landmarkPointId.id))
    })

    val tfBatchLandmarksRenderer = time("Creating batchLandmarksRenderer", {
      TFLandmarksRenderer(model.expressionModel.get.truncate(80, 40, 5), IndexedSeq(0, landmarkPointId.id))
    })

    val tfLandmarksRendererMesh = tfLandmarksRenderer.getInstance(paramTensor)
    val tfBatchLandmarksRendererMesh = tfBatchLandmarksRenderer.batchGetInstance(paramTensorStacked)
    //    println("LandmarksRenderer Mesh:")
    //    println(tfBatchLandmarksRendererMesh.summarize())

    val landmarksRenderer = MoMoRenderer(model, RGBA.BlackTransparent)

    val landmark = landmarksRenderer.renderLandmark(landmarkId, param).get
    val verificationLandmark = landmarksRenderer.renderLandmark(landmarkId, verificationParam).get

    val tfLandmarks = tfLandmarksRenderer.calculateLandmarks(paramTensor, TFPose(initPose), TFCamera(initCamera), image.width, image.height)
    val tfVerificationLandmarks = tfLandmarksRenderer.calculateLandmarks(verificationParamTensor.transpose(), TFPose(TFPose(verificationParam.pose)), TFCamera(TFCamera(verificationParam.camera)), image.width, image.height)

    val tfBatchLandmarks = tfBatchLandmarksRenderer.batchCalculateLandmarks(paramTensorStacked, roll, pitch, yaw,
      translation, cameraNear, cameraFar, cameraSensorSize, focalLength, cameraPrincipalPoint, image.width, image.height)
    //    println("Batch landmarks:")
    //    println(tfBatchLandmarks.summarize())
    val tfBatchLandmarks2D = using(Session())(_.run(fetches = tfBatchLandmarksRenderer.batchCalculateLandmarks2D(paramTensorStacked, roll, pitch, yaw,
      translation, cameraSensorSize, focalLength, cameraPrincipalPoint, image.width, image.height)))

    println(s"Mesh pt0:                         ${mesh.shape.pointSet.point(PointId(0))}")
    //    println(s"TFMesh pt0:                  ${tfMesh.pts(0, 0).scalar}, ${tfMesh.pts(1, 0).scalar}, ${tfMesh.pts(2, 0).scalar}")
    //    println(s"variableModel pt0:           ${results(0, 0).scalar}, ${results(1, 0).scalar}, ${results(2, 0).scalar}")
    println(s"tfLandmarksRendererMesh pt0:      ${tfLandmarksRendererMesh(0, 0).scalar}, ${tfLandmarksRendererMesh(1, 0).scalar}, ${tfLandmarksRendererMesh(2, 0).scalar}")
    println(s"tfBatchLandmarksRendererMesh pt0: ${tfBatchLandmarksRendererMesh(0, 0, 0).scalar}, ${tfBatchLandmarksRendererMesh(0, 0, 1).scalar}, ${tfBatchLandmarksRendererMesh(0, 0, 2).scalar}")
    println()

    val verificationMesh = model.instance(verificationParam.momo.coefficients)

    println(s"Mesh verification pt0:                         ${verificationMesh.shape.pointSet.point(PointId(0))}")
    println(s"tfBatchLandmarksRendererMesh verification pt0: ${tfBatchLandmarksRendererMesh(1, 0, 0).scalar}, ${tfBatchLandmarksRendererMesh(1, 0, 1).scalar}, ${tfBatchLandmarksRendererMesh(1, 0, 2).scalar}")
    println()

    //    val landmarkResults = {
    //      val landmarkPoint = results(::, landmarkPointId.id).expandDims(1)
    //      println(s"Param image size: ${param.imageSize.width}, ${param.imageSize.height}")
    //      println(s"Image size: ${image.width}, ${image.height}")
    //      val normalizedDeviceCoordinates = Transformations.objectToNDC(landmarkPoint, TFPose(initPose), TFCamera(initCamera))
    //      val tfNormalizedDeviceCoordinates = Transformations.ndcToTFNdc(normalizedDeviceCoordinates, image.width, image.height)
    //      // screenCoordinates are the correct landmark points
    //      val screenCoordinates = Transformations.screenTransformation(normalizedDeviceCoordinates, image.width, image.height)
    //      val tfScreenCoordinates = Transformations.screenTransformation(tfNormalizedDeviceCoordinates, image.width, image.height)
    //
    //      using(Session())(_.run(fetches = Seq(screenCoordinates, tfScreenCoordinates)))
    //    }

    println(s"Mesh $landmarkId:                         ${mesh.shape.pointSet.point(landmarkPointId)}")
    //    println(s"variableModel $landmarkId:           ${results(0, landmarkPointId.id).scalar}, ${results(1, landmarkPointId.id).scalar}, ${results(2, landmarkPointId.id).scalar}")
    println(s"tfLandmarksRendererMesh $landmarkId:      ${tfLandmarksRendererMesh(0, 1).scalar}, ${tfLandmarksRendererMesh(1, 1).scalar}, ${tfLandmarksRendererMesh(2, 1).scalar}")
    println(s"tfBatchLandmarksRendererMesh $landmarkId: ${tfBatchLandmarksRendererMesh(0, 1, 0).scalar}, ${tfBatchLandmarksRendererMesh(0, 1, 1).scalar}, ${tfBatchLandmarksRendererMesh(0, 1, 2).scalar}")
    println()

    println(s"Mesh verification $landmarkId:                         ${verificationMesh.shape.pointSet.point(landmarkPointId)}")
    println(s"tfBatchLandmarksRendererMesh verification $landmarkId: ${tfBatchLandmarksRendererMesh(1, 1, 0).scalar}, ${tfBatchLandmarksRendererMesh(1, 1, 1).scalar}, ${tfBatchLandmarksRendererMesh(1, 1, 2).scalar}")
    println()

    println(s"Normal renderer landmark:            ${landmark.point}")
    //    println(s"TF Landmark:                  ${landmarkResults.head(0, 0).scalar}, ${landmarkResults.head(1, 0).scalar}, ${landmarkResults.head(2, 0).scalar}")
    //    println(s"TF Transformed Landmark: ${landmarkResults(1)(0, 0).scalar}, ${landmarkResults(1)(1, 0).scalar}, ${landmarkResults(1)(2, 0).scalar}")
    println(s"TFLandmarksRenderer Landmark:        ${tfLandmarks(0, 1).scalar}, ${tfLandmarks(1, 1).scalar}, ${tfLandmarks(2, 1).scalar}")
    println(s"TFBatchLandmarksRenderer Landmark:   ${tfBatchLandmarks(0, 1, 0).scalar}, ${tfBatchLandmarks(0, 1, 1).scalar}, ${tfBatchLandmarks(0, 1, 2).scalar}")
    println(s"TFBatchLandmarksRenderer Landmark2D: ${tfBatchLandmarks2D(0, 1, 0).scalar}, ${tfBatchLandmarks2D(0, 1, 1).scalar}")
    println()

    println(s"Normal renderer verification landmark:            ${verificationLandmark.point}")
    println(s"TFLandmarksRenderer verification Landmark:        ${tfVerificationLandmarks(0, 1).scalar}, ${tfVerificationLandmarks(1, 1).scalar}, ${tfVerificationLandmarks(2, 1).scalar}")
    println(s"TFBatchLandmarksRenderer verification Landmark:   ${tfBatchLandmarks(1, 1, 0).scalar}, ${tfBatchLandmarks(1, 1, 1).scalar}, ${tfBatchLandmarks(1, 1, 2).scalar}")
    println(s"TFBatchLandmarksRenderer verification Landmark2D: ${tfBatchLandmarks2D(1, 1, 0).scalar}, ${tfBatchLandmarks2D(1, 1, 1).scalar}")
    println()

    val neutralModel = model.neutralModel

    val basicLandmarksRenderer = time("Creating basicLandmarksRenderer", {
      TFLandmarksRenderer.transposed(neutralModel.truncate(80, 40), IndexedSeq(0, landmarkPointId.id))
    })

    val basicBatchLandmarksRenderer = time("Creating basicBatchLandmarksRenderer", {
      TFLandmarksRenderer(neutralModel.truncate(80, 40), IndexedSeq(0, landmarkPointId.id))
    })

    val basicParams = paramTensor(0 :: 80)
    val basicBatchParams = paramTensorStacked(::, 0 :: 80)

    val basicInstance = basicLandmarksRenderer.getInstance(basicParams)
    val basicLandmarks = basicLandmarksRenderer.calculateLandmarks(basicParams, TFPose(initPose), TFCamera(initCamera), image.width, image.height)

    val basicBatchInstance = basicBatchLandmarksRenderer.batchGetInstance(basicBatchParams)
    val basicBatchLandmarks  = basicBatchLandmarksRenderer.batchCalculateLandmarks(basicBatchParams, roll, pitch, yaw,
      translation, cameraNear, cameraFar, cameraSensorSize, focalLength, cameraPrincipalPoint, image.width, image.height)

    //    println("Basic batch instance:")
    //    println(basicBatchInstance.summarize())

    val neutralMesh = neutralModel.instance(param.momo.coefficients)

    val neutralLandmarksRenderer = MoMoRenderer(neutralModel, RGBA.BlackTransparent)
    val neutralLandmark = neutralLandmarksRenderer.renderLandmark(landmarkId, param).get
    val neutralVerificationLandmark = neutralLandmarksRenderer.renderLandmark(landmarkId, verificationParam).get

    println("Neutral model evaluation:")

    println(s"Mesh pt0:                         ${neutralMesh.shape.pointSet.point(PointId(0))}")
    println(s"tfLandmarksRendererMesh pt0:      ${basicInstance(0, 0).scalar}, ${basicInstance(1, 0).scalar}, ${basicInstance(2, 0).scalar}")
    println(s"tfBatchLandmarksRendererMesh pt0: ${basicBatchInstance(0, 0, 0).scalar}, ${basicBatchInstance(0, 0, 1).scalar}, ${basicBatchInstance(0, 0, 2).scalar}")
    println()

    println(s"Mesh $landmarkId:                         ${neutralMesh.shape.pointSet.point(landmarkPointId)}")
    println(s"tfLandmarksRendererMesh $landmarkId:      ${basicInstance(0, 1).scalar}, ${basicInstance(1, 1).scalar}, ${basicInstance(2, 1).scalar}")
    println(s"tfBatchLandmarksRendererMesh $landmarkId: ${basicBatchInstance(0, 1, 0).scalar}, ${basicBatchInstance(0, 1, 1).scalar}, ${basicBatchInstance(0, 1, 2).scalar}")
    println()

    println(s"Normal renderer landmark:          ${neutralLandmark.point}")
    println(s"TFLandmarksRenderer Landmark:      ${basicLandmarks(0, 1).scalar}, ${basicLandmarks(1, 1).scalar}, ${basicLandmarks(2, 1).scalar}")
    println(s"TFBatchLandmarksRenderer Landmark: ${basicBatchLandmarks(0, 1, 0).scalar}, ${basicBatchLandmarks(0, 1, 1).scalar}, ${basicBatchLandmarks(0, 1, 2).scalar}")
    println()

    println(s"Normal renderer verification landmark:          ${neutralVerificationLandmark.point}")
    println(s"TFBatchLandmarksRenderer verification Landmark: ${basicBatchLandmarks(1, 1, 0).scalar}, ${basicBatchLandmarks(1, 1, 1).scalar}, ${basicBatchLandmarks(1, 1, 2).scalar}")
  }
}
