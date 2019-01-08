package meshrenderer

import java.io.File

import breeze.linalg.DenseVector
import org.platanios.tensorflow.api
import org.platanios.tensorflow.api.ops.Gradients
import org.platanios.tensorflow.api.{tf, _}
import scalismo.common.PointId
import scalismo.faces.color.{RGB, RGBA}
import scalismo.faces.io.{MeshIO, MoMoIO, PixelImageIO, RenderParameterIO}
import scalismo.faces.mesh.{ColorNormalMesh3D, VertexColorMesh3D}
import scalismo.faces.momo.MoMo
import scalismo.faces.parameters._
import scalismo.faces.sampling.face.MoMoRenderer
import scalismo.geometry.Point
import scalismo.mesh.TriangleMesh3D


/**
  * Created by andreas on 8/8/18.
  */
object Example {
  def main(args: Array[String]): Unit = {
    // TODO: Find equivalent of Registry.register
    //    Registry.register("RasterizeTriangles", Rasterizer.rasterizeTrianglesGrad)

    /*val param = RenderParameter.defaultSquare.withPose(Pose(1.0, Vector(0,0,-1000), 0,1.1,0)).fitToImageSize(200,250).withEnvironmentMap(
      SphericalHarmonicsLight(SphericalHarmonicsLight.frontal.coefficients ++ IndexedSeq(Vector(0.5,0.5,0.1), Vector(0.2,0.1,0.7), Vector(0.0,0.2,0.0), Vector(0.2,0.1,-0.1), Vector(-0.1,-0.1,-0.1)))
    )*/
    //
    val imageFn = new File("data/JimmyCarterPortrait2.png")
    val image = {
      val img = PixelImageIO.read[RGB](imageFn).get
      img.resample((img.width * 0.1).toInt, (img.height * 0.1).toInt)
    }
    val param = RenderParameterIO.read(new File("data/fit-best.rps")).get.fitToImageSize(image.width, image.height)
    //val image = PixelImageIO.read[RGB](new File("/tmp/tf_rendering_sh_lambert.png")).get

    val model = time("Initializing and loading model",
      {
        scalismo.initialize()
        val momoFn = new File("../face-autoencoder/model2017-1_bfm_nomouth.h5")
        MoMoIO.read(momoFn).get
      })

    //val mesh = MeshIO.read(new File("/home/andreas/export/mean2012_l7_bfm_pascaltex.msh.gz")).get.colorNormalMesh3D.get
    val mesh = model.instance(param.momo.coefficients)

    time("Writing temp model", {MeshIO.write(mesh, new File("/tmp/jimmi_fit.ply")).get})

    val tfMesh = time("Transforming TFMesh", {TFMesh(mesh)})

    val initPose = time("Transforming TFPose", {TFPose(param.pose)})
    val initCamera = time("Transforming TFCamera", {TFCamera(param.camera)})
    val initLight = time("Transforming TFLight", {TFConversions.pointsToTensor(param.environmentMap.coefficients).transpose()})

    //val variableModel = new OffsetFromInitializationModel(tfMesh.pts, tfMesh.colors, initPose, initCamera, initLight)
    val tfModel = time("Creating TFMoMo", {TFMoMo(model.expressionModel.get.truncate(80, 40, 5))})
    /*
    ==================================
    This part used the incorrect mean!
    ==================================
     */
    //    val tfMean = TFMesh(model.neutralModel.mean)
    val tfMean = time("Creating mean TFMesh", {TFMesh(model.mean)})

    val variableModel = time("Creating variable model", {TFMoMoExpressParameterModel(tfModel, tfMean, initPose, initCamera, initLight)})

    val results = {
      val session = Session()
      session.run(targets = tf.globalVariablesInitializer())
      val paramTensor = TFMoMoConversions.toTensor(DenseVector.vertcat(
        param.momo.coefficients.shape,
        DenseVector.zeros[Double](tfModel.shape.shape(1) - param.momo.coefficients.shape.length),
        param.momo.coefficients.expression,
        DenseVector.zeros[Double](tfModel.expression.shape(1) - param.momo.coefficients.expression.length)
      )).transpose()
      val assignOp = variableModel.ptsVar.assign(paramTensor)
      session.run(targets = assignOp)
      session.run(feeds = variableModel.feeds, fetches = variableModel.pts)
    }

    println(s"Mesh pt0: ${mesh.shape.pointSet.point(PointId(0))}")
    println(s"TFMesh pt0: ${tfMesh.pts(0, 0).scalar}, ${tfMesh.pts(1, 0).scalar}, ${tfMesh.pts(2, 0).scalar}")

    println(s"variableModel pt0: ${results(0, 0).scalar}, ${results(1, 0).scalar}, ${results(2, 0).scalar}")
    //    println(tfMesh.pts.summarize(10))
    //    println(results.summarize(10))

    val landmarkId = "left.eye.corner_outer"

    val landmarkPointId = model.landmarkPointId(landmarkId).get

    val landmarkResults = {
      val landmarkPoint = results(::, landmarkPointId.id).expandDims(1)
      println(s"Param image size: ${param.imageSize.width}, ${param.imageSize.height}")
      println(s"Image size: ${image.width}, ${image.height}")
      val normalizedDeviceCoordinates = Transformations.objectToNDC(landmarkPoint, TFPose(initPose), TFCamera(initCamera))
      val tfNormalizedDeviceCoordinates = Transformations.ndcToTFNdc(normalizedDeviceCoordinates, image.width, image.height)
      // screenCoordinates are the correct landmark points
      val screenCoordinates = Transformations.screenTransformation(normalizedDeviceCoordinates, image.width, image.height)
      val tfScreenCoordinates = Transformations.screenTransformation(tfNormalizedDeviceCoordinates, image.width, image.height)

      val session = Session()
      session.run(fetches = Seq(screenCoordinates, tfScreenCoordinates))
    }

    println(s"Mesh $landmarkId: ${mesh.shape.pointSet.point(landmarkPointId)}")

    println(s"variableModel $landmarkId: ${results(0, landmarkPointId.id).scalar}, ${results(1, landmarkPointId.id).scalar}, ${results(2, landmarkPointId.id).scalar}")

    val landmarksRenderer = MoMoRenderer(model, RGBA.BlackTransparent)

    val landmark = landmarksRenderer.renderLandmark(landmarkId, param).get

    println(s"Normal renderer landmark: ${landmark.point}")

    println(s"TF Landmark: ${landmarkResults.head(0, 0).scalar}, ${landmarkResults.head(1, 0).scalar}, ${landmarkResults.head(2, 0).scalar}")
    println(s"TF Transformed Landmark: ${landmarkResults(1)(0, 0).scalar}, ${landmarkResults(1)(1, 0).scalar}, ${landmarkResults(1)(2, 0).scalar}")

    System.exit(0)

    val renderer = TFRenderer(tfMesh, variableModel.pts, variableModel.colors, variableModel.pose, variableModel.camera, variableModel.illumination, param.imageSize.width, param.imageSize.height)

    def renderInitialParametersAndCompareToGroundTruth(): Unit = {
      val sess = Session()
      sess.run(targets = tf.globalVariablesInitializer())

      val result = sess.run(
        feeds = variableModel.feeds,
        fetches = Seq(renderer.shShader)
      )

      val tensorImg = result.head.toTensor
      val img = TFConversions.oneToOneTensorImage3dToPixelImage(tensorImg)
      PixelImageIO.write[RGB](img, new File("/tmp/fit.png")).get

      {
        val img = ParametricRenderer.renderParameterMesh(param, ColorNormalMesh3D(mesh), RGBA.Black)
        PixelImageIO.write[RGBA](img, new File("/tmp/fitgt.png")).get
      }
    }

    val targetImage = TFConversions.image3dToTensor(image)

    val test = TFConversions.oneToOneTensorImage3dToPixelImage(targetImage)
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
    val reg = tf.sum(tf.abs(variableModel.ptsVar))

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
      val part1 = 0.5f * s(n - 1)
      var part2 = Tensor(-0.5f * n * math.log(2 * math.Pi)).toFloat
      part2 - part1
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
    val optimizer = tf.train.AMSGrad(0.1f, name = "adal")
    // TODO: applyGradients requires Variable[Any] here for some reason, bug maybe?
    val gradients: Seq[(OutputLike[Float], api.tf.Variable[Any])] = Seq(
      (grad(0), variableModel.ptsVar),
      (grad(1), variableModel.colorsVar),
      (grad(2), variableModel.illumVar),
      (grad(3), variableModel.poseRotVar)
    )
    val optFn = optimizer.applyGradients(
      gradients
    )


    val session = Session()
    session.run(targets = tf.globalVariablesInitializer())

    for (i <- 0 to 180) {
      val result = session.run(
        feeds = Map(target -> targetImage) ++ variableModel.feeds,
        fetches = Seq(renderer.shShader, loss, rec, lh, shapePrior),
        targets = Seq(optFn)
      )

      println(s"iter ${i}", result(1).toTensor.summarize(), result(2).toTensor.summarize(), result(3).toTensor.summarize(), result(4).toTensor.summarize())

      if (i % 30 == 0) {
        val rendering = TFConversions.oneToOneTensorImage3dToPixelImage(result(0).toTensor)
        PixelImageIO.write(rendering, new File(s"/tmp/${i}_tf_rendering.png")).get
      }
    }

    val fetch = session.run(
      feeds = Map(target -> targetImage) ++ variableModel.feeds,
      fetches = Seq(renderer.pts, loss)
    )

    val finalMesh = {
      val vtx = {
        val finalPts = fetch(0).toTensor
        val n = finalPts.shape(1)

        for (i <- 0 until n) yield {
          val x = finalPts(0, i).entriesIterator.toIndexedSeq(0).asInstanceOf[Float].toDouble
          val y = finalPts(1, i).entriesIterator.toIndexedSeq(0).asInstanceOf[Float].toDouble
          val z = finalPts(2, i).entriesIterator.toIndexedSeq(0).asInstanceOf[Float].toDouble
          Point(x, y, z)
        }
      }

      TriangleMesh3D(vtx, mesh.shape.triangulation)
    }

    val finalFullMesh = VertexColorMesh3D(finalMesh, mesh.color)
    MeshIO.write(finalFullMesh, new File("/tmp/jimmi.ply")).get

  }

  def time[R](name: String, block: => R): R = {
    println(s"$name...")
    val start = System.nanoTime()
    val result = block
    val end = System.nanoTime()
    println(s"$name complete: ${(end - start) / 1e9}s")

    result
  }
}
