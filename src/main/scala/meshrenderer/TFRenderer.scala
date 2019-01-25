package meshrenderer

import org.platanios.tensorflow.api.core.client.FeedMap
import org.platanios.tensorflow.api.{tf, _}

/**
  * Created by andreas on 8/15/18.
  */

case class TFRenderer(mesh: TFMesh, pts: Output[Float], colors: Output[Float], pose: TFPose, camera: TFCamera, illumination: Output[Float], width: Int, height: Int) {

  val normals: Output[Float] = TFMeshOperations.vertexNormals(pts.transpose(), mesh.triangles, mesh.trianglesForPointData)
  val worldNormals: Output[Float] = {
    Transformations.poseRotationTransform(
      normals.transpose(), pose.pitch, pose.yaw, pose.roll
    ).transpose()
  }

  val ndcPts: Output[Float] = Transformations.objectToNDC(pts, pose, camera)
  val ndcPtsTf: Output[Float] = Transformations.transposedNdcToTFNdc(ndcPts, width, height).transpose()

  val triangleIdsAndBCC: Rasterizer.RasterizationOutput = Rasterizer.rasterize_triangles(ndcPtsTf, mesh.triangles, width, height)
  val vtxIdxPerPixel: Output[Int] = tf.gather(mesh.triangles, tf.reshape(triangleIdsAndBCC.triangleIds, Shape(-1)))

  val vtxIdxPerPixelGath: Output[Int] = tf.gatherND(mesh.triangles, tf.expandDims(triangleIdsAndBCC.triangleIds, 2))

  val interpolatedAlbedo: Output[Float] = Shading.interpolateVertexDataPerspectiveCorrect(triangleIdsAndBCC, vtxIdxPerPixel, colors, ndcPtsTf(::, 2))
  val interpolatedNormals: Output[Float] = Shading.interpolateVertexDataPerspectiveCorrect(triangleIdsAndBCC, vtxIdxPerPixel, worldNormals, ndcPtsTf(::, 2))
  //val lambert = Renderer.lambertShader(interpolatedAlbedo, Tensor(0.5f, 0.5f, 0.5f), Tensor(0.5f, 0.5f, 0.5f), Tensor(Seq(0f,0f,1f)), interpolatedNormals)
  val shShader: Output[Float] = Shading.sphericalHarmonicsLambertShader(interpolatedAlbedo, interpolatedNormals, illumination)
}

