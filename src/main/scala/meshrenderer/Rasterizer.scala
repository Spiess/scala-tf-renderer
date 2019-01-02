package meshrenderer

import org.platanios.tensorflow.api.{tf, _}


/**
  * Created by andreas on 8/8/18.
  */
object Rasterizer {
  val path = "lib/rasterize_triangles_kernel.so"

  //keeps the data from the rasterizer
  case class RasterizationOutput(barycetricImage: Output[Float], triangleIds: Output[Int], zBufferImage: Output[Float])

  """Implements a rasterization kernel for rendering mesh geometry.
    |
    |vertices: 2-D tensor with shape [vertex_count, 3]. The 3-D positions of the mesh
    |  vertices in Normalized Device Coordinates.
    |triangles: 2-D tensor with shape [triangle_count, 3]. Each row is a tuple of
    |  indices into vertices specifying a triangle to be drawn. The triangle has an
    |  outward facing normal when the given indices appear in a clockwise winding to
    |  the viewer.
    |image_width: positive int attribute specifying the width of the output image.
    |image_height: positive int attribute specifying the height of the output image.
    |barycentric_coordinates: 3-D tensor with shape [image_height, image_width, 3]
    |  containing the rendered barycentric coordinate triplet per pixel, before
    |  perspective correction. The triplet is the zero vector if the pixel is outside
    |  the mesh boundary. For valid pixels, the ordering of the coordinates
    |  corresponds to the ordering in triangles.
    |triangle_ids: 2-D tensor with shape [image_height, image_width]. Contains the
    |  triangle id value for each pixel in the output image. For pixels within the
    |  mesh, this is the integer value in the range [0, num_vertices] from triangles.
    |  For vertices outside the mesh this is 0; 0 can either indicate belonging to
    |  triangle 0, or being outside the mesh. This ensures all returned triangle ids
    |  will validly index into the vertex array, enabling the use of tf.gather with
    |  indices from this tensor. The barycentric coordinates can be used to determine
    |  pixel validity instead.
    |z_buffer: 2-D tensor with shape [image_height, image_width]. Contains the Z
    |  coordinate in vae.Normalized Device Coordinates for each pixel occupied by a
    |  triangle."""
  def rasterize_triangles(vertices: Output[Float],
                          triangles: Output[Float],
                          image_width: Int,
                          image_height: Int,
                          name: String = "rasterize_triangles"): RasterizationOutput = {
    org.platanios.tensorflow.jni.TensorFlow.loadOpLibrary(path)


    // TODO: Find out how Op registration works in TF Scala 0.4
    val outs = Op.Builder(opType = "RasterizeTriangles", name, input = Seq(vertices, triangles))
      .setAttribute("image_width", image_width)
      .setAttribute("image_height", image_height)
      .build()

    RasterizationOutput(outs(0), outs(1), outs(2))
  }

  def rasterizeTrianglesGrad(op: Op[Float, Float], outputGradients: Seq[Output[Float]]): Seq[Output[Float]] = {
    println("outputGradients", outputGradients.length)
    println("outputGradients", outputGradients(0))
    println("outputGradients", outputGradients(1))
    println("outputGradients", outputGradients(2))
    println("op.outputs", op.outputsSeq(0))
    println("op.outputs", op.outputsSeq(1))
    println("op.inputs", op.inputsSeq.length)
    println("op.outputs", op.inputsSeq.length)
    //outputGradients: dfdBarycentriCoordinates: Output, df_didsIgnored: Output, df_dzIgnored: Output
    // TODO: Find out how Op registration works in TF Scala 0.4
    val outGrad = Op.Builder(opType = "RasterizeTrianglesGrad", "rasterizeTrianglesGrad",
      input = Seq(op.inputsSeq(0).toFloat, op.inputsSeq(1).toFloat, op.outputsSeq(0).toFloat, op.outputsSeq(1).toFloat, outputGradients(0)))
      .setAttribute("image_width", op.longAttribute("image_width"))
      .setAttribute("image_height", op.longAttribute("image_height"))
      .build()

    println("outGrad", outGrad.outputsSeq.length, outGrad)
    Seq(outGrad(0), tf.identity(outGrad(0))) //zBuffer gradients missing but we need to supply something!
  }
}
