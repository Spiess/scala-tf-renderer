package meshrenderer

import org.platanios.tensorflow.api.ops.Gradients
import org.platanios.tensorflow.api.{tf, _}


/**
  * Created by andreas on 8/8/18.
  */
object Rasterizer {
  val path = "lib/rasterize_triangles_kernel.so"

  //keeps the data from the rasterizer
  case class RasterizationOutput(barycetricImage: Output[Float], triangleIds: Output[Int], zBufferImage: Output[Float])

  /**
    * Implements a rasterization kernel for rendering mesh geometry.
    *
    * @param vertices     2-D tensor with shape [vertex_count, 3]. The 3-D positions of the mesh vertices in Normalized
    *                     Device Coordinates.
    * @param triangles    2-D tensor with shape [triangle_count, 3]. Each row is a tuple of
    *                     indices into vertices specifying a triangle to be drawn. The triangle has an
    *                     outward facing normal when the given indices appear in a clockwise winding to
    *                     the viewer.
    * @param image_width  positive int attribute specifying the width of the output image.
    * @param image_height positive int attribute specifying the height of the output image.
    *
    * barycentric_coordinates: 3-D tensor with shape [image_height, image_width, 3]
    *                     containing the rendered barycentric coordinate triplet per pixel, before
    *                     perspective correction. The triplet is the zero vector if the pixel is outside
    *                     the mesh boundary. For valid pixels, the ordering of the coordinates
    *                     corresponds to the ordering in triangles.
    * triangle_ids: 2-D tensor with shape [image_height, image_width]. Contains the
    *                     triangle id value for each pixel in the output image. For pixels within the
    *                     mesh, this is the integer value in the range [0, num_vertices] from triangles.
    *                     For vertices outside the mesh this is 0; 0 can either indicate belonging to
    *                     triangle 0, or being outside the mesh. This ensures all returned triangle ids
    *                     will validly index into the vertex array, enabling the use of tf.gather with
    *                     indices from this tensor. The barycentric coordinates can be used to determine
    *                     pixel validity instead.
    * z_buffer: 2-D tensor with shape [image_height, image_width]. Contains the Z
    *                     coordinate in vae.Normalized Device Coordinates for each pixel occupied by a
    *                     triangle.
    */
  def rasterize_triangles(vertices: Output[Float],
                          triangles: Output[Int],
                          image_width: Int,
                          image_height: Int,
                          name: String = "rasterize_triangles"): RasterizationOutput = {
    org.platanios.tensorflow.jni.TensorFlow.loadOpLibrary(path)


    val gradientFn: Gradients.GradientFn[Seq[Output[Any]], Seq[Output[Float]], Seq[Output[Any]], Seq[Output[Float]]] = rasterizeTrianglesGrad

    val inputs: Seq[Output[Any]] = Seq(vertices, triangles)

    val outs: Op[Seq[Output[Any]], Seq[Output[Float]]] = Op.Builder[Seq[Output[Any]], Seq[Output[Float]]](opType = "RasterizeTriangles", name, inputs, addAsIndividualInputs = true)
      .setAttribute("image_width", image_width)
      .setAttribute("image_height", image_height)
      .setGradientFn(gradientFn)
      .build()

    /*
    In case the Op.Builder quits with the error that it's missing a shape function recompile rasterizer kernel with:
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      int imgWidth;
      int imgHeight;
      c->GetAttr("image_width", &imgWidth);
      c->GetAttr("image_height", &imgHeight);
      c->set_output(0, c->MakeShape({imgHeight, imgWidth, 3}));
      c->set_output(1, c->MakeShape({imgHeight, imgWidth}));
      c->set_output(2, c->MakeShape({imgHeight, imgWidth}));
      return Status::OK();
    })
     */

    RasterizationOutput(outs.output.head, outs.output(1).toInt, outs.output(2))
  }

  def rasterizeTrianglesGrad(op: Op[Seq[Output[Any]], Seq[Output[Float]]], outputGradients: Seq[Output[Float]]): Seq[Output[Float]] = {
    /*
      The outputGradients Seq contains three entries: a gradient for each output
        (barycentric coordinates, triangle ids, z buffer)
      Only the gradient for the barycentric coordinates is used, therefore it doesn't matter that the gradient for the
      triangle ids is null.
     */

    val inputs: Seq[Output[Any]] = Seq(op.input.head.toFloat, op.input(1).toInt, op.output.head.toFloat, op.output(1).toInt, outputGradients.head)
    val outGrad = Op.Builder[Seq[Output[Any]], Seq[Output[Float]]](opType = "RasterizeTrianglesGrad", "rasterizeTrianglesGrad",
      input = inputs, addAsIndividualInputs = true)
      .setAttribute("image_width", op.longAttribute("image_width"))
      .setAttribute("image_height", op.longAttribute("image_height"))
      .build()

    /*
    In case the Op.Builder quits with the error that it's missing a shape function recompile rasterizer kernel with:
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
     */

    println("outGrad", outGrad.output.length, outGrad)
    /*
     The input triangles must be given a gradient. Since these are not supposed to change and don't have any meaningful
     gradient, we assign it a zero gradient.
    */
    Seq(outGrad.output.head.toFloat, tf.zeros[Float](op.input(1).shape))
  }
}
