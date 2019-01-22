# Inverse Renderer in Scala with Tensorflow

Uses the `rasterize_triangles_kernel.so` from [tf_mesh_renderer](https://github.com/google/tf_mesh_renderer)
(commit a6403fbb36a71443ecb822e435e5724550d2b52b or earlier). The kernel must be compiled using the version of
TensorFlow used by the TensorFlow Scala binary used (`tf-nightly-gpu==1.13.0.dev20181121` for TensorFlow Scala GPU
0.4.1).

If TensorFlow Scala claims to require a shape function for the `RasterizeTriangles` operation, you may have to add the
following code snippet to the operation registration in `rasterize_triangles_op.cc` before compiling the
`rasterize_triangles_kernel.so`:
```
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
```

Features:

* interpolate vertex attributes
* spherical harmonics illumination
* optimize for color, pose, illumination, vertex position, camera parameters. 


Please cite: 
```latex
@article{schneider2017efficient,
  title={Efficient global illumination for morphable models},
  author={Schneider, Andreas and Sch{\"o}nborn, Sandro and Frobeen, Lavrenti and Vetter, Thomas and Egger, Bernhard},
  year={2017},
  publisher={IEEE}
}
```

For the rasterizer kernel cite the paper linked in the README of the [tf_mesh_renderer](https://github.com/google/tf_mesh_renderer) project.