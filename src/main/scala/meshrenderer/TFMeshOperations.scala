package meshrenderer

import java.io.File

import org.platanios.tensorflow.api.{tf, _}
import scalismo.common.PointId
import scalismo.faces.io.MeshIO
import scalismo.mesh.{TriangleId, TriangleMesh3D}

import scala.collection.immutable.Seq

/**
  * Created by andreas on 8/11/18.
  */
object TFMeshOperations {

  def triangleNormals(vtx: Output[Float], triangles: Output[Int]): Output[Float] = {
    val vtxsPerTriangle = tf.gather(vtx, triangles)

    val pt1 = vtxsPerTriangle(::, 0)
    val pt2 = vtxsPerTriangle(::, 1)
    val pt3 = vtxsPerTriangle(::, 2)

    val u = pt2 - pt1
    val v = pt3 - pt1

//    println("u", u)

    val cross = tf.cross(u,v)

    tf.l2Normalize(cross, axes=Seq(1))
  }

  def vertexNormals(cellNormals: Output[Float], triangleIdsForPoint: Output[Int]): Output[Float] = {
//    println("cellNromals", cellNormals)
//    println("triangleIdsForPoint", triangleIdsForPoint)
    tf.createWith(nameScope="vertexNormals") {
      val normalsPerVertex = tf.gather(cellNormals, triangleIdsForPoint)

      val validEntries = triangleIdsForPoint > -1

      val sumValidEntries = tf.countNonZero(validEntries.toInt, axes = Seq(1))

//      println("validEntries", validEntries)
//      println("sumValidEntries", sumValidEntries)

//      println("normalsPerVertex", normalsPerVertex)
      val temp0 = tf.expandDims(validEntries, 2)
      val temp1 = tf.tile(temp0, Shape(1, 1, 3))
      tf.l2Normalize(
        tf.sum(normalsPerVertex * temp1.toFloat, axes = Seq(1)),
        axes = Seq(1)
      )
    }
  }


  def vertexNormals(vtx: Output[Float], triangles: Output[Int],  triangleIdsForPoint: Output[Int]): Output[Float] = {
    val cellNormals = triangleNormals(vtx, triangles)
    vertexNormals(cellNormals, triangleIdsForPoint)
  }

  def trianglesForPoint(data: IndexedSeq[(PointId, IndexedSeq[TriangleId])]): Tensor[Int] = {
    val sorted = data.toIndexedSeq.sortBy(_._1.id)
    val maxNeighbouringTriangles = 8
    val listOfTensors = sorted.map { case (_, triangles) =>
      val space = Array.fill(maxNeighbouringTriangles)(-1)
      var i = 0
      for (t <- triangles) {
        space(i) = t.id
        i += 1
      }
      Tensor(space)
    }
    Tensor(listOfTensors).reshape(Shape(data.length, maxNeighbouringTriangles))
  }

  def adjacentPoints(mesh: TriangleMesh3D): Tensor[Int] = {
    val data = mesh.triangulation.pointIds.map { ptId =>
      val adj = mesh.triangulation.adjacentPointsForPoint(ptId)
      (ptId, adj)
    }
    val sorted = data.toIndexedSeq.sortBy(_._1.id)
    val maxNeighs = 8
    val listOfTensors = sorted.map { case (_, neighs) =>
      val space = Array.fill(maxNeighs)(-1)
      var i = 0
      for (n <- neighs) {
        space(i) = n.id
        i += 1
      }
      Tensor(space)
    }
    Tensor(listOfTensors).reshape(Shape(data.length, maxNeighs))
  }

  /** adjacentPoints: #points X maximum possible adjacent points
    * point data:     #points X data dim*/
  def vertexToNeighbourDistance(adjacentPoints: Output[Int], pointData: Output[Int]): Output[Int] = {
    println("adjacentPoints", adjacentPoints)
    println("pointData", pointData)

    val neighValuesPerVertex = tf.gather(pointData, adjacentPoints)
    println("neighValuesPerVertex", neighValuesPerVertex)
    val vertexTiled = tf.tile(tf.expandDims(pointData, 1), Seq(1, adjacentPoints.shape(1), 1))
    println("vertexTiled", vertexTiled)
    val neighsToVertex = tf.subtract(vertexTiled, neighValuesPerVertex)
    println("neighsToVertex", neighsToVertex)

    val validEntries = adjacentPoints > -1
    println("validEntries", validEntries)
    //val sumValidEntries = tf.countNonZero(validEntries, axes=Seq(1))
    val validEntriesTiled = tf.tile(tf.expandDims(validEntries, 2), Seq(1, 1, pointData.shape(1)))
    println("validEntriesTiled", validEntriesTiled)

    val validDifferences = neighsToVertex * validEntriesTiled.toInt
    println("validDifferences", validDifferences)
    val res = tf.sum(tf.abs(validDifferences))
    println("res", res)
    res
  }

  def main(args: Array[String]): Unit = {
    val mesh = MeshIO.read(new File("/home/andreas/export/mean2012_l7_bfm_pascaltex.msh.gz")).get.colorNormalMesh3D.get

    val triForPoint = mesh.shape.pointSet.pointIds.toIndexedSeq.map { id =>
      (id, mesh.shape.triangulation.adjacentTrianglesForPoint(id))
    }

    val trianglesForPointData = trianglesForPoint(triForPoint)

    println("trianglesForPointData", trianglesForPointData)

    val triangles = TFMesh.triangulationAsTensor(mesh.shape.triangulation)
    val pts = TFConversions.pointsToTensorTransposed(mesh.shape.position.pointData)

    val cellNormals = triangleNormals(pts.transpose(), triangles)

    println("cellNormals", cellNormals.shape)

    val vtxNormals = vertexNormals(cellNormals, trianglesForPointData)

    println("vtxNormals", vtxNormals)

    val session = Session()
    val res = session.run(fetches = vtxNormals)

    val tfVtx = res(0)
    println(tfVtx(100,::).summarize())

    println(mesh.shape.vertexNormals.pointData(100))

  }
}
