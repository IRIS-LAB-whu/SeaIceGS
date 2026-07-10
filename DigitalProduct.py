import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm


def rasterize_triangle(dom, tri_2d, color):
    """Fill one projected triangle with a constant RGB color."""
    xs = [p[0] for p in tri_2d]
    ys = [p[1] for p in tri_2d]
    min_x, max_x = int(np.floor(min(xs))), int(np.ceil(max(xs)))
    min_y, max_y = int(np.floor(min(ys))), int(np.ceil(max(ys)))

    height, width, _ = dom.shape
    a, b, c = np.array(tri_2d, dtype=np.float32)
    v0 = b - a
    v1 = c - a
    denom = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(denom) < 1e-6:
        return

    for i in range(max(min_y, 0), min(max_y, height - 1)):
        for j in range(max(min_x, 0), min(max_x, width - 1)):
            p = np.array([j + 0.5, i + 0.5], dtype=np.float32)
            v2 = p - a
            v = (v2[0] * v1[1] - v1[0] * v2[1]) / denom
            w = (v0[0] * v2[1] - v2[0] * v0[1]) / denom
            u = 1.0 - v - w
            if u >= 0 and v >= 0 and w >= 0:
                dom[i, j] = color


def mesh_to_dsm_dom(mesh_file, output_dsm, output_dom, resolution=0.4, nodata=-9999.0):
    mesh = o3d.io.read_triangle_mesh(str(mesh_file))
    if mesh.is_empty():
        raise ValueError(f"Failed to read mesh or mesh is empty: {mesh_file}")
    mesh.compute_triangle_normals()

    has_vertex_color = mesh.has_vertex_colors()
    if not has_vertex_color:
        print("Mesh has no vertex colors; DOM will be gray.")

    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors) if has_vertex_color else None
    triangles = np.asarray(mesh.triangles)

    xs, ys, zs = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    max_z = zs.max()

    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))
    print(f"Raster size: {width} x {height}")

    transform = from_origin(min_x, max_y, resolution, resolution)
    x_coords = np.linspace(min_x + resolution / 2, max_x - resolution / 2, width)
    y_coords = np.linspace(max_y - resolution / 2, min_y + resolution / 2, height)

    dsm = np.full((height, width), nodata, dtype=np.float32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    for i, y in enumerate(tqdm(y_coords, desc="DSM raycasting")):
        for j, x in enumerate(x_coords):
            origin = np.array([x, y, max_z + 10.0], dtype=np.float32)
            direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            rays = o3d.core.Tensor(
                [[origin[0], origin[1], origin[2], direction[0], direction[1], direction[2]]],
                dtype=o3d.core.Dtype.Float32,
            )
            ans = scene.cast_rays(rays)
            t_hit = ans["t_hit"][0].numpy()
            if t_hit != np.inf:
                dsm[i, j] = origin[2] + t_hit * direction[2]

    output_dsm.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_dsm,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(dsm, 1)
    print("DSM saved:", output_dsm)

    dom = np.zeros((height, width, 3), dtype=np.uint8)
    print("Rasterizing triangles for DOM...")
    for tri in tqdm(triangles, desc="DOM triangles"):
        pts = vertices[tri]
        tri_2d = [
            ((pts[k, 0] - min_x) / resolution, (max_y - pts[k, 1]) / resolution)
            for k in range(3)
        ]
        if has_vertex_color:
            color = (colors[tri].mean(axis=0) * 255).clip(0, 255).astype(np.uint8)
        else:
            color = np.array([128, 128, 128], dtype=np.uint8)
        rasterize_triangle(dom, tri_2d, color)

    output_dom.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_dom,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        transform=transform,
    ) as dst:
        for band in range(3):
            dst.write(dom[:, :, band], band + 1)
    print("DOM saved:", output_dom)
    return dsm, dom


def find_mesh(gs_dir, mesh_name=None):
    mesh_dir = gs_dir / "mesh"
    if mesh_name is not None:
        mesh_file = mesh_dir / mesh_name
        if not mesh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
        return mesh_file

    preferred_names = [
        "tsdf_fusion_post.ply",
        "tsdf_fusion.ply",
    ]
    for name in preferred_names:
        mesh_file = mesh_dir / name
        if mesh_file.exists():
            return mesh_file

    ply_files = sorted(mesh_dir.glob("*.ply"))
    if len(ply_files) == 1:
        return ply_files[0]
    if not ply_files:
        raise FileNotFoundError(f"No .ply mesh found in: {mesh_dir}")
    raise ValueError(
        "Multiple .ply files found. Use --mesh-name to select one: "
        + ", ".join(path.name for path in ply_files)
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DSM and DOM GeoTIFFs from a 3DGS output directory.")
    parser.add_argument("--input", required=True, type=Path, help="3DGS output directory containing a mesh/ folder.")
    parser.add_argument("--mesh-name", help="Optional mesh filename inside input/mesh.")
    parser.add_argument("--resolution", type=float, default=0.4, help="Output raster resolution.")
    parser.add_argument("--nodata", type=float, default=-9999.0, help="DSM nodata value.")
    return parser.parse_args()


def main():
    args = parse_args()
    gs_dir = args.input.resolve()
    mesh_file = find_mesh(gs_dir, args.mesh_name)
    output_dir = gs_dir / "digital_products"
    mesh_to_dsm_dom(
        mesh_file=mesh_file,
        output_dsm=output_dir / "DSM.tif",
        output_dom=output_dir / "DOM.tif",
        resolution=args.resolution,
        nodata=args.nodata,
    )


if __name__ == "__main__":
    main()
