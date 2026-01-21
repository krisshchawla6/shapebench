"""
Test script for shape modification pipeline.
Takes CSV input or modification parameters, outputs geometry PNG and meshed domain PNG.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shapes_utils import Shape
from meshes_utils import read_mesh


def load_shape_from_csv(csv_path: str) -> Shape:
    """Load shape from CSV file."""
    shape = Shape()
    shape.read_csv(csv_path)
    shape.generate(centering=False)
    return shape


def apply_modification(shape: Shape, deformation: np.ndarray, 
                       replace: bool = False, pts_list: list = None) -> Shape:
    """Apply deformation to shape control points."""
    shape.modify_shape_from_field(deformation, replace=replace, 
                                  pts_list=pts_list if pts_list else [])
    shape.generate(centering=False)
    return shape


def generate_geometry_image(shape: Shape, output_path: str, 
                            plot_pts: bool = True,
                            xmin: float = -3.0, xmax: float = 3.0,
                            ymin: float = -3.0, ymax: float = 3.0):
    """Generate and save geometry PNG using original fenics style."""
    shape.generate_image(
        plot_pts=plot_pts,
        xmin=xmin, xmax=xmax,
        ymin=ymin, ymax=ymax,
        override_name=output_path
    )
    print(f"Geometry image saved: {output_path}")


def generate_mesh_image(mesh_path: str, output_path: str, shape: Shape = None,
                        xmin: float = -3.0, xmax: float = 3.0,
                        ymin: float = -3.0, ymax: float = 3.0):
    """Generate and save mesh visualization PNG (fenics style)."""
    mesh = read_mesh(mesh_path)
    
    plt.figure()
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Plot triangular mesh
    if mesh.n_tris > 0:
        triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.tris)
        plt.triplot(triang, color='darkblue', linewidth=0.2)
    
    # Fill shape boundary (hole in mesh)
    if shape is not None:
        plt.fill(shape.curve_pts[:, 0], shape.curve_pts[:, 1], 'black')
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0,
                facecolor=(0.784, 0.773, 0.741))
    plt.close()
    print(f"Mesh image saved: {output_path}")


def run_pipeline(csv_path: str, output_dir: str,
                 modification: np.ndarray = None,
                 pts_list: list = None,
                 replace: bool = False,
                 mesh_domain: bool = True,
                 shape_h: float = 0.1,
                 domain_h: float = 0.3,
                 xmin: float = -3.0, xmax: float = 5.0,
                 ymin: float = -3.0, ymax: float = 3.0):
    """
    Full pipeline: load CSV -> (optional) modify -> generate images -> mesh -> mesh image.
    
    Args:
        csv_path: Path to input CSV shape file
        output_dir: Directory for output files
        modification: Optional (N, 3) array of [x, y, edgy] deformations
        pts_list: Optional list of point indices to modify
        replace: If True, replace coords; if False, add to existing
        mesh_domain: Whether to mesh surrounding domain
        shape_h: Mesh size on shape boundary
        domain_h: Mesh size in domain
        xmin, xmax, ymin, ymax: Domain bounds
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load shape
    print(f"Loading shape from: {csv_path}")
    shape = load_shape_from_csv(csv_path)
    print(f"  Control points: {shape.n_control_pts}")
    print(f"  Sampling points: {shape.n_sampling_pts}")
    
    # Apply modification if provided
    if modification is not None:
        print(f"Applying modification to {len(modification)} points...")
        shape = apply_modification(shape, modification, replace=replace, pts_list=pts_list)
    
    # Generate geometry image
    geom_path = os.path.join(output_dir, f"{shape.name}_{shape.index}_geometry.png")
    generate_geometry_image(shape, geom_path, plot_pts=True,
                            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    
    # Save modified CSV
    original_dir = os.getcwd()
    os.chdir(output_dir)
    shape.write_csv()
    csv_out = f"{shape.name}_{shape.index}.csv"
    print(f"CSV saved: {os.path.join(output_dir, csv_out)}")
    os.chdir(original_dir)
    
    # Generate mesh
    print("Generating mesh...")
    os.chdir(output_dir)
    try:
        success, n_tri = shape.mesh(
            mesh_domain=mesh_domain,
            shape_h=shape_h,
            domain_h=domain_h,
            xmin=xmin, xmax=xmax,
            ymin=ymin, ymax=ymax,
            mesh_format='mesh'
        )
        
        if success:
            mesh_file = f"{shape.name}_{shape.index}.mesh"
            print(f"Mesh generated: {mesh_file} ({n_tri} triangles)")
            
            # Generate mesh visualization
            mesh_img_path = f"{shape.name}_{shape.index}_mesh.png"
            generate_mesh_image(mesh_file, mesh_img_path, shape=shape,
                                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        else:
            print("Meshing failed!")
    finally:
        os.chdir(original_dir)
    
    return shape


def parse_modification(mod_str: str) -> np.ndarray:
    """Parse modification string: 'x1,y1,e1;x2,y2,e2;...' """
    if not mod_str:
        return None
    rows = mod_str.split(';')
    return np.array([[float(v) for v in row.split(',')] for row in rows])


def main():
    parser = argparse.ArgumentParser(description='Test shape modification pipeline')
    parser.add_argument('csv_path', help='Path to input CSV shape file')
    parser.add_argument('-o', '--output', default='./output',
                        help='Output directory (default: ./output)')
    parser.add_argument('-m', '--modification', type=str, default=None,
                        help='Modification as "x1,y1,e1;x2,y2,e2;..." or path to numpy file')
    parser.add_argument('-p', '--pts-list', type=str, default=None,
                        help='Point indices to modify as "0,1,2,..."')
    parser.add_argument('-r', '--replace', action='store_true',
                        help='Replace coordinates instead of adding')
    parser.add_argument('--shape-h', type=float, default=0.1,
                        help='Mesh size on shape (default: 0.1)')
    parser.add_argument('--domain-h', type=float, default=0.3,
                        help='Mesh size in domain (default: 0.3)')
    parser.add_argument('--xmin', type=float, default=-3.0)
    parser.add_argument('--xmax', type=float, default=5.0)
    parser.add_argument('--ymin', type=float, default=-3.0)
    parser.add_argument('--ymax', type=float, default=3.0)
    
    args = parser.parse_args()
    
    # Parse modification
    modification = None
    if args.modification:
        if args.modification.endswith('.npy'):
            modification = np.load(args.modification)
        else:
            modification = parse_modification(args.modification)
    
    # Parse pts_list
    pts_list = None
    if args.pts_list:
        pts_list = [int(p) for p in args.pts_list.split(',')]
    
    run_pipeline(
        csv_path=args.csv_path,
        output_dir=args.output,
        modification=modification,
        pts_list=pts_list,
        replace=args.replace,
        shape_h=args.shape_h,
        domain_h=args.domain_h,
        xmin=args.xmin, xmax=args.xmax,
        ymin=args.ymin, ymax=args.ymax
    )


if __name__ == '__main__':
    main()
