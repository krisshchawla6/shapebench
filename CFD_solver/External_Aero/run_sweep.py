"""
Sample Flow 360 API scripts.
Requires a mesh that you are ready to upload and run cases on.
"""

import os
import json
import argparse
from dotenv import load_dotenv
import flow360 as fl
from flow360.log import log

load_dotenv()
api_key = os.getenv("flow360_api_key")
fl.configure(apikey=api_key)

# Variables we want to export in our volume solution files. Many more are available
vol_fields = ["Mach", "Cp", "mut", "mutRatio", "primitiveVars", "qcriterion"]

# Variables we want to export in our surface solution files. Many more are available
surf_fields = ["Cp", "yPlus", "Cf", "CfVec", "primitiveVars", "wallDistance"]

######################################################################################################################

def upload_surface_mesh(file_path, project_name, length_unit="mm"):
    """
    Given a file path and name of the project, this function creates a project and uploads a mesh.
    """
    # length_unit should be 'm', 'mm', 'cm', 'inch' or 'ft'
    project = fl.Project.from_surface_mesh(file_path, length_unit=length_unit, name=project_name)
    log.info(f"The project id is {project.id}")
    return project


######################################################################################################################
def make_params_from_surface_mesh(surface_mesh_object, params_schema):
    """
    Create the params object that contains all the run parameters.
    Needs the mesh_object to get the list of surfaces.
    """
    with fl.SI_unit_system:
        far_field_zone = fl.AutomatedFarfield()

        params = fl.SimulationParams(
            meshing=fl.MeshingParams(
                defaults=fl.MeshingDefaults(
                    boundary_layer_growth_rate=1.2,
                    boundary_layer_first_layer_thickness=0.0003,
                    surface_max_edge_length=1.01,
                ),
                volume_zones=[far_field_zone],
            ),
            # Dimensions can  be either in inches, or m or mm or many other units
            reference_geometry=fl.ReferenceGeometry(
                moment_center=(0, 0, 0) * fl.u.m, moment_length=1 * fl.u.m, area=1 * fl.u.m * fl.u.m
            ),
            operating_condition=fl.AerospaceCondition(
                velocity_magnitude=params_schema['velocity_magnitude'], alpha=params_schema['alpha'] * fl.u.deg, beta=params_schema['beta'] * fl.u.deg
            ),
            time_stepping=fl.Steady(max_steps=2000, CFL=fl.AdaptiveCFL()),
            models=[
                # These boundary names can be taken from the vm.boundary_names printout
                fl.Wall(
                    surfaces=[surface_mesh_object["*"]],
                ),
                fl.Freestream(
                    surfaces=[
                        far_field_zone.farfield
                    ],  # Apply freestream boundary to the far-field zone
                ),
                # Define what sort of physical model of a fluid we will use
                fl.Fluid(
                    navier_stokes_solver=fl.NavierStokesSolver(),
                    turbulence_model_solver=fl.SpalartAllmaras(),
                ),
            ],
            outputs=[
                fl.VolumeOutput(output_format="tecplot", output_fields=vol_fields),
                # This mesh_object['*'] will select all the boundaries in the mesh and export the surface results.
                # Regular expressions can be used to filter for certain boundaries
                fl.SurfaceOutput(
                    surfaces=[surface_mesh_object["*"]], output_fields=surf_fields, output_format="tecplot"
                ),
            ],
        )
    return params


######################################################################################################################
def launch_sweep(params, project, sweep_schema):
    """
    Launch a sweep of cases.
    """

    # for example let's vary alpha:
    values = sweep_schema['values']
    variable = sweep_schema['variable']
    unit = sweep_schema['unit']

    for value in values:
        # modify the alpha
        setattr(params.operating_condition, variable, value * unit)
        # launch the case
        project.run_case(params=params, name=f"{variable}_case ", use_beta_mesher=True)
        log.info(f"The case ID is: {project.case.id} with {variable} ")    


def main(mesh_path=None, params_json_path=None):
    # Handle command line arguments if not provided as parameters
    if mesh_path is None or params_json_path is None:
        parser = argparse.ArgumentParser(description='Run Flow360 sweep with surface mesh and parameters')
        parser.add_argument('mesh_path', help='Path to surface mesh file')
        parser.add_argument('params_json', help='Path to JSON parameters file')
        args = parser.parse_args()
        mesh_path = args.mesh_path
        params_json_path = args.params_json
    
    # Load parameters from JSON
    with open(params_json_path, 'r') as f:
        params_schema = json.load(f)
    
    # Convert unit string to flow360 unit object
    params_schema['sweep_schema']['unit'] = eval(params_schema['sweep_schema']['unit'])
    
    project = fl.Project.from_surface_mesh(mesh_path, params_schema['job_name'], length_unit="mm")

    surface_mesh = project.surface_mesh
    params = make_params_from_surface_mesh(surface_mesh, params_schema)  # define the run params used to launch the run
    volume_mesh = project.generate_volume_mesh(params, use_beta_mesher=True)

    vm = project.volume_mesh  # get the volume mesh entity associated with that project.
    log.info(f"The volume mesh contains the following boundaries:{vm.boundary_names}")
    log.info(f"The volume mesh ID is: {vm.id}")

    launch_sweep(params, project, params_schema['sweep_schema'])

if __name__ == "__main__":
    main()