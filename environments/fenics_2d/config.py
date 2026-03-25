import os


def get_default_config(env_root):
    """
    Default configuration for the FEniCS 2D environment.

    Original source: cfl-minds/drl_shape_optimization (Viquerat et al., 2020)
    Paper: "Direct shape optimization through deep reinforcement learning"
           https://arxiv.org/abs/1908.09885

    Parameter sources:
      - comp_dir, max_deformation, replace_shape, nb_ctrls_per_episode,
        nb_episodes, restart_from_cylinder:
            from original parametered_env.py in cfl-minds/drl_shape_optimization
      - final_time, reynolds, cfl, xmin/xmax/ymin/ymax, output_vtu:
            defaults from fenics_solver.py
      - shape_h, domain_h:
            defaults from shapes_utils.py (mesh size multipliers)
      - nb_pts_to_move, pts_to_move, reset_dir:
            inferred from reset/4/shape_0.csv header (4 control points)
      - restore_model, saving_model_period, cell_limit:
            inferred — verify with original FEniCS env author if needed

    Note on comp_dir:
      Must be '.' (current working directory at launch). orig_environment.py
      generates the deformed mesh (shape_N.xml) in CWD, then compute_reward()
      calls solve_flow with mesh_file = comp_dir + '/shape_N.xml'. Using an
      absolute path here causes a path mismatch — the file is in CWD but the
      solver looks elsewhere. The original repo uses '.' and relies on the
      caller to cd into the appropriate working directory before launching.

    Note on nb_ctrls_per_episode / nb_episodes:
      Original values (0 and 50000) are for a DRL training loop. In the
      ShapeEvolve benchmark context the loop is driven externally by the
      framework, so these are effectively unused — kept at original values
      for fidelity with the source.
    """
    return {
        # Shape control (from original parametered_env.py)
        'nb_pts_to_move':       4,
        'pts_to_move':          [0, 1, 2, 3],
        'max_deformation':      3.0,            # original: 3.0 (was incorrectly 0.1)
        'replace_shape':        True,           # original: True / centering mode

        # Episode structure (from original parametered_env.py)
        'nb_ctrls_per_episode': 0,              # original: 0 (DRL loop; unused here)
        'nb_episodes':          50000,          # original: 50000 (DRL loop; unused here)

        # Paths
        'reset_dir':            os.path.join(env_root, 'reset', '4'),
        'comp_dir':             '.',            # original: '.' — must be CWD at launch

        # Checkpoint / restore
        'restore_model':        False,
        'restart_from_cylinder': True,          # original: True
        'saving_model_period':  1,

        # Solver parameters (from fenics_solver.py defaults)
        'final_time':           15.0,
        'reynolds':             10.0,
        'cfl':                  0.5,
        'output_vtu':           False,

        # Domain bounds (from fenics_solver.py defaults)
        'xmin':                -15.0,
        'xmax':                 30.0,
        'ymin':                -15.0,
        'ymax':                 15.0,

        # Mesh sizing (from shapes_utils.py defaults — multipliers on mesh_size)
        'shape_h':              10.0,
        'domain_h':             20.0,

        # Mesh cell ceiling
        'cell_limit':           10000,
    }
