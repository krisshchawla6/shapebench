import os

DEFAULT_CONFIG = dict(
    reset_dir='reset/4',
    nb_pts_to_move=4,
    pts_to_move=[0, 1, 2, 3],
    nb_ctrls_per_episode=0,
    nb_episodes=50000,
    max_deformation=3.0,
    restart_from_cylinder=True,
    replace_shape=True,
    comp_dir='.',
    restore_model=False,
    saving_model_period=10,
    cfl=0.5,
    reynolds=100.0,
    output_vtu=True,
    shape_h=1.0,
    domain_h=0.8,
    cell_limit=50000,
    xmin=-15.0,
    xmax=30.0,
    ymin=-15.0,
    ymax=15.0,
)

def get_default_config(env_root=None):
    """Return a copy of the default config with reset_dir made absolute."""
    cfg = dict(DEFAULT_CONFIG)
    if env_root:
        cfg['reset_dir'] = os.path.join(env_root, cfg['reset_dir'])
    cfg['final_time'] = 2.0 * (cfg['xmax'] - cfg['xmin'])
    return cfg
