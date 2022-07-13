import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from _ground_wall_brax import (
    create_ground_wall_brax_system,
    print_and_train
)
from omegaconf import OmegaConf

yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, 'ground_wall.yaml'))
cli_cfg = OmegaConf.from_cli()

cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
cfg.substeps = 2
cfg.large_steps = cfg.steps // cfg.brax_substeps # 240
cfg.dt = cfg.simulation_time / cfg.large_steps # 1./240
cfg.name = os.path.basename(__file__)[:-3]
cfg.name += f"_mu_{cfg.customized_mu}"
cfg.THIS_DIR = THIS_DIR

sys, qp_init = create_ground_wall_brax_system(cfg, dynamics_mode='pbd')

print_and_train(cfg, sys, qp_init, dynamics_mode='pbd')
