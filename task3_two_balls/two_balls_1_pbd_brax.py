import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
from _two_balls_1_brax import (
    create_two_balls_1_system,
    print_and_train
)
from omegaconf import OmegaConf

yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, 'two_balls_1.yaml'))
cli_cfg = OmegaConf.from_cli()

cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
cfg.large_steps = cfg.steps // cfg.brax_substeps # 480 / 2
cfg.dt = cfg.simulation_time / cfg.large_steps # 1./240
cfg.name = os.path.basename(__file__)[:-3]
cfg.THIS_DIR = THIS_DIR

sys, qp_init = create_two_balls_1_system(cfg, dynamics_mode='pbd')

print_and_train(cfg, sys, qp_init, dynamics_mode='pbd')