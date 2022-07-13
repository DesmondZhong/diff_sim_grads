import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
import sys
sys.path.append(PARENT_DIR)
from utils.customized_integrator_xpbd import CustomizedXPBDIntegratorForTwoBalls
from _two_balls_1_warp import TwoBalls
import matplotlib.pyplot as plt
import numpy as np

from omegaconf import OmegaConf

yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, 'two_balls_1.yaml'))
cli_cfg = OmegaConf.from_cli()

cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
cfg.dt = cfg.simulation_time / cfg.steps # 1./480
cfg.name = os.path.basename(__file__)[:-3]
cfg.THIS_DIR = THIS_DIR

system = TwoBalls(
    cfg,
    integrator_class=CustomizedXPBDIntegratorForTwoBalls,
    adapter='cpu', 
    render=True
)
loss = system.compute_loss()
print("------------Task 3: Position-based Dynamics (Warp)-----------")
print(f"loss: {loss}")

x_grad = system.check_grad(system.states[0].particle_q)
v_grad = system.check_grad(system.states[0].particle_qd)
ctrl0_grad = system.check_grad(system.states[0].external_particle_f)
print(f"gradient of loss w.r.t. initial position dl/dx0: {x_grad.numpy()[:, 0:2]}")
print(f"gradient of loss w.r.t. initial velocity dl/dv0: {v_grad.numpy()[:, 0:2]}")
print(f"gradient of loss w.r.t. initial ctrl dl/du0: {ctrl0_grad.numpy()[:, 0:2]}")


if cfg.is_train:
    print("---------start training------------")
    loss_np, ctrls_np = system.train()
    print("---------finish training------------")
    np.savez(
        os.path.join(system.save_dir, cfg.name), 
        loss=loss_np, 
        ctrls=ctrls_np
    )