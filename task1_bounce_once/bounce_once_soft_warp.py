import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
import sys
sys.path.append(PARENT_DIR)

from utils.customized_integrator_euler import CustomizedSymplecticEulerIntegrator
from _bounce_once_warp import BounceOnce
import matplotlib.pyplot as plt

from omegaconf import OmegaConf

yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, 'bounce_once.yaml'))
cli_cfg = OmegaConf.from_cli()

cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
cfg.dt = cfg.simulation_time / cfg.steps # 1./480
cfg.name = os.path.basename(__file__)[:-3]
cfg.THIS_DIR = THIS_DIR

system = BounceOnce(
    cfg, 
    integrator_class=CustomizedSymplecticEulerIntegrator,
    adapter='cpu',
    render=True,
)
loss = system.compute_loss()

print("------------Task 1: Compliant Model (Warp)-----------")
print(f"loss: {loss}")

x_grad = system.check_grad(system.states[0].particle_q)
v_grad = system.check_grad(system.states[0].particle_qd)
ctrl0_grad = system.check_grad(system.states[0].external_particle_f)
print(f"gradient of final height w.r.t. initial position dl/dx0: {x_grad.numpy()[0][1]}")
print(f"gradient of final height w.r.t. initial velocity dl/dv0: {v_grad.numpy()[0][1]}")
print(f"gradient of final height w.r.t. initial ctrl dl/du0: {ctrl0_grad.numpy()[0][1]}")
