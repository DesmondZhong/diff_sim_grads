import os
THIS_DIR = os.path.dirname(__file__)
from brax import jumpy as jp
import jax
from _bounce_once_brax import create_bounce_once_system
from omegaconf import OmegaConf

yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, 'bounce_once.yaml'))
cli_cfg = OmegaConf.from_cli()

cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
cfg.large_steps = cfg.steps // cfg.brax_substeps # 240
cfg.dt = cfg.simulation_time / cfg.large_steps # 1./240
cfg.name = os.path.basename(__file__)[:-3]
cfg.THIS_DIR = THIS_DIR


sys, qp_init = create_bounce_once_system(cfg, dynamics_mode='pbd')

@jax.jit
def compute_loss(qp_init, ctrls):

    def do_one_step(state, a):
        next_state, _ = sys.step(state, a)
        return (next_state, state)
    qp, qp_history = jax.lax.scan(do_one_step, qp_init, ctrls)
    loss = qp.pos[1, 2]
    return loss

ctrls = jp.array([[cfg.ctrl_input[0], 0., cfg.ctrl_input[1]] for _ in range(cfg.large_steps)])

loss = compute_loss(qp_init, ctrls)
dldqp, dldctrls = jax.grad(compute_loss, [0, 1])(qp_init, ctrls)

print("------------Task 1: Position-based Dynamics (Brax)-----------")
print(f"loss: {loss}")
print(f"gradient of final height w.r.t. initial position dl/dx0: {dldqp.pos[1][2]}")
print(f"gradient of final height w.r.t. initial velocity dl/dv0: {dldqp.vel[1][2]}")
print(f"gradient of final height w.r.t. initial ctrl dl/du0: {dldctrls[0][2]}")