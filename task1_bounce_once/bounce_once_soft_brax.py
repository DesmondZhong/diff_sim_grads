import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
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


sys, qp_init = create_bounce_once_system(cfg, dynamics_mode='legacy_spring')

@jax.jit
def compute_loss(qp_init, ctrls):

    def do_one_step(state, a):
        next_state, _ = sys.step(state, a)
        return (next_state, state)
    qp, qp_history = jax.lax.scan(do_one_step, qp_init, ctrls)
    loss = qp.pos[1, 2]
    return loss

# the following code snippet is too slow to compile since for loop needs to be unrolled
# @jax.jit
# def compute_loss(qp_init, ctrls):
#     qp = qp_init
#     for i in range(steps):
#         # draw_system(ax, qp.pos, i/480.)
#         qp, _ = sys.step(qp, ctrls[i])
#         # print(f"ball_1_pos: {qp.pos[1]}, ball_1_vel: {qp.vel[1]}")
#     loss = qp.pos[1, 2]
#     return loss

ctrls = jp.array([[cfg.ctrl_input[0], 0., cfg.ctrl_input[1]] for _ in range(cfg.large_steps)])

loss = compute_loss(qp_init, ctrls)
dldqp, dldctrls = jax.grad(compute_loss, [0, 1])(qp_init, ctrls)

print("------------Task 1: Compliant Model (Brax)-----------")
print(f"loss: {loss}")
print(f"gradient of final height w.r.t. initial position dl/dx: {dldqp.pos[1][2]}")
print(f"gradient of final height w.r.t. initial velocity dl/dv: {dldqp.vel[1][2]}")
print(f"gradient of final height w.r.t. initial ctrl dl/du: {dldctrls[0][2]}")