import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import taichi as ti
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf

yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, 'two_balls_1.yaml'))
cli_cfg = OmegaConf.from_cli()

cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
cfg.dt = cfg.simulation_time / cfg.steps # 1./480
cfg.name = os.path.basename(__file__)[:-3]
if cfg.toi:
    cfg.name += "_w_toi"
else:
    cfg.name += "_wo_toi"
cfg.THIS_DIR = THIS_DIR

real = ti.f32
ti.init(default_fp=real, flatten_if=True)

steps = cfg.steps
epsilon = cfg.epsilon
dt = cfg.dt
# alpha = 0.00000
learning_rate = cfg.learning_rate

vis_interval = 8
output_vis_interval = 8
# steps = 1024
# assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

loss = scalar()
x = vec()
x_inc = vec()  # for TOI
v = vec()
impulse = vec()
ctrls = vec()

goal = [0.0, 0.0]
radius = cfg.radius
elasticity = cfg.elasticity

ti.root.dense(ti.i, steps+1).dense(ti.j, 2).place(x, v, x_inc, impulse)
ti.root.dense(ti.i, steps).dense(ti.j, 2).place(ctrls)
ti.root.place(loss)
ti.root.lazy_grad()


@ti.kernel
def collide(t: ti.i32):
    for i in range(2):
        s_id = i
        o_id = 1 - i
        imp = ti.Vector([0.0, 0.0])
        x_inc_contrib = ti.Vector([0.0, 0.0])
        dist = (x[t, s_id] + dt * v[t, s_id]) - (x[t, o_id] + dt * v[t, o_id])
        dist_norm = dist.norm()
        rela_v = v[t, s_id] - v[t, o_id]
        if dist_norm < 2 * radius:
            dir = ti.Vector.normalized(dist)
            projected_v = dir.dot(rela_v)

            if projected_v < 0:
                imp = -(1 + elasticity) * 0.5 * projected_v * dir
                toi = (dist_norm - 2 * radius) / min(
                    -1e-3, projected_v)  # Time of impact
                x_inc_contrib = min(toi - dt, 0) * imp
        x_inc[t + 1, s_id] += x_inc_contrib
        impulse[t + 1, s_id] += imp


@ti.kernel
def advance_wo_toi(t: ti.i32):
    for i in range(2):
        v[t, i] = v[t - 1, i] + impulse[t, i] + ctrls[t - 1, i] * dt
        x[t, i] = x[t - 1, i] + dt * v[t, i]

@ti.kernel
def advance_w_toi(t: ti.i32):
    for i in range(2):
        v[t, i] = v[t - 1, i] + impulse[t, i] + ctrls[t - 1, i] * dt
        x[t, i] = x[t - 1, i] + dt * v[t, i] + x_inc[t, i]

@ti.kernel
def compute_terminal_loss():
    # terminal cost
    loss[None] = (x[steps, 1][0] - goal[0]) ** 2 + (x[steps, 1][1] - goal[1]) ** 2


@ti.kernel
def compute_running_loss():
    # running cost
    for t in range(steps):
        ti.atomic_add(
            loss[None],
            epsilon * (ctrls[t, 0][0] ** 2 + ctrls[t, 0][1] ** 2) * dt
        )


@ti.kernel
def initialize_ctrls():
    for t in range(steps):
        ctrls[t, 0]= [3., 3.]
        ctrls[t, 1] = [0., 0.]


def fit_to_canvas(p):
    return (p + 2.) / 2.

def forward(cfg):

    interval = vis_interval
    pixel_radius = int(radius * 1024) + 1

    for t in range(1, steps + 1):
        collide(t - 1)
        if cfg.toi:
            advance_w_toi(t) # from t - 1 to t
        else:
            advance_wo_toi(t)
        # print(f"Iter {t}", x[t, 0], x[t, 1], v[t, 0], v[t, 1])
        # print(v[t])
        if (t + 1) % interval == 0 and cfg.render_difftaichi:
            gui.clear()
            gui.circle((fit_to_canvas(goal[0]), fit_to_canvas(goal[1])), 0x00000, pixel_radius // 2)
            colors = [0xCCCCCC, 0x3344cc]
            for i in range(2):
                gui.circle((fit_to_canvas(x[t, i][0]), fit_to_canvas(x[t, i][1])), colors[i], pixel_radius// 2)
            gui.show()

    compute_terminal_loss()
    compute_running_loss()


@ti.kernel
def clear():
    for t, i in ti.ndrange(steps + 1, 2):
        impulse[t, i] = ti.Vector([0.0, 0.0])
        x_inc[t, i] = ti.Vector([0.0, 0.0])

@ti.kernel
def step():
    for t in range(steps):
        ctrls[t, 0][0] -= learning_rate * ctrls.grad[t, 0][0]
        ctrls[t, 0][1] -= learning_rate * ctrls.grad[t, 0][1]


def optimize():
    initialize_ctrls()
    # initial condition
    x[0, 0][0] = -2 ; x[0, 0][1] = -2
    x[0, 1][0] = -1 ; x[0, 1][1] = -1

    clear()
    loss_np = []
    for iter in range(cfg.train_iters):
        clear()
        with ti.Tape(loss):
            forward(cfg)
        loss_np.append(loss[None]) # loss[None] is a python scalar
        if cfg.verbose:
            print('Iter=', iter, 'Loss=', loss[None])
        step()
    clear()
    loss_np = np.array(loss_np)
    # ctrls (480, 2, 2)
    ctrls_np = ctrls.to_numpy()[:, 0, :] # 480, 2
    return loss_np, ctrls_np


if __name__ == '__main__':
    if cfg.render_difftaichi:
        gui = ti.GUI("TwoBalls", (1024, 1024), background_color=0x3C733F)

    initialize_ctrls()
    # initial condition
    x[0, 0][0] = -2 ; x[0, 0][1] = -2
    x[0, 1][0] = -1 ; x[0, 1][1] = -1
    with ti.Tape(loss):
        forward(cfg)
    print(f"------------Task 3: Direct Velocity Impulse (TOI: {cfg.toi})-----------")
    print(f"loss: {loss[None]}")
    print(f"gradient of loss w.r.t. initial position dl/dx0: {x.grad[0, 0]} {x.grad[0, 1]}")
    print(f"gradient of loss w.r.t. initial velocity dl/dv0: {v.grad[0, 0]} {v.grad[0, 1]}")
    print(f"gradient of loss w.r.t. initial ctrl dl/du0: {ctrls.grad[0, 0]}")

    if cfg.is_train:
        print("---------start training------------")
        loss_np, ctrls_np = optimize()
        print("---------finish training------------")

        save_dir = os.path.join(cfg.THIS_DIR, cfg.result_dir)
        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            os.path.join(save_dir, cfg.name), 
            loss=loss_np, 
            ctrls=ctrls_np
        )
        