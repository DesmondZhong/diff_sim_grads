import os
import taichi as ti
import sys
import math
import numpy as np
# import matplotlib.pyplot as plt

from omegaconf import OmegaConf

THIS_DIR = os.path.dirname(__file__)

yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, 'bounce_once.yaml'))
cli_cfg = OmegaConf.from_cli()

cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
cfg.dt = cfg.simulation_time / cfg.steps # 1./480
cfg.name = os.path.basename(__file__)[:-3]
cfg.THIS_DIR = THIS_DIR

dt = cfg.dt
steps = cfg.steps

real = ti.f32
ti.init(default_fp=real, flatten_if=True)


scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

loss = scalar()

init_x = vec()
init_v = vec()

x = vec()
x_inc = vec()  # for TOI
v = vec()
impulse = vec()
ctrls = vec()

radius = cfg.radius
elasticity = cfg.elasticity

ti.root.dense(ti.i, steps+1).place(x, v, x_inc, impulse)
ti.root.dense(ti.i, steps).place(ctrls)
ti.root.place(init_x, init_v)
ti.root.place(loss)
ti.root.lazy_grad()

init_x[None] = cfg.init_pos
init_v[None] = cfg.init_vel


@ti.kernel
def collide(t: ti.i32):
    imp = ti.Vector([0.0, 0.0])
    x_inc_contrib = ti.Vector([0.0, 0.0])
    dist_norm = x[t][1] + dt * v[t][1]

    rela_v = v[t]
    if dist_norm < radius:
        dir = ti.Vector([0.0, 1.0])
        projected_v = dir.dot(rela_v)
        if projected_v < 0:
            imp = -(1 + elasticity) * projected_v * dir
            toi = (dist_norm - radius) / min(
                -1e-3, projected_v)  # Time of impact
            x_inc_contrib = min(toi - dt, 0) * imp
    x_inc[t + 1] += x_inc_contrib
    impulse[t + 1] += imp


@ti.kernel
def advance_w_toi(t: ti.i32):
    v[t] = v[t - 1] + impulse[t] + ctrls[t - 1] * dt
    x[t] = x[t - 1] + dt * v[t] + x_inc[t]

@ti.kernel
def advance_wo_toi(t: ti.i32):
    v[t] = v[t - 1] + impulse[t] + ctrls[t - 1] * dt
    x[t] = x[t - 1] + dt * v[t]


@ti.kernel
def compute_loss():
    loss[None] = x[steps][1]


@ti.kernel
def initialize_xv():
    x[0] = init_x[None]
    v[0] = init_v[None]

@ti.kernel
def initialize_ctrls():
    for t in range(steps):
        ctrls[t] = [0., 0.]

def fit_to_canvas(p):
    return (p + 2) / 4.


@ti.kernel
def clear():
    for t in range(steps + 1):
        impulse[t] = ti.Vector([0.0, 0.0])
        x_inc[t] = ti.Vector([0.0, 0.0])

def forward(cfg):
    initialize_xv()
    initialize_ctrls()
    clear()

    pixel_radius = int(radius * 1024) + 1

    for t in range(1, cfg.steps + 1):
        collide(t - 1)
        if cfg.toi:
            advance_w_toi(t) # from t - 1 to t
        else:
            advance_wo_toi(t)
        if cfg.render_difftaichi:
            gui.clear()
            gui.circle((fit_to_canvas(x[t][0]), fit_to_canvas(x[t][1])), 0xCCCCCC, pixel_radius// 4)
            gui.line(begin=[-1, 0.5], end=[1, 0.5])
            gui.show()

    compute_loss()



if __name__ == '__main__':
    if cfg.render_difftaichi:
        gui = ti.GUI("Bounce_once", (1024, 1024), background_color=0x3C733F)
    with ti.Tape(loss):
        forward(cfg)
    print(f"------------Task 1: Direct Velocity Impulse (TOI: {cfg.toi})-----------")
    print(f"loss: {loss[None]}")
    print(f"gradient of final height w.r.t. initial position dl/dx0: {init_x.grad[None][1]}")
    print(f"gradient of final height w.r.t. initial velocity dl/dv0: {init_v.grad[None][1]}")
    print(f"gradient of final height w.r.t. initial ctrl dl/du0: {ctrls.grad[0][1]}")