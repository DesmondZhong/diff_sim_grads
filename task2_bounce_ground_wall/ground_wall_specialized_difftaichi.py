import taichi as ti
import os
import numpy as np
from omegaconf import OmegaConf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, 'ground_wall.yaml'))
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

dt = cfg.dt
steps = cfg.steps
learning_rate = cfg.learning_rate

TARGET_POS = cfg.target
WALL_X = cfg.wall_x
NUM_ITERS = cfg.train_iters

radius = cfg.radius
elasticity = cfg.elasticity

# vis_interval = 8
# output_vis_interval = 8
# steps = 1024
# assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

loss = scalar()

init_x = vec()
init_v = vec()
gravity = vec()

x = vec()
x_inc = vec()  # for TOI
v = vec()
impulse = vec()
ctrls = vec()


ti.root.dense(ti.i, steps+1).place(x, v, x_inc, impulse)
ti.root.dense(ti.i, steps).place(ctrls)
ti.root.place(init_x, init_v, gravity)
ti.root.place(loss)
ti.root.lazy_grad()

init_x[None] = cfg.init_pos
init_v[None] = cfg.init_vel
gravity[None] = [0, -9.8]


@ti.kernel
def collide_ground(t: ti.i32):
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
def collide_wall(t: ti.i32):
    imp = ti.Vector([0.0, 0.0])
    x_inc_contrib = ti.Vector([0.0, 0.0])
    dist_norm = WALL_X - (x[t][0] + dt * v[t][0])

    rela_v = v[t]
    if dist_norm < radius:
        dir = ti.Vector([-1.0, 0.0])
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
    v[t] = v[t - 1] + impulse[t] + gravity[None] * dt + ctrls[t - 1] * dt
    x[t] = x[t - 1] + dt * v[t] + x_inc[t]

@ti.kernel
def advance_wo_toi(t: ti.i32):
    v[t] = v[t - 1] + impulse[t] + gravity[None] * dt + ctrls[t - 1] * dt
    x[t] = x[t - 1] + dt * v[t] 


@ti.kernel
def compute_loss():
    loss[None] = (x[steps][0] - TARGET_POS[0]) ** 2 + (x[steps][1] - TARGET_POS[1]) ** 2


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

def forward(visualize=False):
    initialize_xv()

    pixel_radius = int(radius * 1024) + 1

    for t in range(1, steps + 1):
        collide_ground(t - 1)
        collide_wall(t - 1)
        if cfg.toi:
            advance_w_toi(t) # from t - 1 to t
        else:
            advance_wo_toi(t)
        if visualize:
            gui.clear()
            gui.circle((fit_to_canvas(x[t][0]), fit_to_canvas(x[t][1])), 0xCCCCCC, pixel_radius// 4)
            gui.circle((TARGET_POS[0], TARGET_POS[1]), 0x00000, pixel_radius // 4)
            gui.line(begin=[-1, 0.5], end=[1, 0.5])
            gui.line(begin=[fit_to_canvas(WALL_X), 0.5], end=[fit_to_canvas(WALL_X), 1])
            gui.show()

    compute_loss()


@ti.kernel
def clear():
    for t in range(steps + 1):
        impulse[t] = ti.Vector([0.0, 0.0])
        x_inc[t] = ti.Vector([0.0, 0.0])

# @ti.kernel
# def step():
#     init_v[None][0] -= learning_rate * ctrls.grad[0][0]


def optimize():
    clear()
    # forward(visualize=True, output='initial')
    loss_np = []
    init_vel_np = []
    for iter in range(NUM_ITERS):
        clear()
        with ti.Tape(loss):
            forward(visualize=cfg.render_difftaichi)
        loss_np.append(loss[None])
        init_vel_np.append(init_v[None].to_numpy())
        if cfg.verbose:
            print('Iter=', iter, 'Loss=', loss[None])
        for d in range(2):
            init_v[None][d] -= learning_rate * init_v.grad[None][d]
        # step()

    loss_np = np.array(loss_np)
    init_vel_np.append(init_v[None].to_numpy())
    init_vel_np = np.stack(init_vel_np)
    last_traj_np = x.to_numpy() # (288+1, 2)
    clear()
    return loss_np, init_vel_np, last_traj_np



if __name__ == '__main__':
    if cfg.render_difftaichi:
        gui = ti.GUI("Ground_wall", (1024, 1024), background_color=0x3C733F)
    with ti.Tape(loss):
        forward(visualize=cfg.render_difftaichi)
    print(f"------------Task 2: Direct Velocity Impulse (TOI: {cfg.toi})-----------")
    print(f"loss: {loss[None]}")
    print(f"gradient of loss w.r.t. initial position dl/dx0: {init_x.grad}")
    print(f"gradient of loss w.r.t. initial velocity dl/dv0: {init_v.grad}")
    print(f"gradient of loss w.r.t. initial ctrl dl/du0: {ctrls.grad[0]}")
    if cfg.is_train:
        print("---------start training------------")
        loss_np, init_vel_np, last_traj_np = optimize()
        print("---------finish training------------")
        print(f"optimized_velocity: {init_v[None]}")

        save_dir = os.path.join(cfg.THIS_DIR, cfg.result_dir)
        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            os.path.join(save_dir, cfg.name), 
            loss=loss_np, 
            init_vel=init_vel_np,
            last_traj=last_traj_np,
        )
        
