import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from torchdiffeq import odeint
import numpy as np

from omegaconf import OmegaConf

yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, 'ground_wall.yaml'))
cli_cfg = OmegaConf.from_cli()

cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
cfg.dt = cfg.simulation_time / cfg.steps # 1./480
cfg.name = os.path.basename(__file__)[:-3] + "_" + cfg.method
if cfg.toi:
    cfg.name += "_w_toi"
else:
    cfg.name += "_wo_toi"
cfg.name += f"_mu_{cfg.customized_mu}"
cfg.THIS_DIR = THIS_DIR

dt = cfg.dt
num_steps = cfg.steps
lr = cfg.learning_rate
num_iters = cfg.train_iters
radius = cfg.radius
gravity = [0.0, -9.8]

WALL_X = cfg.wall_x

ELASTICITY = cfg.elasticity
MU = cfg.customized_mu

INIT_POS = cfg.init_pos
INIT_VEL = cfg.init_vel
CTRL_INPUT = cfg.ctrl_input
TARGET_POS = cfg.target


def calculate_vel_impulse(rela_v, J_c, mu, elasticity):

    # set up differentiable optimization problem
    f = cp.Variable((2, 1)) # [f_t, f_n]
    A_decom_p = cp.Parameter((2, 2))
    v_p = cp.Parameter((2, 1))
    mu_p = cp.Parameter((1, 1))
    objective = cp.Minimize(0.5 * cp.sum_squares(A_decom_p @ f) + cp.sum(cp.multiply(f, v_p)))
    constraints = [
        f[0] >= 0, # normal impulse should be creater than zero
        f[1] <= cp.multiply(mu_p[0], f[0]),
        f[1] >= -cp.multiply(mu_p[0], f[0]),
    ]
    problem = cp.Problem(objective, constraints)
    cvxpylayer = CvxpyLayer(problem, parameters=[A_decom_p, v_p, mu_p], variables=[f])

    # feed values to the layer
    A = J_c.transpose(0, 1) @ J_c
    A_decom = torch.linalg.cholesky(A)
    impulse, = cvxpylayer(
        A_decom,
        J_c @ rela_v.reshape(-1, 1),
        mu.reshape(-1, 1),
    )
    return (J_c.transpose(0, 1) @ impulse)[:, 0] * (1+elasticity) # elastic collision

def collide(x, v, dt, mu, elasticity):
    vel_impulse = torch.zeros(2)
    x_inc = torch.zeros(2)
    # collide with ground
    dist_norm = x[1] + v[1] * dt
    rela_v = v
    if dist_norm < radius:
        dir = torch.tensor([0.0, 1.0])
        projected_v = torch.dot(dir, rela_v)
        if projected_v < 0:
            imp = calculate_vel_impulse(
                rela_v,
                J_c=torch.tensor([[0., 1.], [-1., 0.]]),
                mu=mu,
                elasticity=elasticity
            )
            toi = (dist_norm - radius) / min(
                -1e-3, projected_v
            ) # time of impact
            x_inc_contrib = min(toi - dt, 0) * imp
            # udpate
            x_inc = x_inc + x_inc_contrib
            vel_impulse = vel_impulse + imp
    # collide with wall
    dist_norm = WALL_X - (x[0] + v[0] * dt)
    rela_v = v
    if dist_norm < radius:
        dir = torch.tensor([-1.0, 0.0])
        projected_v = torch.dot(dir, rela_v)
        if projected_v < 0:
            imp = calculate_vel_impulse(
                rela_v,
                J_c=torch.tensor([[-1., 0.], [0., -1.]]),
                mu=mu,
                elasticity=elasticity
            )
            toi = (dist_norm - radius) / min(
                -1e-3, projected_v
            ) # time of impact
            x_inc_contrib = min(toi - dt, 0) * imp
            # udpate
            x_inc = x_inc + x_inc_contrib
            vel_impulse = vel_impulse + imp
    return vel_impulse, x_inc


def dynamics_for_odeint(t, x_v_u):
    x, v, u = x_v_u
    dv = u + torch.tensor(gravity)
    dx = v
    du = torch.zeros_like(u)
    return dx, dv, du

def simulate(state, ctrl, dt, method, mu, elasticity):
    x = state[0:2]
    v = state[2:4]
    vel_impulse, x_inc = collide(x, v, dt, mu, elasticity)
    if not cfg.toi:
        x_inc = torch.zeros_like(x_inc)
    if method == "symplectic_euler":
        new_v = v + vel_impulse + ctrl * dt + torch.tensor(gravity) * dt
        new_x = x + v * dt + x_inc
    else:
        new_xs, new_vs, _ = odeint(dynamics_for_odeint, (x, v, ctrl), torch.tensor([0.0, dt]), method=method)
        new_x = new_xs[-1] + x_inc
        new_v = new_vs[-1] + vel_impulse
    return torch.cat([new_x, new_v], dim=0)

# simulate
def compute_loss(state_init, ctrls):
    state = state_init
    states = [state]
    for i in range(num_steps):
        state = simulate(
            state, 
            ctrls[i], 
            cfg.dt, 
            cfg.method, 
            mu, 
            elasticity)
        states.append(state)

    loss = (states[-1][0] - TARGET_POS[0]) ** 2 + (states[-1][1] - TARGET_POS[1]) ** 2
    return loss, states

# initiate torch tensors for simulation
initial_position = torch.tensor(INIT_POS, requires_grad=True)
initial_velocity = torch.tensor(INIT_VEL, requires_grad=True)
ctrls = torch.tensor([CTRL_INPUT for _ in range(num_steps)], requires_grad=True)
mu=torch.tensor([MU], requires_grad=True)
elasticity = torch.tensor(ELASTICITY, requires_grad=True)

state_init = torch.cat([initial_position, initial_velocity], dim=0)
loss, _ = compute_loss(state_init, ctrls)
loss.backward()

print(f"------------Task 2: Convex Optimization Formulation (TOI: {cfg.toi})-----------")
print(f"loss: {loss.item()}")
print(f"gradient of loss w.r.t. initial position dl/dx0: {initial_position.grad}")
print(f"gradient of loss w.r.t. initial velocity dl/dv0: {initial_velocity.grad}")
print(f"gradient of loss w.r.t. initial ctrl dl/du0: {ctrls.grad[0]}")

initial_velocity.grad.zero_()

if cfg.is_train:
    print("---------start training------------")
    loss_np = []
    init_vel_np = []
    for iter in range(num_iters):
        # simulate
        state_init = torch.cat([initial_position, initial_velocity], dim=0)
        loss, states = compute_loss(state_init, ctrls)
        loss_np.append(loss.item())
        init_vel_np.append(initial_velocity.detach().cpu().numpy().copy())
        loss.backward()
        if cfg.verbose:
            print(f"Iter: {iter}, loss: {loss.item()}")
        # update
        with torch.no_grad():
            initial_velocity -= lr * initial_velocity.grad
            initial_velocity.grad.zero_()
    print("---------finish training------------")

    loss_np = np.array(loss_np)
    init_vel_np.append(initial_velocity.detach().cpu().numpy().copy())
    init_vel_np = np.stack(init_vel_np)
    last_traj_np = torch.stack(states, dim=0).detach().cpu().numpy()[:, 0:2]

    save_dir = os.path.join(cfg.THIS_DIR, cfg.result_dir)
    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, cfg.name), 
        loss=loss_np, 
        init_vel=init_vel_np,
        last_traj=last_traj_np,
    )