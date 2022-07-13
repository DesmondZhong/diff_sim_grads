import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from torchdiffeq import odeint
import numpy as np

from omegaconf import OmegaConf

yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, 'two_balls_1.yaml'))
cli_cfg = OmegaConf.from_cli()

cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
cfg.dt = cfg.simulation_time / cfg.steps # 1./480
cfg.name = os.path.basename(__file__)[:-3] + "_" + cfg.method
if cfg.toi:
    cfg.name += "_w_toi"
else:
    cfg.name += "_wo_toi"
cfg.THIS_DIR = THIS_DIR


dt = 1./480
num_steps = cfg.steps
lr = cfg.learning_rate
epsilon = cfg.steps
num_iters = cfg.train_iters
radius = cfg.radius

ELASTICITY = cfg.elasticity


INIT_POS = [-2., -2., -1., -1.]
INIT_VEL = [0., 0., 0., 0.]
CTRL_INPUT = [3., 3.]
TARGET_POS = [0.0, 0.0]


def calculate_vel_impulse(rela_v, J_c, elasticity):

    # set up differentiable optimization problem
    f = cp.Variable((2, 1)) # [f_t, f_n]
    A_decom_p = cp.Parameter((2, 2))
    v_p = cp.Parameter((2, 1))
    objective = cp.Minimize(0.5 * cp.sum_squares(A_decom_p @ f) + cp.sum(cp.multiply(f, v_p)))
    constraints = [
        f[0] >= 0, # normal impulse should be creater than zero
        f[1] == 0,
    ]
    problem = cp.Problem(objective, constraints)
    cvxpylayer = CvxpyLayer(problem, parameters=[A_decom_p, v_p], variables=[f])

    # feed values to the layer
    A = J_c.transpose(0, 1) @ J_c
    A_decom = torch.linalg.cholesky(A)
    impulse, = cvxpylayer(
        A_decom,
        J_c @ rela_v.reshape(-1, 1),
    )
    return (J_c.transpose(0, 1) @ impulse)[:, 0] * (1+elasticity) # elastic collision

def collide(x, v, dt, elasticity):
    vel_impulse = torch.zeros(4)
    x_inc = torch.zeros(4)
    dist = x[0:2] + v[0:2] * dt - (x[2:4] + v[2:4] * dt)
    dist_norm = torch.sqrt(torch.dot(dist, dist))
    rela_v = v[0:2] - v[2:4]
    if dist_norm < 2 * radius:
        dir = dist / dist_norm
        projected_v = torch.dot(dir, rela_v)
        J_c = torch.tensor([[dir[0], dir[1]], [dir[1], -dir[0]]])
        if projected_v < 0:
            imp = calculate_vel_impulse(
                rela_v,
                J_c=J_c,
                elasticity=elasticity
            )
            toi = (dist_norm - 2 * radius) / min(
                -1e-3, projected_v
            ) # time of impact
            x_inc_contrib = min(toi - dt, 0) * imp
            # udpate
            x_inc = x_inc + 0.5 * torch.cat([x_inc_contrib, -x_inc_contrib], dim=0)
            vel_impulse = vel_impulse + 0.5 * torch.cat([imp, -imp], dim=0)
    return vel_impulse, x_inc


def dynamics_for_odeint(t, x_v_u):
    x, v, u = x_v_u
    dv = (torch.tensor([[1., 0.], [0., 1.], [0., 0.], [0., 0.]]) @ u[:, None])[:, 0] * dt
    dx = v
    du = torch.zeros_like(u)
    return dx, dv, du

def simulate(state, ctrl, dt, method, elasticity):
    x = state[0:4]
    v = state[4:8]
    vel_impulse, x_inc = collide(x, v, dt, elasticity)
    if not cfg.toi:
        x_inc = torch.zeros_like(x_inc)
    if method == "symplectic_euler":
        new_v = v + vel_impulse + (torch.tensor([[1., 0.], [0., 1.], [0., 0.], [0., 0.]]) @ ctrl[:, None])[:, 0] * dt
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
            elasticity
        )
        states.append(state)
        # print(f"Iter: {i}, state: {state}")

    loss = (
        (states[-1][2] - TARGET_POS[0]) ** 2 + (states[-1][3] - TARGET_POS[1]) ** 2
        + cfg.epsilon * (ctrls * ctrls).sum() * cfg.dt
    )
    return loss

# initiate torch tensors for simulation
initial_position = torch.tensor(INIT_POS, requires_grad=True)
initial_velocity = torch.tensor(INIT_VEL, requires_grad=True)
ctrls = torch.tensor([CTRL_INPUT for _ in range(num_steps)], requires_grad=True)
# mu=torch.tensor([MU], requires_grad=True)
elasticity = torch.tensor(ELASTICITY, requires_grad=True)


state_init = torch.cat([initial_position, initial_velocity], dim=0)
loss = compute_loss(state_init, ctrls)
loss.backward()

print(f"------------Task 3: Convex Optimization Formulation (TOI: {cfg.toi})-----------")
print(f"loss: {loss.item()}")
print(f"gradient of loss w.r.t. initial position dl/dx0: {initial_position.grad}")
print(f"gradient of loss w.r.t. initial velocity dl/dv0: {initial_velocity.grad}")
print(f"gradient of loss w.r.t. initial ctrl dl/du0: {ctrls.grad[0]}")

ctrls.grad.zero_()

if cfg.is_train:
    print("---------start training------------")
    loss_np = []
    for iter in range(num_iters):
        # simulate
        state_init = torch.cat([initial_position, initial_velocity], dim=0)
        loss = compute_loss(state_init, ctrls)
        loss_np.append(loss.item())
        if cfg.verbose:
            print(f"Iter: {iter}, loss: {loss.item()}")
        loss.backward()
        # update
        with torch.no_grad():
            ctrls -= lr * ctrls.grad
            ctrls.grad.zero_()
    print("---------finish training------------")
    # saving
    loss_np = np.stack(loss_np)
    ctrls_np = ctrls.detach().cpu().numpy()
    save_dir = os.path.join(cfg.THIS_DIR, cfg.result_dir)
    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, cfg.name), 
        loss=loss_np, 
        ctrls=ctrls_np
    )