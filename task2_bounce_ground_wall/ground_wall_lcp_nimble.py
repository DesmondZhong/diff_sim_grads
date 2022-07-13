import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import torch
import nimblephysics as nimble

from omegaconf import OmegaConf
import numpy as np

yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, 'ground_wall.yaml'))
cli_cfg = OmegaConf.from_cli()

cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
cfg.dt = cfg.simulation_time / cfg.steps # 1./480
cfg.name = os.path.basename(__file__)[:-3]
cfg.name += f"_mu_{cfg.customized_mu}"
cfg.THIS_DIR = THIS_DIR


dt = cfg.dt
num_steps = cfg.steps
lr = cfg.learning_rate
num_iters = cfg.train_iters

INIT_POS = cfg.init_pos
INIT_VEL = cfg.init_vel
CTRL_INPUT = cfg.ctrl_input
TARGET_POS = cfg.target

# Set up the world
world = nimble.simulation.World()
world.setGravity([0, -9.8, 0])
world.setTimeStep(dt)

# Set up the the ball

ball_1 = nimble.dynamics.Skeleton()
sphereJoint_1, sphereBody_1 = ball_1.createTranslationalJoint2DAndBodyNodePair()
# dart/dynamics/SphereShape.cpp
sphereShape_1 = sphereBody_1.createShapeNode(nimble.dynamics.SphereShape(cfg.radius))
sphereVisual_1 = sphereShape_1.createVisualAspect()
sphereVisual_1.setColor([0.5, 0.5, 0.5])
sphereShape_1.createCollisionAspect()
sphereBody_1.setFrictionCoeff(cfg.customized_mu)
sphereBody_1.setRestitutionCoeff(cfg.elasticity)
sphereBody_1.setMass(1)
world.addSkeleton(ball_1)

# set up floor
floor = nimble.dynamics.Skeleton()
floorJoint, floorBody = floor.createWeldJointAndBodyNodePair()
floorOffset = nimble.math.Isometry3()
floorOffset.set_translation([0, -0.25, 0])
floorJoint.setTransformFromParentBodyNode(floorOffset)
floorShape = floorBody.createShapeNode(nimble.dynamics.BoxShape([5.0, 0.5, 5.0]))
floorVisual = floorShape.createVisualAspect()
floorVisual.setColor([0.3, 0.3, 0.3])
floorVisual.setCastShadows(False)
floorBody.setFrictionCoeff(1.0)
floorBody.setRestitutionCoeff(1.0)
floorShape.createCollisionAspect()

world.addSkeleton(floor)

# set up wall
wall = nimble.dynamics.Skeleton()
wallJoint, wallBody = wall.createWeldJointAndBodyNodePair()
wallOffset = nimble.math.Isometry3()
wallOffset.set_translation([2.0, 2.0, 0.])
wallJoint.setTransformFromParentBodyNode(wallOffset)
wallShape = wallBody.createShapeNode(nimble.dynamics.BoxShape([0.5, 4.0, 4.0]))
wallVisual = wallShape.createVisualAspect()
wallVisual.setColor([0.3, 0.3, 0.3])
wallVisual.setCastShadows(False)
wallBody.setFrictionCoeff(1.0)
wallBody.setRestitutionCoeff(1.0)
wallShape.createCollisionAspect()

world.addSkeleton(wall)



# Set up the GUI
gui: nimble.NimbleGUI = nimble.NimbleGUI(world)
gui.serve(8080)
gui.nativeAPI().createSphere("goal", radius=0.1, pos=[TARGET_POS[0], TARGET_POS[1], 0], color=[0, 1, 0, 1])

# set up tensors for simulation
initial_position = torch.tensor(INIT_POS, requires_grad=True)
initial_velocity = torch.tensor(INIT_VEL, requires_grad=True)
ctrls = torch.tensor(
    [CTRL_INPUT for _ in range(num_steps)],
    requires_grad=True,
)

def compute_loss(state_init, ctrls):
    state = state_init
    states = [state]
    for i in range(num_steps):
        state = nimble.timestep(world, state, ctrls[i])
        states.append(state)
    loss = (states[-1][0] - TARGET_POS[0]) ** 2 + (states[-1][1] - TARGET_POS[1]) ** 2
    return loss, states

state_init = torch.cat([initial_position, initial_velocity], dim=0)
loss, _ = compute_loss(state_init, ctrls)
loss.backward()

print("----------------Task 2: LCP (Nimble, with TOI)------------------")
print(f"loss: {loss.item()}")
print(f"gradient of loss w.r.t. initial position dl/dx0: {initial_position.grad}")
print(f"gradient of loss w.r.t. initial velocity dl/dv0: {initial_velocity.grad}")
print(f"gradient of loss w.r.t. initial ctrl dl/du0: {ctrls.grad[0]}")

initial_velocity.grad = None
if cfg.is_train:
    print("---------start training------------")
    loss_np = []
    init_vel_np = []
    for iter in range(num_iters):
        state_init = torch.cat([initial_position, initial_velocity], dim=0)
        loss, states = compute_loss(state_init, ctrls)
        loss_np.append(loss.item())
        init_vel_np.append(initial_velocity.detach().cpu().numpy().copy())
        loss.backward()
        if cfg.verbose:
            print(f"Iter: {iter}, loss: {loss.item()}")
        # update gradient
        with torch.no_grad():
            initial_velocity -= lr * initial_velocity.grad
            initial_velocity.grad = None
        gui.loopStates(states)
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
    

# gui.blockWhileServing()
