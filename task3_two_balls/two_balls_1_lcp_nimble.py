import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import torch
import nimblephysics as nimble
import numpy as np

from omegaconf import OmegaConf

yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, 'two_balls_1.yaml'))
cli_cfg = OmegaConf.from_cli()

cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
cfg.dt = cfg.simulation_time / cfg.steps # 1./480
cfg.name = os.path.basename(__file__)[:-3]
cfg.THIS_DIR = THIS_DIR


num_steps = cfg.steps
epsilon = cfg.epsilon
lr = cfg.learning_rate
num_iters = cfg.train_iters

# Set up the world
world = nimble.simulation.World()
world.setGravity([0, 0, 0])
world.setTimeStep(cfg.dt)

# Set up the the first sphere

ball_1 = nimble.dynamics.Skeleton()
sphereJoint_1, sphereBody_1 = ball_1.createTranslationalJoint2DAndBodyNodePair() # createTranslationalJoint2DAndBodyNodePair()
# dart/dynamics/SphereShape.cpp
sphereShape_1 = sphereBody_1.createShapeNode(nimble.dynamics.SphereShape(cfg.radius))
sphereVisual_1 = sphereShape_1.createVisualAspect()
sphereVisual_1.setColor([0.5, 0.5, 0.5])
sphereShape_1.createCollisionAspect()
sphereBody_1.setFrictionCoeff(0.0)
sphereBody_1.setRestitutionCoeff(cfg.elasticity)
world.addSkeleton(ball_1)

ball_2 = nimble.dynamics.Skeleton()
sphereJoint_2, sphereBody_2 = ball_2.createTranslationalJoint2DAndBodyNodePair() # createTranslationalJoint2DAndBodyNodePair()
# dart/dynamics/SphereShape.cpp
sphereShape_2 = sphereBody_2.createShapeNode(nimble.dynamics.SphereShape(0.2))
sphereVisual_2 = sphereShape_2.createVisualAspect()
sphereVisual_2.setColor([0.5, 0.5, 0.5])
sphereShape_2.createCollisionAspect()
sphereBody_2.setFrictionCoeff(0.0)
sphereBody_2.setRestitutionCoeff(1.0)
world.addSkeleton(ball_2)

# ball_1.setPosition(-2, -2)
# ball_1.setVelocity(0, 0)
sphereBody_1.setMass(1)

# ball_2.setPosition(-2, -2)
# ball_2.setVelocity(0, 0)
sphereBody_2.setMass(1)

# print(world.getNumDofs())
# print(ball_1.getPositions())
# print(ball_2.getPositions())

initial_position = torch.tensor([-2., -2., -1., -1.], requires_grad=True)
initial_velocity = torch.zeros(world.getNumDofs(), requires_grad=True)

# only control the first ball, remove the control to the second ball
world.removeDofFromActionSpace(3)
world.removeDofFromActionSpace(2)



# Set up the GUI
gui: nimble.NimbleGUI = nimble.NimbleGUI(world)
gui.serve(8080)
gui.nativeAPI().createSphere("goal", radius=0.2, pos=[0, 0, 0], color=[0, 1, 0, 1])

# set up control 
ctrls = 3 * torch.ones((num_steps, 2))
ctrls.requires_grad = True

def compute_loss(state_init, ctrls):
    state = state_init
    states = [state]
    for i in range(num_steps):
        state = nimble.timestep(world, state, ctrls[i])
        states.append(state)
    terminal_loss = states[-1][2] ** 2 + states[-1][3] ** 2
    running_loss = (ctrls * ctrls).sum() * 1. / num_steps
    loss = terminal_loss + epsilon * running_loss
    return loss, states

state_init = torch.cat([initial_position, initial_velocity], dim=0)
loss, _ = compute_loss(state_init, ctrls)
loss.backward()

print("----------------Task 3: LCP (Nimble, with TOI)------------------")
print(f"loss: {loss.item()}")
print(f"gradient of loss w.r.t. initial position dl/dx0: {initial_position.grad}")
print(f"gradient of loss w.r.t. initial velocity dl/dv0: {initial_velocity.grad}")
print(f"gradient of loss w.r.t. initial ctrl dl/du0: {ctrls.grad[0]}")

ctrls.grad = None
if cfg.is_train:
    print("---------start training------------")
    loss_np = []
    for iter in range(num_iters):
        state_init = torch.cat([initial_position, initial_velocity], dim=0)
        loss, states = compute_loss(state_init, ctrls)
        loss_np.append(loss.item())
        loss.backward()
        if cfg.verbose:
            print(f"Iter: {iter}, loss: {loss.item()}")
        # update gradient
        with torch.no_grad():
            ctrls -= lr * ctrls.grad
            ctrls.grad = None

        gui.loopStates(states)
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
# gui.blockWhileServing()

