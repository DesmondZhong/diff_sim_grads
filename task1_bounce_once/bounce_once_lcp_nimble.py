import os
import torch
import nimblephysics as nimble

from omegaconf import OmegaConf

THIS_DIR = os.path.dirname(__file__)

yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, 'bounce_once.yaml'))
cli_cfg = OmegaConf.from_cli()

cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
cfg.dt = cfg.simulation_time / cfg.steps # 1./480
cfg.name = os.path.basename(__file__)[:-3]
cfg.THIS_DIR = THIS_DIR

# Set up the world
world = nimble.simulation.World()
world.setGravity([0, 0, 0])
world.setTimeStep(cfg.dt)

# Set up the the ball

ball_1 = nimble.dynamics.Skeleton()
sphereJoint_1, sphereBody_1 = ball_1.createTranslationalJoint2DAndBodyNodePair() 
# dart/dynamics/SphereShape.cpp
sphereShape_1 = sphereBody_1.createShapeNode(nimble.dynamics.SphereShape(cfg.radius))
sphereVisual_1 = sphereShape_1.createVisualAspect()
sphereVisual_1.setColor([0.5, 0.5, 0.5])
sphereShape_1.createCollisionAspect()
sphereBody_1.setFrictionCoeff(0.0)
sphereBody_1.setRestitutionCoeff(cfg.elasticity)
sphereBody_1.setMass(1)
world.addSkeleton(ball_1)

# set up floor
floor = nimble.dynamics.Skeleton()
floorJoint, floorBody = floor.createWeldJointAndBodyNodePair()
floorOffset = nimble.math.Isometry3()
floorOffset.set_translation([0, -0.50, 0])
floorJoint.setTransformFromParentBodyNode(floorOffset)
floorShape = floorBody.createShapeNode(nimble.dynamics.BoxShape([5.0, 1.0, 5.0]))
floorVisual = floorShape.createVisualAspect()
floorVisual.setColor([0.5, 0.5, 0.5])
floorVisual.setCastShadows(False)
floorBody.setRestitutionCoeff(1.0)
floorShape.createCollisionAspect()

world.addSkeleton(floor)

# Set up the GUI
gui: nimble.NimbleGUI = nimble.NimbleGUI(world)
gui.serve(8080)

# initiate torch tensors for simulation
initial_position = torch.tensor(cfg.init_pos)
initial_position.requires_grad = True
initial_velocity = torch.tensor(cfg.init_vel)
initial_velocity.requires_grad = True
ctrls = torch.tensor([cfg.ctrl_input for _ in range(cfg.steps)])
ctrls.requires_grad = True

# simulate
state = torch.cat([initial_position, initial_velocity], dim=0)
states = [state]
for i in range(cfg.steps):
    state = nimble.timestep(world, state, ctrls[i])
    states.append(state)

# get gradients
final_height = states[-1][1]
final_height.backward()

print("----------------Task 1: LCP (Nimble, with TOI)------------------")
print(f'final height: {final_height}')
print(f"gradient of final height w.r.t. initial position dh/dx0: {initial_position.grad[1]}")
print(f"gradient of final height w.r.t. initial velocity dh/dv0: {initial_velocity.grad[1]}")
print(f"gradient of final height w.r.t. initial ctrl dh/du0: {ctrls.grad[0][1]}")

gui.loopStates(states)
# gui.blockWhileServing()

