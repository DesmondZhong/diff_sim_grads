import numpy as np
import brax

def create_bounce_once_system(cfg, dynamics_mode):
    bouncy_ball = brax.Config(dt=cfg.dt, substeps=cfg.brax_substeps, dynamics_mode=dynamics_mode)

    # ground is a frozen (immovable) infinite plane
    ground = bouncy_ball.bodies.add(name='ground')
    ground.frozen.all = True
    plane = ground.colliders.add().plane
    plane.SetInParent()  # for setting an empty oneof


    ball_1 = bouncy_ball.bodies.add(name='ball_1', mass=1)
    sphere_1 = ball_1.colliders.add().sphere
    sphere_1.radius = cfg.radius

    thruster = bouncy_ball.forces.add(name="ctrl1", body="ball_1", strength=1).thruster
    thruster.SetInParent()

    # no gracity
    bouncy_ball.gravity.z = 0

    bouncy_ball.elasticity = cfg.elasticity
    bouncy_ball.friction = 0.0

    sys = brax.System(bouncy_ball)

    qp_init = brax.QP(
        # position of each body in 3d (z is up, right-hand coordinates)
        pos = np.array([[0., 0., 0.],       # ground
                        [cfg.init_pos[0], 0., cfg.init_pos[1]]]),     # ball 
        # velocity of each body in 3d
        vel = np.array([[0., 0., 0.],       # ground
                        [cfg.init_vel[0], 0., cfg.init_vel[1]]]),     # ball
        # rotation about center of body, as a quaternion (w, x, y, z)
        rot = np.array([[1., 0., 0., 0.],   # ground
                        [1., 0., 0., 0.]]), # ball
        # angular velocity about center of body in 3d
        ang = np.array([[0., 0., 0.],       # ground
                        [0., 0., 0.]])      # ball
    )
    return sys, qp_init

