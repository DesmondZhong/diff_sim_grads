import numpy as np
import brax
from brax import jumpy as jp
import numpy as np
import jax
import os

def create_ground_wall_brax_system(cfg, dynamics_mode):
    bouncy_ball = brax.Config(dt=cfg.dt, substeps=cfg.brax_substeps, dynamics_mode=dynamics_mode)

    # ground is a frozen (immovable) infinite plane
    ground = bouncy_ball.bodies.add(name='ground')
    ground.frozen.all = True
    plane = ground.colliders.add().plane
    plane.SetInParent()  # for setting an empty oneof

    # wall is a frozen (immovable) infinite plane
    wall = bouncy_ball.bodies.add(name='wall')
    wall.frozen.all = True
    wall_plane = wall.colliders.add().plane
    wall_plane.SetInParent()  # for setting an empty oneof


    ball_1 = bouncy_ball.bodies.add(name='ball_1', mass=1)
    sphere_1 = ball_1.colliders.add().sphere
    sphere_1.radius = cfg.radius

    thruster = bouncy_ball.forces.add(name="ctrl1", body="ball_1", strength=1).thruster
    thruster.SetInParent()

    # no gracity
    bouncy_ball.gravity.z = -9.8
    bouncy_ball.elasticity = cfg.elasticity
    bouncy_ball.friction = cfg.customized_mu

    sys = brax.System(bouncy_ball)

    qp_init = brax.QP(
        # position of each body in 3d (z is up, right-hand coordinates)
        pos = np.array([[0., 0., 0.],       # ground
                        [cfg.wall_x, 0., 0.],      # wall
                        [cfg.init_pos[0], 0., cfg.init_pos[1]]]),     # ball 
        # velocity of each body in 3d
        vel = np.array([[0., 0., 0.],       # ground
                        [0., 0., 0.],       # wall
                        [cfg.init_vel[0], 0., cfg.init_vel[1]]]),     # ball
        # rotation about center of body, as a quaternion (w, x, y, z)
        rot = np.array([[1., 0., 0., 0.],   # ground
                        [1/np.sqrt(2), 0, -1/np.sqrt(2), 0], 
                        [1., 0., 0., 0.]]), # ball
        # angular velocity about center of body in 3d
        ang = np.array([[0., 0., 0.],       # ground
                        [0., 0., 0.],        # wall
                        [0., 0., 0.]])      # ball
    )

    return sys, qp_init


def print_and_train(cfg, sys, qp_init, dynamics_mode):
    @jax.jit
    def compute_loss(qp_init, ctrls):

        def do_one_step(state, a):
            next_state, _ = sys.step(state, a)
            return (next_state, state)
        qp, qp_history = jax.lax.scan(do_one_step, qp_init, ctrls)
        loss = (qp.pos[2, 0] - cfg.target[0]) ** 2 + (qp.pos[2, 2] - cfg.target[1]) ** 2
        return loss, qp_history

    ctrls = jp.array([[cfg.ctrl_input[0], 0., cfg.ctrl_input[1]] for _ in range(cfg.large_steps)])

    loss, _ = compute_loss(qp_init, ctrls)
    grads_tuple, _ = jax.grad(compute_loss, [0, 1], has_aux=True)(qp_init, ctrls)
    dldqp, dldctrls = grads_tuple
    if dynamics_mode == 'pbd':
        print("------------Task 2: Position-based Dynamics (Brax)-----------")
    elif dynamics_mode == 'legacy_spring':
        print("------------Task 2: Compliant Model (Brax)-----------")
    else:
        raise NotImplementedError
    print(f"loss: {loss}")
    print(f"gradient of loss w.r.t. initial position dl/dx0: {dldqp.pos[2][::2]}")
    print(f"gradient of loss w.r.t. initial velocity dl/dv0: {dldqp.vel[2][::2]}")
    print(f"gradient of loss w.r.t. initial ctrl dl/du0: {dldctrls[0][::2]}")

    if cfg.is_train:
        print("---------start training------------")
        grad_loss = jax.jit(jax.grad(compute_loss, has_aux=True))
        loss_np = []
        init_vel_np = []
        for iter in range(cfg.train_iters):
            loss, qp_history = compute_loss(qp_init, ctrls)
            loss_np.append(loss)
            init_vel_np.append(qp_init.vel[2, 0::2].copy())
            if cfg.verbose:
                print(f"Iter: {iter} Loss: {loss}")
            dldqp, _ = grad_loss(qp_init, ctrls)
            qp_init.vel[2] -= cfg.learning_rate * dldqp.vel[2]
        
        loss_np = np.array(loss_np)
        init_vel_np.append(qp_init.vel[2, 0::2].copy())
        init_vel_np = np.stack(init_vel_np)
        last_traj_np = qp_history.pos[:, 2, 0::2]
        
        save_dir = os.path.join(cfg.THIS_DIR, cfg.result_dir)
        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            os.path.join(save_dir, cfg.name), 
            loss=loss_np, 
            init_vel=init_vel_np,
            last_traj=last_traj_np,
        )
        print("---------finish training------------")