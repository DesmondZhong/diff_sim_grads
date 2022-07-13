
import numpy as np
import brax
import jax.numpy as jnp
import jax
import os

def create_two_balls_1_system(cfg, dynamics_mode):
    two_balls = brax.Config(dt=cfg.dt, substeps=cfg.brax_substeps, dynamics_mode=dynamics_mode)

    ball_1 = two_balls.bodies.add(name='ball_1', mass=1)
    sphere_1 = ball_1.colliders.add().sphere
    sphere_1.radius = cfg.radius

    ball_2 = two_balls.bodies.add(name='ball_2', mass=1)
    sphere_2 = ball_2.colliders.add().sphere
    sphere_2.radius = cfg.radius

    thruster = two_balls.forces.add(name="ctrl1", body="ball_1", strength=1).thruster
    thruster.SetInParent()

    two_balls.elasticity = cfg.elasticity
    two_balls.friction =cfg.customized_mu

    sys = brax.System(two_balls)

    qp_init = brax.QP(
        # position of each body in 3d (z is up, right-hand coordinates)
        pos = np.array([[-2., -2., 0.],       # ball1
                        [-1., -1., 0.]]),     # ball2
        # velocity of each body in 3d
        vel = np.array([[0., 0., 0.],       # ball1
                        [0., 0., 0.]]),     # ball2
        # rotation about center of body, as a quaternion (w, x, y, z)
        rot = np.array([[1., 0., 0., 0.],   # ball1
                        [1., 0., 0., 0.]]), # ball2
        # angular velocity about center of body in 3d
        ang = np.array([[0., 0., 0.],       # ball1
                        [0., 0., 0.]])      # ball2
    )
    return sys, qp_init

def print_and_train(cfg, sys, qp_init, dynamics_mode):
    @jax.jit
    def compute_loss(qp_init, ctrls):

        def do_one_step(state, a):
            next_state, _ = sys.step(state, a)
            return (next_state, state)
        qp, qp_history = jax.lax.scan(do_one_step, qp_init, ctrls)
        terminal_loss = qp.pos[1][0] ** 2 + qp.pos[1][1] ** 2
        running_loss = (ctrls[:, 0:2] ** 2).sum() * sys.config.dt
        loss = terminal_loss + cfg.epsilon * running_loss
        return loss

    ctrls = jnp.array([[3., 3., 0.] for _ in range(cfg.large_steps)])

    loss = compute_loss(qp_init, ctrls)

    dldqp, dldctrls = jax.grad(compute_loss, [0, 1])(qp_init, ctrls)

    if dynamics_mode == 'pbd':
        print("------------Task 3: Position-based Dynamics (Brax)-----------")
    elif dynamics_mode == 'legacy_spring':
        print("------------Task 3: Compliant Model (Brax)-----------")
    else:
        raise NotImplementedError
    print(f"loss: {loss}")
    print(f"gradient of loss w.r.t. initial position dl/dx0: {dldqp.pos[:, 0:2]}")
    print(f"gradient of loss w.r.t. initial velocity dl/dv0: {dldqp.vel[:, 0:2]}")
    print(f"gradient of loss w.r.t. initial ctrl dl/du0: {dldctrls[0, 0:2]}")

    if cfg.is_train:
        print("---------start training------------")
        grad_loss = jax.jit(jax.grad(compute_loss, 1))
        loss_np = []
        for iter in range(cfg.train_iters):
            loss = compute_loss(qp_init, ctrls)
            loss_np.append(loss)
            dldctrls = grad_loss(qp_init, ctrls)
            ctrls -= cfg.learning_rate * dldctrls
            if cfg.verbose:
                print(f"Iter: {iter}, loss: {loss}")
        print("---------finish training------------")
        # saving
        loss_np = jnp.array(loss_np)
        save_dir = os.path.join(cfg.THIS_DIR, cfg.result_dir)
        os.makedirs(save_dir, exist_ok=True)
        jnp.savez(
            os.path.join(save_dir, cfg.name), 
            loss=loss_np, 
            ctrls=ctrls[:, 0:2]
        )
