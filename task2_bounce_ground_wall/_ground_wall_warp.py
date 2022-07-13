import os
import warp as wp
import warp.sim
import warp.sim.render
import numpy as np

wp.init()


class GroundWall:

    render_time = 0.0

    def __init__(self, cfg, integrator_class, render=True, adapter='cpu'):

        self.cfg = cfg
        self.frame_dt = 1.0/60.0
        self.frame_steps = int(cfg.simulation_time/self.frame_dt)
        self.sim_dt = cfg.dt
        self.sim_steps = cfg.steps
        self.sim_substeps = int(self.sim_steps/self.frame_steps)
        self.learning_rate = cfg.learning_rate
        self.train_iters = cfg.train_iters

        builder = warp.sim.ModelBuilder()

        # default up axis is y
        builder.add_particle(
            pos=(cfg.init_pos[0], cfg.init_pos[1], 0.0), 
            vel=(cfg.init_vel[0], cfg.init_vel[1], 0.0), 
            mass=1.0
        )
        # for rendering purposes
        builder.add_shape_box(body=-1, pos=(2.0, 1.0, 0.0), hx=0.25, hy=1.0, hz=1.0)


        self.device = adapter

        self.model = builder.finalize(adapter)

        self.model.ground = True
        self.model.particle_radius = cfg.radius
        # type of simulation
        self.model.customized_particle_ground_wall = True 
        self.model.customized_particle_ground = False
        # this will decide the position of the wall in simulation
        self.model.customized_wall_x = 1.75
        # soft contact properties
        self.model.customized_kn = cfg.customized_kn
        self.model.customized_kd = cfg.customized_kd
        # we didn't use the paramter kf
        self.model.customized_mu = cfg.customized_mu

        # this is for the pbd elasticity
        self.model.customized_elasticity = cfg.elasticity

        self.integrator = integrator_class()

        self.target = [cfg.target[0], cfg.target[1], 0.]
        self.loss = wp.zeros(1, dtype=wp.float32, device=adapter, requires_grad=True)

        # allocate sim states for trajectory
        self.states = []
        for i in range(self.sim_steps+1):
            state = self.model.state(requires_grad=True)
            state.external_particle_f = wp.array([
                [cfg.ctrl_input[0], cfg.ctrl_input[1], 0]
            ], dtype=wp.vec3, device=adapter, requires_grad=True)
            self.states.append(state)

        self.save_dir = os.path.join(cfg.THIS_DIR, cfg.result_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        if render:
            self.stage = warp.sim.render.SimRenderer(
                self.model, 
                os.path.join(self.save_dir, cfg.name+".usd")
            )

    @wp.kernel
    def terminal_loss_kernel(
        pos: wp.array(dtype=wp.vec3),
        target: wp.vec3,
        loss: wp.array(dtype=float),
    ):
        delta = pos[0] - target
        wp.atomic_add(loss, 0, wp.dot(delta, delta))

    @wp.kernel
    def step_kernel(
        x: wp.array(dtype=wp.vec3),
        grad: wp.array(dtype=wp.vec3),
        alpha: float
    ):
        tid = wp.tid()
        # gradient descent step
        x[tid] = x[tid] - grad[tid]*alpha
    
    def compute_loss(self):

        self.loss.zero_()
        for i in range(self.sim_steps):
                
            self.states[i].clear_forces()

            self.integrator.simulate(
                self.model, 
                self.states[i], 
                self.states[i+1], 
                self.sim_dt
            )
                    
        # compute loss on final state
        wp.launch(self.terminal_loss_kernel, dim=1, inputs=[self.states[-1].particle_q, self.target, self.loss], device=self.device)

        return self.loss

    def render(self):
        
        for i in range(0, self.sim_steps, self.sim_substeps):

            self.stage.begin_frame(self.render_time)
            self.stage.render(self.states[i])
            self.stage.render_points("particles", self.states[i].particle_q.numpy(), radius=self.model.particle_radius)
            self.stage.render_box(pos=self.target, rot=wp.quat_identity(), extents=(0.1, 0.1, 0.1), name="target")
            self.stage.end_frame()

            self.render_time += self.frame_dt
        
        self.stage.save()

    def train(self):
        tape = wp.Tape()
        loss_np = []
        init_vel_np = []
        for i in range(self.train_iters):
            with tape:
                self.compute_loss()
            loss_np.append(self.loss.numpy()[0])
            init_vel_np.append(self.states[0].particle_qd.numpy().copy()[0, 0:2])

            if self.cfg.verbose:
                print(f"Iter: {i} Loss: {self.loss}")
            tape.backward(self.loss)
            if i % 50 == 0:
                self.render()

            # step
            x = self.states[0].particle_qd
            x_grad = tape.gradients[self.states[0].particle_qd]

            wp.launch(self.step_kernel, dim=len(x), inputs=[x, x_grad, self.learning_rate], device=self.device)

            tape.reset()

        loss_np = np.array(loss_np)
        init_vel_np.append(self.states[0].particle_qd.numpy().copy()[0, 0:2])
        init_vel_np = np.stack(init_vel_np)
        last_traj_np = []
        for i in range(self.sim_steps+1):
            last_traj_np.append(
                self.states[i].particle_q.numpy()[0, 0:2]
            )
        last_traj_np = np.stack(last_traj_np)
        return loss_np, init_vel_np, last_traj_np

    def check_grad(self, param):
        tape = wp.Tape()
        with tape:
            l = self.compute_loss()
        tape.backward(l)
        x_grad_analytic = tape.gradients[param]
        return x_grad_analytic