import os
import numpy as np
import warp as wp
import warp.sim
import warp.sim.render
import matplotlib.pyplot as plt


wp.init()


class TwoBalls:

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
        self.epsilon = cfg.epsilon

        builder = warp.sim.ModelBuilder()

        # default up axis is y
        builder.add_particle(pos=(-2.0, -2.0, -0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        builder.add_particle(pos=(-1.0, -1.0, -0.0), vel=(0.0, 0.0, 0.0), mass=1.0)


        self.device = adapter

        self.model = builder.finalize(adapter)

        self.model.ground = False
        self.model.gravity[1] = 0 # default gravity is along the y axis
        self.model.particle_radius = cfg.radius
        self.model.customized_kn = cfg.customized_kn

        self.integrator = integrator_class()

        self.target = (0.0, 0.0, 0.0)
        self.loss = wp.zeros(1, dtype=wp.float32, device=adapter, requires_grad=True)

        # allocate sim states for trajectory
        self.states = []
        for i in range(self.sim_steps+1):
            state = self.model.state(requires_grad=True)
            state.external_particle_f = wp.array([
                [3, 3, 0],
                [0, 0, 0]
            ], dtype=wp.vec3, device=adapter, requires_grad=True)
            self.states.append(state)
        
        # mask for updating only u_x and u_y
        self.mask = (1, 1, 0)

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
        # distance of the second ball to target
        delta = pos[1] - target
        wp.atomic_add(loss, 0, wp.dot(delta, delta))

    @wp.kernel
    def running_loss_kernel(
        external_particle_f: wp.array(dtype=wp.vec3),
        loss: wp.array(dtype=float),
        epsilon_dt: float,
    ):
        u_x = external_particle_f[0][0]
        u_y = external_particle_f[0][1]
        wp.atomic_add(loss, 0, (u_x * u_x + u_y * u_y) * epsilon_dt)

    @wp.kernel
    def step_kernel(
        f: wp.array(dtype=wp.vec3),
        f_grad: wp.array(dtype=wp.vec3),
        mask: wp.vec3,
        lr: float,
    ):
        f[0] = f[0] - wp.cw_mul(f_grad[0], mask) * lr
    
    def compute_loss(self):

        # run control loop
        self.loss.zero_()
        for i in range(self.sim_steps):
                
            self.states[i].clear_forces()

            self.integrator.simulate(
                self.model, 
                self.states[i], 
                self.states[i+1], 
                self.sim_dt
            )

            wp.launch(self.running_loss_kernel, dim=1, inputs=[self.states[i].external_particle_f, self.loss, self.sim_dt*self.epsilon], device=self.device)
        
        # compute loss on final state
        wp.launch(self.terminal_loss_kernel, dim=1, inputs=[self.states[-1].particle_q, self.target, self.loss], device=self.device)
        return self.loss

    def render(self):
        
        for i in range(0, self.sim_steps, self.sim_substeps):

            self.stage.begin_frame(self.render_time)
            # self.stage.render(self.states[i])
            self.stage.render_points("particles", self.states[i].particle_q.numpy(), radius=self.model.particle_radius)
            self.stage.render_box(pos=self.target, rot=wp.quat_identity(), extents=(0.1, 0.1, 0.1), name="target")
            self.stage.end_frame()

            self.render_time += self.frame_dt
        
        self.stage.save()

    def train(self):
        tape = wp.Tape()
        loss_np = []
        for j in range(self.train_iters):
            with tape:
                self.compute_loss()
            if self.cfg.verbose:
                print(f"Iter: {j} Loss: {self.loss}")
            loss_np.append(self.loss.numpy()[0])
            tape.backward(self.loss)
            if j % 50 == 0:
                self.render()

            # slow step
            ctrl_list = []
            ctrl_grad_list = []
            for i in range(self.sim_steps):
                f = self.states[i].external_particle_f
                f_grad = tape.gradients[self.states[i].external_particle_f]
                wp.launch(self.step_kernel, dim=1, inputs=[f, f_grad, self.mask, self.learning_rate], device=self.device)
                ctrl_list.append(f.numpy()[0])
                ctrl_grad_list.append(f_grad.numpy()[0])
            
            tape.reset()

        loss_np = np.array(loss_np)
        ctrls_np = np.stack(ctrl_list)[:, 0:2]
        return loss_np, ctrls_np

    def plot_ctrls(self, ctrl_np, ctrl_grad_np, loss_np, iter):
        self.ts = np.linspace(start=0, stop=1, num=60*8+1)
        # ctrls
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(self.ts[:-1], ctrl_np[:, 0], label="u_x")
        ax.plot(self.ts[:-1], ctrl_np[:, 1], label="u_y")
        ax.legend()
        file_name = os.path.join(os.path.dirname(__file__), self.fig_dir, f"iter_{iter}_loss_{loss_np}.png")
        fig.savefig(file_name, bbox_inches='tight')
        plt.close(fig)
        # ctrl_grads
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(self.ts[:-1], ctrl_grad_np[:, 0], label="u_x_grad")
        ax.plot(self.ts[:-1], ctrl_grad_np[:, 1], label="u_y_grad")
        ax.legend()
        file_name = os.path.join(os.path.dirname(__file__), self.fig_dir, f"iter_{iter}_loss_{loss_np}_grad.png")
        fig.savefig(file_name, bbox_inches='tight')
        plt.close(fig)


    def check_grad(self, param):
        tape = wp.Tape()
        with tape:
            l = self.compute_loss()

        tape.backward(l)

        x_grad_analytic = tape.gradients[param]

        # print(f"numeric grad: {x_grad_numeric}")
        # print(f"analytic grad: {x_grad_analytic}")
        return x_grad_analytic