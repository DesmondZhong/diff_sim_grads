import os
import numpy as np
import warp as wp
import warp.sim
import warp.sim.render
import math

wp.init()

class BounceOnce:

    render_time = 0.0

    def __init__(self, cfg, integrator_class, render=True, profile=False, adapter='cpu'):

        self.frame_dt = 1.0/60.0
        self.frame_steps = int(cfg.simulation_time/self.frame_dt)
        self.sim_dt = cfg.dt
        self.sim_steps = cfg.steps
        self.sim_substeps = int(self.sim_steps/self.frame_steps)

        builder = warp.sim.ModelBuilder()

        # default up axis is y
        builder.add_particle(pos=(cfg.init_pos[0], cfg.init_pos[1], 0.0), vel=(cfg.init_vel[0], cfg.init_vel[1], 0.0), mass=1.0)

        self.device = adapter
        self.profile = profile

        self.model = builder.finalize(adapter)

        self.model.ground = True
        self.model.gravity[1] = 0 # no gravity
        self.model.particle_radius = cfg.radius
        # type of simulation
        self.model.customized_particle_ground_wall = False 
        self.model.customized_particle_bounce_once = True
        # soft contact properties
        self.model.customized_kn = cfg.customized_kn
        # self.model.customized_kd = self.customized_kd
        # self.model.customized_kf = self.customized_kf
        # self.model.customized_mu = self.customized_mu

        self.integrator = integrator_class()

        self.loss = wp.zeros(1, dtype=wp.float32, device=adapter, requires_grad=True)

        # allocate sim states for trajectory
        self.states = []
        for i in range(self.sim_steps+1):
            state = self.model.state(requires_grad=True)
            state.external_particle_f = wp.array([
                [0, 0, 0]
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
        loss: wp.array(dtype=float),
    ):  
        mask = wp.vec3(0., 1., 0.)
        wp.atomic_add(loss, 0, wp.dot(pos[0], mask))

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
                    
        # # compute loss on final state
        wp.launch(self.terminal_loss_kernel, dim=1, inputs=[self.states[-1].particle_q, self.loss], device=self.device)
        return self.loss

    def render(self):
        
        for i in range(0, self.sim_steps, self.sim_substeps):

            self.stage.begin_frame(self.render_time)
            self.stage.render(self.states[i])
            self.stage.render_points("particles", self.states[i].particle_q.numpy(), radius=self.model.particle_radius)
            self.stage.end_frame()

            self.render_time += self.frame_dt
        
        self.stage.save()

    def get_gradient(self):
        tape = wp.Tape()

        with tape:
            self.compute_loss()
        print(f"Height: {self.loss}")
        loss_np = self.loss.numpy()[0]
        tape.backward(self.loss)

        # get gradient
        x = self.states[0].particle_q
        x_grad = tape.gradients[self.states[0].particle_q]
        return x_grad


    def check_grad(self, param):
        tape = wp.Tape()
        with tape:
            l = self.compute_loss()

        tape.backward(l)

        x_grad_analytic = tape.gradients[param]

        return x_grad_analytic