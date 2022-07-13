# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# https://github.com/NVIDIA/warp/blob/main/LICENSE.md

# This file is modified from 
# https://github.com/NVIDIA/warp/blob/main/warp/sim/integrator_xpbd.py

import warp as wp


@wp.kernel
def integrate_particles(x: wp.array(dtype=wp.vec3),
                        v: wp.array(dtype=wp.vec3),
                        f: wp.array(dtype=wp.vec3),
                        w: wp.array(dtype=float),
                        gravity: wp.vec3,
                        dt: float,
                        x_new: wp.array(dtype=wp.vec3),
                        v_new: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]
    f0 = f[tid]
    inv_mass = w[tid]

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v1 = v0 + (f0 * inv_mass + gravity * wp.step(0.0 - inv_mass)) * dt
    x1 = x0 + v1 * dt

    x_new[tid] = x1
    v_new[tid] = v1


@wp.kernel
def solve_balls_collision_delta(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    invmass: wp.array(dtype=float),
    # kn: wp.array(dtype=float),
    radius: float,
    # dt: float,
    delta: wp.array(dtype=wp.vec3)
):

    # tid = wp.tid()

    # i = spring_indices[tid * 2 + 0]
    # j = spring_indices[tid * 2 + 1]

    # ke = spring_stiffness[tid]
    # kd = spring_damping[tid]
    # rest = spring_rest_lengths[tid]
    xi = x[0]
    xj = x[1]

    vi = v[0]
    vj = v[1]

    xij = xi - xj
    vij = vi - vj

    l = wp.length(xij)
    l_inv = 1.0 / l

    # contact normal from j to i
    dir = xij * l_inv

    c = l - 2. * radius
    if c > 0.: # collision has been resolved
        return
    # dcdt = wp.dot(dir, vij)

    # damping based on relative velocity.
    #fs = dir * (ke * c + kd * dcdt)

    wi = invmass[0]
    wj = invmass[1]

    denom = wi + wj
    # alpha = 1.0/(ke*dt*dt)

    # here c is a negative number
    multiplier = c / (denom)# + alpha)

    xd = dir*multiplier
    # positional updates
    wp.atomic_sub(delta, 0, xd*wi)
    wp.atomic_add(delta, 1, xd*wj)





@wp.kernel
def apply_deltas(x_orig: wp.array(dtype=wp.vec3),
                 v_orig: wp.array(dtype=wp.vec3),
                 x_pred: wp.array(dtype=wp.vec3),
                 delta: wp.array(dtype=wp.vec3),
                 dt: float,
                 x_out: wp.array(dtype=wp.vec3),
                 v_out: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    x0 = x_orig[tid]
    xp = x_pred[tid]

    # constraint deltas
    d = delta[tid]

    x_new = xp + d
    v_new = (x_new - x0)/dt

    x_out[tid] = x_new
    v_out[tid] = v_new

    # clear forces
    # delta[tid] = wp.vec3()

@wp.kernel
def solve_balls_collision_vel_delta(
    q_pred: wp.array(dtype=wp.vec3),
    qd_pred: wp.array(dtype=wp.vec3),
    qd_pred_before_rest: wp.array(dtype=wp.vec3),
    invmass: wp.array(dtype=float),
    radius: float,
    vel_delta: wp.array(dtype=wp.vec3),
):
    xi = q_pred[0]
    xj = q_pred[1]

    vi = qd_pred[0]
    vj = qd_pred[1]

    xij = xi - xj
    vij = vi - vj
    l = wp.length(xij)
    l_inv = 1.0 / l

    c = l - 2. * radius
    if l - 2. * radius > 0.: # collision has been resolved
        return

    # contact normal from j to i
    n = xij * l_inv
    # old relative velocity (normal compoenent), we calculate the velocity after restitution based on this
    v_n_old = wp.dot(vij, n)
    # relative velocity after pbd velocity update
    v_n = wp.dot(qd_pred_before_rest[0] - qd_pred_before_rest[1], n)

    # dv rest should be a positive number
    # assume elasticity 1.0
    dv_rest = - v_n - wp.min(1. * v_n_old, 0.)
    wi = invmass[0]
    wj = invmass[1]

    vd = (dv_rest / (wi + wj)) * n
    wp.atomic_add(vel_delta, 0, vd*wi)
    wp.atomic_sub(vel_delta, 1, vd*wj)

@wp.kernel
def apply_vel_deltas(
    qd_before: wp.array(dtype=wp.vec3),
    vel_delta: wp.array(dtype=wp.vec3),
    #output
    qd_after: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    qd_after[tid] = qd_before[tid] + vel_delta[tid]
    # vel_delta[tid] = wp.vec3()


@wp.kernel
def solve_particle_ground_delta(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    # invmass: wp.array(dtype=float),
    radius: float,
    delta: wp.array(dtype=wp.vec3)
):
    x = particle_x[0]
    v = particle_v[0]

    n = wp.vec3(0., 1., 0.)
    d = wp.dot(x, n)

    c = d - radius
    if c > 0.: # collision has been resolved
        return
    # positional updates
    wp.atomic_sub(delta, 0, n * c)


@wp.kernel
def solve_particle_ground_vel_delta(
    q_pred: wp.array(dtype=wp.vec3),
    qd_pred: wp.array(dtype=wp.vec3),
    qd_pred_before_rest: wp.array(dtype=wp.vec3),
    # invmass: wp.array(dtype=float),
    radius: float,
    vel_delta: wp.array(dtype=wp.vec3),
):
    x = q_pred[0]
    n = wp.vec3(0., 1., 0.)
    d = wp.dot(x, n)
    c = d - radius
    if c > 0.: # collision has been resolved
        return
    # old relative velocity (normal compoenent), we calculate the velocity after restitution based on this
    v_n_old = wp.dot(qd_pred[0], n)
    # relative velocity after pbd velocity update
    v_n = wp.dot(qd_pred_before_rest[0], n)

    # dv rest should be a positive number
    # assume elasticity 1.0
    dv_rest = - v_n - wp.min(1. * v_n_old, 0.)

    wp.atomic_add(vel_delta, 0, n * dv_rest)

@wp.kernel
def solve_particle_ground_wall_delta(
    old_particle_x: wp.array(dtype=wp.vec3),
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    # invmass: wp.array(dtype=float),
    radius: float,
    mu: float,
    wall_x: float,
    delta: wp.array(dtype=wp.vec3)
):
    old_x = old_particle_x[0]
    x = particle_x[0]
    v = particle_v[0]
    diffx = x - old_x

    n = wp.vec3(0., 1., 0.)
    d = wp.dot(x, n)

    c = d - radius
    if c < 0.: # collision with ground
        # positional updates
        dlambda_n = - c
        delta_n = dlambda_n * n
        # static friction

        diffx_t = diffx - wp.dot(diffx, n) * n
        l_diffx_t = wp.length(diffx_t)
        n_t = diffx_t / (l_diffx_t + 1e-6)

        dlambda_t = - l_diffx_t
        # if static friction then tangential position should not be moved. 
        # if slip happens then we don't modify predicted tangential position
        if wp.abs(dlambda_t) < wp.abs(mu * dlambda_n):
            static_mask = 1.
        else:
            static_mask = 0.
        delta_t = static_mask * dlambda_t * n_t
        wp.atomic_add(delta, 0, delta_n + delta_t)

    n_w = wp.vec3(-1., 0., 0.)
    d_w = wall_x + wp.dot(x, n_w)
    c_w = d_w - radius
    if c_w < 0.: # collistion with wall
        # positional updates
        dlambda_n_w = - c_w
        delta_n_w = dlambda_n_w * n_w
        # static friction
        diffx_t_w = diffx - wp.dot(diffx, n_w) * n_w
        l_diffx_t_w = wp.length(diffx_t_w)
        n_t_w = diffx_t_w / (l_diffx_t_w + 1e-6)

        dlambda_t_w = - l_diffx_t_w
        # if static friction then tangential position should not be moved. 
        # if slip happens then we don't modify predicted tangential position
        if wp.abs(dlambda_t_w) < wp.abs(mu * dlambda_n_w):
            static_mask_w = 1.
        else:
            static_mask_w = 0.
        delta_t_w = static_mask_w * dlambda_t_w * n_t_w
        wp.atomic_add(delta, 0, delta_n_w + delta_t_w)
    

@wp.kernel
def solve_particle_ground_wall_vel_delta(
    q_pred: wp.array(dtype=wp.vec3),
    qd_pred: wp.array(dtype=wp.vec3),
    qd_pred_before_rest: wp.array(dtype=wp.vec3),
    # invmass: wp.array(dtype=float),
    radius: float,
    mu: float,
    elasticity: float,
    wall_x: float,
    vel_delta: wp.array(dtype=wp.vec3),
):
    # use q_pred to retrieve contact info, 
    # should enot use q_pred_before_rest to retrieve contact info
    x = q_pred[0] 
    rel_v = qd_pred[0]
    rel_v_old = qd_pred_before_rest[0]

    n = wp.vec3(0., 1., 0.)
    d = wp.dot(x, n)
    c = d - radius
    if c < 0.: # collision with the ground
        # old relative velocity (normal compoenent), we calculate the velocity after restitution based on this
        v_n_old = wp.dot(rel_v_old, n)
        # relative velocity after pbd velocity update
        v_n = wp.dot(rel_v, n)
        # dv rest should be a positive number
        # assume elasticity 1.0
        dv_rest = - v_n - wp.min(elasticity * v_n_old, 0.)

        # dynamic friction calculation
        v_t = rel_v - n * v_n
        v_t_norm = wp.length(v_t)
        if v_t_norm < 1e-6: # static friction
            delta_vt = wp.vec3(0., 0., 0.)
        else:
            v_t_dir = v_t / (v_t_norm)
            diff_v_n = (1. + elasticity) * v_n_old
            delta_vt = - v_t_dir * wp.min(
                mu * wp.abs(diff_v_n), v_t_norm
            )
        wp.atomic_add(vel_delta, 0, n * dv_rest + delta_vt)

    n_w = wp.vec3(-1., 0., 0.)
    d_w = wall_x + wp.dot(x, n_w)
    c_w = d_w - radius
    if c_w < 0.: # collision with the wall
        # old relative velocity (normal compoenent), we calculate the velocity after restitution based on this
        v_n_old_w = wp.dot(rel_v_old, n_w)
        # relative velocity after pbd velocity update
        v_n_w = wp.dot(rel_v, n_w)
        # dv rest should be a positive number
        # assume elasticity 1.0
        dv_rest_w = - v_n_w - wp.min(elasticity * v_n_old_w, 0.)

        # dynamic friction calculation
        v_t_w = rel_v - n_w * v_n_w
        v_t_norm_w = wp.length(v_t_w)
        if v_t_norm_w < 1e-6: # static friction
            delta_vt_w = wp.vec3(0., 0., 0.)
        else:
            v_t_dir_w = v_t_w / (v_t_norm_w)
            diff_v_n_w = (1. + elasticity) * v_n_old_w
            delta_vt_w = - v_t_dir * wp.min(
                mu * wp.abs(diff_v_n_w), v_t_norm_w
            )
        wp.atomic_add(vel_delta, 0, n_w * dv_rest_w + delta_vt_w)


class CustomizedXPBDIntegratorForTwoBalls:
    def __init__(self):
        pass

    def simulate(self, model, state_in, state_out, dt):

        with wp.ScopedTimer("simulate", False):

            # buffers
            q_pred = wp.zeros_like(state_in.particle_q)
            qd_pred = wp.zeros_like(state_in.particle_qd)
            q_pred_before_rest = wp.zeros_like(state_in.particle_q)
            qd_pred_before_rest = wp.zeros_like(state_in.particle_qd)
            delta = wp.zeros_like(state_in.particle_q)
            vel_delta = wp.zeros_like(state_in.particle_q)

            #----------------------------
            # integrate particles

            wp.launch(
                kernel=integrate_particles,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q, 
                    state_in.particle_qd, 
                    state_in.external_particle_f, 
                    model.particle_inv_mass, 
                    model.gravity, 
                    dt
                    ],
                outputs=[q_pred, qd_pred],
                device=model.device
            )


            # for i in range(self.iterations):

            wp.launch(
                kernel=solve_balls_collision_delta,
                dim=1,
                inputs=[
                    q_pred, 
                    qd_pred, 
                    model.particle_inv_mass, 
                    model.particle_radius
                ],
                outputs=[delta],
                device=model.device
            )

            # apply positional delta updates and update velocity
            wp.launch(
                kernel=apply_deltas,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    q_pred,
                    delta,
                    dt
                ],
                outputs=[q_pred_before_rest,
                            qd_pred_before_rest],
                device=model.device)

            # solve restitution velocity
            wp.launch(
                kernel=solve_balls_collision_vel_delta,
                dim=1,
                inputs=[
                    q_pred,
                    qd_pred, 
                    qd_pred_before_rest,
                    model.particle_inv_mass, 
                    model.particle_radius,
                ],
                outputs=[vel_delta],
                device=model.device,
            )
            # update restitution velocity
            wp.launch(
                kernel=apply_vel_deltas,
                dim=model.particle_count,
                inputs=[
                    qd_pred_before_rest,
                    vel_delta,
                ],
                outputs=[state_out.particle_qd],
                device=model.device,
            )

            state_out.particle_q = q_pred_before_rest

            return state_out


class CustomizedXPBDIntegratorForBounceOnce:
    def __init__(self):
        pass

    def simulate(self, model, state_in, state_out, dt):

        with wp.ScopedTimer("simulate", False):

            # buffers
            q_pred = wp.zeros_like(state_in.particle_q)
            qd_pred = wp.zeros_like(state_in.particle_qd)
            q_pred_before_rest = wp.zeros_like(state_in.particle_q)
            qd_pred_before_rest = wp.zeros_like(state_in.particle_qd)
            delta = wp.zeros_like(state_in.particle_q)
            vel_delta = wp.zeros_like(state_in.particle_q)

            #----------------------------
            # integrate particles

            wp.launch(
                kernel=integrate_particles,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q, 
                    state_in.particle_qd, 
                    state_in.external_particle_f,
                    model.particle_inv_mass, 
                    model.gravity, 
                    dt
                    ],
                outputs=[q_pred, qd_pred],
                device=model.device
            )


            # for i in range(self.iterations):

            wp.launch(
                kernel=solve_particle_ground_delta,
                dim=1,
                inputs=[
                    q_pred, 
                    qd_pred, 
                    # model.particle_inv_mass, 
                    model.particle_radius
                ],
                outputs=[delta],
                device=model.device
            )

            # apply positional delta updates and update velocity
            wp.launch(
                kernel=apply_deltas,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    q_pred,
                    delta,
                    dt
                ],
                outputs=[q_pred_before_rest,
                            qd_pred_before_rest],
                device=model.device)

            # solve restitution velocity
            wp.launch(
                kernel=solve_particle_ground_vel_delta,
                dim=1,
                inputs=[
                    q_pred,
                    qd_pred, 
                    qd_pred_before_rest,
                    # model.particle_inv_mass, 
                    model.particle_radius,
                ],
                outputs=[vel_delta],
                device=model.device,
            )
            # update restitution velocity
            wp.launch(
                kernel=apply_vel_deltas,
                dim=model.particle_count,
                inputs=[
                    qd_pred_before_rest,
                    vel_delta,
                ],
                outputs=[state_out.particle_qd],
                device=model.device,
            )

            state_out.particle_q = q_pred_before_rest

            return state_out


class CustomizedXPBDIntegratorForGroundWall:
    def __init__(self):
        pass

    def simulate(self, model, state_in, state_out, dt):

        with wp.ScopedTimer("simulate", False):

            # buffers
            q_pred = wp.zeros_like(state_in.particle_q)
            qd_pred = wp.zeros_like(state_in.particle_qd)
            q_pred_before_rest = wp.zeros_like(state_in.particle_q)
            qd_pred_before_rest = wp.zeros_like(state_in.particle_qd)
            delta = wp.zeros_like(state_in.particle_q)
            vel_delta = wp.zeros_like(state_in.particle_q)

            #----------------------------
            # integrate particles

            wp.launch(
                kernel=integrate_particles,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q, 
                    state_in.particle_qd, 
                    state_in.external_particle_f,
                    model.particle_inv_mass, 
                    model.gravity, 
                    dt
                    ],
                outputs=[q_pred, qd_pred],
                device=model.device
            )


            # for i in range(self.iterations):

            wp.launch(
                kernel=solve_particle_ground_wall_delta,
                dim=1,
                inputs=[
                    state_in.particle_q,
                    q_pred, 
                    qd_pred, 
                    # model.particle_inv_mass, 
                    model.particle_radius,
                    model.customized_mu,
                    model.customized_wall_x,
                ],
                outputs=[delta],
                device=model.device
            )

            # apply positional delta updates and update velocity
            wp.launch(
                kernel=apply_deltas,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    q_pred,
                    delta,
                    dt
                ],
                outputs=[q_pred_before_rest,
                            qd_pred_before_rest],
                device=model.device)

            # solve restitution velocity
            wp.launch(
                kernel=solve_particle_ground_wall_vel_delta,
                dim=1,
                inputs=[
                    q_pred,
                    qd_pred, 
                    qd_pred_before_rest,
                    # model.particle_inv_mass, 
                    model.particle_radius,
                    model.customized_mu,
                    model.customized_elasticity,
                    model.customized_wall_x,
                ],
                outputs=[vel_delta],
                device=model.device,
            )
            # update restitution velocity
            wp.launch(
                kernel=apply_vel_deltas,
                dim=model.particle_count,
                inputs=[
                    qd_pred_before_rest,
                    vel_delta,
                ],
                outputs=[state_out.particle_qd],
                device=model.device,
            )

            state_out.particle_q = q_pred_before_rest

            return state_out

