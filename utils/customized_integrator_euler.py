# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# https://github.com/NVIDIA/warp/blob/main/LICENSE.md

# This file is modified from 
# https://github.com/NVIDIA/warp/blob/main/warp/sim/integrator_euler.py

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
    v1 = v0 + (f0 * inv_mass + gravity * wp.step(0.0 - inv_mass)) *dt
    x1 = x0 + v1 * dt

    x_new[tid] = x1
    v_new[tid] = v1


@wp.kernel
def integrate_bodies(body_q: wp.array(dtype=wp.transform),
                     body_qd: wp.array(dtype=wp.spatial_vector),
                     body_f: wp.array(dtype=wp.spatial_vector),
                     body_com: wp.array(dtype=wp.vec3),
                     m: wp.array(dtype=float),
                     I: wp.array(dtype=wp.mat33),
                     inv_m: wp.array(dtype=float),
                     inv_I: wp.array(dtype=wp.mat33),
                     gravity: wp.vec3,
                     dt: float,
                     body_q_new: wp.array(dtype=wp.transform),
                     body_qd_new: wp.array(dtype=wp.spatial_vector)):

    tid = wp.tid()

    # positions
    q = body_q[tid]
    qd = body_qd[tid]
    f = body_f[tid]

    # masses
    mass = m[tid]
    inv_mass = inv_m[tid]     # 1 / mass

    inertia = I[tid]
    inv_inertia = inv_I[tid]  # inverse of 3x3 inertia matrix

    # unpack transform
    x0 = wp.transform_get_translation(q)
    r0 = wp.transform_get_rotation(q)

    # unpack spatial twist
    w0 = wp.spatial_top(qd)
    v0 = wp.spatial_bottom(qd)

    # unpack spatial wrench
    t0 = wp.spatial_top(f)
    f0 = wp.spatial_bottom(f)

    x_com = x0 + wp.quat_rotate(r0, body_com[tid])
 
    # linear part
    v1 = v0 + (f0 * inv_mass + gravity * wp.nonzero(inv_mass)) * dt
    x1 = x_com + v1 * dt
 
    # angular part (compute in body frame)
    wb = wp.quat_rotate_inv(r0, w0)
    tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb, inertia*wb)   # coriolis forces

    w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)
    r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)

    # angular damping, todo: expose
    w1 = w1*(1.0-0.1*dt)

    body_q_new[tid] = wp.transform(x1 - wp.quat_rotate(r1, body_com[tid]), r1)
    body_qd_new[tid] = wp.spatial_vector(w1, v1)


@wp.kernel
def eval_two_particles(
    particle_x: wp.array(dtype=wp.vec3),
    # particle_v: wp.array(dtype=wp.vec3),
    external_particle_f: wp.array(dtype=wp.vec3),
    radius: float,
    k_n: float,
    # outputs
    particle_f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid > 1:
        return 
    s_id = tid          # self id
    o_id = 1 - tid      # other id

    # add external force
    wp.atomic_add(particle_f, s_id, external_particle_f[s_id])
    # get position and velocity
    x_s = particle_x[s_id]
    x_o = particle_x[o_id]
    # v_s = particle_v[s_id]
    # v_o = particle_v[o_id]

    n = x_s - x_o
    d = wp.length(x_s - x_o)
    n = (x_s - x_o) / d
    err = d - radius * 2.0
    if err > 0: # no contact
        return
    
    # perfect elastic collision, no damping, normal force magnitude
    f_n = - err * k_n
    wp.atomic_add(particle_f, s_id, n * f_n)
    return 

@wp.kernel
def eval_particle_ground(
    particle_x: wp.array(dtype=wp.vec3),
    # particle_v: wp.array(dtype=wp.vec3),
    external_particle_f: wp.array(dtype=wp.vec3),
    radius: float,
    k_n: float,
    # outputs
    particle_f: wp.array(dtype=wp.vec3),
):
    # add external force
    wp.atomic_add(particle_f, 0, external_particle_f[0])
    x = particle_x[0]
    # v = particle_v[0]

    n = wp.vec3(0., 1., 0.)
    d = wp.dot(x, n)
    err = d - radius
    if err > 0: # no contact
        return

    # perfect elastic collision, no damping, normal force magnitude
    f_n = - err * k_n
    wp.atomic_add(particle_f, 0, n * f_n)
    return 

@wp.kernel
def eval_particle_ground_wall(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    external_particle_f: wp.array(dtype=wp.vec3),
    radius: float,
    k_n: float,
    k_d: float,
    # k_f: float,
    mu: float,
    wall_x: float,
    # outputs
    particle_f: wp.array(dtype=wp.vec3),
):
    # add external force
    wp.atomic_add(particle_f, 0, external_particle_f[0])

    x = particle_x[0]
    v = particle_v[0]

    n_g = wp.vec3(0., 1., 0.)
    d_g = wp.dot(x, n_g)
    err_g = d_g - radius
    if err_g < 0.: # contact with ground
        jn_g = err_g * k_n # negative
        # damping
        vn_g = wp.dot(n_g, v)
        jd_g = wp.min(vn_g, 0.) * k_d # negative
        # contact normal force
        fn_g = jn_g + jd_g
        # friction force
        vt_g = v - n_g * vn_g
        vs_g = wp.length(vt_g)
        if (vs_g > 1e-6):
            vt_g = vt_g / vs_g
            ft_g = mu * wp.abs(fn_g)
            wp.atomic_add(particle_f, 0, - vt_g * ft_g)
        # ft_g = wp.min(vs_g*k_f, mu * wp.abs(fn_g))

        wp.atomic_add(particle_f, 0, - n_g * fn_g )

    
    n_w = wp.vec3(-1., 0., 0.)
    d_w = wall_x + wp.dot(x, n_w)
    err_w = d_w - radius
    if err_w < 0.: # contact with wall
        jn_w = err_w * k_n # negative
        # damping
        vn_w = wp.dot(n_w, v)
        jd_w = wp.min(vn_w, 0.) * k_d # negative
        # contact normal force
        fn_w = jn_w + jd_w
        # friction force
        vt_w = v - n_w * vn_w
        vs_w = wp.length(vt_w)
        if (vs_w > 1e-6):
            vt_w = vt_w / vs_w
            ft_w = mu * wp.abs(fn_w)
            wp.atomic_add(particle_f, 0, - vt_w * ft_w)
        # ft_w = wp.min(vs_w*k_f, mu * wp.abs(fn_w))

        wp.atomic_add(particle_f, 0, - n_w * fn_w )


def compute_forces(model, state, particle_f, body_f):
    # balls are modeled as particles (no friction and rotation)
    if (model.particle_count == 2):
        wp.launch(
            kernel=eval_two_particles,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                # state.particle_qd,
                state.external_particle_f,
                # ctrls, 
                model.particle_radius,
                model.customized_kn,
            ],
            outputs=[particle_f],
            device=model.device,
        )
        return
    assert model.particle_count == 1
    if hasattr(model, "customized_particle_bounce_once") and model.customized_particle_bounce_once:
        wp.launch(
            kernel=eval_particle_ground,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                state.external_particle_f,
                model.particle_radius,
                model.customized_kn,
            ],
            outputs=[particle_f],
            device=model.device,
        )
        return
    if hasattr(model, "customized_particle_ground_wall") and model.customized_particle_ground_wall:
        wp.launch(
            kernel=eval_particle_ground_wall,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                state.external_particle_f,
                model.particle_radius,
                model.customized_kn,
                model.customized_kd,
                # model.customized_kf,
                model.customized_mu,
                model.customized_wall_x
            ],
            outputs=[particle_f],
            device=model.device,
        )
        return



class CustomizedSymplecticEulerIntegrator:

    def __init__(self):
        pass


    def simulate(self, model, state_in, state_out, dt):

        with wp.ScopedTimer("simulate", False):

            particle_f = None
            body_f = None

            if state_in.particle_count:
                particle_f = state_in.particle_f
            
            if state_in.body_count:
                body_f = state_in.body_f

            compute_forces(model, state_in, particle_f, body_f)

            #-------------------------------------
            # integrate bodies

            if (model.body_count):

                wp.launch(
                    kernel=integrate_bodies,
                    dim=model.body_count,
                    inputs=[
                        state_in.body_q,
                        state_in.body_qd,
                        state_in.body_f,
                        model.body_com,
                        model.body_mass,
                        model.body_inertia,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        model.gravity,
                        dt,
                    ],
                    outputs=[
                        state_out.body_q,
                        state_out.body_qd
                    ],
                    device=model.device)

            #----------------------------
            # integrate particles

            if (model.particle_count):

                wp.launch(
                    kernel=integrate_particles,
                    dim=model.particle_count,
                    inputs=[
                        state_in.particle_q, 
                        state_in.particle_qd,
                        state_in.particle_f,
                        model.particle_inv_mass, 
                        model.gravity, 
                        dt
                    ],
                    outputs=[
                        state_out.particle_q, 
                        state_out.particle_qd
                        ],
                    device=model.device)

            return state_out