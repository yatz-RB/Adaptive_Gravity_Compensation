from numpy.f2py.crackfortran import groupbegins90
from six import print_

from constants import MDH, DYNAMICS
import os
import mujoco.viewer as mv
import numpy as np
from dm_control import mujoco
import time
from Lagrange import Lagrange
import symengine as se
import sympy as sp
from constants import MDH
from constants import DYNAMICS
from scipy.optimize import least_squares

# 加载自定义Mujoco模型
xml_path = os.path.join('rm65_b/rm65b.xml')
physics = mujoco.Physics.from_xml_path(xml_path)
model = physics.model
data = physics.data


def get_data(physics):
    t_get = physics.data.qfrc_bias
    q = physics.data.qpos
    dq = physics.data.qvel
    ddq = physics.data.qacc
    dq = dq.reshape(6, 1)
    ddq = np.array(ddq).reshape(6, 1)

    return t_get,q,dq,ddq

def add_extra():
    dm = np.random.rand()
    dc = [np.random.rand()*0.1,np.random.rand()*0.1,np.random.rand()*0.1]

    DYNAMICS['m'][5] += dm
    DYNAMICS['x'][5] += dc[0]
    DYNAMICS['y'][5] += dc[1]
    DYNAMICS['z'][5] += dc[2]

    # print(dm,dc)

    return dm,dc

def pri_cal_extra(j):
    if j ==0:
        m_syms = se.symbols(f'm_1:{6 + 1}')
        x_syms = se.symbols(f'x_1:{6 + 1}')
        y_syms = se.symbols(f'y_1:{6 + 1}')
        z_syms = se.symbols(f'z_1:{6 + 1}')

        alpha_syms = se.symbols(f'alpha_1:{6 + 1}')
        a_syms = se.symbols(f'a_1:{6 + 1}')
        d_syms = se.symbols(f'd_1:{6 + 1}')
        offset_syms = se.symbols(f'theta_1:{6 + 1}')

        gravity_a = se.symbols('g')
    else:
        m_syms = DYNAMICS['m'][:5]
        x_syms = DYNAMICS['x'][:5]
        y_syms = DYNAMICS['y'][:5]
        z_syms = DYNAMICS['z'][:5]
        m_syms.append(se.symbols('dm'))
        x_syms.append(se.symbols('dx'))
        y_syms.append(se.symbols('dy'))
        z_syms.append(se.symbols('dz'))
        alpha_syms = MDH['alpha']
        a_syms = MDH['a']
        d_syms = MDH['d']
        offset_syms = MDH['offset']

        gravity_a = 9.81
    q_syms = se.symbols(f'q_1:{6 + 1}')



    cent = [se.Matrix([[x_syms[i]], [y_syms[i]], [z_syms[i]]]) for i in range(6)]
    masses = m_syms[:6]


    T = se.eye(4)
    matrix_list = []

    for i in range(6):
        A_i = se.Matrix([
            [se.cos(q_syms[i] + offset_syms[i]), -se.sin(q_syms[i] + offset_syms[i]), 0, a_syms[i]],
            [se.sin(q_syms[i] + offset_syms[i]) * se.cos(alpha_syms[i]), se.cos(q_syms[i] + offset_syms[i]) * se.cos(alpha_syms[i]), -se.sin(alpha_syms[i]), -se.sin(alpha_syms[i]) * d_syms[i]],
            [se.sin(q_syms[i] + offset_syms[i]) * se.sin(alpha_syms[i]), se.cos(q_syms[i] + offset_syms[i]) * se.sin(alpha_syms[i]), se.cos(alpha_syms[i]), se.cos(alpha_syms[i]) * d_syms[i]],
            [0, 0, 0, 1]
        ])

        T = T * A_i
        matrix_list.append(T)

    centroid_list = []
    for i in range(6):
        centroid = matrix_list[i][:3, 3] + matrix_list[i][:3, :3] * cent[i]
        centroid_list.append(centroid)

    # for i in centroid_list:
    #     print(se.nsimplify(i, tolerance=1e-5))

    P = se.zeros(1, 1)
    for i in range(6):
        m = masses[i]
        P_i = m * se.Matrix([[0], [0], [gravity_a]]).transpose()* centroid_list[i]
        P = P + P_i

    G = se.zeros(6, 1)


    for i in range(6):
        g = se.diff(P, q_syms[i])
        G[i, 0] = g
        print(G[i, 0])

    return G


def cal(model,q,dq,ddq):
    M = model.compute_mass_matrix(q)
    G = model.compute_gravity_matrix(q)
    C = model.compute_coriolis_matrix(q, dq)
    T = M @ ddq + C @ dq + G
    return T.flatten()

def check_extra(pre_list,tau,q_i):
    # know: m_syms[:5],x_syms[:5],y_syms[:5],z_syms[:5]
    # unknow: m_syms[5],x_syms[5],y_syms[5],z_syms[5]
    # getable: tau q_i

    dm, dx, dy, dz = se.symbols('dm dx dy dz')
    q1, q2, q3, q4, q5 ,q6 = se.symbols('q_1 q_2 q_3 q_4 q_5 q_6')

    # 步骤2: 定义符号表达式（假设已知 τ 的表达式）
    # 这里以示例函数形式占位，需替换为实际表达式
    tau1 = pre_list[1]
    tau2 = pre_list[2]
    tau3 = pre_list[3]
    tau4 = pre_list[4]
    tau5 = pre_list[5]

    # 已知数值
    q_values = {q2: q_i[1], q3: q_i[2], q4: q_i[3], q5: q_i[4], q6: q_i[5]}
    tau_num_values = [tau[1], tau[2], tau[3], tau[4], tau[5]]  # τ1到τ5的数值

    # 转换为数值函数
    def residual(vars):
        dm_val, dx_val, dy_val, dz_val = vars
        # 代入 q 和未知数数值，计算表达式值
        subs_dict = {**q_values, dm: dm_val, dx: dx_val, dy: dy_val, dz: dz_val}
        return [
            float(tau1.subs(subs_dict) - tau_num_values[0]),
            float(tau2.subs(subs_dict) - tau_num_values[1]),
            float(tau3.subs(subs_dict) - tau_num_values[2]),
            float(tau4.subs(subs_dict) - tau_num_values[3]),
            float(tau5.subs(subs_dict) - tau_num_values[4]),
        ]

    # 初始猜测
    initial_guess = [1, 0.05,0.05, 0.05]

    # 最小二乘优化
    result0 = least_squares(residual, initial_guess, method='lm')
    print("数值解:", result0.x)

    tau_calculated =[]
    for tt in pre_list:
        v = tt.subs({dm:result0.x[0],dx:result0.x[1],dy:result0.x[2],dz:result0.x[3],q1:q_i[0],q2: q_i[1], q3: q_i[2], q4: q_i[3], q5: q_i[4], q6: q_i[5]})
        tau_calculated.append(v)
    # 计算所有τ值
    print(f"tc: {tau_calculated}")

    return result0.x,tau_calculated





def main():
    # dm,dc = add_extra()
    # pre_list = pri_cal_extra(1)
    # check_extra(pri_cal_extra(), [0, -7.04, 0.44, -0.06, -0.35, 0], [0, 1.85, -2.05, 1.09, 1.21, 0])

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"动力学建模开发: {start_time}")


    dof = 6
    centroids = [se.Matrix([[DYNAMICS['x'][i]], [DYNAMICS['y'][i]], [DYNAMICS['z'][i]]]) for i in range(dof)]

    masses = DYNAMICS['m'][:dof]
    inertia = []

    for i in range(dof):
        I = se.eye(3)
        I[0, 0] = DYNAMICS['l_xx'][i]
        I[0, 1] = DYNAMICS['l_xy'][i]
        I[0, 2] = DYNAMICS['l_xz'][i]

        I[1, 0] = DYNAMICS['l_xy'][i]
        I[1, 1] = DYNAMICS['l_yy'][i]
        I[1, 2] = DYNAMICS['l_yz'][i]

        I[2, 0] = DYNAMICS['l_xz'][i]
        I[2, 1] = DYNAMICS['l_yz'][i]
        I[2, 2] = DYNAMICS['l_zz'][i]
        inertia.append(I)

    lagrange = Lagrange(dof=dof, MDH=MDH, mass=masses, inertia=inertia, centroid=centroids,
                        g=se.Matrix([[0], [0], [9.81]]), cache=True)
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    print(f"动力学建模完成: {end_time}")
    target_qpos = np.array([0, -1.52, 1.2, -0.4, 1.41, 2.1])

    # fr0 = open("tau_without_real.txt","w")
    # fc0 = open("tau_without_cal.txt","w")
    # fr0.write("\n")
    # fc0.write("\n")
    # fr0.close()
    # fc0.close()

    #
    # for i in range(6):
    #     f = open(f"1j{i}.txt","w")
    #     f.write("\n")
    #     f.close()


    # finit = open("init.txt","w")
    # finit.write(str(dm+0.107) + "\ " + str(dc[0]-0.000506)+ "\ " + str(dc[1]-0.000255)+ "\ " + str(dc[2]-0.010801))
    # finit.close()
    # fdm = open("dm.txt","w")
    # fdx = open("dx.txt","w")
    # fdy = open("dy.txt","w")
    # fdz = open("dz.txt","w")
    # fr1 = open("tau_without_real.txt","w")
    # fc1 = open("tau_without_cal.txt","w")
    # fdm.write("\n")
    # fdm.close()
    # fdx.write("\n")
    # fdx.close()
    # fdy.write("\n")
    # fdy.close()
    # fdz.write("\n")
    # fdz.close()
    # fr1.write("\n")
    # fr1.close()
    # fc1.write("\n")
    # fc1.close()




    with mv.launch_passive(physics.model.ptr, physics.data.ptr, key_callback=None) as viewer:
        viewer.sync()
        physics.data.ctrl[:] = target_qpos
        q0 = [0] * 6
        tau0 = np.array(range(6))
        count = 0
        while viewer.is_running():
            physics.step()
            viewer.sync()
            # ___________________________________________________________________
            t_get,q,dq,ddq = get_data(physics)
            # ___________________________________________________________________
            # cal tau
            tau = cal(lagrange,q,dq,ddq)
            # print(f"cal: {tau}")
            # print(f"get: {t_get}")
            # print(tau0)
            # dmdxdydz, tau_cal = check_extra(pre_list, tau, q)
            # print(f"预设值{0.107+dm,-0.000506+dc[0],dc[1]-0.000255,dc[2]-0.010801}")

            # if count < 5000:
            #     for i in range(6):
            #         f = open(f"1j{i}.txt", "a")
            #         f.write(str(tau[i] - tau_cal[i]) + "\n")
            #         f.close()
            #     count += 1
            #     print(count)
            # else:
            #     print("finish")
            # if np.array_equal(tau, tau0):
            #     check_extra(pre_list,tau,q)
            # else:
            #     tau0 = tau

            # print()
            # print(f"q: {q}")
            # print(f"tau0_cal: {tau_0}")
            # print(f"tau _get: {t_get}")
            # print()
            # ___________________________________________________________________



if __name__ == '__main__':
    main()
