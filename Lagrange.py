import os
import time

import dill
import mujoco
import numpy as np
import symengine as se

def generate_inverse_dynamics_function(dof, method_name, dynamic_model):
    q_parts = [f'q_{i + 1} = q[{i}]' for i in range(dof)]
    q_str = '\n\t'.join(q_parts)
    dq_parts = [f'dq_{i + 1} = dq[{i}]' for i in range(dof)]
    dq_str = '\n\t'.join(dq_parts)

    function_str = f"""
from numpy import sin, cos


def {method_name}(q, dq):
	{q_str}
	{dq_str}
	{dynamic_model}
"""
    return function_str


def generate_forward_dynamics_function(dof, method_name, dynamic_model):
    q_parts = [f'q_{i + 1} = q[{i}]' for i in range(dof)]
    q_str = '\n\t'.join(q_parts)
    dq_parts = [f'dq_{i + 1} = dq[{i}]' for i in range(dof)]
    dq_str = '\n\t'.join(dq_parts)
    tau_parts = [f'tau_{i + 1} = tau[{i}]' for i in range(dof)]
    tau_str = '\n\t'.join(tau_parts)

    function_str = f"""
from numpy import sin, cos


def {method_name}(q, dq, tau):
	{q_str}
	{dq_str}
	{tau_str}
	{dynamic_model}
"""
    return function_str


def generate_matrix_function(dof, method_name, dynamic_model):
    q_parts = [f'q_{i + 1} = q[{i}]' for i in range(dof)]
    q_str = '\n\t'.join(q_parts)

    function_str = f"""
from numpy import sin, cos


def {method_name}(q):
	{q_str}
	{dynamic_model}
"""
    return function_str


def generate_coriolis_matrix_function(dof, method_name, dynamic_model):
    q_parts = [f'q_{i + 1} = q[{i}]' for i in range(dof)]
    q_str = '\n\t'.join(q_parts)
    dq_parts = [f'dq_{i + 1} = dq[{i}]' for i in range(dof)]
    dq_str = '\n\t'.join(dq_parts)

    function_str = f"""
from numpy import sin, cos


def {method_name}(q, dq):
	{q_str}
	{dq_str}
	{dynamic_model}
"""
    return function_str


def dump(file, matrices):
    with open(file, 'wb') as f:
        dill.dump(matrices, f)


def load(file):
    with open(file, 'rb') as f:
        matrices = dill.load(f)
    return matrices


class Lagrange:
    def __init__(self, dof, MDH, mass, inertia, centroid, g, cache=True):
        self.dof = dof
        self.MDH = MDH
        self.mass = mass
        self.inertia = inertia
        self.centroid = centroid
        self.g = g

        self.trans_expressions = None
        self.centroid_expressions = None
        self.jacobian_expressions = None
        self.M = None
        self.M_inv = None
        self.G = None
        self.C = None
        self.K = None
        self.P = None
        self.L = None
        # 数值化函数
        self.mass_matrix_numeric_functions = {}
        self.mass_matrix_inv_numeric_functions = {}
        self.gravity_matrix_numeric_functions = {}
        self.coriolis_matrix_numeric_functions = {}
        self.inverse_dynamics_numeric_functions = {}
        self.forward_dynamics_numeric_functions = {}

        self.cache = cache
        self.precompute_expressions()

    @staticmethod
    def mdh_matrix_expression(alpha, a, d, theta):
        return se.Matrix([
            [se.cos(theta), -se.sin(theta), 0, a],
            [se.sin(theta) * se.cos(alpha), se.cos(theta) * se.cos(alpha), -se.sin(alpha), -se.sin(alpha) * d],
            [se.sin(theta) * se.sin(alpha), se.cos(theta) * se.sin(alpha), se.cos(alpha), se.cos(alpha) * d],
            [0, 0, 0, 1]
        ])

    def transformation_matrix_expression(self):
        T = se.eye(4)
        matrix_list = []
        q_syms = se.symbols(f'q_1:{self.dof + 1}')

        for i in range(self.dof):
            A_i = self.mdh_matrix_expression(self.MDH['alpha'][i], self.MDH['a'][i], self.MDH['d'][i], q_syms[i] + self.MDH['offset'][i])
            T = T * A_i
            matrix_list.append(T)
        return matrix_list

    def centroid_expression(self):
        """
        统一质心基坐标
        :return:
        """
        centroid_list = []
        for i in range(self.dof):
            centroid = self.trans_expressions[i][:3, 3] + self.trans_expressions[i][:3, :3] * self.centroid[i]
            centroid_list.append(centroid)
        # print(len(centroid_list))
        return centroid_list

    def jacobian_matrix_expression(self):
        """
        计算质心雅可比矩阵
        todo 貌似这里没有计算质心雅可比矩阵，而是用的雅可比矩阵
        :return:
        """
        jacobian_list = []
        z_axes = [T[:3, 2] for T in self.trans_expressions]

        for j in range(self.dof):
            jacobian = se.zeros(6, self.dof)
            pos = self.trans_expressions[j][:3, 3]

            for i in range(j + 1):
                z_axis_i = z_axes[i]
                # todo 这里应该拿的是质心的位置（以基座坐标系为参考的）
                pos_i = self.trans_expressions[i][:3, 3]

                # 线速度部分 (位置雅可比)
                jacobian[:3, i] = z_axis_i.cross(pos - pos_i)
                # 角速度部分 (旋转雅可比)
                jacobian[3:, i] = z_axis_i
            jacobian_list.append(jacobian)

        return jacobian_list

    def mass_matrix(self):
        """
        计算质量矩阵 n*n
        M(q) = ∑ Jv.T * m_i * Jv + Jw.T * I_i * Jw
        :return:
        """
        M = se.zeros(self.dof, self.dof)
        for i in range(self.dof):
            I = self.inertia[i]
            J = self.jacobian_expressions[i]
            Jv = J[:3, :]
            Jw = J[3:, :]
            matrix = Jv.transpose() * self.mass[i] * Jv + Jw.transpose() * I * Jw
            M += matrix
        return M

    def gravity_matrix(self):
        q_syms = se.symbols(f'q_1:{self.dof + 1}')
        G = se.zeros(self.dof, 1)
        for i in range(self.dof):
            G[i, 0] = se.diff(self.P, q_syms[i])
        return G

    def coriolis_matrix(self):
        """
        计算科氏力矩阵 n*n
        :return:
        """
        C = se.zeros(self.dof, self.dof)
        q_syms = se.symbols(f'q_1:{self.dof + 1}')
        dq_syms = se.symbols(f'dq_1:{self.dof + 1}')
        for i in range(self.dof):
            q_i = q_syms[i]
            for j in range(self.dof):
                M_ij = self.M[i, j]
                q_j = q_syms[j]
                for k in range(self.dof):
                    M_ik = self.M[i, k]
                    M_kj = self.M[k, j]
                    q_k = q_syms[k]
                    dq_k = dq_syms[k]
                    gamma_ijk = (se.diff(M_ij, q_k) + se.diff(M_ik, q_j) - se.diff(M_kj, q_i)) / 2
                    C[i, j] += (gamma_ijk * dq_k)
        return C

    def compute_kinetic_energy(self):
        """
        计算动能
        :return:
        """
        dq_syms = se.Matrix(se.symbols(f'dq_1:{self.dof + 1}'))
        K = (dq_syms.transpose() * self.M * dq_syms) / 2
        return K

    def compute_potential_energy(self):
        """
        计算重力势能
        :return:
        """

        P = se.zeros(1, 1)
        for i in range(self.dof):
            m = self.mass[i]
            P_i = m * self.g.transpose() * self.centroid_expressions[i]
            P = P + P_i
        return P

    def matrix_numeric_function(self):
        mass_matrix_numeric_functions = {}
        mass_matrix_inv_numeric_functions = {}
        coriolis_matrix_numeric_functions = {}
        gravity_matrix_matrix_numeric_functions = {}
        for i in range(self.dof):
            for j in range(self.dof):
                m_equation = [self.M[i, j]]
                m_replacements, m_reduced_expr = se.cse(m_equation)
                m_replacement_str = ''
                for replacement in m_replacements:
                    m_replacement_str += f'{replacement[0]} = {replacement[1]}\n\t'
                m_reduced_str = f"return {m_reduced_expr[0]}"
                m_equation_str = m_replacement_str + m_reduced_str
                m_function_str = generate_matrix_function(dof=self.dof, method_name=f'mass_matrix_{i + 1}_{j + 1}', dynamic_model=m_equation_str)
                exec(m_function_str, mass_matrix_numeric_functions)

                m_inv_equation = [self.M_inv[i, j]]
                m_inv_replacements, m_inv_reduced_expr = se.cse(m_inv_equation)
                m_inv_replacement_str = ''
                for replacement in m_inv_replacements:
                    m_inv_replacement_str += f'{replacement[0]} = {replacement[1]}\n\t'
                m_inv_reduced_str = f"return {m_inv_reduced_expr[0]}"
                m_inv_equation_str = m_inv_replacement_str + m_inv_reduced_str
                m_inv_function_str = generate_matrix_function(dof=self.dof, method_name=f'mass_matrix_inv_{i + 1}_{j + 1}', dynamic_model=m_inv_equation_str)
                exec(m_inv_function_str, mass_matrix_inv_numeric_functions)

                c_equation = [self.C[i, j]]
                c_replacements, c_reduced_expr = se.cse(c_equation)
                c_replacement_str = ''
                for replacement in c_replacements:
                    c_replacement_str += f'{replacement[0]} = {replacement[1]}\n\t'
                c_reduced_str = f"return {c_reduced_expr[0]}"
                c_equation_str = c_replacement_str + c_reduced_str
                c_function_str = generate_coriolis_matrix_function(dof=self.dof, method_name=f'coriolis_matrix_{i + 1}_{j + 1}', dynamic_model=c_equation_str)
                exec(c_function_str, coriolis_matrix_numeric_functions)

            g_equation = [self.G[i, 0]]
            g_replacements, g_reduced_expr = se.cse(g_equation)
            g_replacement_str = ''
            for replacement in g_replacements:
                g_replacement_str += f'{replacement[0]} = {replacement[1]}\n\t'
            g_reduced_str = f"return {g_reduced_expr[0]}"
            g_equation_str = g_replacement_str + g_reduced_str
            g_function_str = generate_matrix_function(dof=self.dof, method_name=f'gravity_matrix_{i + 1}', dynamic_model=g_equation_str)
            exec(g_function_str, gravity_matrix_matrix_numeric_functions)

        return mass_matrix_numeric_functions, mass_matrix_inv_numeric_functions, coriolis_matrix_numeric_functions, gravity_matrix_matrix_numeric_functions

    def forward_dynamics_numeric_function(self):
        dq_syms = se.Matrix(se.symbols(f'dq_1:{self.dof + 1}'))
        tau_syms = se.Matrix(se.symbols(f'tau_1:{self.dof + 1}'))
        M_inv = self.M.inv()
        ddq = M_inv * (tau_syms - self.C * dq_syms - self.G)
        forward_dynamics_numeric_functions = {}
        for i in range(self.dof):
            equation = [ddq[i]]
            replacements, reduced_expr = se.cse(equation)
            replacement_str = ''
            for replacement in replacements:
                replacement_str += f'{replacement[0]} = {replacement[1]}\n\t'
            reduced_str = f"return {reduced_expr[0]}"
            equation_str = replacement_str + reduced_str
            function_str = generate_forward_dynamics_function(dof=self.dof, method_name=f'forward_dynamics_equation{i + 1}', dynamic_model=equation_str)
            exec(function_str, forward_dynamics_numeric_functions)

        return forward_dynamics_numeric_functions

    def inverse_dynamics_numeric_function(self):
        q_syms = se.symbols(f'q_1:{self.dof + 1}')
        dq_syms = se.symbols(f'dq_1:{self.dof + 1}')

        inverse_dynamics_numeric_functions = {}
        equations = []
        for i in range(self.dof):
            equation = se.diff(self.L, dq_syms[i]) - se.diff(self.L, q_syms[i])
            equations.append(equation)
        for i in range(self.dof):
            equation = equations[i]
            replacements, reduced_expr = se.cse(equation)
            replacement_str = ''
            for replacement in replacements:
                replacement_str += f'{replacement[0]} = {replacement[1]}\n\t'
            reduced_str = f"return {reduced_expr[0]}"
            equation_str = replacement_str + reduced_str
            function_str = generate_inverse_dynamics_function(dof=self.dof, method_name=f'inverse_dynamics_equation{i + 1}', dynamic_model=equation_str)
            exec(function_str, inverse_dynamics_numeric_functions)

        return inverse_dynamics_numeric_functions

    def precompute_expressions(self):
        self.trans_expressions = self.transformation_matrix_expression()
        self.centroid_expressions = self.centroid_expression()
        self.jacobian_expressions = self.jacobian_matrix_expression()
        self.M = self.mass_matrix()
        self.M_inv = self.M.inv()
        self.C = self.coriolis_matrix()
        self.K = self.compute_kinetic_energy()
        self.P = self.compute_potential_energy()
        self.G = self.gravity_matrix()
        self.L = self.K - self.P
        self.inverse_dynamics_numeric_functions = self.inverse_dynamics_numeric_function()
        self.forward_dynamics_numeric_functions = self.forward_dynamics_numeric_function()
        (self.mass_matrix_numeric_functions, self.mass_matrix_inv_numeric_functions,
         self.coriolis_matrix_numeric_functions, self.gravity_matrix_numeric_functions) = self.matrix_numeric_function()

    def forward_dynamics(self, q, dq, tau):
        ddq = []
        for i in range(self.dof):
            func = self.forward_dynamics_numeric_functions[f'forward_dynamics_equation{i + 1}']
            a = func(q, dq, tau)
            ddq.append(float(a))
        return ddq

    def inverse_dynamics(self, q, dq):
        torques = []
        for i in range(self.dof):
            func = self.inverse_dynamics_numeric_functions[f'inverse_dynamics_equation{i + 1}']
            torque = func(q, dq)
            torques.append(float(torque))
        return torques

    def compute_mass_matrix(self, q):
        row, column = self.M.shape[0], self.M.shape[1]
        matrix = np.zeros((row, column))
        for i in range(row):
            for j in range(column):
                func = self.mass_matrix_numeric_functions[f'mass_matrix_{i + 1}_{j + 1}']
                matrix[i, j] = func(q)
        return matrix

    def compute_mass_inv_matrix(self, q):
        row, column = self.M_inv.shape[0], self.M_inv.shape[1]
        matrix = np.zeros((row, column))
        for i in range(row):
            for j in range(column):
                func = self.mass_matrix_inv_numeric_functions[f'mass_matrix_inv_{i + 1}_{j + 1}']
                matrix[i, j] = func(q)
        return matrix

    def compute_gravity_matrix(self, q):
        row = self.G.shape[0]
        matrix = np.zeros((row, 1))
        for i in range(row):
            func = self.gravity_matrix_numeric_functions[f'gravity_matrix_{i + 1}']
            matrix[i, 0] = func(q)
        return matrix

    def compute_coriolis_matrix(self, q, dq):
        row, column = self.C.shape[0], self.C.shape[1]
        matrix = np.zeros((row, column))
        for i in range(row):
            for j in range(column):
                func = self.coriolis_matrix_numeric_functions[f'coriolis_matrix_{i + 1}_{j + 1}']
                matrix[i, j] = func(q, dq)
        return matrix

def main():
    dof = 6
    from constants import MDH
    from constants import DYNAMICS
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

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"动力学建模开发: {start_time}")
    lagrange = Lagrange(dof=dof, MDH=MDH, mass=masses, inertia=inertia, centroid=centroids, g=se.Matrix([[0], [0], [9.81]]), cache=True)
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"动力学建模完成: {end_time}")

    q = np.array([0.14797751, -0.57717571,  2.86570107,  1.78682623,  2.62996162,  0.16889559])
    dq = np.array([1.80332186, -0.36981029, 1.58605565, -0.92173959, -3.57522989, -0.17519417])
    ddq = np.array([2.58669297,  12.35849133,  2.12530537, -4.44365651, -37.11187212, 3.70702705])



    verify_forward_dynamics(lagrange, q, dq, ddq)


def verify_forward_dynamics(lagrange, q, dq, ddq):
    # 逆动力学计算力矩
    torques = lagrange.inverse_dynamics(q, dq)

    print(torques)
    # 正向动力学计算加速度
    ddq = lagrange.forward_dynamics(q, dq, torques)
    # print(q)
    # 验证加速度结果: τ = M(q)*ddq + C(q, dq)*dq + G(q)
    M = lagrange.compute_mass_matrix(q)
    G = lagrange.compute_gravity_matrix(q)
    C = lagrange.compute_coriolis_matrix(q, dq)

    dq = dq.reshape(6, 1)
    ddq = np.array(ddq).reshape(6, 1)

    tau = np.array(torques).reshape(6, 1)
    _ddq = np.linalg.inv(M) @ (tau - C @ dq - G)

    T = M @ ddq + C @ dq + G

    print(T.flatten())

np.set_printoptions(precision=5, suppress=True)

if __name__ == '__main__':

    main()
