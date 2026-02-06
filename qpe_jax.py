import jax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)

import numpy

import pennylane as qml

from scipy import sparse
from scipy.linalg import eig, eigh, norm, expm, sqrtm
from scipy.stats import sem
from scipy.optimize import minimize
from functools import reduce, partial
from itertools import product
import qutip as qp
from multiprocessing import Pool
from time import time

X = jnp.array([[0.,1.],
              [1.,0.]])
Y = jnp.array([[0.,-1.j],
              [1.j, 0.]])
Z = jnp.array([[1., 0.],
              [0.,-1.]])
I = jnp.array([[1.,0.],
              [0.,1.]])

I_sp = sparse.csr_array(I)
X_sp = sparse.csr_array(X)
Y_sp = sparse.csr_array(Y)
Z_sp = sparse.csr_array(Z)


# some functions # 

def fidelity(A, B):
    A_sqrt = sqrtm(A)
    res = A_sqrt@B@A_sqrt
    res = sqrtm(res)
    return jnp.trace(res).real**2

def fkron(A, B):
    """ A faster (?) kronecker product. Taken from https://stackoverflow.com/a/56067827 """
    s = len(A)*len(B)
    return (A[:, None, :, None]*B[None, :, None, :]).reshape(s, s)

def fkron_diag(A, B):
    """ A faster (?) kronecker product for vectors. Taken from https://stackoverflow.com/a/56067827 """
    s = len(A)*len(B)
    return (A[:, None]*B[None, :]).reshape(s)


### ansatzes ###

Z_diag = jnp.array([1, -1])

def xz_rot(pars):
    """ xz-rotation """
    e1 = jnp.exp(1j*pars[1])
    e1c0 = e1*jnp.cos(pars[0])
    e1s0 = e1*jnp.sin(pars[0])
    return jnp.array([[e1c0.conjugate(), -1j*e1s0.conjugate()],
                      [        -1j*e1s0,                 e1c0]])

def rzz_diag(N, q1, q2, par):
    d = 2**N
    j, k = sorted([q1, q2])
    term_1 = jnp.full(d, jnp.cos(par))
    term_21 = jnp.kron(jnp.ones(2**j), Z_diag) # I^(j-1) X
    term_21 = jnp.kron(term_21, jnp.ones(2**(k - j - 1))) # I^(j-1) Z I^(k-j-2)
    term_22 = jnp.kron(Z_diag, jnp.ones(2**(N - k - 1))) # Z I^(n-k)
    term_2 = fkron_diag(term_21, term_22) # I^(j-1) Z I^(k-j-1) Z I^(n-k)
    term_2 = -1j*jnp.sin(par)*term_2
    return term_1 + term_2

def hea_rzz(pars, N, L):
    it = iter(pars)
    op = jnp.eye(2**N)
    for l in range(L):
        op_s = xz_rot([next(it), next(it)])
        for q in range(N - 1):
            op_s = fkron(op_s, xz_rot([next(it), next(it)]))
        op = op_s@op
        for q in range(N - 1):
            op = (rzz_diag(N, q, q + 1, next(it))*op.T).T
        if N > 2:
            op = (rzz_diag(N, 0, N - 1, next(it))*op.T).T
    return op

@qml.qnode(qml.device("lightning.qubit"), interface='jax')
def hea(pars, sv, L, N):
    qml.StatePrep(sv, wires=numpy.arange(N))
    pars_iter = iter(pars)
    for l in range(L):
        for q in range(N):    
            qml.RX(next(pars_iter), wires=q)
            qml.RZ(next(pars_iter), wires=q)
        for q in range(N - 1):    
            qml.MultiRZ(next(pars_iter), wires=[q, q + 1])
        if N > 2:
            qml.MultiRZ(next(pars_iter), wires=[0, N - 1])
    return qml.state()

@qml.qnode(qml.device("lightning.qubit", wires=9), interface='jax')
def qcnn_cluster_9q(pars, sv):
    
    qml.StatePrep(sv, wires=numpy.arange(9))
    
    for q in range(9):    
        qml.U3(*pars[0:3], wires=q)
    for q in range(4):
        qml.MultiRZ(pars[3], wires=[2*q, 2*q + 1])
    for q in range(4):
        qml.MultiRZ(pars[3], wires=[2*q + 1, 2*q + 2])
    qml.MultiRZ(pars[3], wires=[0, 8])
    for q in range(9):    
        qml.U3(*pars[4:7], wires=q)
    # qml.Barrier()

    # 0, 2, 4
    qml.U3(*pars[7:10], wires=0)
    qml.U3(*pars[10:13], wires=2)
    qml.ctrl(qml.U3, (0, 2))(*pars[13:16], wires=4)
    qml.adjoint(qml.U3(*pars[7:10], wires=0)) # this
    qml.adjoint(qml.U3(*pars[10:13], wires=2)) # this
    # qml.Barrier()
    
    # 6, 8, 1
    qml.U3(*pars[7:10], wires=6)
    qml.U3(*pars[10:13], wires=8)
    qml.ctrl(qml.U3, (6, 8))(*pars[13:16], wires=1)
    qml.adjoint(qml.U3(*pars[7:10], wires=6)) # this
    qml.adjoint(qml.U3(*pars[10:13], wires=8)) # this
    # qml.Barrier()

    # 3, 5, 7
    qml.U3(*pars[7:10], wires=3)
    qml.U3(*pars[10:13], wires=5)
    qml.ctrl(qml.U3, (3, 5))(*pars[13:16], wires=7)
    qml.adjoint(qml.U3(*pars[7:10], wires=3)) # this
    qml.adjoint(qml.U3(*pars[10:13], wires=5)) # this
    # qml.Barrier()

    # 3, 5, 1
    qml.U3(*pars[7:10], wires=3) # this
    qml.U3(*pars[10:13], wires=5) # this
    qml.ctrl(qml.U3, (3, 5))(*pars[13:16], wires=1)
    qml.adjoint(qml.U3(*pars[7:10], wires=3))
    qml.adjoint(qml.U3(*pars[10:13], wires=5))
    # qml.Barrier()

    # 6, 8, 4
    qml.U3(*pars[7:10], wires=6) # this
    qml.U3(*pars[10:13], wires=8) # this
    qml.ctrl(qml.U3, (6, 8))(*pars[13:16], wires=4)
    qml.adjoint(qml.U3(*pars[7:10], wires=6))
    qml.adjoint(qml.U3(*pars[10:13], wires=8))
    # qml.Barrier()

    # 0, 2, 7
    qml.U3(*pars[7:10], wires=0) # this
    qml.U3(*pars[10:13], wires=2) # this
    qml.ctrl(qml.U3, (0, 2))(*pars[13:16], wires=7)
    qml.adjoint(qml.U3(*pars[7:10], wires=0))
    qml.adjoint(qml.U3(*pars[10:13], wires=2))
    # qml.Barrier()

    # 1, 7, 4
    qml.U3(*pars[7:10], wires=1)
    qml.U3(*pars[10:13], wires=7)
    qml.ctrl(qml.U3, (1, 7))(*pars[13:16], wires=4)
    qml.adjoint(qml.U3(*pars[7:10], wires=1))
    qml.adjoint(qml.U3(*pars[10:13], wires=7))
    # qml.Barrier()

    return qml.state()


@qml.qnode(qml.device("lightning.qubit"), interface='jax')
def qcnn_schwinger(pars, sv, N):
    qml.StatePrep(sv, wires=numpy.arange(N))
    L = int(numpy.log2(N))
    it = iter(pars)
    for l in range(L):
        # print(l)
        pars_cur = [next(it) for _ in range(15)]
        # print("\tconv:")
        # print(pars_cur)
        for a in numpy.arange(0, N, 2**(l + 1)):
            # print("\t\t", a, a + 2**l)
            qml.U3(pars_cur[0], pars_cur[1], pars_cur[2], wires=a)
            qml.U3(pars_cur[3], pars_cur[4], pars_cur[5], wires=a + 2**l)
            qml.PauliRot(pars_cur[6], "XX", wires=[a, a + 2**l])
            qml.PauliRot(pars_cur[7], "YY", wires=[a, a + 2**l])
            qml.PauliRot(pars_cur[8], "ZZ", wires=[a, a + 2**l])
            qml.U3(pars_cur[9], pars_cur[10], pars_cur[11], wires=a)
            qml.U3(pars_cur[12], pars_cur[13], pars_cur[14], wires=a + 2**l)
        for a in numpy.arange(0, N - 2**(l + 1), 2**(l + 1)):
            # print("\t\t", a + l + 1, a + 2**l + l + 1)
            qml.U3(pars_cur[0], pars_cur[1], pars_cur[2], wires=a + l + 1)
            qml.U3(pars_cur[3], pars_cur[4], pars_cur[5], wires=a + 2**l + l + 1)
            qml.PauliRot(pars_cur[6], "XX", wires=[a + l + 1, a + 2**l + l + 1])
            qml.PauliRot(pars_cur[7], "YY", wires=[a + l + 1, a + 2**l + l + 1])
            qml.PauliRot(pars_cur[8], "ZZ", wires=[a + l + 1, a + 2**l + l + 1])
            qml.U3(pars_cur[9], pars_cur[10], pars_cur[11], wires=a + l + 1)
            qml.U3(pars_cur[12], pars_cur[13], pars_cur[14], wires=a + 2**l + l + 1)
        qml.Barrier()
        pars_cur = [next(it) for _ in range(6)]
        # print("\tpool:")
        # print(pars_cur)
        for a in numpy.arange(0, N, 2**(l + 1)):
            # print("\t\t", a, a + 2**l)
            qml.U3(pars_cur[0], pars_cur[1], pars_cur[2], wires=a + 2**l)
            qml.U3(pars_cur[3], pars_cur[4], pars_cur[5], wires=a)
            qml.CNOT(wires=[a + 2**l, a])
            qml.adjoint(qml.U3(pars_cur[3], pars_cur[4], pars_cur[5], wires=a))
        qml.Barrier()
    return qml.state()


@qml.qnode(qml.device("lightning.qubit"), interface='jax')
def qcnn_circ(pars, sv, N):
    qml.StatePrep(sv, wires=numpy.arange(N))
    L = int(numpy.log2(N))
    it = iter(pars)
    for l in range(L):
        # print(l)
        pars_cur = [next(it) for _ in range(15)]
        # print("\tconv:")
        # print(pars_cur)
        for a in numpy.arange(0, N, 2**(l + 1)):
            # print("\t\t", a, a + 2**l)
            qml.U3(pars_cur[0], pars_cur[1], pars_cur[2], wires=a)
            qml.U3(pars_cur[3], pars_cur[4], pars_cur[5], wires=a + 2**l)
            qml.PauliRot(pars_cur[6], "XX", wires=[a, a + 2**l])
            qml.PauliRot(pars_cur[7], "YY", wires=[a, a + 2**l])
            qml.PauliRot(pars_cur[8], "ZZ", wires=[a, a + 2**l])
            qml.U3(pars_cur[9], pars_cur[10], pars_cur[11], wires=a)
            qml.U3(pars_cur[12], pars_cur[13], pars_cur[14], wires=a + 2**l)
        for b in numpy.arange(0, N - 2**(l + 1), 2**(l + 1)):
            # print("\t\t", b + l + 1, b + 2**l + l + 1)
            qml.U3(pars_cur[0], pars_cur[1], pars_cur[2], wires=b + l + 1)
            qml.U3(pars_cur[3], pars_cur[4], pars_cur[5], wires=b + 2**l + l + 1)
            qml.PauliRot(pars_cur[6], "XX", wires=[b + l + 1, b + 2**l + l + 1])
            qml.PauliRot(pars_cur[7], "YY", wires=[b + l + 1, b + 2**l + l + 1])
            qml.PauliRot(pars_cur[8], "ZZ", wires=[b + l + 1, b + 2**l + l + 1])
            qml.U3(pars_cur[9], pars_cur[10], pars_cur[11], wires=b + l + 1)
            qml.U3(pars_cur[12], pars_cur[13], pars_cur[14], wires=b + 2**l + l + 1)
        if l != L - 1:
            print(l)
            qml.Barrier()
            qml.U3(pars_cur[0], pars_cur[1], pars_cur[2], wires=0)
            qml.U3(pars_cur[3], pars_cur[4], pars_cur[5], wires=a + 2**l)
            qml.PauliRot(pars_cur[6], "XX", wires=[0, a + 2**l])
            qml.PauliRot(pars_cur[7], "YY", wires=[0, a + 2**l])
            qml.PauliRot(pars_cur[8], "ZZ", wires=[0, a + 2**l])
            qml.U3(pars_cur[9], pars_cur[10], pars_cur[11], wires=0)
            qml.U3(pars_cur[12], pars_cur[13], pars_cur[14], wires=a + 2**l)
        qml.Barrier()
        pars_cur = [next(it) for _ in range(6)]
        # print("\tpool:")
        # print(pars_cur)
        for c in numpy.arange(0, N, 2**(l + 1)):
            # print("\t\t", c, c + 2**l)
            qml.U3(pars_cur[0], pars_cur[1], pars_cur[2], wires=c + 2**l)
            qml.U3(pars_cur[3], pars_cur[4], pars_cur[5], wires=c)
            qml.CNOT(wires=[c + 2**l, c])
            qml.adjoint(qml.U3(pars_cur[3], pars_cur[4], pars_cur[5], wires=c))
        qml.Barrier()
    return qml.state()


@qml.qnode(qml.device("lightning.qubit"), interface='jax')
def hva_transverse_cluster(pars, sv, L, N):
    qml.StatePrep(sv, wires=numpy.arange(N))
    pars_iter = iter(pars)
    for l in range(L):
        par_loc = next(pars_iter)
        for q in range(N):    
            qml.RX(par_loc, wires=q)
        par_loc = next(pars_iter)
        for q in range(N):    
            qml.RZ(par_loc, wires=q)
        par_loc = next(pars_iter)
        for q in range(N - 2):    
            qml.PauliRot(par_loc, "ZXZ", wires=[q, q+1, q+2])
        if N > 3:
            qml.PauliRot(par_loc, "ZZX", wires=[0, N-2, N-1])
            qml.PauliRot(par_loc, "XZZ", wires=[0, 1, N-1])
    return qml.state()


@qml.qnode(qml.device("lightning.qubit"), interface='jax')
def hva_ising(pars, sv, L, N):
    qml.StatePrep(sv, wires=numpy.arange(N))
    pars_iter = iter(pars)
    for l in range(L):
        par_loc = next(pars_iter)
        for q in range(N):    
            qml.RX(par_loc, wires=q)
        par_loc = next(pars_iter)
        for q in range(N):    
            qml.MultiRZ(par_loc, wires=[q, q+1])
        if N > 3:
            qml.MultiRZ(par_loc, wires=[0, N-1])
    return qml.state()
    
    
### Hamiltonians ###

def ising_ham(n_qubits, h, J=1, bc="closed"):
    d = 2**n_qubits
    Hx = zeros((d, d), dtype=complex)
    for q in range(n_qubits):
        X_op = [I]*q + [X] + [I]*(n_qubits-q-1)
        Hx = Hx + reduce(kron, X_op)
    Hzz = zeros((d, d), dtype=complex)
    for q in range(n_qubits-1):
        Hzz = Hzz + reduce(kron, [I]*q + [Z, Z] + [I]*(n_qubits-q-2))
    if bc == "closed" and n_qubits > 2:
        Hzz = Hzz + reduce(kron, [Z] + [I]*(n_qubits-2) + [Z])
    if n_qubits == 1: # lame
        Hzz = 1*Z
    return -J*(Hzz + h*Hx)

def ising_ham_sparse(N, hx, J=1, closed=True):
    d = 2**N
    Hzz = sparse.csr_array(numpy.zeros([d, d], dtype=complex))
    for j in range(N - 1):
        Hzz += reduce(sparse.kron, [I_sp]*j + [Z_sp, Z_sp] + [I_sp]*(N - j - 2))
    Hx = sparse.csr_array(numpy.zeros([d, d], dtype=complex))
    for j in range(N):
        Hx += reduce(sparse.kron, [I_sp]*j + [X_sp] + [I_sp]*(N - j - 1))
    if closed == True:
        Hzz += reduce(sparse.kron, [Z_sp] + [I_sp]*(N - 2) + [Z_sp])
    return J*Hzz + hx*Hx

def schwinger_ham(N, m, w=1, g=1, e0=0):
    d = 2**N
    sp = (X + 1j*Y)/2
    sm = (X - 1j*Y)/2
    term_1 = zeros([d, d], dtype=complex)
    for j in range(N - 1):
        op = reduce(kron, [I]*j + [sp, sm] + [I]*(N - j - 2)) # optimizable
        term_1 += op + op.conj().T
    term_2 = zeros([d, d], dtype=complex)
    for j in range(N):
        op = reduce(kron, [I]*j + [Z] + [I]*(N - j - 1))
        term_2 += (-1)**(j + 1)*op
    term_3 = zeros([d, d], dtype=complex)
    for j in range(N):
        L_j = zeros([d, d], dtype=complex)
        for l in range(j + 1):
            op = Z + (-1)**(l + 1)*I
            op = reduce(kron, [I]*l + [op] + [I]*(N - l - 1))
            L_j += op
        L_j = e0 - L_j/2
        term_3 += L_j@L_j
    return w*term_1 + m/2*term_2 + g*term_3

def schwinger_ham_sparse(N, m, w=1, g=1, e0=0):
    d = 2**N
    sp = (X_sp + 1j*Y_sp)/2
    sm = (X_sp - 1j*Y_sp)/2
    term_1 = sparse.csr_array(numpy.zeros([d, d], dtype=complex))
    for j in range(N - 1):
        op = reduce(sparse.kron, [I_sp]*j + [sp, sm] + [I_sp]*(N - j - 2))
        term_1 += op + op.conj().T
    term_2 = sparse.csr_array(numpy.zeros([d, d], dtype=complex))
    for j in range(N):
        op = reduce(sparse.kron, [I_sp]*j + [Z_sp] + [I_sp]*(N - j - 1))
        term_2 += (-1)**(j + 1)*op
    term_3 = sparse.csr_array(numpy.zeros([d, d], dtype=complex))
    for j in range(N):
        L_j = sparse.csr_array(numpy.zeros([d, d], dtype=complex))
        for l in range(j + 1):
            op = Z_sp + (-1)**(l + 1)*I_sp
            op = reduce(sparse.kron, [I_sp]*l + [op] + [I_sp]*(N - l - 1))
            L_j += op
        L_j = e0 - L_j/2
        term_3 += L_j@L_j
    return w*term_1 + m/2*term_2 + g*term_3

def transverse_cluster_ham_sparse(N, hx, hz=1e-2, J=1, closed=True):
    d = 2**N
    Hzxz = sparse.csr_array(jnp.zeros([d, d], dtype=complex))
    for j in range(N - 2):
        Hzxz += reduce(sparse.kron, [I_sp]*j + [Z_sp, X_sp, Z_sp] + [I_sp]*(N - j - 3))
    Hx = sparse.csr_array(jnp.zeros([d, d], dtype=complex))
    for j in range(N):
        Hx += reduce(sparse.kron, [I_sp]*j + [X_sp] + [I_sp]*(N - j - 1))
    Hz = sparse.csr_array(jnp.zeros([d, d], dtype=complex))
    for j in range(N):
        Hz += reduce(sparse.kron, [I_sp]*j + [Z_sp] + [I_sp]*(N - j - 1))
    if closed == True:
        Hzxz += reduce(sparse.kron, [Z_sp] + [I_sp]*(N - 3) + [Z_sp, X_sp])
        Hzxz += reduce(sparse.kron, [X_sp, Z_sp] + [I_sp]*(N - 3) + [Z_sp])
    return -J*Hzxz - hx*Hx - hz*Hz

def transverse_cluster_ham_reparametrized_sparse(N, x, eps=1e-2, closed=True):
    d = 2**N
    Hzxz = sparse.csr_array(jnp.zeros([d, d], dtype=complex))
    for j in range(N - 2):
        Hzxz += reduce(sparse.kron, [I_sp]*j + [Z_sp, X_sp, Z_sp] + [I_sp]*(N - j - 3))
    Hx = sparse.csr_array(jnp.zeros([d, d], dtype=complex))
    for j in range(N):
        Hx += reduce(sparse.kron, [I_sp]*j + [X_sp] + [I_sp]*(N - j - 1))
    Hz = sparse.csr_array(jnp.zeros([d, d], dtype=complex))
    for j in range(N):
        Hz += reduce(sparse.kron, [I_sp]*j + [Z_sp] + [I_sp]*(N - j - 1))
    if closed == True:
        Hzxz += reduce(sparse.kron, [Z_sp] + [I_sp]*(N - 3) + [Z_sp, X_sp])
        Hzxz += reduce(sparse.kron, [X_sp, Z_sp] + [I_sp]*(N - 3) + [Z_sp])
    return -jnp.cos(jnp.pi*x/2)*Hzxz - jnp.sin(jnp.pi*x/2)*Hx - eps*Hz

    
### auxiliary functions ###

def aux_info(ham_func, ham_pars, ham_args, ansatz_func, ansatz_args, n_inp, pars, n_meas_list, n_copies=1, dp=1e-5):
    """
        ham_func --- function for generating the Hamiltonian; the first argument is the number of qubits n_inp, the second is the parameter (i.e., transverse field in the Ising Hamiltonian)
        ham_pars --- list of the Hamiltonian parameters (i.e., list of the transverse field values)
        ham_args --- list of the rest Hamiltonain arguments (can be empty, [])
        n_inp --- number of qubits
        pars --- parameters of the ansatz
        Returns: expectations expecs and their derivatives expecs_der, variances disps, classical and quantum Fisher informations CFIs and QFIs
    """
    
    n_tot = n_inp*n_copies
    d = 2**n_tot
    
    pars_ans = pars[:-2**len(n_meas_list)]
    pars_est = pars[-2**len(n_meas_list):]

    basis = product([jnp.diag(jnp.array([1, 0], dtype="complex128")), jnp.diag(jnp.array([0, 1], dtype="complex128"))], repeat=len(n_meas_list))
    projs = []
    for line in basis:
        proj = [jnp.diag(jnp.array([1, 1], dtype="complex128"))]*n_tot
        for q in range(len(n_meas_list)):
            proj[n_meas_list[q]] = line[q]
        proj = reduce(jnp.kron, proj)
        projs.append(proj)
    projs = jnp.array(projs)
        
    v0s = []
    v0s_der = []
    v0s_der_norms = []
    QFIs = []
    U_v0s = []
    U_v0s_der = []
    expecs = []
    expecs_der = []
    disps = []
    CFIs = []
    for j in range(len(ham_pars)): 
        print("\t\tInfering for j:", j, end="\r")
        
        p = ham_pars[j]
        ham = ham_func(n_inp, p, *ham_args)
        ham_p = ham_func(n_inp, p+dp, *ham_args)
        ham_m = ham_func(n_inp, p-dp, *ham_args)
        ham_der = (ham_p - ham_m)/(2*dp)
        
        esys = sparse.linalg.eigsh(ham, k=1, which="SA")
        e0 = esys[0][0]
        v0 = esys[1].reshape(-1)
        e0_der = (v0.conj().T@ham_der@v0).real # Hellmann-Feynman theorem

        mat = ham - e0*sparse.eye(2**n_inp)
        mat_der = ham_der - e0_der*sparse.eye(2**n_inp) # in literature, the second term is absent for some reason

        v0_der = -sparse.linalg.lsqr(mat, mat_der@v0, atol=1e-09, btol=1e-09)[0]
        
        """ checks; should be zeros """
        # print(norm(mat_der@v0 + mat@v0_der))
        # print(norm((ham_der@v0 + ham@v0_der) - (e0_der*v0 + e0*v0_der)))
        # print()
    
        v0_der = sum([reduce(jnp.kron, [v0]*c + [v0_der] + [v0]*(n_copies - c - 1)) for c in range(n_copies)]) # memory-inefficient, but is a fancy one-liner
        v0 = reduce(jnp.kron, [v0]*n_copies)

        QFI = 4*(jnp.vdot(v0_der, v0_der) - jnp.vdot(v0_der, v0)*jnp.vdot(v0, v0_der)).real + 1e-10
        
        U_v0 = ansatz_func(pars_ans, v0, *ansatz_args)
        U_v0_der = ansatz_func(pars_ans, v0_der/norm(v0_der), *ansatz_args)

        probs = [(U_v0.conj().T@proj@U_v0).real for proj in projs]
        probs_der = [norm(v0_der)*2*(U_v0_der.conj().T@proj@U_v0).real for proj in projs]
        expec = sum([ev*prob for ev, prob in zip(pars_est, probs)])
        expec_der = sum([ev*prob_der for ev, prob_der in zip(pars_est, probs_der)])
        disp = sum([ev**2*prob for ev, prob in zip(pars_est, probs)]) - expec**2
        CFI = sum([prob_der**2/prob if prob > 0 else 0 for prob_der, prob in zip(probs_der, probs)]) + 1e-10
        
        expecs.append(expec)
        expecs_der.append(expec_der)
        disps.append(disp)
        CFIs.append(CFI)
        QFIs.append(QFI)

    print()
    
    return numpy.array(expecs), numpy.array(expecs_der), numpy.array(disps), numpy.array(CFIs), numpy.array(QFIs)