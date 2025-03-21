import numpy as np
from numpy import outer, trace, dot, vdot, pi, log2, exp, sin, cos, sqrt, sign, diag, linspace, arange, array, inf, zeros, eye, arccos, arcsin, arctan, mean, std, concatenate, kron, sign, ceil, log, unique
from numpy.random import uniform, normal, randint, choice
from scipy.linalg import eig, eigh, norm, expm, sqrtm
from numpy.linalg import svd, norm, inv, pinv
from scipy.stats import sem
from scipy.optimize import minimize
from functools import reduce, partial
from itertools import product
import qutip as qp
from multiprocessing import Pool
from time import time

P0 = np.array([[1., 0.],
               [0., 0.]]) # |0><0|
P1 = np.array([[0., 0.],
               [0., 1.]]) # |1><1|
X = np.array([[0.,1.],
              [1.,0.]]) # X Pauli matrix
Y = np.array([[0.,-1.j],
              [1.j, 0.]]) # Y Pauli matrix
Z = np.array([[1., 0.],
              [0.,-1.]]) # Z Pauli matrix
I = np.array([[1.,0.],
              [0.,1.]]) # 2x2 identity matrix

# some functions # 

def kron_A_N(A, N): # fast kron(A, eye(N))
    m,n = A.shape
    out = zeros((m, N, n, N), dtype=A.dtype)
    r = arange(N)
    out[:, r, :, r] = A
    out.shape = (m*N, n*N)
    return out
    
def kron_N_A(A, N): # fast kron(eye(N), A)
    m,n = A.shape
    out = zeros((N, m, N, n), dtype=A.dtype)
    r = np.arange(N)
    out[r, :, r, :] = A
    out.shape = (m*N, n*N)
    return out

def kron_A_I_diag(A, N):
    m = len(A)
    out = zeros((N, m), dtype=A.dtype)
    out[arange(N)] = A
    out = out.T.reshape(m*N)
    return out

def kron_I_A_diag(A, N):
    m = len(A)
    out = zeros((N, m), dtype=A.dtype)
    out[arange(N)] = A
    out = out.reshape(m*N)
    return out

def trace_distance(A, B):
    sub = A - B
    return trace(sqrtm(dot(sub.conj().T, sub))).real/2

def fidelity(A, B):
    res = reduce(dot, [sqrtm(A), B, sqrtm(A)])
    res = sqrtm(res)
    return trace(res).real**2

def sup_fidelity(A, B):
    """ An upper bound for the usual fidelity. """
    t1 = trace(A@B).real
    t2 = max(0, 1 - trace(A@A).real)
    t3 = max(0, 1 - trace(B@B).real)
    return t1 + sqrt(t2)*sqrt(t3)

def partial_trace(dm, m=None, n=None, subsystem=0):
    """ Simple and fast, but cuts only in halves. """
    if (m is None) or (n is None): # cut in equal halves
        N = log2(len(dm))
        m = int(N / 2)
        n = int(N - m)
        m = 2**m
        n = 2**n
    if subsystem == 0:
        return trace(dm.reshape((m, n, m, n)), axis1=0, axis2=2)
    elif subsystem == 1:
        return trace(dm.reshape((m, n, m, n)), axis1=1, axis2=3)

def concurrence_pure(dm):
    dm_red = partial_trace(dm)
    return sqrt(2*(1 - trace(dm_red@dm_red).real))

def concurrence(dm):
    YY = kron(Y, Y)
    dm_t = YY@dm.conj()@YY
    R = dm_t@dm
    lambdas = [l if l > 0 else 0 for l in np.sort(eig(R)[0].real)]
    c = sqrt(lambdas[3]) - sqrt(lambdas[2]) - sqrt(lambdas[1]) - sqrt(lambdas[0])
    return max(0, c)

def two_subsys_negativity(dm):
    def partial_transpose(A, n, m):
        A_c = array(A)
        Bt = A[:n, m:].copy()
        Ct = A[n:, :m].copy()
        A_c[:n, m:] = Ct
        A_c[n:, :m] = Bt
        return A_c
    dm_ptrans = partial_transpose(dm, int(len(dm)/2), int(len(dm)/2))
    lambda_min = eigh(dm_ptrans)[0][0]
    return 2*abs(min(0, lambda_min))

def prev_to_next_ansatz(pars, n_tot_p, n_meas_p, n_layers_p, n_tot_n, n_meas_n, n_layers_n, subsval=0):
    """Extends the outcome values and the ansatz hea_cry_rzrx to new n_tot and n_layers, filling the angles with zeros."""
    x0 = []
    it = iter(pars)
    for q in range(n_tot_p):
        x0.append(next(it))
        x0.append(next(it))
    for q in range(n_tot_n - n_tot_p):
        x0.append(subsval)
        x0.append(subsval)   
    for l in range(n_layers_p):
        for q in range(n_tot_p - 1):
            x0.append(next(it))
        for q in range(n_tot_p - 1, n_tot_n - 1):
            x0.append(subsval)
        for q in range(n_tot_p):
            x0.append(next(it))
            x0.append(next(it))
        for q in range(n_tot_p, n_tot_n):
            x0.append(subsval)
            x0.append(subsval)
    for l in range(n_layers_n - n_layers_p):
        for q in range(n_tot_n - 1):
            x0.append(subsval)
        for q in range(n_tot_n):
            x0.append(subsval)
            x0.append(subsval) 
    x0 = x0 + list(kron(diag([next(it) for i in range(2**n_meas_p)]), eye(2**(n_meas_n - n_meas_p))).diagonal())
    return x0


def gen_even_ent_data(n, n_inp=2, mixed=True, marks="neg", n_chunks=100, eps=0):
    """ Generates a data set of states with evenly distributed entanglements. """
    
    d = 2**n_inp
    
    if marks == "neg":
        ent_measure_func = two_subsys_negativity
    elif marks == "con":
        ent_measure_func = concurrence 
    
    ent_count_max = int(ceil(n/n_chunks))
    ent_line = linspace(0, 1, n_chunks + 1)[1:]
    ent_counts = [0]*n_chunks
    count = 0
    states = []
    labels = []
    while count < n:
        print("%d" %count, end="\r")
        if mixed == True:
            state = (qp.rand_dm(d, distribution='ginibre', rank=randint(1, d + 1))).full() # lame, but works faster for mixed states
            ent = ent_measure_func(state)
        else:
            state = (qp.rand_ket(d)).full().reshape(-1)
            ent = ent_measure_func(outer(state, state.conj().T))
        if ent > eps:
            ent_diffs = ent_line - ent
            ind = np.abs(ent_diffs).argmin()
            if sign(ent_diffs[ind]) == -1:
                ind += 1
            if ent_counts[ind] < ent_count_max:
                ent_counts[ind] += 1        
                count += 1
                states.append(state)
                labels.append(ent)
            
    return states, labels


# quantum state generators # 

def rand_sv(n_qubits):
    """ Generates a random pure state as a vector. """
    d = 2**n_qubits
    sv = uniform(-1, 1, d) + 1j*uniform(-1, 1, d)
    return sv/norm(sv)

def rand_dm(n_qubits):
    """ Generates a random mixed state as a full-rank density matrix. """
    d = 2**n_qubits
    H = uniform(-1, 1, [d, d]) + 1j*uniform(-1, 1, [d, d])
    dm = H@H.conj().T
    dm = dm/trace(dm).real
    return dm


# Fisher informations #

def cfi(channel_func, dm, p, channel_args, povm, n_copies=1, n_ext=0, dp=1e-5):
    """ Computes classical Fisher information. Only for channels! """
    dm_n = reduce(kron, [channel_func(dm, p, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_p = reduce(kron, [channel_func(dm, p+dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_m = reduce(kron, [channel_func(dm, p-dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    fi = 0
    for op in povm:
        prob = trace(dot(dm_n, op)).real
        if prob > 0:
            prob_p = trace(dot(dm_n_p, op)).real
            prob_m = trace(dot(dm_n_m, op)).real
            der = (prob_p - prob_m)/(2*dp)
            fi += der**2/prob
    return fi

def qfi(channel_func, dm, p, channel_args, n_copies=1, n_ext=0, dp=1e-2):
    """ Computes quantum Fisher information. Only for channels! """
    dm_n = reduce(kron, [channel_func(dm, p, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_p = reduce(kron, [channel_func(dm, p+dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    fi = 8*(1 - sqrt(fidelity(dm_n, dm_n_p))) / dp**2
    return fi

def qfi_central(channel_func, dm, p, channel_args, n_copies=1, n_ext=0, dp=1e-2):
    """ Computes quantum Fisher information via "central differences". Only for channels! """
    dm_n_p = reduce(kron, [channel_func(dm, p+dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_m = reduce(kron, [channel_func(dm, p-dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    fi = 8*(1 - sqrt(fidelity(dm_n_m, dm_n_p))) / dp**2/4
    return fi

def sup_qfi(channel_func, dm, p, channel_args, n_copies=1, n_ext=0, dp=1e-5):
    """ Computes an upper bound (?) for quantum Fisher information. Only for channels! """
    dm_n = reduce(kron, [channel_func(dm, p, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_p = reduce(kron, [channel_func(dm, p+dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    fi = 8*(1 - sqrt(sup_fidelity(dm_n, dm_n_p))) / dp**2
    return fi


def sld(channel_func, channel_par, channel_args, dm_ini, n_copies=1, n_ext=0, dp=1e-5, return_fi=False):
    """
        Numerically finds the SLD operator L.
        Optionally returns the classical and quantum Fisher informations.
        Only for channels!
    """
    
    n_inp = int(log2(len(dm_ini)))*n_copies
    n_tot = n_inp + n_ext
    d = 2**n_tot
        
    dm_ext = diag([1] + [0]*(2**(n_ext) - 1))
    dm_n = reduce(kron, [channel_func(dm_ini, channel_par, *channel_args)]*n_copies + [dm_ext])
    dm_n_p = reduce(kron, [channel_func(dm_ini, channel_par+dp, *channel_args)]*n_copies + [dm_ext])
    dm_n_m = reduce(kron, [channel_func(dm_ini, channel_par-dp, *channel_args)]*n_copies + [dm_ext])
    dm_n_der = (dm_n_p - dm_n_m)/(2*dp)
    
    evals, evecs = eigh(dm_n)
    evecs = evecs.T
    
    L = zeros([d, d], dtype=complex)
    for i in range(d):
        for j in range(d):
            denom = evals[i] + evals[j]
            if denom > 1e-5:
                numer = evecs[i].conj().T@dm_n_der@evecs[j]
                oper = outer(evecs[i], evecs[j].conj().T)
                L += 2*numer/denom*oper
    # print("Incompliance with the definition of SLD:", norm( (L@dm_n + dm_n@L)/2 - dm_n_der ))
    
    if return_fi == True:
        evecs_L = eigh(L)[1].T
        projs_L = [outer(vec, vec.conj().T) for vec in evecs_L]
        CFI = cfi(channel_func, dm_ini, channel_par, channel_args, projs_L, n_copies=n_copies, n_ext=n_ext, dp=dp)
        QFI = trace(L@L@dm_n).real
        return L, CFI, QFI
    else:
        return L
        

### channels ###

def hw_channel(dm, p):
    """ Holevo-Werner channel """
    d = len(dm)
    return ((d - p)*eye(d) + (d*p - 1)*dm.T)/(d**2 - 1)
    
def depolarizing_channel(dm, p):
    d = len(dm)
    return (1 - p)*dm + p/d*eye(d)

def generalized_amplitude_damping_channel(dm, g, N, target_qubit):
    
    n_qubits = int(log2(len(dm)))
    dl = 2**target_qubit
    dr = 2**(n_qubits - target_qubit - 1)
    
    K1 = array([[1,           0],
                [0, sqrt(1 - g)]])*sqrt(1 - N)
    K1 = reduce(kron, [eye(dl), K1, eye(dr)]) # inefficient
    
    K2 = array([[0, sqrt(g*(1 - N))],
                [0,               0]])
    K2 = reduce(kron, [eye(dl), K2, eye(dr)])
    
    K3 = array([[sqrt(1 - g), 0],
                [          0, 1]])*sqrt(N)
    K3 = reduce(kron, [eye(dl), K3, eye(dr)])
            
    K4 = array([[0,         0],
                [sqrt(g*N), 0]])
    K4 = reduce(kron, [eye(dl), K4, eye(dr)])
    
    dm1 = reduce(dot, [K1, dm, K1.conj().T])
    dm2 = reduce(dot, [K2, dm, K2.conj().T])
    dm3 = reduce(dot, [K3, dm, K3.conj().T])
    dm4 = reduce(dot, [K4, dm, K4.conj().T])
        
    return dm1 + dm2 + dm3 + dm4


def another_generalized_amplitude_damping_channel(dm, g, N, target_qubit):
    """ Adapted from https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.012317 """
    
    n_qubits = int(log2(len(dm)))
    dl = 2**target_qubit
    dr = 2**(n_qubits - target_qubit - 1)
    
    K1 = array([[1,           0],
                [0,     sqrt(g)]])*sqrt(N)
    K1 = reduce(kron, [eye(dl), K1, eye(dr)]) # inefficient
    
    K2 = array([[0, sqrt((1 - g))],
                [0,             0]])*sqrt(N)
    K2 = reduce(kron, [eye(dl), K2, eye(dr)])
    
    K3 = array([[sqrt(g), 0],
                [      0, 1]])*sqrt(1 - N)
    K3 = reduce(kron, [eye(dl), K3, eye(dr)])
            
    K4 = array([[0,           0],
                [sqrt(1 - g), 0]])*sqrt(1 - N)
    K4 = reduce(kron, [eye(dl), K4, eye(dr)])
    
    dm1 = reduce(dot, [K1, dm, K1.conj().T])
    dm2 = reduce(dot, [K2, dm, K2.conj().T])
    dm3 = reduce(dot, [K3, dm, K3.conj().T])
    dm4 = reduce(dot, [K4, dm, K4.conj().T])
        
    return dm1 + dm2 + dm3 + dm4

    
def X_rotations(dm, p):
    n_qubits = int(log2(len(dm)))
    op = reduce(kron, [X]*n_qubits)
    U = expm(-1j*p*op)
    return reduce(dot, [U, dm, U.conj().T])

def Y_rotations(dm, p):
    n_qubits = int(log2(len(dm)))
    op = reduce(kron, [Y]*n_qubits)
    U = expm(-1j*p*op)
    return reduce(dot, [U, dm, U.conj().T])

def Z_rotations(dm, p):
    n_qubits = int(log2(len(dm)))
    op = reduce(kron, [Z]*n_qubits)
    U = expm(-1j*p*op)
    return reduce(dot, [U, dm, U.conj().T])

def z_rot(dm, p, target_qubit):
    """
        z-rotation of the specified qubit.
        Mind the division by two!
    """
    n_qubits = int(log2(len(dm)))
    dl = 2**target_qubit
    dr = 2**(n_qubits - target_qubit - 1)
    U = reduce(kron, [eye(dl), expm(-1j*p/2*Z), eye(dr)]) # inefficient
    return U@dm@U.conj().T


def random_channel(dm, p, pars, p_index, pauli_basis=None):
    """
        Random single-parametrized channel.
        Attaches to a given n-qubit state dm a 2n-qubit pure state,
        applies to the joint state a unitary with (4^(3n) - 1) parameters pars, one of which is the parameter p in question with the index p_index.        
    """
    d = len(dm)
    n_inp = int(log2(d))
    n_ext = 2*n_inp
    n_tot = n_inp + n_ext
    pars_conc = concatenate([pars[:p_index], [p], + pars[p_index:]])
    V = su2n(n_tot, pars_conc, pauli_basis=pauli_basis)
    dm_ext = diag([1] + [0]*(2**n_ext - 1))
    dm_n = V@kron(dm, dm_ext)@V.conj().T
    dm_n = trace(dm_n.reshape(2**n_inp, 2**n_ext, 2**n_inp, 2**n_ext), axis1=1, axis2=3) # partial trace with respect to the extension
    return dm_n


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


def schwinger_ham(N, m, w=1, g=1, e0=0):
    d = 2**N
    sp = (X + 1j*Y)/2
    sm = (X - 1j*Y)/2
    term_1 = zeros([d, d], dtype=complex)
    for j in range(N - 1):
        op = reduce(kron, [I]*j + [sp, sm] + [I]*(N - j - 2))
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


# ansatzes #

def su2(pars):
    """ Universal single-qubit rotation """
    return array([[exp(1j * (-pars[1] - pars[2])) * cos(pars[0]), -exp(1j * (-pars[1] + pars[2])) * sin(pars[0])],
                  [exp(1j * ( pars[1] - pars[2])) * sin(pars[0]),  exp(1j * ( pars[1] + pars[2])) * cos(pars[0])]])

def su2n(n_qubits, pars, pauli_basis=None):
    if pauli_basis is None:
        pauli_basis = [reduce(kron, paulis_tring) for paulis_tring in list(product([I, X, Y, Z], repeat=n_qubits))[1:]]
    op = zeros([2**n_qubits, 2**n_qubits], dtype=complex)
    for par, pauli_string in zip(pars, pauli_basis):
        op += par*pauli_string
    return expm(-1j*op)

def xz_rot(pars):
    """ xz-rotation """
    cos_par_0 = cos(pars[0])
    sin_par_0 = sin(pars[0])
    exp_par_1 = exp(1j * pars[1])
    return np.array([[exp_par_1.conjugate() * cos_par_0, -1j*exp_par_1.conjugate() * sin_par_0],
                     [        -1j*exp_par_1 * sin_par_0,                 exp_par_1 * cos_par_0]])

def cr_y(n_qubits, q1, q2, par):
    """ Controlled y-rotation """
    cry_1 = kron_N_A(P0, 2**q1)
    cry_1 = kron_A_N(cry_1, 2**(n_qubits - q1 -1))
    cos_par = cos(par)
    sin_par = sin(par)
    ry = array([[cos_par, -sin_par], [sin_par, cos_par]])
    if q2 > q1:
        op_l = kron_N_A(P1, 2**q1)
        op_l = kron_A_N(op_l, 2**(q2 - q1 - 1))
        op_r = kron_A_N(ry, 2**(n_qubits - q2 - 1))
        cry_2 = kron(op_l, op_r)
    else:
        op_l = kron_N_A(ry, 2**q2)
        op_l = kron_A_N(op_l, 2**(q1 - q2 - 1))
        op_r = kron_A_N(P1, 2**(n_qubits - q1 - 1))
        cry_2 = kron(op_l, op_r)
    return cry_1 + cry_2
    
def rxx(N, q1, q2, par):
    j, k = sorted([q1, q2])
    term_1 = diag(full(2**n_qubits, cos(par)))
    term_21 = kron_N_A(X, 2**j)
    term_21 = kron_A_N(term_21, 2**(k - j - 1))
    term_22 = kron_A_N(X, 2**(n_qubits - k - 1))
    term_2 = kron(term_21, term_22)
    term_2 = -1j*sin(par)*term_2
    return term_1 + term_2

def cx(n_qubits, q1, q2):
    """ CX gate """
    cx_1 = kron_N_A(P0, 2**q1)
    cx_1 = kron_A_N(cx_1, 2**(n_qubits - q1 -1))
    if q2 > q1:
        op_l = kron_N_A(P1, 2**q1)
        op_l = kron_A_N(op_l, 2**(q2 - q1 - 1))
        op_r = kron_A_N(X, 2**(n_qubits - q2 - 1))
        cx_2 = kron(op_l, op_r)
    else:
        op_l = kron_N_A(X, 2**q2)
        op_l = kron_A_N(op_l, 2**(q1 - q2 - 1))
        op_r = kron_A_N(P1, 2**(n_qubits - q1 - 1))
        cx_2 = kron(op_l, op_r)
    return cx_1 + cx_2

def cx_cascade(n_qubits):
    if n_qubits == 1:
        return eye(2)
    else:
        op_l = []
        for q in range(n_qubits - 1):
            op_l.append(cx(n_qubits, q, q + 1))
        if n_qubits > 2:
            op_l.append(cx(n_qubits, n_qubits - 1, 0))
        return reduce(dot, op_l)

def hea_cx_rzrx(n_qubits, n_layers, CXq, pars):
    """ Requires pre-computed chain of CX-gates, CXq=cx_cascade(n_qubits) """
    it = iter(pars)
    op = reduce(kron, [xz_rot([next(it), next(it)]) for q in range(n_qubits)])
    for l in range(n_layers):
        op = CXq@op
        op = reduce(kron, [xz_rot([next(it), next(it)]) for q in range(n_qubits)])@op        
    return op

def hea_cry_rzrx(n_qubits, n_layers, pars):
    it = iter(pars)
    op = reduce(kron, [xz_rot([next(it), next(it)]) for q in range(n_qubits)])
    for l in range(n_layers):
        for q in range(n_qubits - 1):
            op = cr_y(n_qubits, q, q+1, next(it))@op
        op = reduce(kron, [xz_rot([next(it), next(it)]) for q in range(n_qubits)])@op
    return op

def hea_cry_rzrx_qiskit(n_qubits, n_layers, pars):
    it = iter(pars)
    circ = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        circ.rx(next(it), q)
        circ.rz(next(it), q)
    for l in range(n_layers):
        for q in range(n_qubits - 1):
            circ.cry(next(it), q, q + 1)
        for q in range(n_qubits):
            circ.rx(next(it), q)
            circ.rz(next(it), q)
    return circ


# measure #    

def measure_z_counts(dm, n_shots):
    probs = dm.diagonal().real
    d = len(probs)
    measurements = choice(arange(d), size=n_shots, p=probs)
    measurements = unique(measurements, return_counts=True)
    counts = zeros(d)
    counts[measurements[0]] = measurements[1]
    return counts

def measure_povm_counts(dm, n_shots, povm):
    probs = [trace(dm@el).real for el in povm]
    d = len(probs)
    measurements = choice(arange(d), size=n_shots, p=probs)
    measurements = unique(measurements, return_counts=True)
    counts = zeros(d)
    counts[measurements[0]] = measurements[1]
    return counts


# estimate #

def estimate(dms, labels, n_layers, pars, n_copies=1, n_meas=0, n_shots=1000, n_est=1):
        
    n_inp = int(log2(len(dms[0])))
    n_tot = n_inp*n_copies
    d = 2**n_tot

    pars_ans = pars[:-2**n_meas]
    pars_est = pars[-2**n_meas:]

    ansatz = hea_cry_rzrx(n_tot, n_layers, pars_ans)
    projs = [reduce(kron, [diag(line), eye(2**(n_tot - n_meas))]) for line in eye(2**n_meas)]
    projs_u = [ansatz.conj().T@proj@ansatz for proj in projs]

    dms_cop = [reduce(kron, [dm]*n_copies) for dm in dms]
    
    preds = []
    errors = []
    for e in range(n_est):
        print("Estimation run: %d" %e, end="\r")
        preds_e = []
        errors_e = []
        for j in range(len(dms)):
            pred_j = measure_povm(dms_cop[j], projs_u, pars_est, n_shots)
            error_j = (pred_j - labels[j])**2
            preds_e.append(pred_j)
            errors_e.append(error_j)
        preds.append(array(preds_e))
        errors.append(array(errors_e))
        
    return array(preds), array(errors)


def estimate_pure(svs, labels, n_layers, pars, n_copies=1, n_meas=0, n_shots=1000, n_est=1):
        
    n_inp = int(log2(len(svs[0])))
    n_tot = n_inp*n_copies
    d = 2**n_tot
    
    if n_meas == 0:
        n_meas = n_tot
    
    pars_ans = pars[:-2**n_meas]
    pars_est = pars[-2**n_meas:]

    ansatz = hea_cry_rzrx(n_tot, n_layers, pars_ans)
    projs = [reduce(kron, [diag(line), eye(2**(n_tot - n_meas))]) for line in eye(2**n_meas)]
    projs_u = [ansatz.conj().T@proj@ansatz for proj in projs]
        
    dms_cop = []
    for sv in svs:
        dm = outer(sv, sv.conj().T)
        dms_cop.append(reduce(kron, [dm]*n_copies))
        
    preds = []
    errors = []
    for e in range(n_est):
        print("Estimation run: %d" %e, end="\r")
        preds_e = []
        errors_e = []
        for j in range(len(ham_pars)):
            pred_j = measure_povm(dms[j], projs_u, pars_est, n_shots)
            error_j = (array(pred_j) - array(ham_pars[j]))**2
            preds_e.append(pred_j)
            errors_e.append(error_j)
        preds.append(array(preds_e))
        errors.append(array(errors_e))
        
    return array(preds), array(errors)


def estimate_naimark(dms, labels, n_layers, pars, n_copies=1, n_ext=1, n_shots=1000, n_est=1):

    n_inp = int(log2(len(dms[0])))
    n_tot = n_inp*n_copies + n_ext
    d = 2**n_tot
    
    pars_ans = pars[:-2**n_ext]
    pars_est = pars[-2**n_ext:]

    ansatz = hea_cry_rzrx(n_tot, n_layers, pars_ans)
    projs = [reduce(kron, [eye(2**n_inp), diag(line)]) for line in eye(2**n_ext)]
    projs_u = [ansatz.conj().T@proj@ansatz for proj in projs]
    
    dms_cop = [reduce(kron, [dm]*n_copies + [P0]*n_ext) for dm in dms]
    
    preds = []
    errors = []
    for e in range(n_est):
        print("Estimation run: %d" %e, end="\r")
        preds_e = []
        errors_e = []
        for j in range(len(dms)):
            pred_j = measure_povm(dms_cop[j], projs_u, pars_est, n_shots)
            error_j = (pred_j - labels[j])**2
            preds_e.append(pred_j)
            errors_e.append(error_j)
        preds.append(array(preds_e))
        errors.append(array(errors_e))
        
    return array(preds), array(errors)

    
### train ###

def train(dms, labels, n_layers, n_copies=1, n_meas=0, method="BFGS", w_ls=1e0, w_var=1e-4, x0=None, options={}, save_data=False, file_name=None, fvals=[]):

    n_inp = int(log2(len(dms[0])))
    n_tot = n_inp*n_copies
    d = 2**n_tot
        
    if n_meas == 0:
        n_meas = n_tot
                
    n_pars_est = 2**n_meas
    # n_pars_ans = (3*n_layers + 2)*n_tot # cry  
    n_pars_ans = 2*n_tot + (3*n_tot - 1)*n_layers
    
    d_diff = 2**(n_tot - n_meas)
    
    dms_cop = [reduce(kron, [dm]*n_copies) for dm in dms]

    fval_cont = [0]
    def fun(x):
        x_ans, x_est = x[:n_pars_ans], x[n_pars_ans:]
        U = hea_cry_rzrx(n_tot, n_layers, x_ans)
        H_u = (U.conj().T*kron_A_I_diag(x_est, d_diff))@U
        H_u_sq = H_u@H_u
        expecs = []
        disps = []
        for dm in dms_cop:
            expecs.append(trace(dm@H_u).real)
            disps.append(trace(dm@H_u_sq).real)
        f = w_ls*np.sum((array(expecs) - array(labels))**2)
        f += w_var*np.sum(array(disps) - array(expecs)**2)
        fval_cont[0] = f
        return f
    
    if file_name is None:
        file_name = "pars-c=%d-m=%d=l=%d-w_ls=%f-w_var=%f-n_train=%d" %(n_copies, n_meas, n_layers, w_ls, w_var, len(labels))

    time_loc = time()
    
    def callback(x):
        fvals.append(fval_cont[0])
        print("\t\t\tIteration: %d | Cost: %.8f | Time passed: %d s" %(len(fvals), fval_cont[0], time() - time_loc), end="\r")
        if save_data == True:
            np.save(file_name + "-pars", x)
            np.save(file_name + "-fvals", fvals)
        return None
    
    if method in ["Nelder-Mead", "L-BFGS-B", "SLSQP", "TNC", "Powell", "COBYLA"]:
        bounds = [(0, 2*pi)]*n_pars_ans + [(None,  None)]*n_pars_est
    else:
        bounds = None
    if x0 is None:
        x0_ans = normal(pi, 1, n_pars_ans)
        x0_est = normal(0, 1, n_pars_est)
        x0 = concatenate([x0_ans, x0_est])
    
    optimization_result = minimize(fun=fun, x0=x0, bounds=bounds, method=method, callback=callback, options=options) # "maxiter": int(1e10)

    return fvals, optimization_result


def train_pure(svs, labels, n_layers, n_copies=1, n_meas=0, method="BFGS", w_ls=1e0, w_var=1e-4, x0=None, options={}, save_data=False, file_name=None, fvals=[]):

    n_inp = int(log2(len(svs[0])))
    n_tot = n_inp*n_copies
    d = 2**n_tot
    if n_meas == 0:
        n_meas = n_tot
    d_diff = 2**(n_tot - n_meas)
    
    n_pars_est = 2**n_meas
    n_pars_ans = (3*n_tot - 1)*n_layers + 2*n_tot
    
    svs_cop = []
    for sv in svs:
        svs_cop.append(reduce(kron, [sv]*n_copies))

    fval_cont = [0]
    def fun(x):
        x_ans, x_est = x[:n_pars_ans], x[n_pars_ans:]
        U = hea_cry_rzrx(n_tot, n_layers, x_ans)
        H_u = (U.conj().T*kron_A_I_diag(x_est, d_diff))@U
        # H_u = (U.conj().T*kron(x_est, array([0]*(d_diff - 1) + [1])))@U
        expecs = []
        disps = []
        for j in range(len(labels)):
            H_u_v0 = H_u@svs_cop[j]
            expecs.append((svs_cop[j].conj().T@H_u_v0).real)
            disps.append((H_u_v0.conj().T@H_u_v0).real)
        f = w_ls*np.sum((array(expecs) - array(labels))**2)
        f += w_var*np.sum(array(disps) - array(expecs)**2)
        fval_cont[0] = f
        return f    

    if file_name is None:
        file_name = "pars-c=%d-m=%d=l=%d-w_ls=%f-w_var=%f-n_train=%d" %(n_copies, n_meas, n_layers, w_ls, w_var, len(labels))
    
    time_loc = time()
    
    def callback(x):
        fvals.append(fval_cont[0])
        print("\t\t\tIteration: %d | Cost: %.8f | Time passed: %d s" %(len(fvals), fval_cont[0], time() - time_loc), end="\r")
        if save_data == True:
            np.save(file_name, x)
        return None
    
    if method in ["Nelder-Mead", "L-BFGS-B", "SLSQP", "TNC", "Powell", "COBYLA"]:
        bounds = [(0, 2*pi)]*n_pars_ans + [(None, None)]*n_pars_est
    else:
        bounds = None
    if x0 is None:
        x0_ans = normal(pi/4, 0.1*pi, n_pars_ans)
        x0_est = normal(0, 1, n_pars_est)
        x0 = concatenate([x0_ans, x0_est])
    
    optimization_result = minimize(fun=fun, x0=x0, callback=callback, bounds=bounds, method=method, options=options)

    return fvals, optimization_result


def train_naimark(dms, labels, n_layers, n_copies=1, n_ext=1, method="BFGS", w_ls=1e0, w_var=1e-4, x0=None, options={}, save_data=False, file_name=None):

    n_inp = int(log2(len(dms[0])))
    n_tot = n_inp*n_copies + n_ext
    d_prim =  2**(n_inp*n_copies)

    n_pars_est = 2**n_ext
    n_pars_ans = (3*n_tot - 1)*n_layers + 2*n_tot # cry  
    
    # projs = [reduce(kron, [eye(2**n_inp), diag(line)]) for line in eye(n_pars_est)]
        
    dms_cop = [reduce(kron, [dm]*n_copies + [P0]*n_ext) for dm in dms]

    fvals = []
    fval_cont = [0]
    def fun(x):
        x_ans, x_est = x[:n_pars_ans], x[n_pars_ans:]
        ansatz = hea_cry_rzrx(n_tot,n_layers, x_ans)
        # obs_u = ansatz.conj().T@kron_A_N(diag(x_est), d_prim)@ansatz
        obs_u = (ansatz.conj().T*kron_A_I_diag(x_est, d_diff))@ansatz
        obs_u_sq = obs_u@obs_u
        expecs = []
        disps = []
        for dm in dms_cop:
            expecs.append(trace(dm@obs_u).real)
            disps.append(trace(dm@obs_u_sq).real)
        f = w_ls*np.sum((array(expecs) - array(labels))**2)
        f += w_var*np.sum(array(disps) - array(expecs)**2)
        fval_cont[0] = f
        return f
    
    if save_data == True and file_name is None:
        file_name = "pars-c=%d-e=%d=l=%d-w_ls=%f-w_var=%f-n_train=%d" %(n_copies, n_ext, n_layers, w_ls, w_var, len(labels))
    
    def callback(x):
        fvals.append(fval_cont[0])
        print("Iteration: %d | Function value: %.8f" %(len(fvals), fval_cont[0]), end="\r")
        if save_data == True:
            np.save(file_name, x)
        return None
    
    if method in ["Nelder-Mead", "L-BFGS-B", "SLSQP", "TNC", "Powell", "COBYLA"]:
        bounds = [(0, pi)]*n_pars_ans + [(-10, 10)]*n_pars_est
    else:
        bounds = None
    if x0 is None:
        x0_ans = normal(pi, 0.01*pi, n_pars_ans)
        x0_est = normal(0, 0.01, n_pars_est)
        x0 = concatenate([x0_ans, x0_est])
    
    optimization_result = minimize(fun=fun, x0=x0, bounds=bounds, method=method, callback=callback, options=options) # "maxiter": int(1e10)

    return fvals, optimization_result


def train_pure_naimark(svs, labels, n_layers, n_copies=1, n_ext=1, method="BFGS", w_ls=1e0, w_var=1e-4, x0=None, options={}, save_data=False, file_name=None):

    n_inp = int(log2(len(svs[0])))
    n_tot = n_inp*n_copies + n_ext
    d_prim =  2**(n_inp*n_copies)

    n_pars_est = 2**n_ext
    n_pars_ans = 3*(n_tot + 1)*n_layers 
    
    svs_cop = []
    for sv in svs:
        svs_cop.append(reduce(kron, [sv]*n_copies + [array([1, 0])]*n_ext))

    fvals = []
    fval_cont = [0]
    def fun(x):
        x_ans, x_est = x[:n_pars_ans], x[n_pars_ans:]
        ansatz = hea_cry_rzrx(n_tot, n_layers, x_ans)
        # H_u = ansatz.conj().T@kron_N_A(diag(x_est), d_prim)@ansatz
        H_u = (ansatz.conj().T*x_est)@ansatz
        expecs = []
        disps = []
        for j in range(len(labels)):
            H_u_v0 = H_u@svs_cop[j]
            expecs.append((svs_cop[j].conj().T@H_u_v0).real)
            disps.append((H_u_v0.conj().T@H_u_v0).real)
        f = w_ls*np.sum((array(expecs) - array(labels))**2)
        f += w_var*np.sum(array(disps) - array(expecs)**2)
        fval_cont[0] = f
        return f    

    if file_name is None:
        file_name = "pars-c=%d-e=%d=l=%d-w_ls=%f-w_var=%f-n_train=%d" %(n_copies, n_ext, n_layers, w_ls, w_var, len(labels))
    
    def callback(x):
        fvals.append(fval_cont[0])
        print("Iteration: %d | Function value: %.8f" %(len(fvals), fval_cont[0]), end="\r")
        if save_data == True:
            np.save(file_name, x)
        return None
    
    if method in ["Nelder-Mead", "L-BFGS-B", "SLSQP", "TNC", "Powell", "COBYLA"]:
        bounds = [(0, 2*pi)]*n_pars_ans + [(None, None)]*n_pars_est
    else:
        bounds = None
    if x0 is None:
        x0_ans = normal(pi/4, 0.1*pi, n_pars_ans)
        x0_est = normal(0, 1, n_pars_est)
        x0 = concatenate([x0_ans, x0_est])
    
    optimization_result = minimize(fun=fun, x0=x0, callback=callback, bounds=bounds, method=method, options=options)

    return fvals, optimization_result



### old or unused stuff ###

# def cx_old(n_qubits, q1, q2):z
#     """ Old and inefficient CX gate """
#     cx_1 = reduce(kron, [eye(2**q1), P0, np.eye(2**(n_qubits-q1-1))])
#     if q2 > q1:
#         cx_2 = [eye(2**q1), P1, eye(2**(q2-q1-1)), X, eye(2**(n_qubits - q2 - 1))]
#     else:
#         cx_2 = [eye(2**q2), X, eye(2**(q1-q2-1)), P1, eye(2**(n_qubits - q1 - 1))]
#     cx_2 = reduce(kron, cx_2)
#     return cx_1 + cx_2

# def measure_z_counts(dm, n_shots):
#     probs = dm.diagonal().real # exact measurement outcome probabilities
#     probs_cum = cumsum(probs) # cumulative sum of probabilities
#     ps = uniform(0, 1, n_shots) # generate the probabilities of obtaining a specific outcome
#     counts = zeros(len(dm), dtype=int) # counts of specific measuremet outcomes
#     for s in range(n_shots):
#         probs_cum_diffs = probs_cum - ps[s]
#         ind_min = np.abs(probs_cum_diffs).argmin()
#         if sign(probs_cum_diffs[ind_min]) == 1:
#             counts[ind_min] += 1
#         else:
#             counts[ind_min + 1] += 1
#     return counts

# def measure_povm(density_matrix, povm, outcomes, n_shots, return_probs=False):
#     """
#     Measures an obsrvble given as (POVM, outcomes) in a given state with a given number of shots.
#     Input: density matrix, observable a dict of POVM-elemets and the corresponding measurement outcomes, number of shots.
#     Output: estimated expected value. 
#     """
            
#     probs = {outcomes[j]: 0 for j in range(len(outcomes))} # a dictionary which will contain probabilities of a certain measuremet outcome; just for a case
#     prob_intervals = [] # will contain the probability intervals
#     p_sum = 0 # auxiliary variable; must be equal to 1 at the end of the loop below
#     for j in range(len(povm)):
#         p = trace(dot(density_matrix, povm[j])).real # probability of the outcome E_j
#         probs[outcomes[j]] += p # store the probability
#         prob_intervals.append( (p_sum, p_sum+p, outcomes[j]) ) # store the tripple (previous_probability, previous_probability + current_probability, projector's_binary_number)
#         p_sum += p # this part may be optimized a bit for not obtaining one sum twice
    
#     if n_shots == inf:
#         expectation_exact = 0
#         for outcome, prob in probs.items():
#             expectation_exact += outcome * prob 
#         return expectation_exact
    
#     else:
#         counts = {outcomes[j]: 0 for j in range(len(outcomes))} # a dictionary which will contain counts of a certain measuremet outcomes obtained in an 'experiment'
#         ps = uniform(0, 1, n_shots) # generate the probabilities of obtaining a specific outcome
#         for s in range(n_shots):
#             # find which outcome we obtained in the 'experiment' 
#             for i in range(len(outcomes) - 1):
#                 if (ps[s] >= prob_intervals[i][0]) and (ps[s] < prob_intervals[i][1]):
#                     counts[prob_intervals[i][2]] += 1 # add one 'click'
#                     break
#             # treat the last interval properly
#             if (ps[s] >= prob_intervals[-1][0]) and (ps[s] <= prob_intervals[-1][1]):
#                 counts[prob_intervals[-1][2]] += 1 # add one 'click'
#         # "experimental" probabilities
#         expectation_exper = 0
#         for outcome, count in counts.items():
#             expectation_exper += outcome * count
#         expectation_exper /= n_shots
        
#         if return_probs == True:
#             return expectation_exper, probs
#         else:
#             return expectation_exper
    
# def measure_z(density_matrix, n_shots, n_est=1, outcomes=None, return_probs=False):
#     """
#     Measures Z^n in a given state with a given number of shots. Unfortunately, not faster.
#     Input: density matrix, number of shots.
#     Output: estimated expected value. 
#     """
    
#     d = len(density_matrix)
#     if outcomes is None:
#         outcomes = [(-1)**j for j in range(d)]
#     probs_dict = {outcomes[j]: 0 for j in range(d)} # dictionary with probabilities of a certain measuremet outcome
#     probs = [] # probabilities
#     probs_cum = [] # cumulative sum of probabilities
#     prob_intervals = [] # probability intervals
#     p_cum = 0 # auxiliary variable; must be equal to 1 at the end of the loop below
#     for j in range(d):
#         p = density_matrix[j][j].real # probability of the outcome j
#         p_cum += p # this part may be optimized a bit for not obtaining one sum twice
#         probs_dict[outcomes[j]] += p # store the probability
#         probs.append(p)
#         probs_cum.append(p_cum)
#         prob_intervals.append( (p_cum, p_cum + p, outcomes[j]) ) # store the tripple (previous_probability, previous_probability + current_probability, projector's_binary_number)
#     probs_cum = array(probs_cum)
      
#     if n_shots == inf:
#         expec = 0
#         for outcome, prob in probs_dict.items():
#             expec += outcome * prob 
#         return expec
#     else:
#         expecs = [] 
#         ps = uniform(0, 1, (n_est, n_shots)) # generate the probabilities of obtaining a specific outcome
#         for e in range(n_est):
#             counts = {outcomes[j]: 0 for j in range(d)} # dict with counts of certain measuremet outcomes obtained in the 'experiment'
#             for s in range(n_shots):
#                 probs_cum_diffs = probs_cum - ps[e][s]
#                 ind_min = np.abs(probs_cum_diffs).argmin()
#                 if sign(probs_cum_diffs[ind_min]) == 1:
#                     counts[outcomes[ind_min]] += 1
#                 else:
#                     counts[outcomes[ind_min + 1]] += 1
#             expec = 0
#             for outcome, count in counts.items():
#                 expec += outcome * count
#             expecs.append(expec/n_shots)
#             # print(counts)

#         if return_probs == True:
#             return array(expecs), probs
#         else:
#             return array(expecs)