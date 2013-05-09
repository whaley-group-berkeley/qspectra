
def pp_4ways(rho0, H_1, t_max, dipoles,
             polarization=polarization_setup(['x','x']),
             sample_rate=0, rw_freq=12500):
    N_1 = dipoles.shape[0]
    ss = memoize(lambda s: density_subset(s, N_1))

    X = [np.triu(dipole_matrices(dipoles).dot(p)) for p in polarization]
    V = S_commutator(X[0].T)[np.ix_(ss('eg'), ss('ee'))]
    Vf = S_left(X[1])
    tr = vec(np.diag(np.ones(N_2_from_1(N_1), dtype=complex)))
    tr_Vf = np.einsum('i,ij', tr, Vf)[ss('eg')]

    V_rho0 = V.dot(rho0)

    if sample_rate == 0:
        sample_rate = default_sample_rate(H_1, rw_freq)

    L = pi*6e-5 * L_redfield(H_1, subspace='012', basis='sites',
                             rw_freq=rw_freq)[np.ix_(ss('eg'), ss('eg'))]

    t = np.arange(0, t_max, 1.0/sample_rate)

    def dtr_Vf(t, tr_Vf):
        return tr_Vf.dot(L)
    tr_Vf_G = odeint(dtr_Vf, tr_Vf, t, 'zvode', method='adams', rtol=1e-8)
    S1 = MetaArray(np.einsum('ni,i->n', tr_Vf_G, V_rho0),
                   ticks=t, rw_freq=rw_freq)

    def drho(t, rho):
        return L.dot(rho)
    G_V_rho0 = odeint(drho, V_rho0, t, 'zvode', method='adams', rtol=1e-8)
    S2 = MetaArray(np.einsum('i,ni->n', tr_Vf, G_V_rho0),
                   ticks=t, rw_freq=rw_freq)

    G_ss = np.ix_(range(len(t)), ss('eg'), ss('eg'))
    G = greens_function(H_1, t_max, piecewise=True, desired_subset='eg')[G_ss]
    S3 = MetaArray(np.einsum('i,nij,j->n', tr_Vf, G, V_rho0),
                   ticks=t, rw_freq=rw_freq)

    tr_Vf_G_V = np.einsum('ni,ij->nj', tr_Vf_G, V)
#    return tr_Vf_G_V
    S4 = MetaArray(np.einsum('ni,i->n', tr_Vf_G_V, rho0),
                   ticks=t, rw_freq=rw_freq)

    return S1, S2, S3, S4
#pump_probe_projector_ge_iso = isotropic_avg(pump_probe_projector_ge)


def pump_probe_from_density(rho0, H_1, t_max, dipoles,
                            polarization=polarization_setup(['x','x']),
                            sample_rate=0, rw_freq=12500):
    N_1 = dipoles.shape[0]
    ss = memoize(lambda txt: density_subset(txt, N_1))

    X = [np.triu(dipole_matrices(dipoles).dot(p)) for p in polarization]
    V = S_commutator(X[0].T)
    Vf = S_left(X[1])
    tr = vec(np.diag(np.ones(N_2_from_1(N_1), dtype=complex)))
    tr_Vf = tr.dot(Vf)

    tr_Vf_ge = tr_Vf[ss('ge')]

    V_rho = V[np.ix_(ss('ge'), ss('ee'))].dot(rho0)

    if sample_rate == 0:
        sample_rate = default_sample_rate(H_1, rw_freq)

    L = pi*6e-5 * L_redfield(H_1, subspace='012', basis='sites',
                             rw_freq=rw_freq)
    L_ge = L[np.ix_(ss('ge'), ss('ge'))]

    t_range = np.arange(0, t_max, 1.0/sample_rate)

    def drho(t, rho):
        return L_ge.dot(rho)

    G_V_rho = odeint(drho, V_rho, t_range, 'zvode', method='adams', rtol=1e-8)

    tr_Vf_G_V = np.einsum('i,ti->t', tr_Vf_ge, G_V_rho)
    return MetaArray(tr_Vf_G_V, ticks=t_range, rw_freq=rw_freq)