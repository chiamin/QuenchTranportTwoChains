#ifndef __TWOCHAINTIGHTBINDINGBASIS_H_CMC__
#define __TWOCHAINTIGHTBINDINGBASIS_H_CMC__
#include "OneParticleBasis.h"
#include "IUtility.h"

// Two chains in a one-dimension labeling
//
//  1--3--5--...
//
//  0--2--4--...

// Map from two-chain to one-dimension labeling
auto two_chain_global_label (bool up, int i)
{
    i *= 2;
    if (up)
        i += 1;
    return i;
}

Matrix two_chain_tight_binding_Hamilt (int L, Real t_up, Real t_dn, Real mu_up, Real mu_dn, Real damp_fac=1., bool damp_from_right=true, bool verbose=false)
{
    if (verbose)
    {
        cout << "Two chain tight binding" << endl;
        cout << "L = " << L << endl;
    }
    Matrix H (2*L,2*L);


    for(int j = 0; j < L; j++)
    {
        int damp_dist = (damp_from_right ? L-2-j : j);
        for(bool up : vector<bool>({false,true}))
        {
            int i1 = two_chain_global_label (up, j);
            int i2 = two_chain_global_label (up, j+1);
            auto mu = (up ? mu_up : mu_dn);
            auto t = (up ? t_up : t_dn);

            H(i1,i1) = -mu;
            if (j != L-1)
            {
                Real ti = t * pow (damp_fac, damp_dist);
                H(i1,i2) = -ti;
                H(i2,i1) = -ti;
                if (verbose)
                    cout << "t_" << up << " (" << j << "," << j+1 << ")->(" << i1 << "," << i2 << ") = " << ti << endl;
            }
        }
    }
    return H;
}

// Two uncoupled tight-binding chains.
// Labeling:    1--3--5--...
//
//              0--2--4--...
class TwoChainTightBindingBasis : public OneParticleBasis
{
    public:
        TwoChainTightBindingBasis () {}
        TwoChainTightBindingBasis (const string& name, int L, Real t_up, Real t_dn, Real mu_up, Real mu_dn,
                                   Real damp_fac=1., bool damp_from_right=true, bool verbose=false);
};

// Since the two chains are uncoupled, get eigenstates and eigenenergies from the two separate chains
TwoChainTightBindingBasis :: TwoChainTightBindingBasis
(const string& name, int L, Real t_up, Real t_dn, Real mu_up, Real mu_dn, Real damp_fac, bool damp_from_right, bool verbose)
{
    _name = name;
    _H = Matrix (2*L, 2*L);
    _Uik = Matrix (2*L, 2*L);

    auto Hup = tight_binding_Hamilt (L, t_up, mu_up, damp_fac, damp_from_right, verbose);
    auto Hdn = tight_binding_Hamilt (L, t_dn, mu_dn, damp_fac, damp_from_right, verbose);

    Matrix Uup, Udn;
    Vector en_up, en_dn;
    diagHermitian (Hup, Uup, en_up);
    diagHermitian (Hdn, Udn, en_dn);

    // Upper chain
    for(int i = 0; i < L; i++)
        for(int j = 0; j < L; j++)
        {
            int i2 = two_chain_global_label (true, i),
                j2 = two_chain_global_label (true, j);
            _Uik(i2,j2) = Uup(i,j);
            _H(i2,j2) = Hup(i,j);
        }
    // Lower chain
    for(int i = 0; i < L; i++)
        for(int j = 0; j < L; j++)
        {
            int i2 = two_chain_global_label (false, i),
                j2 = two_chain_global_label (false, j);
            _Uik(i2,j2) = Udn(i,j);
            _H(i2,j2) = Hdn(i,j);
        }

    // _ens
    Matrix U;
    diagHermitian (_H, U, _ens);

    // Check
    {
        auto Hd = transpose(_Uik) * _H * _Uik;
        mycheck (is_diagonal(Hd), "Not the correct unitary matrix");
        Vector ens = Vector(diagonal(Hd));
        Real d = norm (ens - _ens);
        mycheck (d < 1e-12, "Energy not match");
    }
}
#endif
