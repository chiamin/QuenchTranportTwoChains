#ifndef _TWOCHAINBDGBASIS_H_CMC__
#define _TWOCHAINBDGBASIS_H_CMC__
#include "BdGBasis.h"
#include "IUtility.h"

// Two chains in a one-dimension labeling
//
//  1--3--5--...
//
//  0--2--4--...
Matrix two_chain_BdG_Hamilt (int L, Real t_up, Real t_dn, Real mu_up, Real mu_dn, Real Delta_up, Real Delta_dn)
{
    Matrix H (4*L, 4*L);
    // mu
    // Upper chain
    for(int i = 0; i < L; i++)
        set_mu_BdG_Hamilt (H, mu_up, i);
    // Lower chain
    for(int i = L; i < 2*L; i++)
        set_mu_BdG_Hamilt (H, mu_dn, i);

    // t and Delta
    for(int i = 0; i < L-1; i++)
    {
        int iup = i*2 + 1,
            idn = i*2;
        int jup = iup + 2,
            jdn = idn + 2;
        set_t_BdG_Hamilt (H, t_up, iup, jup);
        set_t_BdG_Hamilt (H, t_dn, idn, jdn);
        set_Delta_BdG_Hamilt (H, Delta_up, iup, jup);
        set_Delta_BdG_Hamilt (H, Delta_dn, idn, jdn);
    }
    return 0.5*H;
}


// Two decoupled BdG chains in a one-dimension labeling
//
//  1--3--5--...
//
//  0--2--4--...
class TwoChainBdGBasis : public BdGBasis
{
    public:
        TwoChainBdGBasis () {}
        TwoChainBdGBasis (const string& name, int L, Real t_up, Real t_dn, Real mu_up, Real mu_dn, Real Delta_up, Real Delta_dn);
};


void sort_by_energy (Vector& en, vector<bool>& up, Matrix& u, Matrix& v)
{
    // Sort _ens, _u, _v (, up2) by the energies
    // If up chain and down chain have the same energy states, the up-chain state is in front of the down-chain state
    int N = en.size();
    for(int i = 0; i < N; i++)
        for(int j = i; j >= 1; j--)
        {
            Real dE = en(j-1) - en(j);
            if (dE > 0. or
                ( abs(dE) < 1e-14 and (!up.at(j-1) and  up.at(j)) ) )
                // 1. same energy,    2. up chain at the right
            {
                swap (en(j-1), en(j));
                swap_column (u, j-1, j);
                swap_column (v, j-1, j);
                swap (up.at(j-1), up.at(j));
            }
            else break;
        }
}

// Since the two chains are uncoupled, get eigenstates and eigenenergies from the two separate chains
TwoChainBdGBasis :: TwoChainBdGBasis
(const string& name, int L, Real t_up, Real t_dn, Real mu_up, Real mu_dn, Real Delta_up, Real Delta_dn)
{
    _name = name;
    _H = Matrix (4*L, 4*L);
    _u = Matrix (2*L, 2*L);
    _v = Matrix (2*L, 2*L);
    _ens = Vector (2*L);

    auto bdg_up = BdGBasis ("up", L, t_up, mu_up, Delta_up);
    auto bdg_dn = BdGBasis ("dn", L, t_up, mu_up, Delta_up);

    vector<bool> up2 (2*L); // Record the state in two-chain basis belongs to the upper or the lower chain
    for(bool up : {false, true})
    {
        const auto& bdg = (up ? bdg_up : bdg_dn);
        for(int i = 0; i < L; i++)
            for(int j = 0; j < L; j++)
            {
                // Two-chain lebals to one-chain label
                int i2 = two_chain_global_label (up, i),
                    j2 = two_chain_global_label (up, j);
                // BdG extended-block index
                int i2e = two_chain_global_label (up, i+L),
                    j2e = two_chain_global_label (up, j+L);
                // H
                // Upper left block
                _H(i2,j2) = bdg.H(i,j);
                // Upper right block
                _H(i2,j2e) = bdg.H(i,j+L);
                // Lower left block
                _H(i2e,j2) = bdg.H(i+L,j);
                // Lower right block
                _H(i2e,j2e) = bdg.H(i+L,j+L);

                // u and v
                _u(i2,j2) = bdg.u(i,j);
                _v(i2,j2) = bdg.v(i,j);
                // Record if the j2-th state belongs to upper or lower chain
                up2.at(j2) = up;
            }
    }

    // Get _ens
    int N = 2*L;
    auto U = this->U();
    auto Hd = transpose(U) * _H * U;    // Hd is a 2Nx2N matrix
    for(int i = 0; i < N; i++)
        _ens(i) = 2.*Hd(i,i);

    // Check
    {
        mycheck (is_diagonal(Hd), "Not the correct unitary matrix");
        for(int i = 0; i < N; i++)
        {
            Real d = Hd(i,i) + Hd(i+N,i+N);
            mycheck (abs(d) < 1e-14, "Energy is not antisymmetric");
            d = 2*Hd(i,i) - _ens(i);
            cout << i << " " << Hd(i,i) << " " << _ens(i) << endl;
            mycheck (abs(d) < 1e-14, "Energy not match");
        }
    }

}
#endif
