#ifndef __CSETS_H_CMC__
#define __CSETS_H_CMC__
#include "itensor/all.h"
using namespace std;
using namespace itensor;

ITensor multSite (ITensor A, ITensor const& B)
{
    A *= B;
    A.noPrime();
    return A;
}

Index make_index (int m)
{
    return Index (QN({"Sz=",0},{"Nf=",0}), m, Out, "Mix");
}

ITensor make_expand_proj (Index ii, int m_extra=1)
{
    ii.dag();
    // Make projector between the original space (m) and the enlarged space (m+m_extra)
    Index inew = make_index (dim(ii)+m_extra);
    auto P_expand = ITensor (ii, inew);
    for(int i = 1; i <= dim(ii); i++)
        P_expand.set (ii=i,inew=i,1.);
    return P_expand;
}

class Cdag_Set
{
  public:
           Cdag_Set () {}
           Cdag_Set (const MPS& psi, int ell, const string& spin_str);
    void   to_right (const MPS& psi, int ell, const string& spin_str, Args const& args = Args::global());
    Vector apply_C  (const MPS& psi, int j, string spin_str) const;
    int    m        () const { return dim(_si); }

  private:
    vector<ITensor> _Us, _Proj;
    ITensor         _V;
    int             _ilast;
    Index           _si;	// The "site" (left) index of _V
};

Cdag_Set :: Cdag_Set (const MPS& psi, int ell, const string& spin_str)
: _ilast (0)
{
    if (orthoCenter(psi) <= ell) {
        cout << "Error: Cdag_Set :: init: orthogonality center must be > ell" << endl;
        cout << "       " << orthoCenter(psi) << ", " << ell << endl;
        throw;
    }

    Electron sp (siteInds(psi));
    _V = multSite(sp.op("Cdag"+spin_str,ell),multSite(sp.op("F",ell),psi.A(ell)));
    _V *= prime(dag(psi.A(ell)),rightLinkIndex(psi,ell));

    // Make a dummy site-index to C
    _si = make_index (1);
    _V *= setElt(_si(1));
}

void Cdag_Set :: to_right (const MPS& psi, int ell, const string& spin_str, Args const& args)
{
    if (orthoCenter(psi) <= ell) {
        cout << "Error: Cdag_Set :: to_right: orthogonality center must be > ell" << endl;
        cout << "       " << orthoCenter(psi) << ", " << ell << endl;
        throw;
    }

    Electron sp (siteInds(psi));

    // Apply the i-th transfer matrix
    _V *= prime(dag(psi.A(ell)),"Link");
    _V *= multSite(sp.op("F",ell),psi.A(ell));

    // Compute the element for c_{i=ell}
    ITensor c_l = multSite(sp.op("Cdag"+spin_str,ell),multSite(sp.op("F",ell),psi.A(ell)));
    c_l *= prime(dag(psi.A(ell)),rightLinkIndex(psi,ell));

    // ------- Add c_l to Cdag -------
    //
    ITensor P_ex = make_expand_proj (_si);
    _Proj.push_back (P_ex);

    _V *= P_ex; // expand _V

    // Add c_l into _V
    Index si = findIndex (_V, "Mix");
    int m = dim(si);
    _V += c_l * setElt(si=m);
    // -------------------------------------------

    // Truncate by SVD; keep the UV form
    _Us.emplace_back (si);
    ITensor Vtmp, D;
    svd (_V, _Us.back(), D, Vtmp, args);
    _V = D * Vtmp;
    _si = commonIndex (_V, _Us.back());

    _ilast++;
}

Vector Cdag_Set :: apply_C (const MPS& psi, int j, string spin_str) const
{
    Electron sp (siteInds(psi));

    // Apply Cj
    ITensor V = _V * multSite(sp.op("C"+spin_str,j), psi.A(j));
    V *= prime(dag(psi.A(j)),rightLinkIndex(psi,j-1));

    // Contract back _Us to get CidagCj
    Vector cdagc (_ilast+1);
    for(int i = _ilast; i > 0; i--)
    {
        int im = i-1;
        V *= _Us.at(im);
        const Index& ii = V.inds().index(1);
        cdagc(i) = V.real(ii(dim(ii)));

        V *= dag(_Proj.at(im)); // Project to left space (the column of U)
    }

    Index si = V.inds().index(1);
    cdagc(0) = V.real(si(1));

    return cdagc;
}
#endif
