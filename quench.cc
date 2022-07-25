#include <iomanip>
#include "itensor/all.h"
#include "Timer.h"
Timers timer;
#include "ReadInput.h"
#include "IUtility.h"
#include "MyObserver.h"
#include "MixedBasis.h"
#include "TDVPObserver.h"
#include "tdvp.h"
#include "basisextension.h"
#include "InitState.h"
#include "Hamiltonian.h"
#include "ReadWriteFile.h"
#include "OneParticleBasis.h"
#include "BdGBasis.h"
using namespace itensor;
using namespace std;

void print_orbs (const vector<SortInfo>& orbs)
{
    cout << "Orbitals: name, ki, energy" << endl;
    for(int i = 1; i <= orbs.size(); i++)
    {
        auto [name, ki, en] = orbs.at(i-1);
        cout << i << ": " << name << " " << ki << " " << en << endl;
    }
}

template <typename Basis1, typename Basis2, typename SiteType>
MPO get_current_mpo (const SiteType& sites, const Basis1& basis1, const Basis2& basis2, int i1, int i2, const ToGlobDict& to_glob)
{
    AutoMPO ampo (sites);
    add_CdagC (ampo, basis1, basis2, i1, i2, 1., to_glob);
    auto mpo = toMPO (ampo);
    return mpo;
}

inline Real get_current (const MPO& JMPO, const MPS& psi)
{
    auto J = innerC (psi, JMPO, psi);
    return -2. * imag(J);
}

tuple<int,int> find_scatterer_region (const ToLocDict& to_loc, string Sname1, string Sname2, string Cname, bool include_charge=true)
{
    bool start_scatter = false;
    int i1 = -1,
        i2 = to_loc.size();
    for(int i = 1; i < to_loc.size(); i++)
    {
        auto [name, ind] = to_loc.at(i);
        bool is_scatter = (name == Sname1 or name == Sname2 or (include_charge and name == Cname));
        if (!start_scatter and is_scatter)
        {
            start_scatter = true;
            i1 = i;
        }
        if (start_scatter and !is_scatter)
        {
            i2 = i-1;
            break;
        }
    }
    mycheck (i1 > 0, "Do not search a system site");

    // Check all the scatterer sites are together
    for(int i = i2+1; i < to_loc.size(); i++)
    {
        auto [name, ind] = to_loc.at(i);
        bool is_scatter = (name == Sname1 or name == Sname2 or (include_charge and name == Cname));
        mycheck (!is_scatter, "Scatterer sites are not all together");
    }
    return {i1, i2};
}

int main(int argc, char* argv[])
{
    string infile = argv[1];
    InputGroup input (infile,"basic");

    auto quench_type   = input.getString("quench_type");
    auto dt            = input.getReal("dt");
    auto time_steps    = input.getInt("time_steps");
    auto NumCenter     = input.getInt("NumCenter");
    auto Truncate      = input.getYesNo("Truncate");
    auto mixNumCenter  = input.getYesNo("mixNumCenter",false);
    auto globExpanNStr       = input.getString("globExpanN","inf");
    int globExpanN;
    if (globExpanNStr == "inf" or globExpanNStr == "Inf" or globExpanNStr == "INF")
        globExpanN = std::numeric_limits<int>::max();
    else
        globExpanN = std::stoi (globExpanNStr);
    auto globExpanItv        = input.getInt("globExpanItv",1);
    auto globExpanCutoff     = input.getReal("globExpanCutoff",1e-8);
    auto globExpanKrylovDim  = input.getInt("globExpanKrylovDim",3);
    auto globExpanHpsiCutoff = input.getReal("globExpanHpsiCutoff",1e-8);
    auto globExpanHpsiMaxDim = input.getInt("globExpanHpsiMaxDim",300);
    auto globExpanMethod     = input.getString("globExpanMethod","DensityMatrix");

    auto measure_entropy_cutoff = input.getReal("measure_entropy_cutoff",1e-14);
    auto measure_entropy_maxdim = input.getReal("measure_entropy_maxdim",std::numeric_limits<int>::max());

    auto UseSVD        = input.getYesNo("UseSVD",true);
    auto SVDmethod     = input.getString("SVDMethod","gesdd");  // can be also "ITensor"
    auto WriteDim      = input.getInt("WriteDim");

    auto write         = input.getYesNo("write",false);
    auto write_dir     = input.getString("write_dir",".");
    auto write_file    = input.getString("write_file","");
    auto read          = input.getYesNo("read",false);
    auto read_dir      = input.getString("read_dir",".");
    auto read_file     = input.getString("read_file","");

    auto sweeps        = iut::Read_sweeps (infile, "TDVP");
    auto sweeps_DMRG   = iut::Read_sweeps (infile, "DMRG");

    cout << setprecision(14) << endl;

    MPS psi;
    MPO H;

    int step = 1;
    auto sites = MixedBasis();
    Args args_basis;

    ToGlobDict to_glob;
    ToLocDict to_loc;
    OneParticleBasis leadL_up, leadL_dn, leadR_up, leadR_dn;
    OneParticleBasis charge;
    BdGBasis scatt_up, scatt_dn;

    // -- Initialization --
    if (!read)
    {
        // Read parameters from input file
        auto L_lead   = input.getInt("L_lead");
        auto L_device   = input.getInt("L_device");
        auto t_lead     = input.getReal("t_lead");
        auto t_device   = input.getReal("t_device");
        auto tcL_up     = input.getReal("t_contactL_up");
        auto tcL_dn     = input.getReal("t_contactL_dn");
        auto tcR_up     = input.getReal("t_contactR_up");
        auto tcR_dn     = input.getReal("t_contactR_dn");
        auto mu_leadL   = input.getReal("mu_leadL");
        auto mu_leadR   = input.getReal("mu_leadR");
        auto mu_device  = input.getReal("mu_device");
        auto mu_biasL_up = input.getReal("mu_biasL_up");
        auto mu_biasL_dn = input.getReal("mu_biasL_dn");
        auto mu_biasR_up = input.getReal("mu_biasR_up");
        auto mu_biasR_dn = input.getReal("mu_biasR_dn");
        auto Delta       = input.getReal("Delta");
        auto Ec          = input.getReal("Ec");
        auto Ng          = input.getReal("Ng");
        auto EJ          = input.getReal("EJ");
        auto damp_decay_length = input.getInt("damp_decay_length",0);
        auto maxCharge   = input.getInt("maxCharge");

        // Factor for exponentially decaying hoppings
        Real damp_fac = (damp_decay_length == 0 ? 1. : exp(-1./damp_decay_length));

        // Create bases for the leads
        cout << "H left lead" << endl;
        leadL_up = OneParticleBasis ("Lup", L_lead, t_lead, mu_leadL, damp_fac, true, true);
        leadL_dn = OneParticleBasis ("Ldn", L_lead, t_lead, mu_leadL, damp_fac, true, true);

        cout << "H right lead" << endl;
        leadR_up = OneParticleBasis ("Rup", L_lead, t_lead, mu_leadR, damp_fac, false, true);
        leadR_dn = OneParticleBasis ("Rdn", L_lead, t_lead, mu_leadR, damp_fac, false, true);

        // Create basis for scatterer
        cout << "H dev" << endl;
        scatt_up = BdGBasis ("Sup", L_device, t_device, mu_device, Delta);
        scatt_dn = BdGBasis ("Sdn", L_device, t_device, mu_device, Delta);

        // Create basis for the charge site
        charge = OneParticleBasis ("C", 1);

        // Combine and sort all the basis states
        auto info_leads = sort_by_energy (leadL_up, leadL_dn, leadR_up, leadR_dn);
        auto info_scatt = sort_by_energy (scatt_up, scatt_dn);
        auto info_charg = sort_by_energy (charge);
        auto info = info_leads;
        insert_at_energy (0., info_charg, info);
        insert_at_energy (0., info_scatt, info);
        tie(to_glob, to_loc) = make_orb_dicts (info);
        print_orbs(info);

        // SiteSet
        int N = to_glob.size();
        int charge_site = to_glob.at({"C",1});
        // Find the scatterer sites in the global indices
        vector<int> scatter_sites;
        for(int i = 1; i <= scatt_up.size(); i++)
            scatter_sites.push_back (to_glob.at({"Sup",i}));
        for(int i = 1; i <= scatt_dn.size(); i++)
            scatter_sites.push_back (to_glob.at({"Sdn",i}));
        // Make SiteSet
        auto systype = (EJ == 0. ? "SC_scatter" : "SC_Josephson_scatter");
        args_basis = {"MaxOcc",maxCharge,"SystemType",systype};
        sites = MixedBasis (N, scatter_sites, charge_site, args_basis);
        cout << "charge site = " << charge_site << endl;

        if (quench_type == "density_quench")
        {
            // Initialze MPS
            // psi: Ground state of disconnected leads and scatterer with bias potentials
            psi = get_ground_state_BdG_scatter (leadL_up, leadL_dn, leadR_up, leadR_dn, scatt_up, scatt_dn,
                                                mu_biasL_up, mu_biasL_dn, mu_biasR_up, mu_biasR_dn, Ec, Ng,
                                                sites, maxCharge, to_glob);
            psi.position(1);

            // Make Hamiltonian MPO for time evolution
            // H: Connected system with no bias potential
            auto ampo = get_ampo_two_Kitaev_chains (leadL_up, leadL_dn, leadR_up, leadR_dn, scatt_up, scatt_dn, charge,
                                                    0., 0., 0., 0.,
                                                    tcL_up, tcL_dn, tcR_up, tcR_dn, Ec, Ng, EJ,
                                                    sites, to_glob);
            H = toMPO (ampo);
            cout << "MPO dim = " << maxLinkDim(H) << endl;
            cout << "Initial energy = " << inner (psi,H,psi) << endl;
        }
        else if (quench_type == "mu_quench")
        {
            // Make Hamiltonian MPO for initial state
            // H0: No bias potential
            auto ampoi = get_ampo_two_Kitaev_chains (leadL_up, leadL_dn, leadR_up, leadR_dn, scatt_up, scatt_dn, charge,
                                                     0., 0., 0., 0.,
                                                     tcL_up, tcL_dn, tcR_up, tcR_dn, Ec, Ng, EJ,
                                                     sites, to_glob);
            auto H0 = toMPO (ampoi);
            cout << "H0 MPO dim = " << maxLinkDim(H0) << endl;

            // Initialze MPS which is the ground state of H0
            psi = get_ground_state_BdG_scatter (leadL_up, leadL_dn, leadR_up, leadR_dn, scatt_up, scatt_dn,
                                                0., 0., 0., 0., Ec, Ng,
                                                sites, maxCharge, to_glob);
            psi.position(1);
            itensor::Real en0 = dmrg (psi, H0, sweeps_DMRG);
            cout << "Initial energy = " << en0 << endl;

            // Make Hamiltonian MPO for time evolution
            // H: Applying bias potential
            auto ampo = get_ampo_two_Kitaev_chains (leadL_up, leadL_dn, leadR_up, leadR_dn, scatt_up, scatt_dn, charge,
                                                    mu_biasL_up, mu_biasL_dn, mu_biasR_up, mu_biasR_dn,
                                                    tcL_up, tcL_dn, tcR_up, tcR_dn, Ec, Ng, EJ,
                                                    sites, to_glob);
            H = toMPO (ampo);
            cout << "MPO dim = " << maxLinkDim(H) << endl;
        }
        else
        {
            cout << "Unknown quench type: " << quench_type << endl;
            throw;
        }
    }
    else
    {
        ifstream ifs = open_file (read_dir+"/"+read_file);
        iut::read_all (ifs, psi, H, args_basis, step, to_glob, to_loc, leadL_up, leadL_dn, leadR_up, leadR_dn, scatt_up, scatt_dn, charge);
        sites = MixedBasis (siteInds(psi), args_basis);
    }
    // -- End of initialization --

    // -- Observer --
    auto obs = TDVPObserver (sites, psi, {"charge_site",to_glob.at({"C",1})});
    // Current MPO
    auto jmpoL_up = get_current_mpo (sites, leadL_up, leadL_up, -2, -1, to_glob);
    auto jmpoL_dn = get_current_mpo (sites, leadL_dn, leadL_dn, -2, -1, to_glob);
    auto jmpoR_up = get_current_mpo (sites, leadR_up, leadR_up, 1, 2, to_glob);
    auto jmpoR_dn = get_current_mpo (sites, leadR_dn, leadR_dn, 1, 2, to_glob);

    // Find the scatterer region, which will be used in compaute the entanglement entropy
    // (Assume that all the scatterer sites are together; otherwise raise error.)
    auto [si1, si2] = find_scatterer_region (to_loc, scatt_up.name(), scatt_dn.name(), charge.name());
    cout << "charge and scatterer is between sites " << si1 << " " << si2 << endl;

    // -- Time evolution --
    cout << "Start time evolution" << endl;
    cout << sweeps << endl;
    psi.position(1);
    Real en, err;
    Args args_tdvp_expansion = {"Cutoff",globExpanCutoff, "Method","DensityMatrix",
                                "KrylovOrd",globExpanKrylovDim, "DoNormalize",true, "Quiet",true};
    Args args_tdvp  = {"Quiet",true,"NumCenter",NumCenter,"DoNormalize",true,"Truncate",Truncate,
                       "UseSVD",UseSVD,"SVDmethod",SVDmethod,"WriteDim",WriteDim,"mixNumCenter",mixNumCenter};
    LocalMPO PH (H, args_tdvp);
    while (step <= time_steps)
    {
        cout << "step = " << step << endl;

        // Subspace expansion
        if (maxLinkDim(psi) < sweeps.mindim(1) or (step < globExpanN and (step-1) % globExpanItv == 0))
        {
            timer["glob expan"].start();
            addBasis (psi, H, globExpanHpsiCutoff, globExpanHpsiMaxDim, args_tdvp_expansion);
            PH.reset();
            timer["glob expan"].stop();
        }

        // Time evolution
        timer["tdvp"].start();
        //tdvp (psi, H, 1_i*dt, sweeps, obs, args_tdvp);
        TDVPWorker (psi, PH, 1_i*dt, sweeps, obs, args_tdvp);
        timer["tdvp"].stop();
        auto d1 = maxLinkDim(psi);

        // Measure currents by MPO
        timer["current mps"].start();
        auto jL_up = get_current (jmpoL_up, psi);
        auto jL_dn = get_current (jmpoL_dn, psi);
        auto jR_up = get_current (jmpoR_up, psi);
        auto jR_dn = get_current (jmpoR_dn, psi);
        cout << "\tI L/R = " << jL_up << " " << jL_dn << " " << jR_up << " " << jR_dn << endl;
        timer["current mps"].stop();

        // Measure entanglement entropy
        timer["entang entropy"].start();
        Real EE = get_entang_entropy (psi, si1, si2, {"Cutoff",measure_entropy_cutoff,"MaxDim",measure_entropy_maxdim});
        timer["entang entropy"].stop();
        cout << "\tEE = " << EE << endl;

        step++;
        if (write)
        {
            timer["write"].start();
            ofstream ofs (write_dir+"/"+write_file);
            iut::write_all (ofs, psi, H, args_basis, step, to_glob, to_loc, leadL_up, leadL_dn, leadR_up, leadR_dn, scatt_up, scatt_dn, charge);
            timer["write"].stop();
        }
    }
    timer.print();
    return 0;
}
