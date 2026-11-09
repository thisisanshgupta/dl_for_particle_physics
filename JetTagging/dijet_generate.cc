/*
 Jet generator for W-vs-QCD tagging  (PYTHIA 8 + FastJet + ROOT)

 Usage:  ./gen_jets <qcd|signal> [nEvent] [seed] [pTHatMin] [jetPtMin] [prefix]

 Configurations used:
    ./gen_jets qcd    200000 1 200 250 jets
    ./gen_jets signal  50000 2 200 250 jets

 Output (per mode):
    <prefix>_<mode>.bin           fixed-length float32 records, one per selected jet
    <prefix>_<mode>_jetMass.root  leading-jet mass histogram
    <prefix>_<mode>_jetMass.pdf   same histogram, log scale

 Record layout (RECORD = 6 + 100*6 floats):
    header      : label (0 = QCD, 1 = W), jet pT, eta, phi, mass, n constituents
    constituent : pT, eta, phi, E, PDG id, charge   (up to 100, pT-ordered, zero padded)
*/

#include "Pythia8/Pythia.h"
#include "fastjet/ClusterSequence.hh"
#include "TFile.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TPad.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
    constexpr int    MAXC     = 100;
    constexpr int    NFEAT    = 6;
    constexpr int    HEADER   = 6;
    constexpr int    RECORD   = HEADER + MAXC * NFEAT;
    constexpr double kDataR   = 1.0;
    constexpr double kMatchDR = 0.6;
    constexpr double kEtaParticleMax = 4.5;
    constexpr double kEtaJetMax      = 2.5;
    constexpr int    kWBosonId       = 24;
}

static void findBosons(const Pythia8::Event& ev, int bosonId,
                       std::vector<fastjet::PseudoJet>& out) {
    out.clear();
    for (int i = 0; i < ev.size(); ++i) {
        const Pythia8::Particle& p = ev[i];
        if (std::abs(p.id()) != bosonId) continue;
        int d1 = p.daughter1();
        if (d1 > 0 && std::abs(ev[d1].id()) == bosonId) continue;
        out.emplace_back(p.px(), p.py(), p.pz(), p.e());
    }
}

int main(int argc, char** argv) {
    const std::string mode   = (argc > 1) ? argv[1] : "qcd";
    const int    nEvent      = (argc > 2) ? std::atoi(argv[2]) : 15000;
    const int    seed        = (argc > 3) ? std::atoi(argv[3]) : 9999;
    const double pTHatMin    = (argc > 4) ? std::atof(argv[4]) : 150.;
    const double jetPtMin    = (argc > 5) ? std::atof(argv[5]) : 200.;
    const std::string prefix = (argc > 6) ? argv[6] : "jets";

    if (mode != "signal" && mode != "qcd") {
        std::cerr << "mode must be 'qcd' or 'signal'\n";
        return 1;
    }
    if (nEvent <= 0) { std::cerr << "nEvent must be > 0\n"; return 1; }
    if (seed < 0 || seed > 900000000) {
        std::cerr << "seed must be in [0, 900000000]\n";
        return 1;
    }

    const bool  isSignal   = (mode == "signal");
    const int   bosonId    = isSignal ? kWBosonId : 0;
    const float classLabel = isSignal ? 1.f : 0.f;

    Pythia8::Pythia pythia;
    std::vector<std::string> config = {
        "Beams:idA = 2212",
        "Beams:idB = 2212",
        "Beams:eCM = 13000.",
        "PhaseSpace:pTHatMin = " + std::to_string(pTHatMin),
        "PhaseSpace:pTHatMax = -1.",
        "PartonLevel:MPI = off",
        "Random:setSeed = on",
        "Random:seed = " + std::to_string(seed),
        "Print:quiet = on",
        "Next:numberCount = 0"
    };
    if (isSignal) {
        config.push_back("WeakDoubleBoson:ffbar2WW = on");
        config.push_back("24:onMode = off");
        config.push_back("24:onIfAny = 1 2 3 4 5");
    } else {
        config.push_back("HardQCD:all = on");
    }
    for (const auto& cmd : config) pythia.readString(cmd);
    if (!pythia.init()) { std::cerr << "Pythia init failed\n"; return 1; }

    const fastjet::JetDefinition jetDef(fastjet::antikt_algorithm, kDataR);

    TH1::AddDirectory(kFALSE);
    const std::string rs     = std::to_string(kDataR).substr(0, 3);
    const std::string hname  = "hJetMass_R" + rs + "_" + mode;
    const std::string htitle = "Leading Jet Invariant Mass (" + mode + "), R=" + rs +
                               ";M_{jet} [GeV];Jets / 2 GeV";
    TH1D hist(hname.c_str(), htitle.c_str(), 100, 0.0, 200.0);

    const std::string binName = prefix + "_" + mode + ".bin";
    std::vector<char> ioBuf(1 << 22);
    std::ofstream bin;
    bin.rdbuf()->pubsetbuf(ioBuf.data(), static_cast<std::streamsize>(ioBuf.size()));
    bin.open(binName, std::ios::binary);
    if (!bin) { std::cerr << "cannot open " << binName << "\n"; return 1; }

    std::vector<float> rec(RECORD);
    long nWritten = 0, nTruncated = 0;

    std::vector<fastjet::PseudoJet> particles, bosons, jets;
    particles.reserve(2048);

    for (int iEvent = 0; iEvent < nEvent; ++iEvent) {
        if (!pythia.next()) continue;
        const Pythia8::Event& event = pythia.event;

        particles.clear();
        for (int i = 0; i < event.size(); ++i) {
            const Pythia8::Particle& p = event[i];
            if (!p.isFinal()) continue;
            const int id = std::abs(p.id());
            if (id == 12 || id == 14 || id == 16) continue;
            if (std::abs(p.eta()) > kEtaParticleMax) continue;
            fastjet::PseudoJet pj(p.px(), p.py(), p.pz(), p.e());
            pj.set_user_index(i);
            particles.push_back(pj);
        }
        if (particles.empty()) continue;

        if (isSignal) findBosons(event, bosonId, bosons);

        fastjet::ClusterSequence cs(particles, jetDef);
        jets = fastjet::sorted_by_pt(cs.inclusive_jets());
        if (jets.size() < 2) continue;

        const fastjet::PseudoJet* lead[2] = {&jets[0], &jets[1]};
        if (std::abs(lead[0]->eta()) > kEtaJetMax) continue;
        if (std::abs(lead[1]->eta()) > kEtaJetMax) continue;

        hist.Fill(lead[0]->m());
        hist.Fill(lead[1]->m());

        for (const fastjet::PseudoJet* jp : lead) {
            const fastjet::PseudoJet& jet = *jp;
            if (jet.pt() < jetPtMin) continue;

            float label = 0.f;
            if (isSignal) {
                bool matched = false;
                for (const auto& b : bosons)
                    if (jet.delta_R(b) < kMatchDR) { matched = true; break; }
                if (!matched) continue;
                label = classLabel;
            }

            const std::vector<fastjet::PseudoJet> cons = fastjet::sorted_by_pt(jet.constituents());
            std::fill(rec.begin(), rec.end(), 0.f);
            rec[0] = label;
            rec[1] = static_cast<float>(jet.pt());
            rec[2] = static_cast<float>(jet.eta());
            rec[3] = static_cast<float>(jet.phi_std());
            rec[4] = static_cast<float>(jet.m());
            const int nc = std::min<int>(static_cast<int>(cons.size()), MAXC);
            if (static_cast<int>(cons.size()) > MAXC) ++nTruncated;
            rec[5] = static_cast<float>(nc);
            for (int c = 0; c < nc; ++c) {
                const Pythia8::Particle& p = event[cons[c].user_index()];
                float* f = &rec[HEADER + c * NFEAT];
                f[0] = static_cast<float>(cons[c].pt());
                f[1] = static_cast<float>(cons[c].eta());
                f[2] = static_cast<float>(cons[c].phi_std());
                f[3] = static_cast<float>(cons[c].e());
                f[4] = static_cast<float>(p.id());
                f[5] = static_cast<float>(p.charge());
            }
            bin.write(reinterpret_cast<const char*>(rec.data()),
                      static_cast<std::streamsize>(RECORD * sizeof(float)));
            ++nWritten;
        }
    }
    bin.flush();
    if (!bin) { std::cerr << "write error on " << binName << "\n"; return 1; }
    bin.close();

    const std::string stem = prefix + "_" + mode + "_jetMass";
    {
        TFile outFile((stem + ".root").c_str(), "RECREATE");
        outFile.cd();
        hist.Write();
        outFile.Close();
    }

    gStyle->SetOptStat(1111);
    TCanvas c("c", "c", 800, 600);
    c.cd();
    gPad->SetLogy();
    hist.SetLineColor(kBlue + 1);
    hist.SetLineWidth(2);
    hist.SetFillColorAlpha(kBlue - 9, 0.4);
    hist.Draw("HIST");
    c.SaveAs((stem + ".pdf").c_str());

    pythia.stat();
    std::cout << "Wrote " << nWritten << " " << mode << " jets to " << binName
              << " (" << nTruncated << " jets had > " << MAXC << " constituents and were truncated)\n";
    return 0;
}
