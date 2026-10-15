import csv
import json
import os

from charset_normalizer.md import is_arabic
from dimod import Binary, ConstrainedQuadraticModel, Integer, Real, cqm_to_bqm
from dimod.binary import quicksum
from pulp import *
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

def open_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class BendersKesit:
    """
    is_feasibility=False ise (Optimallik): t_var >= sabit + sum(kat_w * w) + sum(kat_y * y)
    is_feasibility=True  ise (Fizibilite):      0 >= sabit + sum(kat_w * w) + sum(kat_y * y)
    """
    sabit: float
    katsayilar_w: Dict[str, float] = field(default_factory=dict)
    katsayilar_y: Dict[str, float] = field(default_factory=dict)
    is_feasibility: bool = False


@dataclass
class BendersSonucu:
    """Tek fazlı Benders döngüsünün nihai çözüm, metrik ve yakınsama çıktıları."""
    durum: str
    hedef_deger: float
    karar: Dict[str, Any]
    metrikler: Dict[str, Any] = field(default_factory=dict)
    yakinsama_gecmisi: List[Dict[str, float]] = field(default_factory=dict)

@dataclass
class AltProblemSonucu:
    """Alt problem (PuLP/LP) çözümünden dönen tüm operasyonel ve dual veriler."""
    feasible: bool
    maliyet: float
    w_bar: Dict[str, int] = field(default_factory=dict)
    y_bar: Dict[str, int] = field(default_factory=dict)

    akislar: Dict[str, float] = field(default_factory=dict)
    kaynak_uretimi: Dict[str, float] = field(default_factory=dict)
    karsilanmayan: Dict[str, float] = field(default_factory=dict)
    depo_hacimleri: Dict[str, float] = field(default_factory=dict)
    depo_giris: Dict[str, float] = field(default_factory=dict)
    depo_cikis: Dict[str, float] = field(default_factory=dict)
    depo_ihlalleri: Dict[str, float] = field(default_factory=dict)

    # Benders Kesiti Katsayıları ve Sabiti
    source_duals: Dict[str, float] = field(default_factory=dict)
    pump_duals: Dict[str, float] = field(default_factory=dict)
    kesit_sabiti: float = 0.0


class HBD:
    def __init__(self, sampler: str = "hybrid"):
        self.sampler = sampler
        self._kesit_no = 0
        self._read_datasets()
        self._build_model()
        self._build_master_problem(sampler=self.sampler)

    def _read_datasets(self):
        self.arc_availability = open_json("arc_availability.json")
        self.critical_demands = open_json("critical_demands.json")

        self.demands = open_json("demands.json")["demands"]
        self.demands_set = {demand["node_id"]: demand for demand in self.demands}

        self.failures = open_json("failures.json")["failures"]
        self.network_arcs = open_json("network_arcs.json")["arcs"]
        self.network_nodes = open_json("network_nodes.json")["nodes"]

        pumps = open_json("pumps.json")
        self.pumps = pumps["pumps"]
        self.pump_groups = pumps["pump_groups"]

        sources = open_json("sources.json")
        self.sources = sources["sources"]
        self.treatment_plants = sources["treatment_plants"]
        self.source_packages = sources["source_packages"]

        self.tanks = open_json("tanks.json")["tanks"]
        self.zones = open_json("zones.json")["zones"]

        self.src_id_to_node = {s["source_id"]: s["node_id"] for s in self.sources}
        self.pmp_id_to_node = {p["pump_id"]: p["node_id"] for p in self.pumps}

        self.node_incoming_arcs = {n["node_id"]: [] for n in self.network_nodes}
        self.node_outgoing_arcs = {n["node_id"]: [] for n in self.network_nodes}

        for a in self.network_arcs:
            if a["to_node"] in self.node_incoming_arcs:
                self.node_incoming_arcs[a["to_node"]].append(a["arc_id"])
            if a["from_node"] in self.node_outgoing_arcs:
                self.node_outgoing_arcs[a["from_node"]].append(a["arc_id"])

    def _build_model(self):
        self.w = {s["node_id"]: Binary(f"w_{s['node_id']}") for s in self.sources}
        self.y = {p["node_id"]: Binary(f"y_{p['node_id']}") for p in self.pumps}

    def _build_master_problem(self, sampler: str = "hybrid", t_var_ust_sinir: float = 1e12):
        cqm = ConstrainedQuadraticModel()

        if sampler == "hybrid":
            self.t_var = Real("t_var", lower_bound=0.0, upper_bound=t_var_ust_sinir)
        else:
            self.t_var = Integer("t_var", lower_bound=0, upper_bound=int(t_var_ust_sinir))

        q1 = quicksum(self.w[s["node_id"]] * s["activation_cost_tl"] for s in self.sources)
        q2 = quicksum(self.y[p["node_id"]] * p["fixed_on_cost"] for p in self.pumps)
        cqm.set_objective(q1 + q2 + self.t_var)

        for g in self.pump_groups:
            pmp_nodes = [self.pmp_id_to_node[pid] for pid in g["member_pump_ids"] if pid in self.pmp_id_to_node]
            cqm.add_constraint(
                quicksum(self.y[nid] for nid in pmp_nodes) <= g["max_active"],
                label=f"K15_PumpGroup_{g.get('group_id', '')}",
            )

        for g in self.source_packages:
            src_nodes = [self.src_id_to_node[sid] for sid in g["member_source_ids"] if sid in self.src_id_to_node]
            cqm.add_constraint(
                quicksum(self.w[nid] for nid in src_nodes) <= g["max_active"],
                label=f"K16_SourcePkg_{g.get('package_id', '')}",
            )

        self.cqm = cqm
        self._kesit_no = 0

    def kesit_ekle(self, kesit: BendersKesit | AltProblemSonucu):
        if isinstance(kesit, AltProblemSonucu):
            sabit = kesit.kesit_sabiti
            katsayilar_w = kesit.source_duals
            katsayilar_y = kesit.pump_duals
            is_feas = not kesit.feasible
        else:
            sabit = kesit.sabit
            katsayilar_w = kesit.katsayilar_w
            katsayilar_y = kesit.katsayilar_y
            is_feas = kesit.is_feasibility

        sol_taraf = (
                quicksum(kat * self.w[node_id] for node_id, kat in katsayilar_w.items() if node_id in self.w)
                + quicksum(kat * self.y[node_id] for node_id, kat in katsayilar_y.items() if node_id in self.y)
        )

        if is_feas:
            self.cqm.add_constraint(-sol_taraf >= sabit, label=f"benders_feas_kesit_{self._kesit_no}")
        else:
            self.cqm.add_constraint(self.t_var - sol_taraf >= sabit, label=f"benders_opt_kesit_{self._kesit_no}")

        self._kesit_no += 1

    def coz_master(self, time_limit: float = 10.0, num_reads: int = 1500):
        if self.sampler == "hybrid":
            from dwave.system import LeapHybridCQMSampler
            sampler_obj = LeapHybridCQMSampler(token="RNNK-f9847339b1ae9e1c6f5b2030fbf184bddb46c9f3")
            sampleset = sampler_obj.sample_cqm(self.cqm, time_limit=time_limit, label="su_agi_benders_master")
            feasibleset = sampleset.filter(lambda row: row.is_feasible)

            if len(feasibleset) == 0:
                print(
                    "UYARI: Master problem için geçerli (feasible) çözüm bulunamadı! En düşük enerjili örnek alınıyor.")
                en_iyi = sampleset.first.sample
            else:
                en_iyi = feasibleset.first.sample

        elif self.sampler == "local":
            from neal import SimulatedAnnealingSampler
            bqm, invert = cqm_to_bqm(self.cqm)
            ham_sonuc = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads)
            en_iyi = invert(ham_sonuc.first.sample)
        else:
            raise ValueError(f"Bilinmeyen sampler tipi: {self.sampler}")

        w_bar = {s["node_id"]: int(round(en_iyi[f"w_{s['node_id']}"])) for s in self.sources}
        y_bar = {p["node_id"]: int(round(en_iyi[f"y_{p['node_id']}"])) for p in self.pumps}
        t_bar = float(en_iyi.get("t_var", 0.0))

        return w_bar, y_bar, t_bar

    def master_sabit_maliyeti(self, w_bar: Dict[str, int], y_bar: Dict[str, int]) -> float:
        toplam = sum(w_bar[s["node_id"]] * s["activation_cost_tl"] for s in self.sources)
        toplam += sum(y_bar[p["node_id"]] * p["fixed_on_cost"] for p in self.pumps)
        return toplam
#sub problem kismi

    def _build_sub_problem(self, w_val: dict, y_val: dict) -> AltProblemSonucu:
        params = {
            "T": 24,
            "tau": 0.002,
            "rho": 0.001,
            "Pi_tal": 25000,
            "Pi_min": 100,
            "Pi_kri": 800,
            "Pi_tuk": 25,
            "Sigma_hed": 1.0,
            "Pi_slack": 100000.0,
        }

        sub_prob = LpProblem("Benders_SubProblem", LpMinimize)

        # Değişkenler
        sp_x = {a["arc_id"]: LpVariable(f"x_{a['arc_id']}", lowBound=0, upBound=a["capacity_m3h"]) for a in
                self.network_arcs}
        sp_q = {s["node_id"]: LpVariable(f"q_{s['node_id']}", lowBound=0) for s in self.sources}
        sp_t_plus = {r["node_id"]: LpVariable(f"t_plus_{r['node_id']}", lowBound=0) for r in self.tanks}
        sp_t_minus = {r["node_id"]: LpVariable(f"t_minus_{r['node_id']}", lowBound=0) for r in self.tanks}
        sp_v = {r["node_id"]: LpVariable(f"V_{r['node_id']}", lowBound=0, upBound=r["capacity_m3"]) for r in self.tanks}
        sp_u = {z["zone_id"]: LpVariable(f"u_{z['zone_id']}", lowBound=0,
                                         upBound=self.demands_set[z["node_id"]]["total_demand_m3h"]) for z in
                self.zones}

        sp_e_min = {r["node_id"]: LpVariable(f"e_min_{r['node_id']}", lowBound=0) for r in self.tanks}
        sp_e_kri = {r["node_id"]: LpVariable(f"e_kri_{r['node_id']}", lowBound=0) for r in self.tanks}
        sp_e_tuk = {r["node_id"]: LpVariable(f"e_tuk_{r['node_id']}", lowBound=0) for r in self.tanks}

        sp_slack_src = {s["node_id"]: LpVariable(f"slack_src_{s['node_id']}", lowBound=0) for s in self.sources}
        sp_slack_pmp = {p["node_id"]: LpVariable(f"slack_pmp_{p['node_id']}", lowBound=0) for p in self.pumps}

        # Amaç Fonksiyonu
        cost_q = params["T"] * lpSum(s["cost_tl_m3"] * sp_q[s["node_id"]] for s in self.sources)
        cost_pump_energy = params["T"] * lpSum(
            p["variable_energy_cost"] * sp_x[a_id]
            for p in self.pumps
            for a_id in p["out_arc_ids"]
        )
        cost_transport = params["T"] * lpSum(
            (a["unit_cost_tl_m3"] + params["rho"]) * sp_x[a["arc_id"]] for a in self.network_arcs)
        cost_tank_io = params["T"] * params["tau"] * lpSum(
            sp_t_plus[r["node_id"]] + sp_t_minus[r["node_id"]] for r in self.tanks)
        cost_unmet_demand = params["T"] * params["Pi_tal"] * lpSum(sp_u[z["zone_id"]] for z in self.zones)
        cost_tank_penalties = (
                params["Pi_min"] * lpSum(sp_e_min[r["node_id"]] for r in self.tanks)
                + params["Pi_kri"] * lpSum(sp_e_kri[r["node_id"]] for r in self.tanks)
                + params["Pi_tuk"] * lpSum(sp_e_tuk[r["node_id"]] for r in self.tanks)
        )
        cost_slacks = params["Pi_slack"] * (
                lpSum(sp_slack_src[s["node_id"]] for s in self.sources)
                + lpSum(sp_slack_pmp[p["node_id"]] for p in self.pumps)
        )

        sub_prob += (
                    cost_q + cost_pump_energy + cost_transport + cost_tank_io + cost_unmet_demand + cost_tank_penalties + cost_slacks)

        # K1 & K2: Kaynak limitleri
        for s in self.sources:
            s_id = s["node_id"]
            sub_prob += sp_q[s_id] <= s["capacity_m3h"] * w_val[s_id], f"K1_SourceCap_{s_id}"
            sub_prob += sp_q[s_id] + sp_slack_src[s_id] >= s["min_flow_if_active_m3h"] * w_val[
                s_id], f"K2_SourceMinFlow_{s_id}"

        # K4 & K5: Pompa limitleri
        for p in self.pumps:
            p_id = p["node_id"]
            pump_arcs = p["out_arc_ids"]
            if pump_arcs:
                sub_prob += lpSum(sp_x[a_id] for a_id in pump_arcs) <= p["capacity_m3h"] * y_val[
                    p_id], f"K4_PumpCap_{p_id}"
                sub_prob += lpSum(sp_x[a_id] for a_id in pump_arcs) + sp_slack_pmp[p_id] >= p["min_flow_if_on"] * y_val[
                    p_id], f"K5_PumpMinFlow_{p_id}"

        # K6: Kütle korunum dengesi
        # K6: Kütle korunum dengesi (Hızlı lookup ile O(1))
        for n in self.network_nodes:
            n_id = n["node_id"]
            incoming = [sp_x[a_id] for a_id in self.node_incoming_arcs[n_id]]
            outgoing = [sp_x[a_id] for a_id in self.node_outgoing_arcs[n_id]]

            if n["node_type"] in ("source", "treatment_plant"):
                incoming.append(sp_q[n_id])
            elif n["node_type"] == "tank":
                incoming.append(sp_t_minus[n_id])
                outgoing.append(sp_t_plus[n_id])
            elif n["node_type"] == "demand_zone":
                outgoing.append(self.demands_set[n_id]["total_demand_m3h"] - sp_u[n_id])

            sub_prob += lpSum(incoming) == lpSum(outgoing), f"K6_FlowBalance_{n_id}"

        # K7 - K13: Depo Kısıtları
        for r in self.tanks:
            r_id = r["node_id"]
            v_init, v_min, v_max, v_kri = r["initial_volume_m3"], r["min_level_m3"], r["capacity_m3"], r[
                "critical_level_m3"]
            sigma_hed, T = params["Sigma_hed"], params["T"]

            sub_prob += sp_v[r_id] == v_init + T * (sp_t_plus[r_id] - sp_t_minus[r_id]), f"K7_TankBalance_{r_id}"
            sub_prob += T * sp_t_minus[r_id] <= v_init, f"K9_TankExtractionLimit_{r_id}"
            sub_prob += T * sp_t_plus[r_id] <= v_max, f"K10_TankInjectionLimit_{r_id}"
            sub_prob += sp_v[r_id] + sp_e_min[r_id] >= v_min, f"K11_TankMinLevel_{r_id}"
            sub_prob += sp_v[r_id] + sp_e_kri[r_id] >= v_kri, f"K12_TankCritLevel_{r_id}"
            sub_prob += sp_v[r_id] + sp_e_tuk[r_id] >= sigma_hed * v_init, f"K13_TankTargetLevel_{r_id}"

        self.sub_prob = sub_prob

        # Çözüm
        solver = PULP_CBC_CMD(timeLimit=15, msg=False)
        status = sub_prob.solve(solver)

        if status != constants.LpStatusOptimal:
            return AltProblemSonucu(
                feasible=False,
                maliyet=float("inf"),
                w_bar=dict(w_val),
                y_bar=dict(y_val)
            )

        sub_cost = float(value(sub_prob.objective))
        is_valid = is_feasible(
            w=w_val, y=y_val,
            x=sp_x, q=sp_q,
            t_plus=sp_t_plus, t_minus=sp_t_minus,
            V=sp_v, u=sp_u,
            e_min=sp_e_min, e_kri=sp_e_kri, e_tuk=sp_e_tuk
        )

        # Dual Katsayılar: d(Cost)/d(RHS)
        source_duals = {}
        for s in self.sources:
            s_id = s["node_id"]
            c1 = sub_prob.constraints.get(f"K1_SourceCap_{s_id}")
            c2 = sub_prob.constraints.get(f"K2_SourceMinFlow_{s_id}")
            pi1 = c1.pi if (c1 is not None and c1.pi is not None) else 0.0
            pi2 = c2.pi if (c2 is not None and c2.pi is not None) else 0.0
            source_duals[s_id] = pi1 * float(s["capacity_m3h"]) + pi2 * float(s["min_flow_if_active_m3h"])

        pump_duals = {}
        for p in self.pumps:
            p_id = p["node_id"]
            c4 = sub_prob.constraints.get(f"K4_PumpCap_{p_id}")
            c5 = sub_prob.constraints.get(f"K5_PumpMinFlow_{p_id}")
            pi4 = c4.pi if (c4 is not None and c4.pi is not None) else 0.0
            pi5 = c5.pi if (c5 is not None and c5.pi is not None) else 0.0
            pump_duals[p_id] = pi4 * float(p["capacity_m3h"]) + pi5 * float(p["min_flow_if_on"])

        # Kesit Sabiti: SubCost - sum(kat * val)
        kesit_sabiti = sub_cost - sum(source_duals[s_id] * w_val[s_id] for s_id in source_duals) - sum(
            pump_duals[p_id] * y_val[p_id] for p_id in pump_duals)

        return AltProblemSonucu(
            feasible=True,
            maliyet=sub_cost,
            w_bar=dict(w_val),
            y_bar=dict(y_val),
            akislar={a["arc_id"]: round(sp_x[a["arc_id"]].varValue, 3) for a in self.network_arcs if
                     sp_x[a["arc_id"]].varValue is not None},
            kaynak_uretimi={s["node_id"]: round(sp_q[s["node_id"]].varValue, 3) for s in self.sources if
                            sp_q[s["node_id"]].varValue is not None},
            karsilanmayan={z["zone_id"]: round(sp_u[z["zone_id"]].varValue, 3) for z in self.zones if
                           sp_u[z["zone_id"]].varValue is not None},
            depo_hacimleri={r["node_id"]: round(sp_v[r["node_id"]].varValue, 3) for r in self.tanks if
                            sp_v[r["node_id"]].varValue is not None},
            depo_giris={r["node_id"]: round(sp_t_plus[r["node_id"]].varValue, 3) for r in self.tanks if
                        sp_t_plus[r["node_id"]].varValue is not None},
            depo_cikis={r["node_id"]: round(sp_t_minus[r["node_id"]].varValue, 3) for r in self.tanks if
                        sp_t_minus[r["node_id"]].varValue is not None},
            source_duals=source_duals,
            pump_duals=pump_duals,
            kesit_sabiti=kesit_sabiti,
        )

    def _dual_dogrulama_testi(self, w_bar, y_bar):
        alt_sonuc = self._build_sub_problem(w_bar, y_bar)
        print("Ana maliyet:", alt_sonuc.maliyet)
        2883505130
        for s in self.sources[:3]:  # ilk birkaç kaynakla test yeter
            s_id = s["node_id"]
            if w_bar[s_id] == 1:
                w_test = dict(w_bar)
                w_test[s_id] = 0
                alt_test = self._build_sub_problem(w_test, y_bar)
                gercek_fark = alt_sonuc.maliyet - alt_test.maliyet  # w=1 iken maliyet - w=0 iken maliyet
                tahmini_fark = alt_sonuc.source_duals[s_id] * 1  # w'yi 1 azaltmanın etkisi
                print(f"{s_id}: gerçek_fark(w:1->0 kapatınca kazanç)={gercek_fark:.2f} "
                      f"vs source_dual={alt_sonuc.source_duals[s_id]:.2f}")

    def run_benders(self, max_iterasyon: int = 10, epsilon: float = 1.0) -> BendersSonucu:
        """Pencereleme olmadan tam 24 saatlik Benders orkestrasyon döngüsü."""
        w_bar = {s["node_id"]: 0 for s in self.sources}
        for g in self.source_packages:
            for sid in g["member_source_ids"][:g["max_active"]]:
                if sid in self.src_id_to_node:
                    w_bar[self.src_id_to_node[sid]] = 1

        # 2. Pompaları grup limitlerini (max_active) aşmayacak şekilde aç
        y_bar = {p["node_id"]: 0 for p in self.pumps}
        for g in self.pump_groups:
            for pid in g["member_pump_ids"][:g["max_active"]]:
                if pid in self.pmp_id_to_node:
                    y_bar[self.pmp_id_to_node[pid]] = 1

        ub = float("inf")
        lb = -float("inf")
        en_iyi_w, en_iyi_y, en_iyi_alt_sonuc = None, None, None
        iter_kaydi = []

        print("\n" + "=" * 65)
        print(f" BENDERS OPTİMİZASYON DÖNGÜSÜ BAŞLATILDI (Sampler: {self.sampler})")
        print("=" * 65)

        self._dual_dogrulama_testi(w_bar, y_bar)

        for iterasyon in range(max_iterasyon):
            # 1. Alt Problemi (LP) Çöz
            alt_sonuc = self._build_sub_problem(w_val=w_bar, y_val=y_bar)
            if not alt_sonuc.feasible:
                print(f"[UYARI] İterasyon {iterasyon + 1}: Alt problem çözülemedi.")
                break

            # 2. Üst Sınır (UB) Güncellemesi
            sabit_maliyet = self.master_sabit_maliyeti(w_bar, y_bar)
            toplam_maliyet = sabit_maliyet + alt_sonuc.maliyet

            if toplam_maliyet < ub:
                ub = toplam_maliyet
                en_iyi_w = w_bar
                en_iyi_y = y_bar
                en_iyi_alt_sonuc = alt_sonuc

            # 3. Optimallik Kesitini Master Problema Ekle
            self.kesit_ekle(alt_sonuc)

            # 4. Ana Problemi (CQM) Çöz
            w_bar, y_bar, t_bar = self.coz_master()

            # 5. Alt Sınır (LB) Hesabı: Master Sabit + t_var
            yeni_sabit = self.master_sabit_maliyeti(w_bar, y_bar)
            lb = yeni_sabit + t_bar

            gap = ub - lb
            iter_kaydi.append({"iter": iterasyon + 1, "ub": round(ub, 2), "lb": round(lb, 2), "gap": round(gap, 2)})
            print(f" İter {iterasyon + 1:02d} | UB: {ub:12,.2f} TL | LB: {lb:12,.2f} TL | Gap: {gap:10,.2f}")

            # 6. Yakınsama Kriteri
            if gap <= epsilon or (ub > 0 and gap / ub <= 0.01):
                print(f"\n[OK] Yakınsama sağlandı (|UB - LB| <= {epsilon}).")
                break

        if en_iyi_w is None:
            raise RuntimeError("Benders döngüsü hiçbir iterasyonda olurlu çözüm üretemedi.")
        # self.sub_prob'u en iyi çözüme eşitlemek için son çalıştırma:
        self._build_sub_problem(w_val=en_iyi_w, y_val=en_iyi_y)

        return BendersSonucu(
            durum="Feasible" if abs(ub - lb) <= epsilon else "Infeasible",
            hedef_deger=ub,
            karar={
                "w": en_iyi_w,
                "y": en_iyi_y,
                "akislar": en_iyi_alt_sonuc.akislar,
                "kaynak_uretimi": en_iyi_alt_sonuc.kaynak_uretimi,
                "rezervuar": en_iyi_alt_sonuc.depo_hacimleri,
                "karsilanmayan": en_iyi_alt_sonuc.karsilanmayan,
            },
            metrikler={
                "iterasyon_sayisi": len(iter_kaydi),
                "en_iyi_ub": round(ub, 2),
                "son_lb": round(lb, 2),
                "final_gap": round(ub - lb, 2),
            },
            yakinsama_gecmisi=iter_kaydi
        )
    def extract_solution(self):
        status_code = self.sub_prob.status
        status_str = LpStatus[status_code]

        result = {
            "status": status_str,
            "objective_value": value(self.sub_prob.objective) if self.sub_prob.objective is not None else None,
            "flows": {},
            "sources": {},
            "unmet_demands": {},
            "tank_volumes": {},
            "tank_inflows": {},
            "tank_outflows": {},
            "tank_violations": {},
            "duals": {},
        }

        if status_str != "Optimal":
            return result

        var_dict = self.sub_prob.variablesDict()
        for name, var in var_dict.items():
            val = var.varValue if var.varValue is not None else 0.0
            if val < 1e-4:
                continue
            if name.startswith("x_"):
                result["flows"][name[2:]] = round(val, 3)
            elif name.startswith("q_"):
                result["sources"][name[2:]] = round(val, 3)
            elif name.startswith("u_"):
                result["unmet_demands"][name[2:]] = round(val, 3)
            elif name.startswith("V_"):
                result["tank_volumes"][name[2:]] = round(val, 3)
            elif name.startswith("t_plus_"):
                result["tank_inflows"][name[7:]] = round(val, 3)
            elif name.startswith("t_minus_"):
                result["tank_outflows"][name[8:]] = round(val, 3)
            elif name.startswith("e_"):
                result["tank_violations"][name] = round(val, 3)

        for c_name, constraint in self.sub_prob.constraints.items():
            if constraint.pi is not None and abs(constraint.pi) > 1e-5:
                result["duals"][c_name] = constraint.pi

        return result

    def print_summary(self):
        sol = self.extract_solution()
        print("\n" + "=" * 45)
        print(f" DURUM: {sol['status']}")
        if sol["objective_value"] is not None:
            print(f" TOPLAM ALT PROBLEM MALİYETİ: {sol['objective_value']:,.2f} TL")
        print("=" * 45)

        if sol["status"] == "Optimal":
            total_prod = sum(sol["sources"].values())
            total_unmet = sum(sol["unmet_demands"].values())
            print(f" Toplam Su Üretimi    : {total_prod:,.2f} m3/sa")
            print(f" Karşılanamayan Talep : {total_unmet:,.2f} m3/sa")
            print(f" Aktif Hat Sayısı      : {len(sol['flows'])}")
            print(f" Kısıntılı Bölge Sayısı: {len(sol['unmet_demands'])}")
            print(f" Seviye İhlali Depolar : {len(sol['tank_violations'])}")
            print("=" * 45 + "\n")

    def export_submission_files(self, w_val, y_val, output_dir="problem_02"):
        os.makedirs(output_dir, exist_ok=True)
        var_dict = self.sub_prob.variablesDict()

        def get_val(var_name):
            v = var_dict.get(var_name)
            return round(v.varValue, 4) if (v is not None and v.varValue is not None) else 0.0

        with open(os.path.join(output_dir, "source_dispatch.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source_id", "production_m3h", "is_active"])
            for s in self.sources:
                writer.writerow([s["source_id"], get_val(f"q_{s['node_id']}"), int(w_val.get(s["node_id"], 0))])

        with open(os.path.join(output_dir, "pump_operation.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pump_id", "is_on"])
            for p in self.pumps:
                writer.writerow([p["pump_id"], int(y_val.get(p["node_id"], 0))])

        with open(os.path.join(output_dir, "arc_flows.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["arc_id", "flow_m3h"])
            for a in self.network_arcs:
                writer.writerow([a["arc_id"], get_val(f"x_{a['arc_id']}")])

        with open(os.path.join(output_dir, "tank_levels.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tank_id", "fill_rate_m3h", "discharge_rate_m3h", "end_volume_m3",
                "min_level_shortfall_m3", "critical_level_shortfall_m3", "depletion_vs_initial_m3",
            ])
            for r in self.tanks:
                node_id = r["node_id"]
                writer.writerow([
                    r["tank_id"], get_val(f"t_plus_{node_id}"), get_val(f"t_minus_{node_id}"),
                    get_val(f"V_{node_id}"), get_val(f"e_min_{node_id}"), get_val(f"e_kri_{node_id}"),
                    get_val(f"e_tuk_{node_id}"),
                ])

        with open(os.path.join(output_dir, "demand_service.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["zone_id", "unmet_m3h"])
            for z in self.zones:
                writer.writerow([z["zone_id"], get_val(f"u_{z['zone_id']}")])

        obj_val = value(self.sub_prob.objective) if self.sub_prob.objective is not None else 0.0
        summary = {
            "problem_id": "problem_02",
            "status": LpStatus[self.sub_prob.status],
            "objective_value": round(obj_val, 2),
        }
        with open(os.path.join(output_dir, "solution_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[OK] Teslimat dosyaları '{output_dir}/' klasörüne başarıyla yazıldı.")


def main():
    # Yerel prova için sampler="local", D-Wave API için sampler="hybrid"
    hbd = HBD(sampler="hybrid")

    sonuc = hbd.run_benders(max_iterasyon=30, epsilon=1.0)

    # 1. Raporlama ve Özet
    toplam_talep = sum(d["total_demand_m3h"] for d in hbd.demands)
    toplam_karsilanmayan = sum(sonuc.karar["karsilanmayan"].values())
    toplam_saglanan = toplam_talep - toplam_karsilanmayan
    karsilanma_orani = (toplam_saglanan / toplam_talep) * 100 if toplam_talep > 0 else 0

    print("\n" + "=" * 45)
    print(" NİHAİ BENDERS SONUÇLARI")
    print("=" * 45)
    print(f" Durum                 : {sonuc.durum}")
    print(f" Toplam Hedef Değer    : {sonuc.hedef_deger:,.2f} TL")
    print(f" Toplam Beklenen Talep : {toplam_talep:,.2f} m³")
    print(f" Karşılanamayan Talep  : {toplam_karsilanmayan:,.2f} m³")
    print(f" Talep Karşılama Oranı : %{karsilanma_orani:.2f}")
    print(f" İterasyon Sayısı      : {sonuc.metrikler['iterasyon_sayisi']}")
    print(f" Final Gap             : {sonuc.metrikler['final_gap']:,.2f}")
    print("=" * 45)

    # 2. Çıktıları Diske Yaz
    hbd.export_submission_files(w_val=sonuc.karar["w"], y_val=sonuc.karar["y"], output_dir="solution")


import math


def is_feasible(
        x: dict, q: dict, w: dict, y: dict,
        t_plus: dict, t_minus: dict, V: dict, u: dict,
        e_min: dict, e_kri: dict, e_tuk: dict,
) -> bool:
    print("is_feasible basliyor")
    T = 24
    sigma_hed = 1.0
    network_nodes = open_json("network_nodes.json")["nodes"]
    network_arcs = open_json("network_arcs.json")["arcs"]

    sources_raw = open_json("sources.json")
    sources = sources_raw["sources"]
    source_packages = sources_raw["source_packages"]

    pumps_raw = open_json("pumps.json")
    pumps = pumps_raw["pumps"]
    pump_groups = pumps_raw["pump_groups"]

    tanks = open_json("tanks.json")["tanks"]
    zones = open_json("zones.json")["zones"]
    demands = open_json("demands.json")["demands"]
    feasible = True

    def viol(msg: str):
        nonlocal feasible
        feasible = False
        print(f"[IHLAL] {msg}")

    def warn(msg: str):
        print(f"[UYARI] {msg}")

    def isnum(v) -> bool:
        if v is None:
            return False
        try:
            v = float(v)
        except (TypeError, ValueError):
            return False
        return not (math.isnan(v) or math.isinf(v))

    def is_binary(v) -> bool:
        return isnum(v) and (v == 0 or v == 1)

    demand_by_zone = {d["zone_id"]: d["total_demand_m3h"] for d in demands}

    print("dumen kontrolden basariyla gecti")
    # --------------------------------------------------------
    # K1, K2 - kaynak kapasitesi / aktifse minimum uretim
    # --------------------------------------------------------
    for s in sources:
        nid = s["node_id"]
        q_bar = s.get("capacity_m3h")
        q_min = s.get("min_flow_if_active_m3h", 0)
        qs = q.get(nid)
        ws = w.get(nid)

        if qs.varValue > q_bar * ws:
            viol(f"K1: q[{nid}]={qs.varValue} > q_bar*w={q_bar * ws}")

        if qs.varValue < q_min * ws:
            viol(f"K2: q[{nid}]={qs.varValue} < q_min*w={q_min * ws}")

    # --------------------------------------------------------
    # K15 - kaynak paketi
    # --------------------------------------------------------
    for pkg in source_packages:
        pid = pkg["package_id"]
        max_active = pkg.get("max_active")
        members = pkg.get("member_source_ids", [])
        toplam = sum(w.get(sid[2:], 0) for sid in members)
        if max_active is not None and toplam > max_active:
            viol(f"K15: source package {pid} toplam aktif={toplam} > max_active={max_active}")

    # --------------------------------------------------------
    # K3 - ark kapasitesi
    # --------------------------------------------------------
    for a in network_arcs:
        aid = a["arc_id"]
        xa = x.get(aid)
        cap = a.get("capacity_m3h")

        if xa.varValue > cap:
            viol(f"K3: x[{aid}]={xa} > capacity={cap}")

    # --------------------------------------------------------
    # K4, K5 - pompa kapasitesi / acikken minimum debi
    # --------------------------------------------------------
    for pmp in pumps:
        pid = pmp["node_id"]
        f_bar = pmp.get("capacity_m3h")
        f_min = pmp.get("min_flow_if_on", 0)
        yp = y[pid]
        out_arc_ids = pmp.get("out_arc_ids", [])

        total_flow = 0.0
        for aid in out_arc_ids:
            xa = x[aid]
            total_flow += xa.varValue

        if round(total_flow, 2) > f_bar * yp:
            viol(f"K4: pump {pid} toplam debi={total_flow} > f_bar*y={f_bar * yp}")
        if round(total_flow, 2) < f_min * yp:
            viol(f"K5: pump {pid} toplam debi={total_flow} < f_min*y={f_min * yp}")

    # --------------------------------------------------------
    # K14 - pompa grubu
    # --------------------------------------------------------
    for g in pump_groups:
        gid = g["group_id"]
        max_active = g.get("max_active")
        members = g.get("member_pump_ids", [])
        toplam = sum(y[pid[2:]] for pid in members)
        if toplam > max_active:
            viol(f"K14: pump group {gid} toplam aktif={toplam} > max_active={max_active}")

    # --------------------------------------------------------
    # K6 - dugum akis korunumu
    # gel(n)/cik(n) network_nodes.json'da yok -> network_arcs'tan burada kuruluyor
    # --------------------------------------------------------
    incoming = {}
    outgoing = {}
    for a in network_arcs:
        outgoing.setdefault(a.get("from_node"), []).append(a["arc_id"])
        incoming.setdefault(a.get("to_node"), []).append(a["arc_id"])

    source_by_node = {s["node_id"]: s["source_id"] for s in sources}
    tank_by_node = {t["node_id"]: t["tank_id"] for t in tanks}
    zone_by_node = {z["node_id"]: z["zone_id"] for z in zones}

    for n in network_nodes:
        nid = n["node_id"]
        gelen = sum(x[aid].varValue for aid in incoming.get(nid, []))
        cikan = sum(x[aid].varValue for aid in outgoing.get(nid, []))

        kaynak_enj = 0.0
        depo_cekim = 0.0
        depo_dolum = 0.0
        karsilanan_talep = 0.0

        if nid in source_by_node:
            kaynak_enj = q[nid].varValue

        if nid in tank_by_node:
            depo_cekim = t_minus[nid].varValue
            depo_dolum = t_plus[nid].varValue

        if nid in zone_by_node:
            dz = demand_by_zone[nid]
            uz = u[nid].varValue
            karsilanan_talep = dz - uz

        lhs = round(gelen + kaynak_enj + depo_cekim, 2)
        rhs = round(cikan + depo_dolum + karsilanan_talep, 2)

        if abs(lhs - rhs) > 1e-6:
            viol(f"K6: node {nid} akis dengesi saglanmiyor. lhs={lhs}, rhs={rhs}, fark={lhs - rhs}")

    # --------------------------------------------------------
    # K7-K10 - depo stok dengesi
    # --------------------------------------------------------
    for r in tanks:
        rid = r["node_id"]
        V0 = r.get("initial_volume_m3")
        V_bar = r.get("capacity_m3")
        Vr = V[rid].varValue
        tp = t_plus[rid].varValue
        tm = t_minus[rid].varValue

        beklenen_V = V0 + T * (tp - tm)
        if abs(Vr - beklenen_V) > 1e-4:
            viol(f"K7: tank {rid} V={Vr} != V0+T*(t+-t-)={beklenen_V}")

        if Vr > V_bar:
            viol(f"K8: tank {rid} V={Vr} > capacity={V_bar}")

        if T * tm > V0:
            viol(f"K9: tank {rid} T*t-={T * tm} > V0={V0}")

        if T * tp > V_bar:
            viol(f"K10: tank {rid} T*t+={T * tp} > capacity={V_bar}")

    # --------------------------------------------------------
    # K11, K12, K13 - depo seviye kisitlari (yumusak)
    # --------------------------------------------------------
    for r in tanks:
        rid = r["node_id"]
        Vr = V[rid].varValue
        V0 = r.get("initial_volume_m3")
        V_min = r.get("min_level_m3")
        V_kri = r.get("critical_level_m3")
        if Vr is None:
            continue

        em = e_min[rid].varValue
        ek = e_kri[rid].varValue
        et = e_tuk[rid].varValue

        if round(Vr + em, 2) < V_min:
            viol(f"K11: tank {rid} V+e_min={Vr + em} < V_min={V_min}")
        if round(Vr + ek, 2) < V_kri:
            viol(f"K12: tank {rid} V+e_kri={Vr + ek} < V_kri={V_kri}")
        if round(Vr + et, 2) < sigma_hed * V0:
            viol(f"K13: tank {rid} V+e_tuk={Vr + et} < sigma_hed*V0={sigma_hed * V0}")

    # --------------------------------------------------------
    # K16 - talep karsilama
    # --------------------------------------------------------
    for z in zones:
        zid = z["node_id"]
        dz = demand_by_zone[zid]
        uz = u[zid].varValue
        if uz > dz:
            viol(f"K16: u[{zid}]={uz} > demand={dz}")

    # --------------------------------------------------------
    # ÖZET VE PERFORMANS HESAPLAMALARI (Sadece Print)
    # --------------------------------------------------------
    total_demand_m3 = sum(demand_by_zone.values()) * T
    total_unmet_m3 = sum(u[z["node_id"]].varValue for z in zones) * T
    total_served_m3 = total_demand_m3 - total_unmet_m3
    service_rate = (total_served_m3 / total_demand_m3 * 100.0) if total_demand_m3 > 0 else 100.0

    cost_u = total_unmet_m3 * 25000
    cost_emin = sum(e_min[r["node_id"]].varValue for r in tanks) * 100
    cost_ekri = sum(e_kri[r["node_id"]].varValue for r in tanks) * 800
    cost_etuk = sum(e_tuk[r["node_id"]].varValue for r in tanks) * 25
    total_penalties = cost_u + cost_emin + cost_ekri + cost_etuk

    print("\n" + "=" * 50)
    print("        KUANTUM / LP ÇÖZÜM ANALİZ ÖZETİ        ")
    print("=" * 50)
    print(f" ÇÖZÜM FİZİBİL Mİ?           : {'EVET' if feasible else 'HAYIR (0 PUAN)'}")
    print(f" TALEP KARŞILAMA ORANI (%)   : %{service_rate:.2f}")
    print(f" Toplam Talep                : {total_demand_m3:,.2f} m3")
    print(f" Karşılanamayan Talep (u)    : {total_unmet_m3:,.2f} m3")
    print("-" * 50)
    print(f" TOPLAM CEZA MALİYETİ        : {total_penalties:,.2f} TL")
    print(f"   ├─ Talep Kısıntı Cezası   : {cost_u:,.2f} TL")
    print(f"   ├─ Min Seviye İhlal Cezası: {cost_emin:,.2f} TL")
    print(f"   ├─ Kritik Seviye Cezası   : {cost_ekri:,.2f} TL")
    print(f"   └─ Dönem Sonu Hedef Cezası: {cost_etuk:,.2f} TL")
    print("=" * 50 + "\n")

    return feasible

if __name__ == "__main__":
    main()
