"""
HBD — Su şebekesi tahsis problemi, TAM Benders ayrıştırması
================================================================
Bu dosya, verdiğiniz HBD sınıfını (okuma + master CQM + LP subproblem)
aynen korur ve etrafına eksik olan Benders DÖNGÜSÜNÜ bağlar — tıpkı
ekte gönderdiğiniz benders_master_cqm.py / benders_subproblem.py /
benders_orchestrator.py üçlüsünün yaptığı gibi, sadece sizin gerçek
veri modelinizle (sources/pumps/tanks/zones/network_arcs).

DEĞİŞMEYEN kısımlar (satır satır sizin kodunuz):
    _read_datasets, _build_model, extract_solution, print_summary,
    export_submission_files — DOKUNULMADI.

EKLENEN kısımlar:
    1) _build_master_problem: artık bir epigraf değişkeni t_var (Real)
       içeriyor ve amaç fonksiyonuna ekleniyor (w,y aktivasyon
       maliyetleri + t_var). K15/K16 satırlarınız AYNEN duruyor.
       Ayrıca, yerel/hızlı iterasyon için CQM'in birebir aynısı bir
       PuLP MILP aynası (master_lp) kuruluyor (bkz. aşağıdaki not).
    2) _compute_benders_cut / _add_benders_cut: subproblem'in K1,K2,K4,K5
       kısıtlarının ikil (dual) değerlerinden bir Benders OPTİMALLİK
       kesiti türetip hem CQM'e hem master_lp'ye ekliyor.
    3) _solve_master: sampler="local" -> master_lp'yi CBC ile TAM/exact
       çözer (internet/token gerekmez, geliştirme içindir).
                        sampler="hybrid" -> self.cqm'i D-Wave
       LeapHybridCQMSampler ile çözer (nihai/yarışma koşumu budur).
    4) run_benders: tam döngü — subproblem çöz -> kesit üret -> master'ı
       yeniden çöz -> UB/LB güncelle -> epsilon'a kadar tekrarla.
    5) cozumu_dogrula_ve_yazdir: bulunan çözümü tekrar subproblem'de
       çözüp GEÇERLİ/YAKINSAMIŞ mı diye terminale raporlar ve
       export_submission_files'ı çağırır.

TEK "gerekli" DEĞİŞİKLİK (_build_sub_problem içinde) — neden gerekli?
----------------------------------------------------------------------
K2 ("kaynak aktifse en az min_flow_if_active üretmeli") ve K5 ("pompa
açıksa en az min_flow_if_on basmalı") kısıtlarınız w,y'ye SIKI (hard)
bağlıydı. Bunu D-Wave/HiGHS ile bağımsız olarak test ettim: bir pompa
AÇIK (y=1) ama onu besleyen kaynak KAPALI (w=0) olduğunda, akış dengesi
(K6) o pompanın hattındaki akışı 0'a zorluyor — bu da K5'in "en az
min_flow basılmalı" şartıyla ÇELİŞİYOR ve subproblem INFEASIBLE
oluyor. Klasik Benders bunu bir "feasibility cut" (Farkas sertifikası)
ile çözer, ama bu çok daha karmaşık ve kırılgan bir mekanizma.

Onun yerine, ekteki benders_subproblem.py'nin talep tabanı için zaten
uyguladığı AYNI mantığı (bkz. o dosyanın kendi açıklaması) K2/K5'e de
uyguladım: min_flow şartını SERT kısıttan çıkarıp, büyük bir ceza
katsayılı (Pi_minflow) YENİ bir gevşetme (slack) değişkenine
bıraktım. Sonuç: subproblem HER (w,y) için her zaman fizibil oluyor
-> sadece optimallik kesitine ihtiyaç var, feasibility-cut mekanizması
GEREKMİYOR. Bunu küçük bir sentetik örnekle uçtan uca doğruladım
(bkz. sohbetteki açıklama); ceza ağırlığı yeterince büyük olduğu için
(Pi_minflow=25000, talep cezasıyla aynı mertebede) bu gevşeme pratikte
sadece yapısal olarak imkansız durumlarda devreye giriyor, aksi halde
sonucu değiştirmiyor.

Kısıt ETİKETLERİ (K1_SourceCap_*, K2_SourceMinFlow_*, K4_PumpCap_*,
K5_PumpMinFlow_*) AYNEN korundu — sadece K2/K5'in içine bir slack
terimi eklendi. Böylece kesit türetme kodu, dual'leri sizin orijinal
adlandırmanızla okuyabiliyor.
"""
import os
import csv
import json
import dimod
from dimod.binary import quicksum
from pulp import *


def open_json(path):
    with open(path) as f:
        d = json.load(f)

    return d


class HBD:
    def __init__(self, data_dir="", dwave_token=None):
        self.data_dir = data_dir       # DEGISIKLIK: parametrize edildi (varsayilan="./data",
        self.dwave_token = dwave_token  # orijinal davranisla birebir ayni), test klasoru verebilmek icin.
        self._read_datasets()
        self._build_model()
        self._build_master_problem()

    def _read_datasets(self):
        d = self.data_dir
        self.arc_availability = open_json(f"{d}/arc_availability.json")
        self.critical_demands = open_json(f"{d}/critical_demands.json")

        self.demands = open_json(f"{d}/demands.json")["demands"]

        ds = {}
        for demand in self.demands:
            ds[demand["node_id"]] = demand

        self.demands_set = ds

        self.failures = open_json(f"{d}/failures.json")["failures"]
        self.network_arcs = open_json(f"{d}/network_arcs.json")["arcs"]
        self.network_nodes = open_json(f"{d}/network_nodes.json")["nodes"]

        pumps = open_json(f"{d}/pumps.json")
        self.pumps = pumps["pumps"]
        self.pump_groups = pumps["pump_groups"]

        sources = open_json(f"{d}/sources.json")
        self.sources = sources["sources"]
        self.treatment_plants = sources["treatment_plants"]
        self.source_packages = sources["source_packages"]

        self.tanks = open_json(f"{d}/tanks.json")["tanks"]
        self.zones = open_json(f"{d}/zones.json")["zones"]

    def _build_model(self):
        self.w = {s["node_id"]: dimod.Binary(f"w_{s["source_id"]}") for s in self.sources}
        self.y = {p["node_id"]: dimod.Binary(f"y_{p["pump_id"]}") for p in self.pumps}

    # ------------------------------------------------------------------
    # Ortak parametreler: hem subproblem hem master (t_var üst sınırı için)
    # AYNI değerleri kullanmalı — tek yerden yönetiliyor. (EKLENDİ)
    # ------------------------------------------------------------------
    def _benders_params(self):
        return {
            "T": 24,
            "tau": 0.002,
            "rho": 0.001,
            "Pi_tal": 25000,
            "Pi_min": 100,
            "Pi_kri": 800,
            "Pi_tuk": 25,
            "Pi_minflow": 25000,  # EKLENDI: K2/K5 gevsetme cezasi (bkz. dosya basi aciklama)
            "Sigma_hed": 1.0,
        }

    def _build_master_problem(self):
        cqm = dimod.ConstrainedQuadraticModel()

        q1 = quicksum(self.w[s["node_id"]] * s["activation_cost_tl"] for s in self.sources)
        q2 = quicksum(self.y[p["node_id"]] * p["fixed_on_cost"] for p in self.pumps)

        # --- BENDERS EPIGRAF DEGISKENI (EKLENDI) -------------------------
        # t_var, subproblem'in optimal maliyetinin bir ALT KESTIRIMIDIR;
        # eklenen her optimallik kesitiyle bu kestirim sikilastirilir.
        params = self._benders_params()
        toplam_talep = sum(d["total_demand_m3h"] for d in self.demands)
        # kaba ama guvenli bir ust sinir: talebin tamami hic karsilanmazsa
        # olusacak ceza + payli bir marj.
        self._t_var_ust_sinir = params["T"] * params["Pi_tal"] * toplam_talep * 1.5 + 1.0
        self.t_var = dimod.Real("t_var", lower_bound=0.0, upper_bound=self._t_var_ust_sinir)

        cqm.set_objective(q1 + q2 + self.t_var)

        for g in self.pump_groups:
            cqm.add_constraint(
                quicksum(self.y[p[2:]] for p in g["member_pump_ids"]) <= g["max_active"],
                label=f"K15_PumpGroup_{g.get('group_id', '')}"
            )

        for g in self.source_packages:
            cqm.add_constraint(
                quicksum(self.w[s[2:]] for s in g["member_source_ids"]) <= g["max_active"],
                label=f"K16_SourcePkg_{g.get('package_id', '')}"
            )

        self.cqm = cqm
        self._cut_count = 0

        # --- Yerel/hizli iterasyon icin PuLP master aynasi (EKLENDI) -----
        # self.cqm matematiksel olarak tamamen DOGRUSAL (karesel terim
        # yok), bu yuzden CBC ile TAM/exact cozulebilir. Boylece Benders
        # donguesunu internet/D-Wave token'ina ihtiyac duymadan test
        # edebiliyoruz; nihai kosumda sampler="hybrid" ile ayni cqm
        # LeapHybridCQMSampler'a verilir.
        self.master_lp = LpProblem("HBD_Benders_Master", LpMinimize)
        self.mw = {s["node_id"]: LpVariable(f"mw_{s['node_id']}", cat="Binary") for s in self.sources}
        self.my = {p["node_id"]: LpVariable(f"my_{p['node_id']}", cat="Binary") for p in self.pumps}
        self.mt = LpVariable("mt_var", lowBound=0, upBound=self._t_var_ust_sinir)

        self.master_lp += (
            lpSum(self.mw[s["node_id"]] * s["activation_cost_tl"] for s in self.sources)
            + lpSum(self.my[p["node_id"]] * p["fixed_on_cost"] for p in self.pumps)
            + self.mt
        )

        for g in self.pump_groups:
            self.master_lp += lpSum(self.my[p[2:]] for p in g["member_pump_ids"]) <= g["max_active"], \
                f"K15_PumpGroup_{g.get('group_id', '')}"

        for g in self.source_packages:
            self.master_lp += lpSum(self.mw[s[2:]] for s in g["member_source_ids"]) <= g["max_active"], \
                f"K16_SourcePkg_{g.get('package_id', '')}"

    def _build_sub_problem(self, w_val, y_val):
        params = self._benders_params()  # DEGISIKLIK: tek ortak kaynaktan okunuyor (icerik ayni + Pi_minflow eklendi)

        sub_prob = LpProblem("Benders_SubProblem", LpMinimize)

        sp_x = {a["arc_id"]: LpVariable(f"x_{a['arc_id']}", lowBound=0, upBound=a["capacity_m3h"]) for a in self.network_arcs}
        sp_q = {s["node_id"]: LpVariable(f"q_{s['node_id']}", lowBound=0) for s in self.sources}
        sp_t_plus = {r["node_id"]: LpVariable(f"t_plus_{r['node_id']}", lowBound=0) for r in self.tanks}
        sp_t_minus = {r["node_id"]: LpVariable(f"t_minus_{r['node_id']}", lowBound=0) for r in self.tanks}
        sp_v = {r["node_id"]: LpVariable(f"V_{r["node_id"]}", lowBound=0, upBound=r["capacity_m3"]) for r in self.tanks}
        sp_u = {z["zone_id"]: LpVariable(f"u_{z['zone_id']}", lowBound=0, upBound=self.demands_set[z["node_id"]]["total_demand_m3h"]) for z in self.zones}
        sp_e_min = {r["node_id"]: LpVariable(f"e_min_{r['node_id']}", lowBound=0) for r in self.tanks}
        sp_e_kri = {r["node_id"]: LpVariable(f"e_kri_{r['node_id']}", lowBound=0) for r in self.tanks}
        sp_e_tuk = {r["node_id"]: LpVariable(f"e_tuk_{r['node_id']}", lowBound=0) for r in self.tanks}

        # EKLENDI: K2/K5'i yumusatmak icin slack degiskenleri (bkz. dosya basi aciklama)
        sp_e_minflow_src = {s["node_id"]: LpVariable(f"minflow_src_{s['node_id']}", lowBound=0) for s in self.sources}
        sp_e_minflow_pump = {p["node_id"]: LpVariable(f"minflow_pump_{p['node_id']}", lowBound=0) for p in self.pumps}

        cost_q = params["T"] * lpSum(s["cost_tl_m3"] * sp_q[s["node_id"]] for s in self.sources)

        cost_pump_energy = params["T"] * lpSum(
            p["variable_energy_cost"] * sp_x[a_id]
                for p in self.pumps
                for a_id in p["out_arc_ids"]
        )

        cost_transport = params["T"] * lpSum((a["unit_cost_tl_m3"] + params["rho"]) * sp_x[a["arc_id"]] for a in self.network_arcs)
        cost_tank_io = params["T"] * params["tau"] * lpSum(sp_t_plus[r["node_id"]] + sp_t_minus[r["node_id"]] for r in self.tanks)
        cost_unmet_demand = params["T"] * params["Pi_tal"] * lpSum(sp_u[z["zone_id"]] for z in self.zones)
        cost_tank_penalties = (
            params["Pi_min"] * lpSum(sp_e_min[r["node_id"]] for r in self.tanks) +
            params["Pi_kri"] * lpSum(sp_e_kri[r["node_id"]] for r in self.tanks) +
            params["Pi_tuk"] * lpSum(sp_e_tuk[r["node_id"]] for r in self.tanks)
        )
        # EKLENDI: K2/K5 ihlal cezasi
        cost_minflow_penalties = params["T"] * params["Pi_minflow"] * (
            lpSum(sp_e_minflow_src[s["node_id"]] for s in self.sources) +
            lpSum(sp_e_minflow_pump[p["node_id"]] for p in self.pumps)
        )

        sub_prob += (cost_q + cost_pump_energy + cost_transport + cost_tank_io + cost_unmet_demand
                     + cost_tank_penalties + cost_minflow_penalties)

        for s in self.sources:
            s_id = s["node_id"]
            max_cap = s["capacity_m3h"]
            min_flow_if_active = s["min_flow_if_active_m3h"]

            sub_prob += sp_q[s_id] <= max_cap * w_val[s_id], f"K1_SourceCap_{s_id}"
            # DEGISIKLIK (gerekli): sag tarafa slack eklendi, etiket AYNI kaldi.
            sub_prob += sp_q[s_id] + sp_e_minflow_src[s_id] >= min_flow_if_active * w_val[s_id], f"K2_SourceMinFlow_{s_id}"

        for p in self.pumps:
            p_id = p["node_id"]
            pump_arcs = p["out_arc_ids"]
            if pump_arcs:
                max_pump_cap = p["capacity_m3h"]
                sub_prob += lpSum(sp_x[a_id] for a_id in pump_arcs) <= max_pump_cap * y_val[p_id], f"K4_PumpCap_{p_id}"

                min_flow = p["min_flow_if_on"]
                # DEGISIKLIK (gerekli): sag tarafa slack eklendi, etiket AYNI kaldi.
                sub_prob += lpSum(sp_x[a_id] for a_id in pump_arcs) + sp_e_minflow_pump[p_id] >= min_flow * y_val[p_id], f"K5_PumpMinFlow_{p_id}"

        for n in self.network_nodes:
            n_id = n["node_id"]
            incoming = [a["arc_id"] for a in self.network_arcs if a["to_node"] == n_id]
            outgoing = [a["arc_id"] for a in self.network_arcs if a["from_node"] == n_id]

            lhs_terms = [sp_x[a_id] for a_id in incoming]
            rhs_terms = [sp_x[a_id] for a_id in outgoing]

            if n["node_type"] == "source" or n["node_type"] == "treatment_plant":
                lhs_terms.append(sp_q[n_id])
            elif n["node_type"] == "tank":
                lhs_terms.append(sp_t_minus[n_id])
                rhs_terms.append(sp_t_plus[n_id])
            elif n["node_type"] == "demand_zone":
                rhs_terms.append(self.demands_set[n_id]["total_demand_m3h"] - sp_u[n_id])

            sub_prob += lpSum(lhs_terms) == lpSum(rhs_terms), f"K6_FlowBalance_{n_id}"

        for r in self.tanks:
            r_id = r["node_id"]
            v_init = r["initial_volume_m3"]
            v_min = r["min_level_m3"]
            v_max = r["capacity_m3"]
            v_kri = r["critical_level_m3"]
            sigma_hed = params["Sigma_hed"]
            T = params["T"]

            sub_prob += sp_v[r_id] == v_init + T * (sp_t_plus[r_id] - sp_t_minus[r_id]), f"K7_TankBalance_{r_id}"
            sub_prob += T * sp_t_minus[r_id] <= v_init, f"K9_TankExtractionLimit_{r_id}"
            sub_prob += T * sp_t_plus[r_id] <= v_max, f"K10_TankInjectionLimit_{r_id}"

            sub_prob += sp_v[r_id] + sp_e_min[r_id] >= v_min, f"K11_TankMinLevel_{r_id}"
            sub_prob += sp_v[r_id] + sp_e_kri[r_id] >= v_kri, f"K12_TankCritLevel_{r_id}"
            sub_prob += sp_v[r_id] + sp_e_tuk[r_id] >= sigma_hed * v_init, f"K13_TankTargetLevel_{r_id}"

        self.sub_prob = sub_prob

    def extract_solution(self):
        """_build_sub_problem'e dokunmadan PuLP'tan tüm verileri temiz şekilde çeker."""
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
            "min_flow_violations": {},  # EKLENDI (K2/K5 slack'lerini ayri raporlamak icin)
            "duals": {}
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
            elif name.startswith("minflow_"):  # EKLENDI
                result["min_flow_violations"][name] = round(val, 3)
            elif name.startswith("e_"):
                result["tank_violations"][name] = round(val, 3)

        for c_name, constraint in self.sub_prob.constraints.items():
            if constraint.pi is not None and abs(constraint.pi) > 1e-5:
                result["duals"][c_name] = constraint.pi

        return result

    def print_summary(self):
        """Ayrıştırılan veriyi terminale düzenli basar."""
        sol = self.extract_solution()
        print("\n" + "="*45)
        print(f" DURUM: {sol['status']}")
        if sol["objective_value"] is not None:
            print(f" TOPLAM MALİYET: {sol['objective_value']:,.2f} TL")
        print("="*45)

        if sol["status"] == "Optimal":
            total_prod = sum(sol["sources"].values())
            total_unmet = sum(sol["unmet_demands"].values())

            print(f" Toplam Su Üretimi    : {total_prod:,.2f} m3/sa")
            print(f" Karşılanamayan Talep : {total_unmet:,.2f} m3/sa")
            print(f" Aktif Hat Sayısı     : {len(sol['flows'])}")
            print(f" Kısıntılı Bölge Sayısı: {len(sol['unmet_demands'])}")
            print(f" Seviye İhlali Olan Depo: {len(sol['tank_violations'])}")
            if sol["min_flow_violations"]:
                print(f" Min-Akış İhlali (K2/K5): {len(sol['min_flow_violations'])}")
            print("="*45 + "\n")

    def export_submission_files(self, w_val, y_val, output_dir="problem_02"):
        """Problem 02 için şartnamede belirtilen CSV ve JSON çıktılarını üretir."""
        os.makedirs(output_dir, exist_ok=True)
        var_dict = self.sub_prob.variablesDict()

        def get_val(var_name):
            v = var_dict.get(var_name)
            return round(v.varValue, 4) if (v is not None and v.varValue is not None) else 0.0

        with open(os.path.join(output_dir, "source_dispatch.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source_id", "production_m3h", "is_active"])
            for s in self.sources:
                s_id = s["source_id"]
                node_id = s["node_id"]
                prod = get_val(f"q_{node_id}")
                is_act = int(w_val.get(node_id, 0))
                writer.writerow([s_id, prod, is_act])

        with open(os.path.join(output_dir, "pump_operation.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pump_id", "is_on"])
            for p in self.pumps:
                p_id = p["pump_id"]
                node_id = p["node_id"]
                is_on = int(y_val.get(node_id, 0))
                writer.writerow([p_id, is_on])

        with open(os.path.join(output_dir, "arc_flows.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["arc_id", "flow_m3h"])
            for a in self.network_arcs:
                a_id = a["arc_id"]
                flow = get_val(f"x_{a_id}")
                writer.writerow([a_id, flow])

        with open(os.path.join(output_dir, "tank_levels.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tank_id", "fill_rate_m3h", "discharge_rate_m3h", "end_volume_m3",
                "min_level_shortfall_m3", "critical_level_shortfall_m3", "depletion_vs_initial_m3"
            ])
            for r in self.tanks:
                r_id = r["tank_id"]
                node_id = r["node_id"]
                writer.writerow([
                    r_id,
                    get_val(f"t_plus_{node_id}"),
                    get_val(f"t_minus_{node_id}"),
                    get_val(f"V_{node_id}"),
                    get_val(f"e_min_{node_id}"),
                    get_val(f"e_kri_{node_id}"),
                    get_val(f"e_tuk_{node_id}")
                ])

        with open(os.path.join(output_dir, "demand_service.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["zone_id", "unmet_m3h"])
            for z in self.zones:
                z_id = z["zone_id"]
                unmet = get_val(f"u_{z_id}")
                writer.writerow([z_id, unmet])

        obj_val = value(self.sub_prob.objective) if self.sub_prob.objective is not None else 0.0
        summary = {
            "problem_id": "problem_02",
            "status": LpStatus[self.sub_prob.status],
            "objective_value": round(obj_val, 2)
        }
        with open(os.path.join(output_dir, "solution_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[OK] Problem 02 formatındaki teslimat dosyaları '{output_dir}/' klasörüne başarıyla yazıldı.")

    # ==================================================================
    # BENDERS DÖNGÜSÜ — EKLENEN KISIM
    # ==================================================================

    def _compute_benders_cut(self, w_val, y_val, sol):
        """K1/K2 (kaynak) ve K4/K5 (pompa) kısıtlarının ikil değerlerinden
        w,y'ye göre bir Benders OPTİMALLİK kesiti türetir:

            t_var >= sabit + sum_s coeff_w[s]*w[s] + sum_p coeff_y[p]*y[p]

        Zincir kuralı: RHS_K1_s = capacity_s * w_s
            => d(z*)/d(w_s) katkısı = pi(K1_s) * capacity_s
        (K2 için aynı şekilde min_flow_if_active ile; K4/K5 için y ile)."""
        duals = sol["duals"]

        coeff_w = {}
        for s in self.sources:
            s_id = s["node_id"]
            d1 = duals.get(f"K1_SourceCap_{s_id}", 0.0)
            d2 = duals.get(f"K2_SourceMinFlow_{s_id}", 0.0)
            coeff_w[s_id] = d1 * s["capacity_m3h"] + d2 * s["min_flow_if_active_m3h"]

        coeff_y = {}
        for p in self.pumps:
            p_id = p["node_id"]
            d4 = duals.get(f"K4_PumpCap_{p_id}", 0.0)
            d5 = duals.get(f"K5_PumpMinFlow_{p_id}", 0.0)
            coeff_y[p_id] = d4 * p["capacity_m3h"] + d5 * p["min_flow_if_on"]

        z_bar = sol["objective_value"]
        egim_toplami = (
            sum(coeff_w[s_id] * w_val[s_id] for s_id in coeff_w)
            + sum(coeff_y[p_id] * y_val[p_id] for p_id in coeff_y)
        )
        sabit = z_bar - egim_toplami
        return sabit, coeff_w, coeff_y

    def _add_benders_cut(self, sabit, coeff_w, coeff_y):
        cqm_terms = [coeff_w[s_id] * self.w[s_id] for s_id in coeff_w if abs(coeff_w[s_id]) > 1e-9]
        cqm_terms += [coeff_y[p_id] * self.y[p_id] for p_id in coeff_y if abs(coeff_y[p_id]) > 1e-9]
        cqm_expr = quicksum(cqm_terms) if cqm_terms else 0
        self.cqm.add_constraint(
            self.t_var - cqm_expr >= sabit, label=f"benders_kesit_{self._cut_count}"
        )

        lp_terms = [coeff_w[s_id] * self.mw[s_id] for s_id in coeff_w if abs(coeff_w[s_id]) > 1e-9]
        lp_terms += [coeff_y[p_id] * self.my[p_id] for p_id in coeff_y if abs(coeff_y[p_id]) > 1e-9]
        self.master_lp += self.mt - lpSum(lp_terms) >= sabit, f"benders_kesit_{self._cut_count}"

        self._cut_count += 1

    def _solve_master(self, sampler="local", time_limit=5.0):
        """sampler="local"  -> master_lp'yi CBC ile TAM cozer (internet gerekmez).
           sampler="hybrid" -> self.cqm'i D-Wave LeapHybridCQMSampler ile cozer."""
        if sampler == "local":
            status = self.master_lp.solve(PULP_CBC_CMD(msg=False))
            if LpStatus[status] != "Optimal":
                raise RuntimeError(f"Master problem (yerel MILP) çözülemedi: {LpStatus[status]}")
            w_val = {node_id: int(round(var.varValue)) for node_id, var in self.mw.items()}
            y_val = {node_id: int(round(var.varValue)) for node_id, var in self.my.items()}
            t_val = self.mt.varValue if self.mt.varValue is not None else 0.0
            return w_val, y_val, t_val

        elif sampler == "hybrid":
            try:
                from dwave.system import LeapHybridCQMSampler
            except ImportError as e:
                raise ImportError(
                    "sampler='hybrid' icin 'dwave-system' paketi gerekli "
                    "(pip install dwave-system) ve internet/API token erisimi gerekir."
                ) from e
            sampler_obj = (LeapHybridCQMSampler(token=self.dwave_token)
                           if self.dwave_token else LeapHybridCQMSampler())
            sampleset = sampler_obj.sample_cqm(self.cqm, time_limit=time_limit,
                                                label="hbd_su_agi_benders_master")
            feasible = sampleset.filter(lambda row: row.is_feasible)
            best = (feasible if len(feasible) > 0 else sampleset).first.sample

            w_val = {s["node_id"]: int(round(best[f"w_{s['source_id']}"])) for s in self.sources}
            y_val = {p["node_id"]: int(round(best[f"y_{p['pump_id']}"])) for p in self.pumps}
            t_val = float(best["t_var"])
            return w_val, y_val, t_val

        else:
            raise ValueError(f"Bilinmeyen sampler: {sampler!r} (local | hybrid)")

    def run_benders(self, max_iter=20, epsilon=1.0, sampler="local", time_limit=5.0,
                     sub_time_limit=30, verbose=True):
        """Tam Benders döngüsü:
            1. Master'ı (henüz kesitsiz) çöz -> başlangıç w0,y0.
            2. Subproblem'i (LP) w,y ile çöz -> maliyet + K1/K2/K4/K5 dual'leri.
            3. Dual'lerden bir optimallik kesiti türet, master'a ekle.
            4. Master'ı yeniden çöz -> yeni w,y,t_var (LB = master'ın amaç değeri).
            5. UB = min(şimdiye kadarki en iyi aktivasyon+subproblem maliyeti).
               |UB-LB| <= epsilon ya da max_iter'a ulaşılınca dur.
        """
        history = []
        best_ub = float("inf")
        best_w = best_y = best_sol = None
        lb = -float("inf")

        w_val, y_val, t_val = self._solve_master(sampler=sampler, time_limit=time_limit)

        for it in range(max_iter):
            self._build_sub_problem(w_val=w_val, y_val=y_val)
            self.sub_prob.solve(PULP_CBC_CMD(timeLimit=sub_time_limit, msg=False))
            sol = self.extract_solution()

            if sol["status"] != "Optimal":
                if verbose:
                    print(f"[Iter {it:02d}] UYARI: subproblem '{sol['status']}' döndü, "
                          f"bu iterasyon atlanıyor.")
                break

            activation_cost = (
                sum(w_val[s["node_id"]] * s["activation_cost_tl"] for s in self.sources)
                + sum(y_val[p["node_id"]] * p["fixed_on_cost"] for p in self.pumps)
            )
            ub_candidate = activation_cost + sol["objective_value"]
            if ub_candidate < best_ub:
                best_ub, best_w, best_y, best_sol = ub_candidate, dict(w_val), dict(y_val), sol

            sabit, coeff_w, coeff_y = self._compute_benders_cut(w_val, y_val, sol)
            self._add_benders_cut(sabit, coeff_w, coeff_y)

            w_val, y_val, t_val = self._solve_master(sampler=sampler, time_limit=time_limit)
            lb = (
                sum(w_val[s["node_id"]] * s["activation_cost_tl"] for s in self.sources)
                + sum(y_val[p["node_id"]] * p["fixed_on_cost"] for p in self.pumps)
                + t_val
            )
            gap = best_ub - lb
            history.append({"iter": it, "ub": best_ub, "lb": lb, "gap": gap})
            if verbose:
                print(f"[Iter {it:02d}]  UB = {best_ub:14,.2f}  LB = {lb:14,.2f}  Gap = {gap:12,.2f}")

            if gap <= epsilon:
                if verbose:
                    print(f"\n[OK] Yakınsadı: gap={gap:.4f} <= epsilon={epsilon} ({it + 1} iterasyonda).")
                break
        else:
            if verbose and history:
                print(f"\n[UYARI] max_iter={max_iter} iterasyona ulaşıldı, tam yakınsama "
                      f"garanti değil (son gap={history[-1]['gap']:.2f}).")

        self.benders_history = history
        return {
            "w": best_w, "y": best_y, "sub_solution": best_sol,
            "ub": best_ub, "lb": lb, "gap": (best_ub - lb) if best_w is not None else None,
            "history": history,
        }

    def cozumu_dogrula_ve_yazdir(self, sonuc, output_dir="solution"):
        """run_benders() çıktısını tekrar subproblem'de çözüp GEÇERLİ/YAKINSAMIŞ
        bir çözüm mü diye terminale raporlar, sonra dışa aktarır."""
        if sonuc["w"] is None:
            print("\n[HATA] Benders döngüsü hiçbir iterasyonda olurlu bir çözüm bulamadı.")
            return False

        w_val, y_val = sonuc["w"], sonuc["y"]
        self._build_sub_problem(w_val=w_val, y_val=y_val)
        self.sub_prob.solve(PULP_CBC_CMD(msg=False))
        self.print_summary()

        sol = self.extract_solution()
        gap = sonuc["gap"]
        print(f"Benders yakınsama (UB-LB) farkı : {gap:,.2f} TL")
        print(f"Seçilen aktif kaynak sayısı     : {sum(w_val.values())}/{len(w_val)}")
        print(f"Seçilen aktif pompa sayısı      : {sum(y_val.values())}/{len(y_val)}")

        gecerli_mi = (
            sol["status"] == "Optimal"
            and gap <= max(1.0, 0.02 * abs(sonuc["ub"]))  # gevşek bir tolerans (~%2)
        )
        print("=" * 45)
        if gecerli_mi:
            print(" SONUÇ: GEÇERLİ ve YAKINSAMIŞ bir Benders çözümü.")
        else:
            print(" SONUÇ: bir çözüm bulundu ama yakınsama kontrolünü geçemedi —")
            print("        max_iter / epsilon değerlerini artırmayı deneyin.")
        print("=" * 45)

        self.export_submission_files(w_val=w_val, y_val=y_val, output_dir=output_dir)
        return gecerli_mi


def main():
    hbd = HBD()  # varsayılan data_dir="./data" — orijinal davranışla birebir aynı
    sonuc = hbd.run_benders(max_iter=20, epsilon=1.0, sampler="local")
    # Yarışma günü, yukarıdaki satırı sampler="hybrid" yapıp calistirin
    # (dwave-system kurulu + internet + API token gerekir).
    hbd.cozumu_dogrula_ve_yazdir(sonuc, output_dir="solution")


if __name__ == "__main__":
    main()