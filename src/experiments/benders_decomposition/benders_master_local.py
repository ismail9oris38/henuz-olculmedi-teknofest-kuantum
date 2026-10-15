"""
Benders ANA PROBLEMİ (master problem) — TAMAMEN YEREL, PuLP + CBC sürümü
==========================================================================
HBD sınıfındaki `_build_master_problem` metodu bir `dimod.ConstrainedQuadraticModel`
kuruyor ama hiçbir yerde çözülmüyor (D-Wave hibrit sampler internet + API
token istiyor, `neal` ise sadece sezgisel/olurlu-garantisiz bir sonuç verir).
Benders ayrıştırmasının LB (alt sınır) garantisinin geçerli olması için ana
problemin HER iterasyonda TAM OLARAK (kesin/exact) optimuma çözülmesi gerekir;
aksi halde LB monoton artmayabilir ve yakınsama garantisi çöker.

Bu yüzden burada `self.cqm`'e HİÇ dokunmuyoruz (hbd_model.py aynen duruyor);
onun yerine, aynı karar değişkenlerini (kaynak aç/kapa W[node_id], pompa
aç/kapa Y[node_id]) ve aynı K15/K16 grup kısıtlarını PuLP ile kuran, CBC
(`PULP_CBC_CMD`) ile KESİN çözülen paralel bir ana problem inşa ediyoruz.

Epigraf değişkeni ETA, alt problemin (subproblem) maliyetinin bir ALT
KESTİRİMİDİR (t_var'ın CQM şablonundaki karşılığı). Amaç fonksiyonu:

    min  sum_s activation_cost_tl[s] * W[s]
       + sum_p fixed_on_cost[p]     * Y[p]
       + ETA

ETA >= 0 (alt problemdeki tüm maliyet terimleri —üretim, enerji, iletim,
depo giriş/çıkışı, karşılanmayan talep cezası, depo seviye ihlali cezaları—
negatif olamayacağı için bu geçerli bir alt sınır).

Kesitler (optimality cuts), K1/K2/K4/K5 kısıtlarının ikil (dual) değerlerinden
LP duyarlılık analizi (parametrik LP: RHS = kapasite * W ya da min_akış * W)
ile türetilir; bkz. benders_orchestrator_local.py -> _kesit_hesapla.
"""
from dataclasses import dataclass, field
from typing import Dict

from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, PULP_CBC_CMD, lpSum, value


@dataclass
class BendersKesit:
    """Optimallik kesiti:
       ETA >= sabit + sum_s w_katsayi[s] * W[s] + sum_p y_katsayi[p] * Y[p]
    """
    sabit: float
    w_katsayi: Dict[str, float] = field(default_factory=dict)
    y_katsayi: Dict[str, float] = field(default_factory=dict)


class MasterProblemKuruculocal:
    """HBD ana problemin PuLP + CBC (tamamen yerel, token gerektirmeyen) sürümü.

    Kaynak/pompa aç-kapa kararları HBD.sources / HBD.pumps üzerinden aynen
    okunur; K15 (pompa grubu) ve K16 (kaynak paketi) kısıtları HBD'deki
    `_build_master_problem` ile BİREBİR aynı mantıkla (aynı prefiks-kırpma
    kuralı: `member_*_ids` elemanlarının ilk 2 karakteri atılır) kurulur.
    """

    def __init__(self, hbd, eta_ust_sinir: float = 1e9):
        self.hbd = hbd
        self._kesit_no = 0

        self.prob = LpProblem("Benders_Master", LpMinimize)

        self.W = {s["node_id"]: LpVariable(f"W_{s['source_id']}", cat="Binary") for s in hbd.sources}
        self.Y = {p["node_id"]: LpVariable(f"Y_{p['pump_id']}", cat="Binary") for p in hbd.pumps}
        self.ETA = LpVariable("ETA", lowBound=0, upBound=eta_ust_sinir)

        sabit_maliyet = (
            lpSum(s["activation_cost_tl"] * self.W[s["node_id"]] for s in hbd.sources)
            + lpSum(p["fixed_on_cost"] * self.Y[p["node_id"]] for p in hbd.pumps)
        )
        self.prob += sabit_maliyet + self.ETA, "toplam_maliyet"

        # K15: Pompa grubu kısıtı (HBD._build_master_problem ile aynı)
        for g in hbd.pump_groups:
            self.prob += (
                lpSum(self.Y[p[2:]] for p in g["member_pump_ids"]) <= g["max_active"],
                f"K15_PumpGroup_{g.get('group_id', '')}",
            )

        # K16: Kaynak paketi kısıtı (HBD._build_master_problem ile aynı)
        for g in hbd.source_packages:
            self.prob += (
                lpSum(self.W[s[2:]] for s in g["member_source_ids"]) <= g["max_active"],
                f"K16_SourcePkg_{g.get('package_id', '')}",
            )

    def kesit_ekle(self, kesit: BendersKesit):
        ifade = (
            lpSum(kat * self.W[node_id] for node_id, kat in kesit.w_katsayi.items() if node_id in self.W)
            + lpSum(kat * self.Y[node_id] for node_id, kat in kesit.y_katsayi.items() if node_id in self.Y)
        )
        self.prob += self.ETA >= kesit.sabit + ifade, f"benders_kesit_{self._kesit_no}"
        self._kesit_no += 1

    def coz(self, msg: bool = False, time_limit: float = None):
        """CBC ile KESİN (exact) MILP çözümü. İnternet / API token gerektirmez.
        Döndürür: (w_bar, y_bar, eta_bar, master_obj_degeri, durum_str)
        """
        solver = PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
        status = self.prob.solve(solver)
        durum_str = LpStatus[status]

        w_bar = {node_id: int(round(var.varValue)) for node_id, var in self.W.items()}
        y_bar = {node_id: int(round(var.varValue)) for node_id, var in self.Y.items()}
        eta_bar = self.ETA.varValue if self.ETA.varValue is not None else 0.0
        master_obj = value(self.prob.objective)

        return w_bar, y_bar, eta_bar, master_obj, durum_str

    def sabit_maliyet(self, w_bar: Dict[str, int], y_bar: Dict[str, int]) -> float:
        """Verilen w,y için SADECE aktivasyon/fixed-on maliyetini hesaplar
        (UB takibi için; master'ı yeniden çözmeden)."""
        toplam = 0.0
        for s in self.hbd.sources:
            toplam += s["activation_cost_tl"] * w_bar.get(s["node_id"], 0)
        for p in self.hbd.pumps:
            toplam += p["fixed_on_cost"] * y_bar.get(p["node_id"], 0)
        return toplam
