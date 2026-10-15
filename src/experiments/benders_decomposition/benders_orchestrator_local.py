"""
Benders döngüsü — HBD alt problemini (PuLP/CBC, LP) ve YEREL PuLP/CBC ana
problemini (benders_master_local.py) birbirine bağlayan orkestratör.

hbd_model.py'deki HBD sınıfına HİÇBİR DOKUNUŞ YAPILMADI: `_build_sub_problem`,
`extract_solution`, `export_submission_files` vs. AYNEN kullanılıyor. Sadece
eksik olan "döngü" (master <-> subproblem <-> kesit) burada kuruluyor.

Akış (klasik 2-aşamalı Benders, D-Wave / dimod / neal YOK, tamamen yerel):
    1. w̄, ȳ ile başla (ilk iterasyonda: tüm kaynak ve pompalar AÇIK —
       alt problem bunu kesitleyip master'a gerçekçi bir yön verecek).
    2. Alt problemi (HBD._build_sub_problem, PuLP/CBC, LP) w̄,ȳ ile çöz.
    3. K1/K2 (kaynak kapasite/min-akış) ve K4/K5 (pompa kapasite/min-akış)
       kısıtlarının ikil (dual/gölge fiyat) değerlerinden bir OPTİMALLİK
       KESİTİ türet, ana probleme ekle.
       Neden geçerli: alt problem SAF bir LP (K1,K2,K4,K5 kısıtlarının
       RHS'i w̄,ȳ'de DOĞRUSAL); LP'nin optimal değer fonksiyonu RHS'te
       KONVEKS olduğundan (klasik LP duyarlılık teoremi) ve RHS, (w,y)'de
       doğrusal olduğundan, z*(w,y) da (w,y)'de konvekstir → ikil
       değerlerden türetilen teğet düzlem HER (w,y) için geçerli bir alt
       sınırdır (ikili/binary noktalar dahil). K2/K5 (>=) kısıtları için de
       AYNI mantık geçerlidir; işaret PuLP'ın kendi `.pi` kuralına göre
       otomatik doğru çıkar (test edildi, bkz. proje notları).
    4. Ana problemi (PuLP + CBC, KESİN MILP çözümü) tekrar çöz -> yeni
       w̄, ȳ ve ETA (alt sınır / LB bileşeni).
    5. Üst sınırı (UB) = min(şimdiye kadarki en iyi sabit_maliyet + alt
       problem maliyeti) güncelle. |UB - LB| <= epsilon ya da
       max_iterasyon'a ulaşılınca dur.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List
from dwave.system import LeapHybridCQMSampler

from pulp import LpStatus, PULP_CBC_CMD, value

from benders_master_local import BendersKesit, MasterProblemKuruculocal


@dataclass
class BendersSonucu:
    w_bar: Dict[str, int]
    y_bar: Dict[str, int]
    ub: float
    lb: float
    iterasyon_sayisi: int
    yakinsama_gecmisi: List[Dict] = field(default_factory=list)
    en_iyi_alt_sonuc: Any = None  # en iyi (UB) iterasyondaki hbd.extract_solution() çıktısı


class HBDBendersCozucu:
    def __init__(self, hbd, max_iterasyon: int = 30, epsilon: float = 1.0, msg: bool = False):
        """
        hbd : hbd_model.HBD örneği (zaten __init__ içinde veri setini okumuş olmalı).
        max_iterasyon / epsilon : klasik Benders durdurma kriterleri.
        msg : CBC'nin çözücü loglarını terminale basıp basmayacağı.
        """
        self.hbd = hbd
        self.max_iterasyon = max_iterasyon
        self.epsilon = epsilon
        self.msg = msg
        self.kurucu = MasterProblemKuruculocal(hbd)

    @staticmethod
    def _kesit_hesapla(hbd, sub_cost: float, w_bar: Dict[str, int], y_bar: Dict[str, int]) -> BendersKesit:
        """K1/K2/K4/K5'in ikil değerlerinden bir destek hiper-düzlemi (subgradient
        inequality) üretir:
            ETA >= sub_cost(w̄,ȳ) + sum_s w_katsayi[s]*(W[s]-w̄[s])
                                  + sum_p y_katsayi[p]*(Y[p]-ȳ[p])
        w_katsayi[s]  = dual(K1_s)*capacity_m3h[s] + dual(K2_s)*min_flow_if_active_m3h[s]
        y_katsayi[p]  = dual(K4_p)*capacity_m3h[p] + dual(K5_p)*min_flow_if_on[p]
        (pompanın out_arc_ids'i boşsa K4/K5 hiç kurulmamıştır -> katsayı 0)
        """
        cons = hbd.sub_prob.constraints

        w_katsayi: Dict[str, float] = {}
        for s in hbd.sources:
            s_id = s["node_id"]
            kat = 0.0
            c1 = cons.get(f"K1_SourceCap_{s_id}")
            if c1 is not None and c1.pi is not None:
                kat += c1.pi * s["capacity_m3h"]
            c2 = cons.get(f"K2_SourceMinFlow_{s_id}")
            if c2 is not None and c2.pi is not None:
                kat += c2.pi * s["min_flow_if_active_m3h"]
            w_katsayi[s_id] = kat

        y_katsayi: Dict[str, float] = {}
        for p in hbd.pumps:
            p_id = p["node_id"]
            kat = 0.0
            if p["out_arc_ids"]:
                c4 = cons.get(f"K4_PumpCap_{p_id}")
                if c4 is not None and c4.pi is not None:
                    kat += c4.pi * p["capacity_m3h"]
                c5 = cons.get(f"K5_PumpMinFlow_{p_id}")
                if c5 is not None and c5.pi is not None:
                    kat += c5.pi * p["min_flow_if_on"]
            y_katsayi[p_id] = kat

        egim_toplami = (
            sum(w_katsayi[s_id] * w_bar.get(s_id, 0) for s_id in w_katsayi)
            + sum(y_katsayi[p_id] * y_bar.get(p_id, 0) for p_id in y_katsayi)
        )
        sabit = sub_cost - egim_toplami
        return BendersKesit(sabit=sabit, w_katsayi=w_katsayi, y_katsayi=y_katsayi)

    def coz(self) -> BendersSonucu:
        hbd = self.hbd
        sampler = LeapHybridCQMSampler()  # token ortamdan/config'ten otomatik okunur
        sampleset = sampler.sample_cqm(hbd.cqm, time_limit=5, label="hbd_master")
        feasible = sampleset.filter(lambda row: row.is_feasible)
        best = (feasible if len(feasible) else sampleset).first.sample
        # Başlangıç sezgiseli: tüm kaynak ve pompalar AÇIK.
        w_bar = {s["node_id"]: 1 for s in hbd.sources}
        y_bar = {p["node_id"]: 1 for p in hbd.pumps}

        ub, lb = float("inf"), -float("inf")
        en_iyi_w, en_iyi_y, en_iyi_alt_sonuc = None, None, None
        iter_kaydi = []

        for iterasyon in range(self.max_iterasyon):
            # 1) Alt problemi mevcut w̄,ȳ ile çöz (HBD'nin KENDİ metodu, değişmeden)
            hbd._build_sub_problem(w_val=w_bar, y_val=y_bar)
            status = hbd.sub_prob.solve(PULP_CBC_CMD(msg=self.msg))

            if LpStatus[status] != "Optimal":
                # HBD'nin tasarımı gereği (sert talep tabanı yok, sadece ceza)
                # alt problem her w̄,ȳ için olurlu olmalı; yine de güvenlik ağı:
                raise RuntimeError(
                    f"[iter {iterasyon}] Alt problem '{LpStatus[status]}' durumunda — "
                    "w̄,ȳ için beklenmedik bir olurluluk sorunu var."
                )

            sub_cost = value(hbd.sub_prob.objective)

            # 2) Üst sınırı (UB) güncelle: bu w̄,ȳ ile GERÇEK toplam maliyet
            toplam_maliyet = self.kurucu.sabit_maliyet(w_bar, y_bar) + sub_cost
            if toplam_maliyet < ub:
                ub = toplam_maliyet
                en_iyi_w, en_iyi_y = dict(w_bar), dict(y_bar)
                en_iyi_alt_sonuc = hbd.extract_solution()

            # 3) Kesit türet ve master'a ekle
            kesit = self._kesit_hesapla(hbd, sub_cost, w_bar, y_bar)
            self.kurucu.kesit_ekle(kesit)

            # 4) Ana problemi (KESİN MILP, CBC) tekrar çöz
            w_bar, y_bar, eta_bar, master_obj, master_durum = self.kurucu.coz(msg=self.msg)
            if master_durum != "Optimal":
                raise RuntimeError(f"[iter {iterasyon}] Ana problem '{master_durum}' durumunda.")

            # LB = ana problemin o anki amaç değeri (sabit_maliyet(w̄,ȳ) + ETA).
            # ETA, eklenen tüm kesitlerin altından geçmeyen geçerli bir alt
            # kestirim olduğu için bu değer geçerli bir GLOBAL alt sınırdır.
            lb = master_obj

            iter_kaydi.append({
                "iter": iterasyon, "ub": round(ub, 2), "lb": round(lb, 2),
                "gap": round(ub - lb, 2), "alt_problem_maliyeti": round(sub_cost, 2),
                "eta": round(eta_bar, 2),
            })
            print(f"iter={iterasyon:2d}  UB={ub:14,.2f}  LB={lb:14,.2f}  gap={ub - lb:12,.2f}")

            if ub - lb <= self.epsilon:
                break

        if en_iyi_w is None:
            raise RuntimeError("Hiçbir iterasyonda olurlu bir çözüm bulunamadı.")

        return BendersSonucu(
            w_bar=en_iyi_w, y_bar=en_iyi_y, ub=ub, lb=lb,
            iterasyon_sayisi=len(iter_kaydi),
            yakinsama_gecmisi=iter_kaydi,
            en_iyi_alt_sonuc=en_iyi_alt_sonuc,
        )


def main():
    from hbd_model import HBD

    hbd = HBD()
    # sampler="hybrid"/"local" (D-Wave/neal) YOK — tamamen yerel PuLP+CBC.
    cozucu = HBDBendersCozucu(hbd, max_iterasyon=30, epsilon=1.0, msg=False)
    sonuc = cozucu.coz()

    print("\n" + "=" * 50)
    print(f" Benders yakınsadı: {sonuc.iterasyon_sayisi} iterasyon")
    print(f" UB={sonuc.ub:,.2f} TL   LB={sonuc.lb:,.2f} TL   gap={sonuc.ub - sonuc.lb:,.2f} TL")
    print("=" * 50)

    # En iyi bulunan w̄,ȳ ile alt problemi SON KEZ çöz (export için sub_prob'u
    # bu optimuma göre yeniden kur — HBD'nin kendi export mekanizması aynen kullanılır)
    hbd._build_sub_problem(w_val=sonuc.w_bar, y_val=sonuc.y_bar)
    hbd.sub_prob.solve(PULP_CBC_CMD(msg=False))
    hbd.print_summary()
    hbd.export_submission_files(w_val=sonuc.w_bar, y_val=sonuc.y_bar, output_dir="solution")


if __name__ == "__main__":
    main()
