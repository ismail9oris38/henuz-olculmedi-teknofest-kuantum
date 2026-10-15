# Benders Ayrıştırması Deneyi (Benders Decomposition Experiment)

Bu dizin, TEKNOFEST Kuantum Hackathonu sürecinde **Problem 02** için geliştirilen ilk matematiksel ayrıştırma yaklaşımını ve deney kodlarını içermektedir.

---

## 📌 Yaklaşımın Amacı ve Hipotez

Su dağıtım şebekesi optimizasyon problemi (7.141 düğüm, 45.702 yönlü ark, 221 kaynak, 1.800 pompa, 520 depo ve 600 tüketim bölgesi) toplam **2.021 adet ikili (binary) karar değişkeni** içermektedir. Mevcut kuantum donanımının tam bağlı gömme kapasitesinin çok üzerinde olan bu ölçek için ilk hipotezimiz problemi **Benders Ayrıştırması (Benders Decomposition)** ile ikiye bölmekti:

1. **Ana Problem (Master Problem):** İkili kararlar (kaynak aç/kapa $w$, pompa aç/kapa $y$).
2. **Alt Problem (Subproblem - PuLP/CBC):** Verilen ikili kararlar altında sürekli akış ve depo dengesi (LP relaksasyonu, dual gölge fiyatları ve fizibilite/optimallik kesitleri türetimi).

---

## 📂 Dosya Yapısı

- **`main.py`**: Benders döngüsünün, veri okumasının ve çözüm doğrulamasının ana çalıştırma betiği.
- **`benders_master_local.py`**: Yerel PuLP + CBC tabanlı Benders ana problemi (Master Problem). İkili değişkenleri çözer ve Benders kesitlerini (cuts) biriktirir.
- **`benders_orchestrator_local.py`**: Ana problem ile alt problem arasındaki iteratif döngüyü, kesit üretimini ve yakınsama kriterlerini yöneten orkestratör.
- **`hbd_model.py`**: Su şebekesi akış ve denge alt problemini (Subproblem LP) kuran, kısıtları denetleyen ve dual çarpanları hesaplayan model.

---

## 🔍 Elde Edilen Bulgular ve Nihai Yönteme Geçiş

Bu yaklaşımın test edilmesi sonucunda şu kritik tespitler yapılmıştır:
- **Zayıf Benders Kesitleri (Weak Cuts):** Sabit ücretli ağ akışı (*fixed-charge network flow*) yapısında üretilen Benders kesitleri yeterince sıkı (*tight*) olmadığından yakınsama çok yavaş kalmıştır.
- **Değişken Sayısı:** Master problem hâlâ 2.021 ikili değişkeni çözmek zorundaydı.

Bu deneysel süreç takımımıza doğru soruyu sordurmuştur:
> *"Bu 2.021 kararın kaç tanesi gerçekten kombinatoryal belirsizlik içeriyor?"*

Yapılan LP gevşetmesi analiziyle 2.021 değişkenin **1.806'sının doğrudan 0 veya 1 değerini aldığı**, kombinatoryal belirsizliğin yalnızca **215 değişkende yoğunlaştığı** tespit edilmiş; bu sayede nihai **Dual Yönlendirmeli Hibrit CQM Boru Hattı (`src/run.py`)** geliştirilmiştir.
