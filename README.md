# AI-Powered RIS Phase Shift Optimization for 6G Networks

> 🎓 **Not / Acknowledgement:**  
> Bu proje, **University of Glasgow - 6G Vision: ML, Intelligent Surfaces & Optical Networks** eğitimi katkısıyla hazırlanmıştır.

---

## 🚀 Temel Özellikler
* **Gerçekçi Kanal Modellemesi:** 3D koordinatlar, Rician (LoS) ve Rayleigh (NLoS) küçük ölçekli sönümleme ve yol kaybı (Path loss).
* **Unsupervised Policy Network:** Geleneksel çözümlere kıyasla milisaniyeler mertebesinde optimum faz kestirimi.
* **Kapsamlı Kıyaslama (Baselines):** No-RIS, Random Phase, 2-bit Discrete Phase ve Alternating Optimization (AO - Üst Sınır).
* **Donanım Kısıtı Analizi:** Sürekli fazların 1-bit / 2-bit ayrık seviyelere kuantizasyonu.

## 📂 Dosya Yapısı
* `channel_model.py`: 3D kablosuz kanal matrisleri üretimi.
* `baselines.py`: No-RIS, Random Phase ve Alternating Optimization çözücüleri.
* `environment.py`: Gymnasium uyumlu MISO-RIS simülasyon ortamı.
* `train.py`: PyTorch tabanlı model eğitimi ve ağırlık kaydı (`ris_model.pth`).
* `plots_and_benchmarks.py`: Transmit Power vs. Spectral Efficiency eğrileri ve çıkarım süresi analizi.

## 🛠️ Kurulum ve Çalıştırma
```bash
pip install -r requirements.txt

# 1. Modeli Eğit:
python train.py

# 2. Karşılaştırma Grafikleri ve Zaman Analizini Üret:
python plots_and_benchmarks.py
