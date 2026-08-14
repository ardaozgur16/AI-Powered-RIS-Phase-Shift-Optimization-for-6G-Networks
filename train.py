import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from channel_model import WirelessChannel
from baselines import BaselineSolvers


# --- 1. SİNİR AĞI MİMARİSİ (RIS Faz Tahmin Edici) ---
class RISPhaseNet(nn.Module):
    def __init__(self, input_dim, num_elements=16):
        super(RISPhaseNet, self).__init__()
        self.N = num_elements

        # Çok Katmanlı Algılayıcı (MLP)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.N),
            nn.Sigmoid()  # Çıkışı [0, 1] aralığına sıkıştırır
        )

    def forward(self, x):
        # [0, 1] aralığını [0, 2*pi] faz açılarına ölçekler
        phases = self.net(x) * 2.0 * np.pi
        return phases


# --- 2. VERİ ÜRETİCİ VE YARDIMCI FONKSİYONLAR ---
def generate_batch_data(channel_sim, batch_size):
    """
    Model eğitimi için toplu (batch) kanal matrisleri ve model giriş vektörlerini üretir.
    """
    H_d_list, G_list, H_r_list = [], [], []
    obs_list = []

    scale_factor = 1e3

    for _ in range(batch_size):
        h_d, G, h_r = channel_sim.get_channel_realization()
        H_d_list.append(h_d)
        G_list.append(G)
        H_r_list.append(h_r)

        # Sinir ağı girişi için vektörleştirme
        obs = np.concatenate([
            h_d.flatten().real * scale_factor,
            h_d.flatten().imag * scale_factor,
            G.flatten().real * scale_factor,
            G.flatten().imag * scale_factor,
            h_r.flatten().real * scale_factor,
            h_r.flatten().imag * scale_factor
        ])
        obs_list.append(obs)

    return (
        torch.tensor(np.array(obs_list), dtype=torch.float32),
        H_d_list, G_list, H_r_list
    )


# --- 3. EĞİTİM DÖNGÜSÜ ---
def train():
    # Hiperparametreler
    M = 4  # BS Anten Sayısı
    N = 16  # RIS Eleman Sayısı
    tx_power_dbm = 30.0  # 30 dBm = 1 Watt
    noise_power_dbm = -90.0

    P_t = 10 ** ((tx_power_dbm - 30.0) / 10.0)
    sigma2 = 10 ** ((noise_power_dbm - 30.0) / 10.0)

    epochs = 200
    batch_size = 64
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ortam ve Model Tanımları
    channel_sim = WirelessChannel(num_antennas=M, num_elements=N)
    obs_dim = 2 * (M + N * M + N)

    model = RISPhaseNet(input_dim=obs_dim, num_elements=N).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    print(f"Eğitim Başlatılıyor ({device})...")

    for epoch in range(1, epochs + 1):
        model.train()

        # Batch kanal üret
        x_batch, H_d_list, G_list, H_r_list = generate_batch_data(channel_sim, batch_size)
        x_batch = x_batch.to(device)

        # Ağ faz açılarını tahmin eder: (Batch, N)
        predicted_phases = model(x_batch)

        # Efektif hızları (Loss = -Rate) hesaplama
        total_rate = 0.0
        phases_np = predicted_phases.detach().cpu().numpy()

        rates_batch = []
        for i in range(batch_size):
            theta = np.exp(1j * phases_np[i])
            Theta = np.diag(theta)

            cascaded = np.dot(np.dot(H_r_list[i].conj().T, Theta), G_list[i])
            h_eff = H_d_list[i].conj().T + cascaded

            norm_sq = np.linalg.norm(h_eff) ** 2
            rate = np.log2(1.0 + (P_t * norm_sq) / sigma2)
            rates_batch.append(rate)

        # Unsupervised Loss: Spektral verimliliği maksimize etmek için negatifini minimize ediyoruz
        mean_rate = np.mean(rates_batch)

        # Gradyan güncellemesi için türevlenebilir loss yaklaşımı
        # (Ağın çıktılarını ödülle ağırlıklandırarak policy gradient adımı)
        phases_tensor = predicted_phases
        # Basit surrogate loss
        reward_tensor = torch.tensor(rates_batch, device=device).unsqueeze(1)
        loss = -torch.mean(phases_tensor * reward_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(mean_rate)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] - Ortalama Rate: {mean_rate:.4f} bps/Hz")

    # Modeli Kaydet
    torch.save(model.state_dict(), "ris_model.pth")
    print("\nModel 'ris_model.pth' olarak kaydedildi.")

    return model, channel_sim, M, N


# --- 4. TEST VE GRAFİK ÇİZİMİ ---
def evaluate(model, channel_sim, M, N, test_samples=100):
    print(f"\n--- {test_samples} Test Kanalı Üzerinde Karşılaştırma Yapılıyor ---")
    model.eval()
    solvers = BaselineSolvers(num_antennas=M, num_elements=N)

    rates_no_ris = []
    rates_random = []
    rates_ai = []
    rates_ao = []

    for _ in range(test_samples):
        h_d, G, h_r = channel_sim.get_channel_realization()

        # 1. No RIS
        rates_no_ris.append(solvers.no_ris(h_d))

        # 2. Random RIS
        r_rand, _ = solvers.random_phase(h_d, G, h_r)
        rates_random.append(r_rand)

        # 3. Alternating Optimization (Üst Sınır)
        r_ao, _ = solvers.alternating_optimization(h_d, G, h_r)
        rates_ao.append(r_ao)

        # 4. AI Model
        scale_factor = 1e3
        obs = np.concatenate([
            h_d.flatten().real * scale_factor,
            h_d.flatten().imag * scale_factor,
            G.flatten().real * scale_factor,
            G.flatten().imag * scale_factor,
            h_r.flatten().real * scale_factor,
            h_r.flatten().imag * scale_factor
        ])
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_phase = model(obs_tensor).numpy().squeeze()

        Theta_ai = np.diag(np.exp(1j * pred_phase))
        cascaded_ai = np.dot(np.dot(h_r.conj().T, Theta_ai), G)
        h_eff_ai = h_d.conj().T + cascaded_ai
        rates_ai.append(solvers._compute_rate(h_eff_ai))

    print("\n--- TEST SONUÇLARI (Ortalama Spektral Verimlilik) ---")
    print(f"1. No RIS (Doğrudan Yol)       : {np.mean(rates_no_ris):.4f} bps/Hz")
    print(f"2. Random Phase RIS            : {np.mean(rates_random):.4f} bps/Hz")
    print(f"3. AI-Powered RIS (Modelimiz)  : {np.mean(rates_ai):.4f} bps/Hz")
    print(f"4. Alternating Optimization (AO): {np.mean(rates_ao):.4f} bps/Hz")

    # Çıktı Bar Grafiği
    methods = ["No RIS", "Random Phase", "AI-Powered RIS", "Upper Bound (AO)"]
    avg_rates = [np.mean(rates_no_ris), np.mean(rates_random), np.mean(rates_ai), np.mean(rates_ao)]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, avg_rates, color=["#7f7f7f", "#bcbd22", "#1f77b4", "#2ca02c"])
    plt.ylabel("Spectral Efficiency (bps/Hz)")
    plt.title("RIS Phase Shift Optimization - Performance Comparison")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom',
                 fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    trained_model, sim, M, N = train()
    evaluate(trained_model, sim, M, N)