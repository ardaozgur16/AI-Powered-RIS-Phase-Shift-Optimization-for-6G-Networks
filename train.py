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

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.N),
            nn.Sigmoid()  # [0, 1] aralığı
        )

    def forward(self, x):
        # [0, 1] aralığını [0, 2*pi] faz açılarına dönüştür
        return self.net(x) * (2.0 * np.pi)


# --- 2. PURE PYTORCH DIFFERENTIABLE LOSS FONKSİYONU ---
def compute_spectral_efficiency_loss(phases, H_d, G, H_r, P_t, sigma2):
    """
    Türevlenebilir (Differentiable) Spektral Verimlilik Kaybı.
    Geriye yayılım (backprop) faz açılarına kadar kesintisiz akar!
    
    phases: (Batch, N) - float
    H_d   : (Batch, M, 1) - complex
    G     : (Batch, N, M) - complex
    H_r   : (Batch, N, 1) - complex
    """
    # 1. Fazları karmaşık sayılara çevir: e^(j * theta)
    # phases -> (Batch, N)
    theta_complex = torch.complex(torch.cos(phases), torch.sin(phases))  # (Batch, N)
    
    # Theta matrisi: diag(e^(j * theta)) -> (Batch, N, N)
    Theta = torch.diag_embed(theta_complex)

    # 2. h_r^H hesapla: (Batch, 1, N)
    H_r_H = H_r.conj().transpose(-2, -1)

    # 3. Yansıyan kanal: H_r^H * Theta * G -> (Batch, 1, M)
    cascaded = torch.bmm(torch.bmm(H_r_H, Theta), G)

    # 4. h_d^H hesapla: (Batch, 1, M)
    H_d_H = H_d.conj().transpose(-2, -1)

    # 5. Toplam efektif kanal: h_eff = h_d^H + cascaded -> (Batch, 1, M)
    H_eff = H_d_H + cascaded

    # 6. Alınan Sinyal Gücü: P_t * ||h_eff||^2
    # complex norm karesi: |real|^2 + |imag|^2
    norm_sq = torch.sum(H_eff.real ** 2 + H_eff.imag ** 2, dim=(-2, -1))  # (Batch,)

    # 7. SNR ve Spektral Verimlilik (bits/s/Hz)
    snr = (P_t * norm_sq) / sigma2
    rate = torch.log2(1.0 + snr)  # (Batch,)

    # Amacımız Rate'i maksimize etmek, bu yüzden Loss = -Mean(Rate)
    loss = -torch.mean(rate)
    return loss, torch.mean(rate).item()


# --- 3. BATCH VERİ ÜRETİCİ ---
def generate_batch_tensors(channel_sim, batch_size, device):
    obs_list = []
    H_d_list, G_list, H_r_list = [], [], []
    scale_factor = 1e3

    for _ in range(batch_size):
        h_d, G, h_r = channel_sim.get_channel_realization()
        
        # PyTorch tensörlerine aktar
        H_d_list.append(torch.tensor(h_d, dtype=torch.complex64))
        G_list.append(torch.tensor(G, dtype=torch.complex64))
        H_r_list.append(torch.tensor(h_r, dtype=torch.complex64))

        obs = np.concatenate([
            h_d.flatten().real * scale_factor,
            h_d.flatten().imag * scale_factor,
            G.flatten().real * scale_factor,
            G.flatten().imag * scale_factor,
            h_r.flatten().real * scale_factor,
            h_r.flatten().imag * scale_factor
        ])
        obs_list.append(obs)

    x_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32).to(device)
    H_d_tensor = torch.stack(H_d_list).to(device)
    G_tensor = torch.stack(G_list).to(device)
    H_r_tensor = torch.stack(H_r_list).to(device)

    return x_tensor, H_d_tensor, G_tensor, H_r_tensor


# --- 4. EĞİTİM DÖNGÜSÜ ---
def train():
    M = 4                  # BS Anten Sayısı
    N = 16                 # RIS Eleman Sayısı
    tx_power_dbm = 30.0    # 1 Watt
    noise_power_dbm = -90.0
    
    P_t = 10 ** ((tx_power_dbm - 30.0) / 10.0)
    sigma2 = 10 ** ((noise_power_dbm - 30.0) / 10.0)

    epochs = 200
    batch_size = 64
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channel_sim = WirelessChannel(num_antennas=M, num_elements=N)
    obs_dim = 2 * (M + N * M + N)

    model = RISPhaseNet(input_dim=obs_dim, num_elements=N).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Gerçek Differentiable Loss ile Eğitim Başlatılıyor ({device})...")

    for epoch in range(1, epochs + 1):
        model.train()
        
        # Batch tensörleri üret
        x_batch, H_d_batch, G_batch, H_r_batch = generate_batch_tensors(channel_sim, batch_size, device)

        # Ağ doğrudan fazları üretir
        predicted_phases = model(x_batch)

        # Tamamen PyTorch üzerinde türevlenebilir kayıp hesabı
        loss, avg_rate = compute_spectral_efficiency_loss(
            predicted_phases, H_d_batch, G_batch, H_r_batch, P_t, sigma2
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] - Ortalama Rate: {avg_rate:.4f} bps/Hz")

    # Modeli Kaydet
    torch.save(model.state_dict(), "ris_model.pth")
    print("\n[OK] Model 'ris_model.pth' olarak başarıyla üretildi ve kaydedildi!")

    return model, channel_sim, M, N


# --- 5. TEST VE DEĞERLENDİRME ---
def evaluate(model, channel_sim, M, N, test_samples=100):
    print(f"\n--- {test_samples} Test Kanalı Üzerinde Karşılaştırma Yapılıyor ---")
    model.eval()
    solvers = BaselineSolvers(num_antennas=M, num_elements=N)

    rates_no_ris, rates_random, rates_ai, rates_ao = [], [], [], []

    for _ in range(test_samples):
        h_d, G, h_r = channel_sim.get_channel_realization()

        # Baselines
        rates_no_ris.append(solvers.no_ris(h_d))
        r_rand, _ = solvers.random_phase(h_d, G, h_r)
        rates_random.append(r_rand)
        r_ao, _ = solvers.alternating_optimization(h_d, G, h_r)
        rates_ao.append(r_ao)

        # AI Model
        scale_factor = 1e3
        obs = np.concatenate([
            h_d.flatten().real * scale_factor, h_d.flatten().imag * scale_factor,
            G.flatten().real * scale_factor, G.flatten().imag * scale_factor,
            h_r.flatten().real * scale_factor, h_r.flatten().imag * scale_factor
        ])
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pred_phase = model(obs_tensor).cpu().numpy().squeeze()

        Theta_ai = np.diag(np.exp(1j * pred_phase))
        cascaded_ai = np.dot(np.dot(h_r.conj().T, Theta_ai), G)
        h_eff_ai = h_d.conj().T + cascaded_ai
        rates_ai.append(solvers._compute_rate(h_eff_ai))

    print("\n--- TEST SONUÇLARI (Ortalama Spektral Verimlilik) ---")
    print(f"1. No RIS (Doğrudan Yol)       : {np.mean(rates_no_ris):.4f} bps/Hz")
    print(f"2. Random Phase RIS            : {np.mean(rates_random):.4f} bps/Hz")
    print(f"3. AI-Powered RIS (Modelimiz)  : {np.mean(rates_ai):.4f} bps/Hz")
    print(f"4. Alternating Optimization (AO): {np.mean(rates_ao):.4f} bps/Hz")

    # Grafik
    methods = ["No RIS", "Random Phase", "AI-Powered RIS", "Upper Bound (AO)"]
    avg_rates = [np.mean(rates_no_ris), np.mean(rates_random), np.mean(rates_ai), np.mean(rates_ao)]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, avg_rates, color=["#7f7f7f", "#bcbd22", "#1f77b4", "#2ca02c"])
    plt.ylabel("Spectral Efficiency (bps/Hz)")
    plt.title("RIS Phase Shift Optimization - Performance Comparison")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    trained_model, sim, M, N = train()
    evaluate(trained_model, sim, M, N)
