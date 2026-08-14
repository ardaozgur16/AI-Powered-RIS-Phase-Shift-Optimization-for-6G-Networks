import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from channel_model import WirelessChannel
from baselines import BaselineSolvers
from train import RISPhaseNet


def quantize_phases(phases, bits=2):
    """Sürekli faz açılarını 1-bit veya 2-bit ayrık değerlere yuvarlar."""
    levels = 2 ** bits
    step = 2.0 * np.pi / levels
    return np.round(phases / step) * step % (2.0 * np.pi)


def run_comprehensive_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Eğitilmiş Modeli Yükle
    M, N = 4, 16
    obs_dim = 2 * (M + N * M + N)
    model = RISPhaseNet(input_dim=obs_dim, num_elements=N).to(device)

    try:
        model.load_state_dict(torch.load("ris_model.pth", map_location=device))
        print("Kayıtlı 'ris_model.pth' başarıyla yüklendi.")
    except FileNotFoundError:
        print("Model dosyası bulunamadı, lütfen önce train.py dosyasını çalıştırın!")
        return

    model.eval()
    channel_sim = WirelessChannel(num_antennas=M, num_elements=N)

    # --- ANALİZ 1: Transmit Power vs. Spectral Efficiency ---
    power_dbm_range = np.linspace(10, 35, 6)  # 10 dBm ile 35 dBm arası
    samples = 100

    rates_no_ris_p = []
    rates_random_p = []
    rates_ai_p = []
    rates_ai_2bit_p = []
    rates_ao_p = []

    # Zamanlama sayaçları
    time_ai = 0.0
    time_ao = 0.0

    print("\n[1/2] İletim Gücü Analizi Başlatılıyor...")
    for p_dbm in power_dbm_range:
        solvers = BaselineSolvers(num_antennas=M, num_elements=N, tx_power_dbm=p_dbm)

        r_no_ris, r_rand, r_ai, r_ai_2b, r_ao = [], [], [], [], []

        for _ in range(samples):
            h_d, G, h_r = channel_sim.get_channel_realization()

            # No RIS & Random
            r_no_ris.append(solvers.no_ris(h_d))
            r_rnd, _ = solvers.random_phase(h_d, G, h_r)
            r_rand.append(r_rnd)

            # AO (Geleneksel Optimizasyon & Zaman Ölçümü)
            t0 = time.perf_counter()
            r_opt, _ = solvers.alternating_optimization(h_d, G, h_r)
            time_ao += (time.perf_counter() - t0)
            r_ao.append(r_opt)

            # AI Model Çıkarımı & Zaman Ölçümü
            scale_factor = 1e3
            obs = np.concatenate([
                h_d.flatten().real * scale_factor, h_d.flatten().imag * scale_factor,
                G.flatten().real * scale_factor, G.flatten().imag * scale_factor,
                h_r.flatten().real * scale_factor, h_r.flatten().imag * scale_factor
            ])
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                pred_phase = model(obs_tensor).cpu().numpy().squeeze()
            time_ai += (time.perf_counter() - t0)

            # Sürekli AI
            Theta_ai = np.diag(np.exp(1j * pred_phase))
            h_eff_ai = h_d.conj().T + np.dot(np.dot(h_r.conj().T, Theta_ai), G)
            r_ai.append(solvers._compute_rate(h_eff_ai))

            # 2-bit Kuantize Edilmiş AI
            phase_2bit = quantize_phases(pred_phase, bits=2)
            Theta_2b = np.diag(np.exp(1j * phase_2bit))
            h_eff_2b = h_d.conj().T + np.dot(np.dot(h_r.conj().T, Theta_2b), G)
            r_ai_2bit.append(solvers._compute_rate(h_eff_2b))

        rates_no_ris_p.append(np.mean(r_no_ris))
        rates_random_p.append(np.mean(r_rand))
        rates_ai_p.append(np.mean(r_ai))
        rates_ai_2bit_p.append(np.mean(r_ai_2bit))
        rates_ao_p.append(np.mean(r_ao))

    # Zaman Kıyaslama Raporu
    total_calls = len(power_dbm_range) * samples
    print("\n--- Çıkarım / Optimizasyon Süreleri ---")
    print(f"Alternating Optimization (AO) : {(time_ao / total_calls) * 1000:.3f} ms / kanal")
    print(f"AI-Powered Neural Network     : {(time_ai / total_calls) * 1000:.3f} ms / kanal")
    print(f"Hızlanma Faktörü              : {(time_ao / time_ai):.1f}x Kat Daha Hızlı!")

    # --- GRAFİK ÇİZİMİ ---
    plt.figure(figsize=(9, 6))
    plt.plot(power_dbm_range, rates_ao_p, 'g-^', label='Alternating Opt (Ideal Upper Bound)')
    plt.plot(power_dbm_range, rates_ai_p, 'b-o', linewidth=2, label='AI-Powered (Continuous Phase)')
    plt.plot(power_dbm_range, rates_ai_2bit_p, 'c--s', label='AI-Powered (2-bit Discrete Phase)')
    plt.plot(power_dbm_range, rates_random_p, 'y--d', label='Random Phase')
    plt.plot(power_dbm_range, rates_no_ris_p, 'k:', label='No RIS (Direct Path Only)')

    plt.xlabel('Transmit Power $P_t$ (dBm)', fontsize=12)
    plt.ylabel('Average Spectral Efficiency (bps/Hz)', fontsize=12)
    plt.title(f'Spectral Efficiency vs. Transmit Power (M={M}, N={N})', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("power_vs_rate.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_comprehensive_benchmark()