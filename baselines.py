import numpy as np
from channel_model import WirelessChannel

class BaselineSolvers:
    """
    RIS sistemleri için referans optimizasyon ve karşılaştırma algoritmaları.
    """

    def __init__(self, num_antennas=4, num_elements=16, tx_power_dbm=30.0, noise_power_dbm=-90.0):
        self.M = num_antennas
        self.N = num_elements
        self.P_t = 10 ** ((tx_power_dbm - 30.0) / 10.0)
        self.sigma2 = 10 ** ((noise_power_dbm - 30.0) / 10.0)

    def _compute_rate(self, h_eff):
        """Efektif kanaldan MRT hüzmeleme ile spektral verimliliği (bps/Hz) hesaplar."""
        # MRT Beamforming ile alınan güç: P_t * ||h_eff||^2
        norm_h_eff_sq = np.linalg.norm(h_eff) ** 2
        snr = (self.P_t * norm_h_eff_sq) / self.sigma2
        rate = np.log2(1.0 + snr)
        return float(rate)

    def no_ris(self, h_d):
        """
        1. RIS Olmayan Durum (Sadece BS -> User doğrudan yolu)
        """
        h_eff = h_d.conj().T  # Boyut: (1, M)
        return self._compute_rate(h_eff)

    def random_phase(self, h_d, G, h_r):
        """
        2. Rastgele Faz Durumu (Theta elemanları rastgele [0, 2*pi))
        """
        random_angles = np.random.uniform(0, 2 * np.pi, size=self.N)
        Theta = np.diag(np.exp(1j * random_angles))

        cascaded = np.dot(np.dot(h_r.conj().T, Theta), G)
        h_eff = h_d.conj().T + cascaded
        return self._compute_rate(h_eff), random_angles

    def alternating_optimization(self, h_d, G, h_r, max_iters=20):
        """
        3. Eş-Faz Hizalama / Alternating Optimization (Üst Sınır)
        Yansıyan sinyalleri doğrudan kanalın fazıyla yapıcı (constructive)
        olarak üst üste bindiren analitik faz hizalama yöntemi.
        """
        # Başlangıçta rastgele faz ata
        theta_angles = np.zeros(self.N)

        # Basit MISO için kapalı form / iteratif faz hizalama:
        # phi_n = arg(h_d^H * G_n^H * h_r_n) faz farkını sıfırlamak hedeflenir.
        for _ in range(max_iters):
            for n in range(self.N):
                # n. eleman dışındaki toplam kanal
                Theta_temp = np.diag(np.exp(1j * theta_angles))
                Theta_temp[n, n] = 0  # n. elemanı çıkar
                h_other = h_d.conj().T + np.dot(np.dot(h_r.conj().T, Theta_temp), G)

                # n. elemanın katkısı: h_r[n]^* * G[n, :]
                a_n = h_r[n].conj() * G[n:n + 1, :]

                # h_other ve a_n arasındaki faz farkını sıfırlayan optimum açı
                inner_prod = np.dot(h_other, a_n.conj().T)[0, 0]
                theta_angles[n] = (np.angle(inner_prod)) % (2 * np.pi)

        # Optimum faz matrisi ile veri hızını hesapla
        Theta_opt = np.diag(np.exp(1j * theta_angles))
        cascaded = np.dot(np.dot(h_r.conj().T, Theta_opt), G)
        h_eff = h_d.conj().T + cascaded

        return self._compute_rate(h_eff), theta_angles


# --- MONTE CARLO TEST VE KARŞILAŞTIRMA ---
if __name__ == "__main__":
    M, N = 4, 32
    num_trials = 100

    channel_sim = WirelessChannel(num_antennas=M, num_elements=N)
    solvers = BaselineSolvers(num_antennas=M, num_elements=N)

    rates_no_ris = []
    rates_random = []
    rates_ao = []

    print(f"{num_trials} Monte Carlo kanal denemesi yapılıyor...")

    for _ in range(num_trials):
        h_d, G, h_r = channel_sim.get_channel_realization()

        r_no_ris = solvers.no_ris(h_d)
        r_rand, _ = solvers.random_phase(h_d, G, h_r)
        r_ao, _ = solvers.alternating_optimization(h_d, G, h_r)

        rates_no_ris.append(r_no_ris)
        rates_random.append(r_rand)
        rates_ao.append(r_ao)

    print("\n--- SONUÇLAR (Ortalama Spektral Verimlilik) ---")
    print(f"1. No-RIS Baseline       : {np.mean(rates_no_ris):.4f} bps/Hz")
    print(f"2. Random Phase RIS      : {np.mean(rates_random):.4f} bps/Hz")
    print(f"3. Alternating Opt (AO)  : {np.mean(rates_ao):.4f} bps/Hz")