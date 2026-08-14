import gymnasium as gym
from gymnasium import spaces
import numpy as np

from channel_model import WirelessChannel

class RISEnvironment(gym.Env):
    """
    RIS Destekli MISO Sistemi için Gymnasium Ortamı.
    Pekiştirmeli Öğrenme ajanları (DDPG, TD3, PPO, SAC vb.) ile tam uyumludur.
    """
    metadata = {"render_modes": []}

    def __init__(
            self,
            num_antennas=4,  # M: Baz İstasyonu anten sayısı
            num_elements=16,  # N: RIS eleman sayısı
            tx_power_dbm=30.0,  # İletim gücü (dBm) -> 30 dBm = 1 Watt
            noise_power_dbm=-90.0,  # Gürültü gücü (dBm) -> -90 dBm = 1e-9 Watt
            rician_factor_k=3.0
    ):
        super(RISEnvironment, self).__init__()

        self.M = num_antennas
        self.N = num_elements

        # Güç birimlerini dBm'den Watt'a (Lineer) dönüştür
        self.P_t = 10 ** ((tx_power_dbm - 30.0) / 10.0)
        self.sigma2 = 10 ** ((noise_power_dbm - 30.0) / 10.0)

        # Kanal modelini başlat
        self.channel_sim = WirelessChannel(
            num_antennas=self.M,
            num_elements=self.N,
            rician_factor_k=rician_factor_k
        )

        # AKSİYON UZAYI: N adet RIS elemanı için faz açıları [0, 2*pi]
        self.action_space = spaces.Box(
            low=0.0,
            high=2.0 * np.pi,
            shape=(self.N,),
            dtype=np.float32
        )

        # GÖZLEM UZAYI:
        # h_d: M kompleks eleman (2M float)
        # G  : N x M kompleks eleman (2NM float)
        # h_r: N kompleks eleman (2N float)
        self.obs_dim = 2 * (self.M + self.N * self.M + self.N)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # Anlık kanal durumları
        self.h_d = None
        self.G = None
        self.h_r = None

    def _get_obs(self):
        """Kanal katsayılarını düzleştirip reel ve imajiner olarak birleştirir."""
        # Ölçekleme: Kanal katsayıları path loss nedeniyle çok küçük olduğundan
        # ağın daha kolay öğrenmesi için normalize edilebilir (opsiyonel scaling)
        scale_factor = 1e3  # Gözlemleri daha sayısal kararlı aralığa çeker

        obs = np.concatenate([
            self.h_d.flatten().real * scale_factor,
            self.h_d.flatten().imag * scale_factor,
            self.G.flatten().real * scale_factor,
            self.G.flatten().imag * scale_factor,
            self.h_r.flatten().real * scale_factor,
            self.h_r.flatten().imag * scale_factor,
        ])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Yeni bir kanal durumu üret
        self.h_d, self.G, self.h_r = self.channel_sim.get_channel_realization()

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        """
        Action: (N,) boyutlu float dizi [0, 2*pi] arasındaki faz açıları
        """
        # 1. Faz matrisini oluştur: Theta = diag(e^(j * theta))
        phase_shifts = np.clip(action, 0.0, 2.0 * np.pi)
        theta_diag = np.exp(1j * phase_shifts)
        Theta = np.diag(theta_diag)

        # 2. Efektif kanal vektörünü hesapla:
        # h_eff = h_d^H + h_r^H * Theta * G
        # h_eff boyutu: (1, M)
        cascaded_channel = np.dot(np.dot(self.h_r.conj().T, Theta), self.G)
        h_eff = self.h_d.conj().T + cascaded_channel

        # 3. İdeal Hüzmeleme (MRT - Maximum Ratio Transmission):
        # w = sqrt(P_t) * (h_eff^H / ||h_eff||)
        norm_h_eff = np.linalg.norm(h_eff)
        if norm_h_eff > 1e-12:
            w = np.sqrt(self.P_t) * (h_eff.conj().T / norm_h_eff)
        else:
            w = np.zeros((self.M, 1), dtype=complex)

        # 4. Alınan Sinyal Gücü, SNR ve Veri Hızı (Data Rate):
        received_power = np.abs(np.dot(h_eff, w)[0, 0]) ** 2
        snr = received_power / self.sigma2
        rate = np.log2(1.0 + snr)

        # Ödül (Reward) doğrudan spektral verimliliktir (bits/s/Hz)
        reward = float(rate)

        # Tek zaman dilimli (block fading) optimizasyon:
        # Her adımdan sonra kanal değiştiği için episode tek adımda biter.
        terminated = True
        truncated = False

        info = {
            "rate": rate,
            "snr_db": 10.0 * np.log10(snr) if snr > 0 else -np.inf
        }

        # Bir sonraki gözlem için resetlenmiş kanal dönülür
        obs, _ = self.reset()

        return obs, reward, terminated, truncated, info


# --- ORTAM TESTİ ---
if __name__ == "__main__":
    env = RISEnvironment(num_antennas=4, num_elements=16)
    obs, _ = env.reset()

    print(f"Gözlem Boyutu: {obs.shape}")
    print(f"Aksiyon Boyutu: {env.action_space.shape}")

    # Rastgele bir aksiyon deneyelim (Random Phase Baseline)
    random_action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(random_action)

    print(f"\nRastgele Faz ile Elde Edilen:")
    print(f"  - Spektral Verimlilik (Rate): {info['rate']:.4f} bps/Hz")
    print(f"  - Alınan SNR: {info['snr_db']:.2f} dB")