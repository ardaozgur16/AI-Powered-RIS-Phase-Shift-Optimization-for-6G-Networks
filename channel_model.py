import numpy as np


class PathLossModel:
    """
    3D koordinatlara dayalı mesafe ve Path Loss (Yol Kaybı) hesaplama sınıfı.
    """

    def __init__(self, pl_exponent_direct=3.5, pl_exponent_ris=2.2, c_0_db=-30.0):
        # c_0_db: 1 metredeki referans yol kaybı (dB cinsinden)
        self.c_0 = 10 ** (c_0_db / 10.0)
        self.alpha_d = pl_exponent_direct  # Doğrudan yol (NLoS, daha yüksek kayıp)
        self.alpha_r = pl_exponent_ris  # RIS yolları (LoS baskın, daha düşük kayıp)

    def calculate_distance(self, pos1, pos2):
        """İki 3D nokta arasındaki Öklid mesafesini hesaplar."""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def get_path_loss(self, distance, alpha):
        """Mesafe ve sönümleme katsayısına göre doğrusal (linear) kazancı döner."""
        return np.sqrt(self.c_0 * (distance ** (-alpha)))


class WirelessChannel:
    """
    RIS Destekli MISO Sistemi için Kanal Üreticisi.
    BS (M anten) -> RIS (N eleman) -> User (1 anten)
    """

    def __init__(
            self,
            num_antennas=4,  # M: Baz istasyonu anten sayısı
            num_elements=16,  # N: RIS yansıtıcı eleman sayısı
            bs_pos=(0, 0, 10),  # BS Koordinatı (x, y, z) metre
            ris_pos=(50, 10, 10),  # RIS Koordinatı
            user_pos=(60, 0, 1.5),  # Kullanıcı Koordinatı
            rician_factor_k=3.0  # Rician K-faktörü (Line-of-Sight baskınlığı)
    ):
        self.M = num_antennas
        self.N = num_elements
        self.bs_pos = bs_pos
        self.ris_pos = ris_pos
        self.user_pos = user_pos
        self.K = rician_factor_k

        self.path_loss_engine = PathLossModel()

    def _generate_rayleigh(self, shape):
        """Rayleigh (Tamamen NLoS) kanal matrisi üretir (CN(0, 1))."""
        real = np.random.randn(*shape)
        imag = np.random.randn(*shape)
        return (real + 1j * imag) / np.sqrt(2.0)

    def _generate_rician(self, shape):
        """Rician (LoS + NLoS bileşenli) kanal matrisi üretir."""
        # NLoS (Dağınık) bileşen
        h_nlos = self._generate_rayleigh(shape)

        # LoS (Görüş hattı deterministik) bileşen - Basit düzlem dalga modeli
        h_los = np.ones(shape, dtype=complex)

        # Rician birleşimi
        h_rician = np.sqrt(self.K / (self.K + 1.0)) * h_los + \
                   np.sqrt(1.0 / (self.K + 1.0)) * h_nlos
        return h_rician

    def get_channel_realization(self):
        """
        Anlık kanal matrislerini (büyük ve küçük ölçekli kayıplarla birleştirilmiş) üretir.

        Dönen Matrisler:
        - h_d: Doğrudan kanal (BS -> User) [Boyut: M x 1]
        - G  : BS -> RIS kanalı [Boyut: N x M]
        - h_r: RIS -> User kanalı [Boyut: N x 1]
        """
        # 1. Mesafeleri hesapla
        d_bs_user = self.path_loss_engine.calculate_distance(self.bs_pos, self.user_pos)
        d_bs_ris = self.path_loss_engine.calculate_distance(self.bs_pos, self.ris_pos)
        d_ris_user = self.path_loss_engine.calculate_distance(self.ris_pos, self.user_pos)

        # 2. Path loss katsayılarını hesapla
        pl_d = self.path_loss_engine.get_path_loss(d_bs_user, self.path_loss_engine.alpha_d)
        pl_G = self.path_loss_engine.get_path_loss(d_bs_ris, self.path_loss_engine.alpha_r)
        pl_hr = self.path_loss_engine.get_path_loss(d_ris_user, self.path_loss_engine.alpha_r)

        # 3. Küçük ölçekli sönümleme ile birleştir
        # Doğrudan yol NLoS -> Rayleigh
        h_d = pl_d * self._generate_rayleigh((self.M, 1))

        # BS -> RIS yolu LoS baskın -> Rician,
        G = pl_G * self._generate_rician((self.N, self.M))

        # RIS -> User yolu LoS baskın -> Rician
        h_r = pl_hr * self._generate_rician((self.N, 1))

        return h_d, G, h_r


# --- TEST ÇALIŞTIRMASI ---
if __name__ == "__main__":
    # 4 BS anten, 32 RIS elemanı ile:
    channel_sim = WirelessChannel(num_antennas=4, num_elements=32)
    h_d, G, h_r = channel_sim.get_channel_realization()

    print("--- Kanal Boyutları ve Örnekleri ---")
    print(f"h_d (Doğrudan Kanal) Boyutu : {h_d.shape} | Ortalama Güç: {np.mean(np.abs(h_d) ** 2):.2e}")
    print(f"G   (BS -> RIS Kanalı) Boyutu: {G.shape}   | Ortalama Güç: {np.mean(np.abs(G) ** 2):.2e}")
    print(f"h_r (RIS -> User Kanalı) Boyutu: {h_r.shape} | Ortalama Güç: {np.mean(np.abs(h_r) ** 2):.2e}")