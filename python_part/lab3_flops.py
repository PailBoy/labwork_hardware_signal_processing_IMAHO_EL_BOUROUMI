import torch
import torch.nn as nn
from thop import profile

# On définit un bloc simple typique d'un modèle de diffusion (Conv2d)
class SimpleDiffusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Convolution standard 3x3
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))

def calculate_flops():
    print("=== PARTIE 3 : CALCUL DES FLOPS ===\n")

    # --- Paramètres du Modèle ---
    # Image typique : Batch=1, Canaux=64, Taille=32x32
    N, C, H, W = 1, 64, 32, 32
    out_channels = 128
    kernel_size = 3
    
    # Création du modèle et de l'input
    model = SimpleDiffusionBlock(C, out_channels)
    input_tensor = torch.randn(N, C, H, W)

    print(f"Input : ({N}, {C}, {H}, {W})")
    print(f"Couche : Conv2d {C} -> {out_channels}, kernel={kernel_size}x{kernel_size}")

    # --- MÉTHODE 1 : Calcul Théorique (Formule) ---
    # Formule pour Conv2d : 2 * K * K * Cin * Cout * Hout * Wout
    # (Le x2 vient du fait qu'une opération MAC = 1 Mult + 1 Add = 2 FLOPs)
    
    # Dimensions de sortie (padding=1 conserve la taille)
    H_out, W_out = H, W
    
    ops_per_pixel = 2 * kernel_size * kernel_size * C * out_channels
    total_flops_theoretical = N * ops_per_pixel * H_out * W_out
    
    print(f"\n[Méthode 1] Calcul Théorique (Formule) :")
    print(f"Formule : 2 * K² * Cin * Cout * H * W")
    print(f"Résultat : {total_flops_theoretical / 1e6:.2f} MFLOPs (Mega FLOPS)")

    # --- MÉTHODE 2 : Calcul Automatique (Librairie thop) ---
    print(f"\n[Méthode 2] Calcul Automatique (via thop) :")
    # thop renvoie souvent les MACs (Multiply-Accumulate), donc FLOPS ~ 2 * MACs
    macs, params = profile(model, inputs=(input_tensor, ), verbose=False)
    flops_thop = 2 * macs 
    
    print(f"MACs détectés : {macs / 1e6:.2f} M")
    print(f"Résultat (approx 2*MACs) : {flops_thop / 1e6:.2f} MFLOPs")

    # --- COMPARAISON ---
    diff = abs(total_flops_theoretical - flops_thop)
    print(f"\nDifférence entre les méthodes : {diff:.0f} FLOPs")
    
    if diff < 1000:
        print("-> Les méthodes concordent parfaitement !")
    else:
        print("-> Légère différence (souvent due aux biais ou optimisations internes).")

if __name__ == "__main__":
    calculate_flops()