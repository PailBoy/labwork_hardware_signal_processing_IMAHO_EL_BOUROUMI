import torch
import time
from lab2_spd import spd_log, get_eigen # On réutilise tes fonctions

def run_benchmark():
    # Configuration Device (avec le fix pour Mac)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("--- Benchmark sur GPU Apple Silicon (MPS) ---")
    else:
        device = torch.device("cpu")
        print("--- Benchmark sur CPU ---")

    # Paramètres
    BATCH_SIZE = 500  # Nombre de matrices
    DIM = 30          # Taille 30x30
    
    print(f"\n1. Génération de {BATCH_SIZE} matrices SPD ({DIM}x{DIM})...")
    A = torch.randn(BATCH_SIZE, DIM, DIM, device=device)
    # Création SPD : M = A*A^T + epsilon*I
    M_batch = A @ A.transpose(-2, -1) + 1e-3 * torch.eye(DIM, device=device)

    # --- VERSION 1 : NAÏVE (Boucle For - Question 2.1 style) ---
    print("\n2. Exécution version NAÏVE (Boucle For)...")
    if device.type == 'mps': torch.mps.synchronize()
    start_naive = time.time()
    
    results_naive = []
    # On itère une par une (ce qu'il ne faut pas faire en PyTorch)
    for i in range(BATCH_SIZE):
        single_matrix = M_batch[i] # Extraction (D, D)
        # On doit ajouter une dim pour que spd_log marche (1, D, D) puis retirer
        res = spd_log(single_matrix.unsqueeze(0)) 
        results_naive.append(res.squeeze(0))
    
    if device.type == 'mps': torch.mps.synchronize()
    end_naive = time.time()
    time_naive = end_naive - start_naive
    print(f"-> Temps Naïf : {time_naive:.4f} secondes")

    # --- VERSION 2 : OPTIMISÉE (Batch - Question 2.2 style) ---
    print("\n3. Exécution version OPTIMISÉE (Vectorisée)...")
    if device.type == 'mps': torch.mps.synchronize()
    start_opt = time.time()
    
    # On envoie tout le paquet d'un coup !
    results_opt = spd_log(M_batch)
    
    if device.type == 'mps': torch.mps.synchronize()
    end_opt = time.time()
    time_opt = end_opt - start_opt
    print(f"-> Temps Optimisé : {time_opt:.4f} secondes")

    # --- CONCLUSION ---
    speedup = time_naive / time_opt
    print(f"\n=== RÉSULTAT ===")
    print(f"Accélération (Speedup) : x{speedup:.2f}")
    print("L'optimisation vectorielle est validée !")

if __name__ == "__main__":
    run_benchmark()