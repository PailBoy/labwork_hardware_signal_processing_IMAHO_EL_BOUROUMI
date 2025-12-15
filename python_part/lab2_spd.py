import torch
import time

# 1. Vectorize
def vectorize(M):
    rows, cols = torch.tril_indices(M.shape[-2], M.shape[-1])
    return M[..., rows, cols]

# 2. Devectorize
def devectorize(v, dim):
    batch_shape = v.shape[:-1]
    M = torch.zeros(batch_shape + (dim, dim), dtype=v.dtype, device=v.device)
    rows, cols = torch.tril_indices(dim, dim)
    M[..., rows, cols] = v
    mask_diag = torch.eye(dim, device=v.device).bool()
    M_diag = M * mask_diag
    return M + M.transpose(-2, -1) - M_diag

# Helper pour valeurs propres (Version Robuste Mac M1/M2)
def get_eigen(M):
    # Si on est sur MPS (Mac GPU), on passe temporairement sur CPU
    # car linalg.eigh n'est pas encore implémenté sur MPS
    if M.device.type == 'mps':
        vals, vecs = torch.linalg.eigh(M.cpu())
        return vals.to(M.device), vecs.to(M.device)
    else:
        return torch.linalg.eigh(M)

# 3. Matrix Square Root
def spd_sqrt(M):
    vals, vecs = get_eigen(M)
    sqrt_vals = torch.sqrt(vals.clamp(min=1e-6))
    return vecs @ torch.diag_embed(sqrt_vals) @ vecs.transpose(-2, -1)

# 4. Matrix Logarithm
def spd_log(M):
    vals, vecs = get_eigen(M)
    log_vals = torch.log(vals.clamp(min=1e-6))
    return vecs @ torch.diag_embed(log_vals) @ vecs.transpose(-2, -1)

# --- QUESTION 2.2 : PERFORMANCE & BENCHMARK ---
def benchmark():
    # Détection du device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"--- Utilisation du GPU Apple Silicon (MPS) ---")
    else:
        device = torch.device("cpu")
        print("--- Utilisation du CPU ---")

    # Paramètres du test
    BATCH_SIZE = 1000  # On traite 1000 matrices d'un coup
    DIM = 20           # Matrices 20x20
    
    print(f"Génération de {BATCH_SIZE} matrices SPD de taille {DIM}x{DIM}...")
    
    # Génération aléatoire
    A = torch.randn(BATCH_SIZE, DIM, DIM, device=device)
    M = A @ A.transpose(-2, -1) + 1e-3 * torch.eye(DIM, device=device)
    
    # Mesure pour Logarithme Matriciel
    print("Démarrage du calcul 'spd_log'...")
    
    if device.type == 'mps': torch.mps.synchronize()
    start = time.time()
    
    # Calcul
    res = spd_log(M)
    
    if device.type == 'mps': torch.mps.synchronize()
    end = time.time()
    
    print(f"Temps écoulé : {end - start:.4f} secondes")
    print(f"Moyenne par matrice : {(end - start)/BATCH_SIZE * 1000:.4f} ms")

if __name__ == "__main__":
    # Test unitaire rapide
    print("Test unitaire rapide...")
    D = 3
    # On force le device MPS si dispo
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    M_test = torch.eye(D).to(dev)
    
    # Le test ne devrait plus planter grâce au fix dans get_eigen
    spd_sqrt(M_test) 
    print("Test OK (Correction appliquée avec succès).\n")

    # Lancement du benchmark
    benchmark()