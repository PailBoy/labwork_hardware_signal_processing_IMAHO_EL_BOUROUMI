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



def spd_log_naive(M_batch):
    batch_size = M_batch.shape[0]
    result_list = []
    # On itère une par une (ce qu'il ne faut pas faire en temps normal)
    for i in range(batch_size):
        # On traite la matrice i
        single_res = spd_log(M_batch[i]) 
        result_list.append(single_res)
    return torch.stack(result_list)

def benchmark():
    # 1. Configuration du Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("--- Mode : NVIDIA GPU (CUDA) ---")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("--- Mode : Apple Silicon (MPS) ---")
        print("Note : Le benchmark peut être biaisé par les transferts CPU/GPU dans get_eigen.")
    else:
        device = torch.device("cpu")
        print("--- Mode : CPU ---")

    # 2. Génération des Données
    BATCH_SIZE = 1000   # Nombre de matrices
    DIM = 20            # Taille 20x20
    
    print(f"\nGénération de {BATCH_SIZE} matrices SPD ({DIM}x{DIM})...")
    # Création de matrices définies positives (A * A^T + epsilon*I)
    A = torch.randn(BATCH_SIZE, DIM, DIM, device=device)
    M = A @ A.transpose(-2, -1) + 1e-3 * torch.eye(DIM, device=device)

    # Fonction utilitaire pour la synchro GPU (pour avoir un temps précis)
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()

    print("\n--- Début du Benchmark : Logarithme Matriciel ---")

    # 3. Mesure Version NAÏVE (Boucle For)
    synchronize()
    start_naive = time.time()
    
    _ = spd_log_naive(M)
    
    synchronize()
    end_naive = time.time()
    time_naive = end_naive - start_naive
    print(f"1. Version Naïve (Boucle) : {time_naive:.4f} secondes")

    # 4. Mesure Version OPTIMISÉE (Batch Vectorisé)
    synchronize()
    start_opt = time.time()
    
    _ = spd_log(M)
    
    synchronize()
    end_opt = time.time()
    time_opt = end_opt - start_opt
    print(f"2. Version Optimisée (Batch) : {time_opt:.4f} secondes")

    # 5. Résultats
    print("-" * 40)
    if time_opt > 0:
        speedup = time_naive / time_opt
        print(f"SPEEDUP (Accélération) : x{speedup:.2f}")
    else:
        print("Temps trop court pour calculer le speedup.")
    print("-" * 40)

if __name__ == "__main__":
    benchmark()