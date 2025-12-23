import torch
import time
import numpy as np
from model_gaussian import SimpleMLP

def count_flops_theoretical(model, input_dim, t_dim):
    """
    Calcule manuellement les FLOPS théoriques basés sur l'architecture de SimpleMLP.
    Rappel: Pour une couche Linear(In, Out), Ops = 2 * In * Out (Mul + Add)
    """
    flops = 0
    
    print("--- Détail du calcul théorique (pour 1 exemple) ---")
    
    # 1. Analyse de Time Embedding (Sequential)
    # Layer 1: Linear(t_dim, t_dim)
    l1_ops = 2 * t_dim * t_dim
    flops += l1_ops
    print(f"Time Embed L1 ({t_dim}->{t_dim}): {l1_ops} FLOPs")
    
    # Layer 2: Linear(t_dim, t_dim) (Après GELU)
    l2_ops = 2 * t_dim * t_dim
    flops += l2_ops
    print(f"Time Embed L2 ({t_dim}->{t_dim}): {l2_ops} FLOPs")

    # 2. Analyse du Main Net
    # L'entrée concaténée est de taille : input_dim (2) + t_dim (50) = 52
    concat_dim = input_dim + t_dim
    hidden_dim = 128
    output_dim = 2
    
    # Main Layer 1: Linear(52 -> 128)
    m1_ops = 2 * concat_dim * hidden_dim
    flops += m1_ops
    print(f"Main Net L1   ({concat_dim}->{hidden_dim}): {m1_ops} FLOPs")
    
    # Main Layer 2: Linear(128 -> 128)
    m2_ops = 2 * hidden_dim * hidden_dim
    flops += m2_ops
    print(f"Main Net L2   ({hidden_dim}->{hidden_dim}): {m2_ops} FLOPs")
    
    # Main Layer 3: Linear(128 -> 2)
    m3_ops = 2 * hidden_dim * output_dim
    flops += m3_ops
    print(f"Main Net L3   ({hidden_dim}->{output_dim}):   {m3_ops} FLOPs")
    
    # Note: On néglige souvent les fonctions d'activation (GELU, Sin, Cos) 
    # car elles sont coûteuses mais moins nombreuses que les matmul.
    
    print(f"TOTAL Théorique par échantillon : {flops} FLOPs")
    return flops

def benchmark_inference(model, batch_size=10000, device='cpu'):
    model.to(device)
    model.eval()
    
    # Création de données bidons
    x = torch.randn(batch_size, 2).to(device)
    t = torch.randint(0, 100, (batch_size,)).float().to(device)
    
    # Warmup (chauffe)
    for _ in range(10):
        with torch.no_grad():
            _ = model(x, t)
            
    if device == 'cuda': torch.cuda.synchronize()
    
    # Mesure
    iterations = 100
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x, t)
            if device == 'cuda': torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iterations
    return avg_time

def main():
    # Paramètres du modèle (identiques à model_gaussian.py)
    T_DIM = 50
    BATCH_SIZE = 10000 # On prend un gros batch pour saturer le CPU/GPU
    
    print(f"=== Analyse de Performance : SimpleMLP ===")
    
    # 1. Calcul Théorique
    model = SimpleMLP(t_dim=T_DIM)
    flops_per_sample = count_flops_theoretical(model, input_dim=2, t_dim=T_DIM)
    
    total_flops_batch = flops_per_sample * BATCH_SIZE
    print(f"\nPour un batch de {BATCH_SIZE}, Total = {total_flops_batch/1e6:.2f} MegaFLOPs")

    # 2. Mesure Pratique
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Si vous êtes sur Mac M1/M2, utilisez "mps" si disponible, sinon "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    
    print(f"\n--- Mesure du temps réel sur {device.upper()} ---")
    
    avg_time = benchmark_inference(model, batch_size=BATCH_SIZE, device=device)
    print(f"Temps moyen pour un batch de {BATCH_SIZE} : {avg_time:.4f} secondes")
    
    # 3. Calcul des GFLOPS
    # GFLOPS = (Operations Totales) / (Temps en secondes * 10^9)
    real_gflops = total_flops_batch / avg_time / 1e9
    
    print(f"\n=== RÉSULTATS FINAUX ===")
    print(f"Performance mesurée : {real_gflops:.4f} GFLOPS")
    
    if device == "cpu":
        print("(Note: Sur CPU, c'est souvent bas. Sur GPU, ce sera beaucoup plus haut.)")

if __name__ == "__main__":
    main()