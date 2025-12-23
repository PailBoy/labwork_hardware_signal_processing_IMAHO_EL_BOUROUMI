import torch
import time
import numpy as np
from model_gaussian import SimpleMLP

def count_flops_theoretical(model, input_dim, t_dim):

    flops = 0
    
    print("--- Détail du calcul théorique (pour 1 exemple) ---")

    l1_ops = 2 * t_dim * t_dim
    flops += l1_ops
    print(f"Time Embed L1 ({t_dim}->{t_dim}): {l1_ops} FLOPs")
    
    l2_ops = 2 * t_dim * t_dim
    flops += l2_ops
    print(f"Time Embed L2 ({t_dim}->{t_dim}): {l2_ops} FLOPs")

    concat_dim = input_dim + t_dim
    hidden_dim = 128
    output_dim = 2
    
    m1_ops = 2 * concat_dim * hidden_dim
    flops += m1_ops
    print(f"Main Net L1   ({concat_dim}->{hidden_dim}): {m1_ops} FLOPs")
    
    m2_ops = 2 * hidden_dim * hidden_dim
    flops += m2_ops
    print(f"Main Net L2   ({hidden_dim}->{hidden_dim}): {m2_ops} FLOPs")
    
    m3_ops = 2 * hidden_dim * output_dim
    flops += m3_ops
    print(f"Main Net L3   ({hidden_dim}->{output_dim}):   {m3_ops} FLOPs")

    
    print(f"TOTAL Théorique par échantillon : {flops} FLOPs")
    return flops

def benchmark_inference(model, batch_size=10000, device='cpu'):
    model.to(device)
    model.eval()
    
    x = torch.randn(batch_size, 2).to(device)
    t = torch.randint(0, 100, (batch_size,)).float().to(device)
    
    for _ in range(10):
        with torch.no_grad():
            _ = model(x, t)
            
    if device == 'cuda': torch.cuda.synchronize()
    
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
    T_DIM = 50
    BATCH_SIZE = 10000 
    
    print(f"=== Analyse de Performance : SimpleMLP ===")
    
    model = SimpleMLP(t_dim=T_DIM)
    flops_per_sample = count_flops_theoretical(model, input_dim=2, t_dim=T_DIM)
    
    total_flops_batch = flops_per_sample * BATCH_SIZE
    print(f"\nPour un batch de {BATCH_SIZE}, Total = {total_flops_batch/1e6:.2f} MegaFLOPs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    
    print(f"\n--- Mesure du temps réel sur {device.upper()} ---")
    
    avg_time = benchmark_inference(model, batch_size=BATCH_SIZE, device=device)
    print(f"Temps moyen pour un batch de {BATCH_SIZE} : {avg_time:.4f} secondes")
    
    real_gflops = total_flops_batch / avg_time / 1e9
    
    print(f"\n=== RÉSULTATS FINAUX ===")
    print(f"Performance mesurée : {real_gflops:.4f} GFLOPS")
    
    if device == "cpu":
        print("(Note: Sur CPU, c'est souvent bas. Sur GPU, ce sera beaucoup plus haut.)")

if __name__ == "__main__":
    main()