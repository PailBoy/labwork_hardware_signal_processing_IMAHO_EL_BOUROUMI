#include "Wallet.hpp"
#include <iostream>
#include <chrono>

Wallet::Wallet() : rupees(0), virtual_rupees(0) {}

Wallet::~Wallet() {
    for (std::thread &t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }
}

// --- PARTIE PHYSIQUE (Lente) ---
void Wallet::credit(unsigned int val) {
    for (unsigned int i = 0; i < val; ++i) {
        {
            std::lock_guard<std::mutex> lock(myMutex);
            rupees++;
            std::cout << "   [PHYSIQUE] +1 (Reel: " << rupees << ")" << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 10 rubis/sec
    }
}

void Wallet::debit(unsigned int val) {
    for (unsigned int i = 0; i < val; ++i) {
        bool paye = false;
        
        // On essaye de payer tant que ce n'est pas fait
        while (!paye) {
            {
                std::lock_guard<std::mutex> lock(myMutex);
                if (rupees > 0) {
                    rupees--;
                    std::cout << "   [PHYSIQUE] -1 (Reel: " << rupees << ")" << std::endl;
                    paye = true; 
                }
            }
            
            // Si on n'a pas pu payer, on attend un tout petit peu que l'argent rentre
            if (!paye) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        // Pause normale entre deux retraits
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// --- PARTIE VIRTUELLE (rapide) ---
bool Wallet::virtual_credit(unsigned int val) {
    {
        std::lock_guard<std::mutex> lock(virtualMutex);
        virtual_rupees += val;
    }
    // On lance le travail physique en arrière-plan
    workers.push_back(std::thread(&Wallet::credit, this, val));
    return true;
}

bool Wallet::virtual_debit(unsigned int val) {
    std::lock_guard<std::mutex> lock(virtualMutex);
    if (virtual_rupees >= val) {
        virtual_rupees -= val;
        // On a l'argent virtuel, donc on lance le débit physique
        workers.push_back(std::thread(&Wallet::debit, this, val));
        return true;
    } else {
        std::cout << "Transaction refusee : Pas assez de fonds virtuels !" << std::endl;
        return false;
    }
}

unsigned int Wallet::balance() {
    std::lock_guard<std::mutex> lock(virtualMutex);
    return virtual_rupees;
}