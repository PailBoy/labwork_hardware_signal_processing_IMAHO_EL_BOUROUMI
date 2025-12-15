#ifndef WALLET_HPP
#define WALLET_HPP

#include <mutex>
#include <vector>
#include <thread>

class Wallet {
    // Partie Physique (ce qu'on a déjà)
    unsigned int rupees;
    std::mutex myMutex;

    // Partie Virtuelle (Nouveau !)
    unsigned int virtual_rupees;
    std::mutex virtualMutex;

    // Gestion des threads pour éviter les crashs
    std::vector<std::thread> workers;

public:
    Wallet();
    ~Wallet(); // Destructeur pour nettoyer les threads

    // Méthodes physiques (privées ou utilisées par les threads internes)
    void credit(unsigned int val);
    void debit(unsigned int val);

    // Méthodes virtuelles (publiques pour l'utilisateur)
    bool virtual_credit(unsigned int val);
    bool virtual_debit(unsigned int val);

    unsigned int balance(); // Renvoie maintenant le virtuel
};

#endif
