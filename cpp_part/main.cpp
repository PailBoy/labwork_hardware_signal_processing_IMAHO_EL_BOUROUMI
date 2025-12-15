#include <iostream>
#include "Wallet.hpp"

int main() {
    Wallet myWallet;

    std::cout << "Solde initial : " << myWallet.balance() << std::endl;

    // On fait les opérations via le système virtuel
    std::cout << "Demande d'ajout de 100 rubis..." << std::endl;
    myWallet.virtual_credit(100);

    std::cout << "Demande de retrait de 50 rubis..." << std::endl;
    bool success = myWallet.virtual_debit(50);

    std::cout << "Operation reussie ? " << (success ? "OUI" : "NON") << std::endl;

    // Le solde virtuel doit être correct TOUT DE SUITE (100 - 50 = 50)
    std::cout << "Solde Virtuel (Immediat) : " << myWallet.balance() << std::endl;

    std::cout << "--- Attente de la fin des operations physiques ---" << std::endl;
    // Le destructeur de myWallet sera appelé ici au 'return', 
    // et il attendra sagement que les compteurs physiques finissent.
    
    return 0;
}