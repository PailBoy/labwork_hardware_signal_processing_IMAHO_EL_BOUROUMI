#include <iostream>
#include "Wallet.hpp"

int main() {
    Wallet myWallet;

    std::cout << "Solde initial : " << myWallet.balance() << std::endl;

    std::cout << "Demande d'ajout de 100 rubis..." << std::endl;
    myWallet.virtual_credit(100);

    std::cout << "Demande de retrait de 50 rubis..." << std::endl;
    bool success = myWallet.virtual_debit(50);

    std::cout << "Operation reussie ? " << (success ? "OUI" : "NON") << std::endl;

    std::cout << "Solde Virtuel (Immediat) : " << myWallet.balance() << std::endl;

    std::cout << "--- Attente de la fin des operations physiques ---" << std::endl;
    return 0;
}