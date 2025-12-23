#ifndef WALLET_HPP
#define WALLET_HPP

#include <mutex>
#include <vector>
#include <thread>

class Wallet {

    unsigned int rupees;
    std::mutex myMutex;

    unsigned int virtual_rupees;
    std::mutex virtualMutex;

    std::vector<std::thread> workers;

public:
    Wallet();
    ~Wallet(); 

    void credit(unsigned int val);
    void debit(unsigned int val);

    bool virtual_credit(unsigned int val);
    bool virtual_debit(unsigned int val);

    unsigned int balance(); 
};

#endif
