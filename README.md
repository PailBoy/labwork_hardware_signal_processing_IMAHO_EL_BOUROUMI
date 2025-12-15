# Labwork Hardware for Signal Processing

This repository contains the work for the multi-threading and performance optimization lab.

## Part 1: C++ Multi-threading (The Wallet)

### 1.2 Sequential Version
[cite_start]**Why parallelize?** [cite: 19]
In a real video game or RPG, we cannot freeze the entire screen or stop the gameplay just to count coins. The addition of money must happen in the background while the player continues to play. That is why we need threads.

### 1.3 Parallel Payments
[cite_start]**Problems encountered:** [cite: 22]
When we ran the threads in parallel without protection, the console output was completely mixed up. More importantly, the final calculation was wrong (we got 91 instead of 50).
[cite_start]**Solution:** [cite: 23]
This was a race condition. Both threads were trying to modify the variable at the exact same time. The solution is to use a Mutex to protect the access to the shared variable.

### 1.4 Mutex
[cite_start]**Did it solve everything?** [cite: 26, 27]
The mutex solved the data corruption and the display issues, but a logic problem remained. We ended up with a balance of 51 instead of 50. This happened because the debit thread checked the balance when it was still 0 (because the credit thread was sleeping) and decided to skip the payment.
[cite_start]**Is it solvable?** [cite: 28]
It is difficult to solve with the simple architecture because the debit thread needs to wait for the money to arrive physically, rather than just skipping the turn.

### 1.5 Instant Wallet
[cite_start]**Why call debit/credit from virtual methods?** [cite: 42]
We use the virtual methods to validate the transaction immediately for the user (UI), and then we launch the slow physical counting in a background thread so it does not block the program.
[cite_start]**Persisting problems:** [cite: 43]
Even with the virtual wallet, the physical threads can still desynchronize if the debit tries to run before the credit thread has physically added the coins. I solved this by adding a loop in the debit function that waits for the physical funds to arrive before proceeding.
