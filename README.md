# Labwork Hardware for Signal Processing - Compte rendu

Ce fichier résume les réponses aux questions posées lors du développement du système de portefeuille multi-threadé.

## Partie 1 : C++ Multi-threading

### 1.2 Version séquentielle
L'intérêt principal de la parallélisation dans ce contexte est de ne pas bloquer le programme principal. Dans un jeu vidéo réel, il est inconcevable de figer l'écran ou d'empêcher le joueur d'agir le temps que l'animation du comptage des pièces se termine. Le multi-threading permet donc de gérer ces transactions en arrière-plan tout en maintenant le jeu fluide.

### 1.3 Paralléliser les paiements
En lançant les threads sans protection, nous avons rencontré une "race condition" (concurrence critique). Les différents threads tentaient de lire et modifier la variable du solde simultanément, ce qui a causé un mélange des sorties dans la console et surtout un résultat final erroné. Pour résoudre cela, il est impératif d'utiliser un Mutex qui verrouille l'accès à la variable partagée, assurant qu'un seul thread peut la modifier à la fois.

### 1.4 Multi-threading et Mutex
Si le Mutex a corrigé la corruption des données, il n'a pas résolu le problème de logique temporelle. Nous avons observé que le thread de débit pouvait s'exécuter alors que le thread de crédit n'avait pas encore fini d'ajouter "physiquement" les rubis (à cause des délais simulés). Le débit échouait donc par manque de fonds immédiats, même si le solde final devait être positif. Ce problème est difficilement solvable avec cette architecture simple car elle dépend trop de la vitesse d'exécution des threads physiques.

### 1.5 Portefeuille instantané
L'approche du portefeuille instantané sépare la logique comptable de la représentation physique. Nous utilisons les méthodes virtuelles pour garantir et valider la transaction instantanément auprès de l'utilisateur, ce qui évite les refus de paiement injustifiés. Ces méthodes lancent ensuite les fonctions physiques (debit/credit) dans des threads séparés pour l'aspect visuel. Le problème de synchronisation physique persiste néanmoins : pour éviter que le compteur physique ne tombe dans le négatif ou ne refuse le débit, j'ai dû implémenter une boucle d'attente dans la fonction de débit physique qui patiente jusqu'à ce que les fonds soient réellement arrivés.