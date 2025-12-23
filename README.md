# Labwork Hardware for Signal Processing - Compte rendu

## Partie 1 : C++ Multi-threading

### 1.2 Version séquentielle
L'intérêt principal de la parallélisation dans ce contexte est de ne pas bloquer le programme principal. Dans un jeu vidéo réel, il est inconcevable de figer l'écran ou d'empêcher le joueur d'agir le temps que l'animation du comptage des pièces se termine. Le multi-threading permet donc de gérer ces transactions en arrière-plan tout en maintenant le jeu fluide.

### 1.3 Paralléliser les paiements
En lançant les threads sans protection, nous avons rencontré une "race condition" (concurrence critique). Les différents threads tentaient de lire et modifier la variable du solde simultanément, ce qui a causé un mélange des sorties dans la console et surtout un résultat final erroné. Pour résoudre cela, il est impératif d'utiliser un Mutex qui verrouille l'accès à la variable partagée, assurant qu'un seul thread peut la modifier à la fois.

### 1.4 Multi-threading et Mutex
Si le Mutex a corrigé la corruption des données, il n'a pas résolu le problème de logique temporelle. Nous avons observé que le thread de débit pouvait s'exécuter alors que le thread de crédit n'avait pas encore fini d'ajouter "physiquement" les rubis (à cause des délais simulés). Le débit échouait donc par manque de fonds immédiats, même si le solde final devait être positif. Ce problème est difficilement solvable avec cette architecture simple car elle dépend trop de la vitesse d'exécution des threads physiques.

### 1.5 Portefeuille instantané
L'approche du portefeuille instantané sépare la logique comptable de la représentation physique. Nous utilisons les méthodes virtuelles pour garantir et valider la transaction instantanément auprès de l'utilisateur, ce qui évite les refus de paiement injustifiés. Ces méthodes lancent ensuite les fonctions physiques (debit/credit) dans des threads séparés pour l'aspect visuel. Le problème de synchronisation physique persiste néanmoins : pour éviter que le compteur physique ne tombe dans le négatif ou ne refuse le débit, j'ai dû implémenter une boucle d'attente dans la fonction de débit physique qui patiente jusqu'à ce que les fonds soient réellement arrivés.

## Partie 2 : Vectorisation avec Pytorch

### 2.2 et 2.3 Optimisation et Benchmark
Pour évaluer l'efficacité de la vectorisation, nous avons comparé deux implémentations du logarithme matriciel sur un lot de 1000 matrices SPD de taille 20x20. La première approche, dite "naïve", traite les matrices séquentiellement via une boucle, tandis que la seconde exploite les dimensions de batch de Pytorch pour traiter l'ensemble simultanément.

Les tests effectués sur une architecture Mac montrent une différence drastique. La version naïve s'exécute en 0.77 seconde, contre seulement 0.036 seconde pour la version vectorisée. Nous obtenons ainsi un facteur d'accélération (speedup) supérieur à 21. Ce résultat illustre que l'overhead de l'interpréteur Python dans les boucles est extrêmement coûteux et que la parallélisation des opérations matricielles est indispensable pour le calcul haute performance, même en tenant compte des contraintes de transfert mémoire spécifiques au backend MPS.

## Partie 3 : Mesures de performance (FLOPS)

### 3.1 & 3.2 Calculs et Mesures
Nous avons estimé la complexité théorique du modèle `SimpleMLP` (2 opérations par poids) et mesuré sa performance réelle sur GPU (Apple Silicon MPS) avec un batch de 10 000 échantillons.

* **Complexité théorique :** 56 592 FLOPS par échantillon.
* **Temps d'exécution :** 0.0005 seconde par batch.
* **Performance mesurée :** 1244 GFLOPS (1.2 TFLOPS).

Ce résultat démontre l'efficacité massive du parallélisme sur GPU pour les larges batchs, rendant le coût d'inférence négligeable.