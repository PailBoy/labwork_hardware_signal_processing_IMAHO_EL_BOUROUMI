# Compilateur (sur Mac, g++ est un alias pour clang++)
CXX = g++

# Options de compilation
# -std=c++17 : Utilise le standard C++17 (moderne et stable)
# -Wall -Wextra : Active les avertissements utiles pour éviter les bugs
# -pthread : Indispensable pour le multi-threading (std::thread)
# -g : Ajoute les informations de débogage (utile pour VS Code)
CXXFLAGS = -std=c++17 -Wall -Wextra -pthread -g

# Nom de l'exécutable final
TARGET = wallet_app

# Fichiers sources (on assume que tu auras ces deux fichiers comme demandé dans le TP)
SRCS = main.cpp Wallet.cpp

# Transformation des .cpp en .o (fichiers objets)
OBJS = $(SRCS:.cpp=.o)

# Règle par défaut (ce qui se lance quand tu tapes juste "make")
all: $(TARGET)

# Édition de liens : crée l'exécutable à partir des objets
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Compilation des fichiers sources en objets
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Nettoyage des fichiers générés
clean:
	rm -f $(OBJS) $(TARGET)

# Pour éviter des conflits avec des fichiers qui s'appelleraient "all" ou "clean"
.PHONY: all clean