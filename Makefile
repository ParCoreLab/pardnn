SRC_DIR := src
BIN_DIR := bin

EXE := $(BIN_DIR)/pardnn
SRC := $(wildcard $(SRC_DIR)/*.cpp)

CXXFLAGS = -Wall -Iinclude

.PHONY: all clean

all: $(EXE)
	cp $(EXE) pardnn

$(EXE): $(SRC)
	$(CXX) $^ -o $@ 

$(SRC): $(BIN_DIR)

$(BIN_DIR):
	mkdir -p $@

clean:
	@$(RM) -rv pardnn $(BIN_DIR)
