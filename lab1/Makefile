DEBUGFLAGS = #-DDEBUG
CC = mpicc
CXX = mpicxx
CXXFLAGS = -std=c++17 -O3 $(DEBUGFLAGS)
CFLAGS = -O3
TARGETS = lab1

.PHONY: all
all: $(TARGETS)

.PHONY: debug
debug: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
