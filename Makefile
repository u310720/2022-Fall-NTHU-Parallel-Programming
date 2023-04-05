CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3 #-DTIMER #-DDEBUG #-g -fsanitize=address
hw2a: CFLAGS += -pthread
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a hw2b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
