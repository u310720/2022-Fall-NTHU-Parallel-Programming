CC = mpicc
CXX = mpicxx
CXXFLAGS = -O3 -lm
CFLAGS = -O3 -lm
TARGETS = hw1

.PHONY: all
all: $(TARGETS)

.PHONY: debug
debug:
	$(CXX) $(CXXFLAGS) -g -DDEBUG $(TARGETS).cc -o $(TARGETS)


.PHONY: clean
clean:
	rm -f $(TARGETS) *.in *.out
