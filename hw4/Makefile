CXX:= mpicxx
CXXFLAGS := -O0 #-Wall
LIBS := -lpthread -fopenmp
TARGET := hw4
SOURCE = $(wildcard *.cc)
SOURCE := $(filter-out test_thread_pool.cc, $(SOURCE))
SOURCE := $(filter-out verifier.cc, $(SOURCE))
OBJECT = $(SOURCE:.cc=.o)


all:$(OBJECT)
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $(TARGET)

%.o:%.cc
	$(CXX) $(CXXFLAGS) $(LIBS) -c $<

test_thread_pool: test_thread_pool.cc thread_pool.h
	$(CXX) $< -O3 -lgtest -lgtest_main $(LIBS) -o $@

verifier: verifier.cc
	$(CXX) $< -O3 -o $@

.PHONY:clean clean_out clean_inter
clean:
	rm -rf $(OBJECT) $(TARGET) test_thread_pool verifier *.out *.intermediate

clean_out:
	rm *.out

clean_inter:
	rm *.intermediate