CXX = mpic++
CXXFLAGS = -Wall -Wextra -std=c++20 -O2

all: bin/var_1

bin/var_1: src/var_1.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) src/var_1.cpp -o bin/var_1

clean:
	rm -rf bin/* obj/*
