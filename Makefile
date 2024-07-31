CXX = g++
CUDA_PATH ?= /usr/local/cuda

IDIR = ./
SRCS = cudnn_logestic.cpp main.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = cudnn_logestic

CXXFLAGS = --std=c++17 -I$(IDIR) -I$(CUDA_PATH)/include -I$(CUDA_PATH)/targets/x86_64-linux/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcuda -lcudnn -lcudart

all: clean build

build: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

run:
	./$(TARGET) $(ARGS) > result

clean:
	rm -f $(OBJS) $(TARGET)