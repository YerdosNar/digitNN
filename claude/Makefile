# Makefile - Build Configuration for MNIST Neural Network Project
# Usage: make all     (compile everything)
#        make train   (train the network)
#        make test    (test the network)
#        make draw    (run drawing interface)
#        make clean   (remove compiled files)

CC = gcc
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm

# SDL flags (if available)
SDL_CFLAGS := $(shell pkg-config --cflags sdl2 SDL2_ttf 2>/dev/null)
SDL_LIBS := $(shell pkg-config --libs sdl2 SDL2_ttf 2>/dev/null)

# Object files
OBJS = nn_lib.o

# Targets
TARGETS = train_nn test_nn

# Check if SDL2 is available
ifneq ($(SDL_LIBS),)
    TARGETS += draw_test
endif

.PHONY: all clean train test draw download-mnist

all: $(TARGETS)

# Compile the library
nn_lib.o: nn_lib.c nn_lib.h
	$(CC) $(CFLAGS) -c $< -o $@

# Main training program
train_nn: train_nn.c nn_lib.o nn_lib.h
	$(CC) $(CFLAGS) $< nn_lib.o -o $@ $(LDFLAGS)

# Test program
test_nn: test_nn.c nn_lib.o nn_lib.h
	$(CC) $(CFLAGS) $< nn_lib.o -o $@ $(LDFLAGS)

# Drawing interface (only if SDL2 is available)
draw_test: draw_test.c nn_lib.o nn_lib.h
	$(CC) $(CFLAGS) $< nn_lib.o -o $@ $(SDL_CFLAGS) $(SDL_LIBS) $(LDFLAGS)

clean:
	rm -f $(TARGETS) $(OBJS) weights.bin

# Download MNIST dataset
download-mnist:
	@echo "Downloading MNIST dataset..."
	@wget -nc http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	@wget -nc http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	@wget -nc http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	@wget -nc http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
	@echo "Extracting files..."
	@gunzip -f *.gz
	@echo "MNIST dataset ready!"

# Run the training
train: train_nn
	./train_nn

# Run the test
test: test_nn
	./test_nn

# Run the drawing interface
draw: draw_test
	./draw_test

# Complete workflow
run: all train test
