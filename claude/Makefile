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

# Targets
TARGETS = nn test_nn

# Check if SDL2 is available
ifneq ($(SDL_LIBS),)
    TARGETS += draw_test
endif

.PHONY: all clean train test draw download-mnist

all: $(TARGETS)

# Main training program (self-contained)
nn: nn.c nn.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Test program (self-contained)
test_nn: test_nn.c nn.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Drawing interface (self-contained, only if SDL2 is available)
draw_test: draw_test.c nn.h
	$(CC) $(CFLAGS) $< -o $@ $(SDL_CFLAGS) $(SDL_LIBS) $(LDFLAGS)

clean:
	rm -f $(TARGETS) weights.bin

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
train: nn
	./nn

# Run the test
test: test_nn
	./test_nn

# Run the drawing interface
draw: draw_test
	./draw_test

# Complete workflow
run: all train test
