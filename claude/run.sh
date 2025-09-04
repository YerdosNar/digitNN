#!/usr/bin/env bash
# run.sh - Complete Build and Run Script for MNIST Neural Network
# Usage: chmod +x run.sh && ./run.sh

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MNIST Neural Network Training Script ===${NC}\n"

# Function to validate numeric input
validate_number() {
    local value=$1
    local min=$2
    local max=$3
    
    if [[ ! "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt "$min" ] || [ "$value" -gt "$max" ]; then
        return 1
    fi
    return 0
}

# Function to validate float input
validate_float() {
    local value=$1
    local min=$2
    local max=$3
    
    if [[ ! "$value" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        return 1
    fi
    
    # Use bc for float comparison
    if (( $(echo "$value < $min" | bc -l) )) || (( $(echo "$value > $max" | bc -l) )); then
        return 1
    fi
    return 0
}

# Check if bc is installed (needed for float comparison)
if ! command -v bc &> /dev/null; then
    echo -e "${YELLOW}Warning: 'bc' not found. Installing...${NC}"
    if command -v apt-get >/dev/null; then
        sudo apt-get install -y bc
    elif command -v yum >/dev/null; then
        sudo yum install -y bc
    elif command -v pacman >/dev/null; then
        sudo pacman -S --noconfirm bc
    else
        echo -e "${RED}Error: Could not install 'bc'. Please install it manually.${NC}"
        exit 1
    fi
fi

# Network architecture configuration
echo -e "${GREEN}Network Architecture Configuration${NC}"
echo "The network has 3 layers: Input (784) -> Hidden1 -> Hidden2 -> Output (10)"
echo ""

# Hidden layer 1
read -p "Enter number of neurons in the first hidden layer [default=128]: " hid1
hid1=${hid1:-128}
while ! validate_number "$hid1" 10 512; do
    echo -e "${RED}Invalid input!${NC}"
    read -p "Enter a valid number between 10-512: " hid1
done

# Hidden layer 2
read -p "Enter number of neurons in the second hidden layer [default=64]: " hid2
hid2=${hid2:-64}
while ! validate_number "$hid2" 10 "$hid1"; do
    echo -e "${RED}Invalid input!${NC}"
    read -p "Enter a valid number between 10-${hid1}: " hid2
done

# Training parameters
echo -e "\n${GREEN}Training Parameters${NC}"

# Epochs
read -p "Enter number of epochs [default=30]: " epochs
epochs=${epochs:-30}
while ! validate_number "$epochs" 1 100; do
    echo -e "${RED}Invalid input!${NC}"
    read -p "Enter a valid number between 1-100: " epochs
done

# Batch size
read -p "Enter batch size [default=64]: " batch_size
batch_size=${batch_size:-64}
while ! validate_number "$batch_size" 1 256; do
    echo -e "${RED}Invalid input!${NC}"
    read -p "Enter a valid number between 1-256: " batch_size
done

# Learning rate
read -p "Enter learning rate [default=0.001]: " learn_rate
learn_rate=${learn_rate:-0.001}
while ! validate_float "$learn_rate" 0.0001 0.1; do
    echo -e "${RED}Invalid input!${NC}"
    read -p "Enter a valid learning rate between 0.0001-0.1: " learn_rate
done

# Momentum
read -p "Enter momentum [default=0.9]: " momentum
momentum=${momentum:-0.9}
while ! validate_float "$momentum" 0 0.99; do
    echo -e "${RED}Invalid input!${NC}"
    read -p "Enter a valid momentum between 0-0.99: " momentum
done

# L2 regularization
read -p "Enter L2 regularization lambda [default=0.0001]: " l2_lambda
l2_lambda=${l2_lambda:-0.0001}
while ! validate_float "$l2_lambda" 0 0.01; do
    echo -e "${RED}Invalid input!${NC}"
    read -p "Enter a valid L2 lambda between 0-0.01: " l2_lambda
done

echo -e "\n${GREEN}Configuration Summary:${NC}"
echo "Architecture: 784 -> ${hid1} -> ${hid2} -> 10"
echo "Epochs: ${epochs}"
echo "Batch size: ${batch_size}"
echo "Learning rate: ${learn_rate}"
echo "Momentum: ${momentum}"
echo "L2 regularization: ${l2_lambda}"
echo ""

# Update the source files with the configuration
echo -e "${BLUE}Updating source files...${NC}"

# Update train_nn.c
sed -i "s/#define HID1 [0-9]*/#define HID1 ${hid1}/" train_nn.c 2>/dev/null || \
sed -i '' "s/#define HID1 [0-9]*/#define HID1 ${hid1}/" train_nn.c

sed -i "s/#define HID2 [0-9]*/#define HID2 ${hid2}/" train_nn.c 2>/dev/null || \
sed -i '' "s/#define HID2 [0-9]*/#define HID2 ${hid2}/" train_nn.c

sed -i "s/#define EPOCHS [0-9]*/#define EPOCHS ${epochs}/" train_nn.c 2>/dev/null || \
sed -i '' "s/#define EPOCHS [0-9]*/#define EPOCHS ${epochs}/" train_nn.c

sed -i "s/#define BATCH_SIZE [0-9]*/#define BATCH_SIZE ${batch_size}/" train_nn.c 2>/dev/null || \
sed -i '' "s/#define BATCH_SIZE [0-9]*/#define BATCH_SIZE ${batch_size}/" train_nn.c

sed -i "s/#define LEARNING_RATE [0-9]*\.*[0-9]*f*/#define LEARNING_RATE ${learn_rate}f/" train_nn.c 2>/dev/null || \
sed -i '' "s/#define LEARNING_RATE [0-9]*\.*[0-9]*f*/#define LEARNING_RATE ${learn_rate}f/" train_nn.c

sed -i "s/#define MOMENTUM [0-9]*\.*[0-9]*f*/#define MOMENTUM ${momentum}f/" train_nn.c 2>/dev/null || \
sed -i '' "s/#define MOMENTUM [0-9]*\.*[0-9]*f*/#define MOMENTUM ${momentum}f/" train_nn.c

sed -i "s/#define L2_LAMBDA [0-9]*\.*[0-9]*f*/#define L2_LAMBDA ${l2_lambda}f/" train_nn.c 2>/dev/null || \
sed -i '' "s/#define L2_LAMBDA [0-9]*\.*[0-9]*f*/#define L2_LAMBDA ${l2_lambda}f/" train_nn.c

# Check for training data
echo -e "\n${BLUE}Checking for MNIST data files...${NC}"

training_files=("../files/train-images-idx3-ubyte" "../files/train-labels-idx1-ubyte")
test_files=("../files/t10k-images-idx3-ubyte" "../files/t10k-labels-idx1-ubyte")
missing_files=()

for file in "${training_files[@]}" "${test_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo -e "${YELLOW}Missing MNIST data files:${NC}"
    printf '%s\n' "${missing_files[@]}"
    echo ""
    echo "Please download the MNIST dataset from:"
    echo "http://yann.lecun.com/exdb/mnist/"
    echo ""
    read -p "Do you want me to try downloading them automatically? (y/n): " download_choice
    
    if [[ "$download_choice" =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Downloading MNIST dataset...${NC}"
        
        # Download training set
        wget -nc http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
        wget -nc http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
        
        # Download test set
        wget -nc http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
        wget -nc http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
        
        # Extract files
        echo -e "${BLUE}Extracting files...${NC}"
        gunzip -f *.gz
    else
        echo -e "${RED}Cannot proceed without MNIST data files.${NC}"
        exit 1
    fi
fi

# Check for SDL2 dependencies
echo -e "\n${BLUE}Checking SDL2 dependencies...${NC}"

sdl2_missing=false
sdl2_ttf_missing=false

if ! pkg-config --exists sdl2 2>/dev/null; then
    sdl2_missing=true
fi

if ! pkg-config --exists SDL2_ttf 2>/dev/null; then
    sdl2_ttf_missing=true
fi

if $sdl2_missing || $sdl2_ttf_missing; then
    echo -e "${YELLOW}SDL2 libraries not found. Drawing interface will not be compiled.${NC}"
    echo "To enable the drawing interface, install SDL2 and SDL2_ttf:"
    echo "  Ubuntu/Debian: sudo apt-get install libsdl2-dev libsdl2-ttf-dev"
    echo "  Fedora/RHEL: sudo yum install SDL2-devel SDL2_ttf-devel"
    echo "  Arch: sudo pacman -S sdl2 sdl2_ttf"
    echo "  macOS: brew install sdl2 sdl2_ttf"
    echo ""
fi

# Compilation
echo -e "${BLUE}Compiling neural network...${NC}"

# Clean previous builds
make clean >/dev/null 2>&1

# Build all targets
if ! make all; then
    echo -e "${RED}Compilation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Compilation successful!${NC}"

# Start training
echo -e "\n${GREEN}Starting neural network training...${NC}"
echo "This may take several minutes depending on your hardware."
echo ""

time ./train_nn

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Training completed successfully!${NC}"
    
    # Test the network
    echo -e "\n${BLUE}How would you like to test the neural network?${NC}"
    echo "[1] Drawing interface (interactive)"
    echo "[2] Test set evaluation (batch)"
    echo "[3] Both"
    echo "[4] Skip testing"
    
    while true; do
        read -p "Enter your choice (1-4): " choice
        case $choice in
            1)
                if [ -f ./draw_test ]; then
                    echo -e "${GREEN}Starting drawing interface...${NC}"
                    ./draw_test
                else
                    echo -e "${RED}Drawing interface not available (SDL2 required)${NC}"
                fi
                break
                ;;
            2)
                echo -e "${GREEN}Running test set evaluation...${NC}"
                ./test_nn
                break
                ;;
            3)
                echo -e "${GREEN}Running test set evaluation...${NC}"
                ./test_nn
                echo ""
                if [ -f ./draw_test ]; then
                    read -p "Press Enter to start drawing interface..."
                    ./draw_test
                else
                    echo -e "${RED}Drawing interface not available (SDL2 required)${NC}"
                fi
                break
                ;;
            4)
                echo -e "${BLUE}Testing skipped.${NC}"
                break
                ;;
            *)
                echo -e "${RED}Invalid choice. Please enter 1, 2, 3, or 4.${NC}"
                ;;
        esac
    done
else
    echo -e "${RED}Training failed!${NC}"
    exit 1
fi

echo -e "\n${GREEN}Done!${NC}"
