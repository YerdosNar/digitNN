#!/usr/bin/env bash

read -p "Enter number of neurons in the first hidden layer [default=16]: " hid1
hid1=${hid1:-16}
while ! [[ "$hid1" =~ ^[0-9]+$ ]] || [ "$hid1" -lt 10 ] || [ "$hid1" -gt 600 ]; do
    read -p "Enter a valid number betwee 10~600: " hid1
done

read -p "Enter number of neurons in the second hidden layer [default=16]: " hid2
hid2=${hid2:-16}
while ! [[ "$hid2" =~ ^[0-9]+$ ]] || [ "$hid2" -lt 10 ] || [ "$hid2" -gt "$hid1" ]; do
    read -p "Enter a valid number between 10~${hid1}" hid2
done

read -p "Enter number of epochs [default=50]: " epochs
epochs=${epochs:-50}
while ! [[ "$epochs" =~ ^[0-9]+$ ]] || [ "$epochs" -lt 1 ]; do
    read -p "Enter a valid number of epochs (1 or more): " epochs
done

read -p "Enter learning rate [default=0.01]: " learn_rate
learn_rate=${learn_rate:-0.01}
while ! [[ "$learn_rate" =~ ^[0-9]+(\.[0-9]+)?$ ]] || [ "$(echo "$learn_rate <= 0" | bc)" -eq 1 ]; do
    read -p "Enter a valid learning rate (greate than 0): " learn_rate
done


sed -i "s/#define HID1 [0-9]*/#define HID1 ${hid1}/" nn.c draw_test.c test_nn.c
sed -i "s/#define HID2 [0-9]*/#define HID2 ${hid2}/" nn.c draw_test.c test_nn.c
sed -i "s/#define EPOCHS [0-9]*/#define EPOCHS ${epochs}/" nn.c
sed -i "s/#define LEARNING_RATE [0-9]*\.*[0-9]*/#define LEARNING_RATE ${learn_rate}/" nn.c

gcc nn.c -o nn -O3 -lm || { echo "Compilation of \"nn.c\" failed... And I don't know why."; exit 1; }
gcc test_nn.c -o test_nn -lm ||{ echo "Compilation of \"test_nn.c\" failed... And I don't know why."; exit 1; }

sdl_exist=false
ttf_exist=false

if ! pkg-config --exists sdl2; then
    sdl_exist=true
fi

if ! pkg-config --exists SDL2_ttf; then
    ttf_exist=true
fi


if $sdl_exist || $ttf_exist; then
    echo "SDL2 or SDL2_ttf not found. Attempting to install..."
    # For DebIan like systems
    if command -v apt-get >/dev/null; then
        sudo apt-get update
        if $sdl_exist; then
            sudo apt-get install libsdl2-dev -y
        fi
        if $ttf_exist; then
            sudo apt-get install libsdl-2ttf-dev -y
        fi
    # As I remember Arch uses yum and pacman
    elif command -v yum >/dev/null; then
        if $sdl_exist; then
            sudo yum install libsdl2-dev -y
        fi
        if $ttf_exist; then
            sudo yum install libsdl-2ttf-dev -y
        fi
    elif command -v pacman >/dev/null; then
        packages=()
        if $sdl_exist; then
            packages+=("sdl2")
        fi
        if $ttf_exist; then
            packages+=("sdl2_ttf")
        fi
        sudo pacman -S --noconfirm "${packages[@]}"
    else
        echo "Unsupported package manager. Please install SDL2 and SDL2_ttf manually."
        exit 1
    fi
fi

# If the packages exist or downloaded, we can compile
gcc draw_test.c -o draw_test -lSDL2 -lSDL2_ttf -lm || { echo "Compilation of \"draw_test.c\" failed... And I don't know why."; exit 1; }

echo "Training neural network started..."
time ./nn

echo "How do you want to test the Neural Network?"
echo "By drawing = d; By running test = t"

while true; do
    read -p "Enter your choice (d/t): " choice
    if [ "$choice" == "d" ] || [ "$choice" == "t" ]; then
        break
    else
        echo "Invalid choice. Please enter 'd' or 't'."
    fi
done

if [ "$choice" == "d" ]; then
    ./draw_test
elif [ "$choice" == "t" ]; then
    ./test_nn
fi
