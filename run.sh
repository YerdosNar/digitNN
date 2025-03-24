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

gcc nn.c -o nn -lm || { echo "Compilation of \"nn.c\" failed... And I don't know why."; exit 1; }
gcc test_nn.c -o test_nn -lm ||{ echo "Compilation of \"test_nn.c\" failed... And I don't know why."; exit 1; }

if ! pkg-config --exists sdl2 SDL_ttf; then
    echo "SDL2 or SDL2_ttf not found. Attempting to install..."
    if command -v apt-get >/dev/null; then
        sudo apt-get update && sudo apt-get install libsdl2-dev libsdl2-ttf-dev -y 
    elif command -v yum >/dev/null; then
        sudo yum install sdl2-devel sdl2_ttf_devel -y
    elif command -v pacman >/dev/null; then
        sudo pacman -S --noconfirm sdl2 sdl2-ttf
    else 
        echo "Unsupported package manager. Please install SDL2 and SDL2_ttf by yourself."
        exit 1 
    fi 
fi
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
