#!/usr/bin/env bash

read -p "Enter number of neurons in the first hidden layer: " hid1
while ! [[ "$hid1" =~ ^[0-9]+$ ]]; do 
    read -p "Enter a valid number betwee 600~10: " hid1
done

read -p "Enter number of neurons in the second hidden layer: " hid2
while ! [[ "$hid2" =~ ^[0-9]+$ ]]; do 
    read -p "Enter a valid number between ${hid1}~10" hid2 
done

read -p "Enter number of epochs: " epochs
while ! [[ "$epochs" =~ ^[0-9]+$ ]]; do 
    read -p "Enter a valid number of epochs: " epochs
done

read -p "Enter learning rate: " learn_rate
while ! [[ "$learn_rate" =~ ^[0-9]+(\.[0-9]+)?$ ]]; do 
    read -p "Enter a valid learning rate: " learn_rate
done


sed -i "s/#define HID1 [0-9]*/#define HID1 ${hid1}/" nn.c draw_test.c test_nn.c
sed -i "s/#define HID2 [0-9]*/#define HID2 ${hid2}/" nn.c draw_test.c test_nn.c 
sed -i "s/#define EPOCHS [0-9]*/#define EPOCHS ${epochs}/" nn.c
sed -i "s/#define LEARNING_RATE [0-9]*\.*[0-9]*/#define LEARNING_RATE ${learn_rate}/" nn.c

gcc nn.c -o nn -lm
gcc draw_test.c -o draw_test -lSDL2 -lSDL2_ttf -lm
gcc test_nn.c -o test_nn -lm

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
