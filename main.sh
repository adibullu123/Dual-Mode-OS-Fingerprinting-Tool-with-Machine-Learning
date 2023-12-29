#!/bin/bash

interface=$1
choice=$2

# Replace these with the actual paths to your Python files
python_file1="pythontest.py"
python_file2="localAnalysis.py"
python_file3="tcp_connector.py"

# Check the user's input and run scripts accordingly
if [ "$choice" == "-a" ]; then
    gnome-terminal -- bash -c "python3 $python_file3; exec bash"
elif [ "$choice" == "-p" ]; then
    # Command to open the first terminal and run the first Python file
    gnome-terminal -- bash -c "python3 $python_file1 $interface; exec bash"

    # Command to open the second terminal and run the second Python file
    gnome-terminal -- bash -c "python3 $python_file2 $interface; exec bash"
else
    echo "Invalid choice. Please choose 'a' or 'p'."
fi
