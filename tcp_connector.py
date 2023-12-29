import subprocess
import re
import socket
import platform
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pyshark

def train_random_forest_classifier(file_name='fm.xlsx', feature_indices=[1, 2], target_index=4, major_version_index=5, minor_version_index=6, test_size=0.2, random_state=48, n_estimators=100):
    # Read the dataset from Excel
    df = pd.read_excel(file_name)

    # Drop any rows with missing values
    df.dropna(inplace=True)

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Extract features based on the specified indices
    X = df.iloc[:, feature_indices]

    # Extract the target variable 'os'
    y = df.iloc[:, target_index]

    # Encode the target variable 'os' using label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create the RandomForestClassifier with n_estimators
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Calculate accuracy on the test set
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy on the test set: {accuracy * 100:.2f}%')

    # Reverse the encoding of class labels to get the original string labels
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred)

    # Generate a classification report with string class labels
    classification_rep = classification_report(y_test_original, y_pred_original)

    print("Classification Report:")
    print(classification_rep)

    return rf_classifier, label_encoder, df

def find_closest_os_row(df, predicted_os, syn_size, window_size):
    # Function to find the closest row in the dataset for a predicted OS
    filtered_df = df[df['os'] == predicted_os]

    if filtered_df.empty:
        return None  # No matching OS in the dataset

    distances = np.linalg.norm(filtered_df.iloc[:, [1, 2]].values - [syn_size, window_size], axis=1)

    closest_index = distances.argmin()
    predicted_major = filtered_df.iloc[closest_index, df.columns.get_loc('major version')]
    predicted_minor = filtered_df.iloc[closest_index, df.columns.get_loc('minor version')]

    return predicted_major, predicted_minor

def predict_os_version(rf_classifier, label_encoder, df, syn_size, window_size):
    # Function to predict OS major and minor version using win_size and syn_size
    new_data_point = np.array([syn_size, window_size])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        predicted_class = rf_classifier.predict([new_data_point])

    predicted_label = label_encoder.inverse_transform(predicted_class)

    major_version, minor_version = None, None
    if predicted_label[0] in df['os'].values:
        major_version, minor_version = find_closest_os_row(df, predicted_label[0], syn_size, window_size)

    return predicted_label[0], major_version, minor_version

def get_ips_from_arp_scan():
    # Function to run arp-scan and capture its output
    arp_scan_output = subprocess.check_output(["arp-scan", "--localnet"]).decode("utf-8")
    ip_pattern = re.compile(r'(\d+\.\d+\.\d+\.\d+)')
    ip_list = ip_pattern.findall(arp_scan_output)
    return list(set(ip_list))

def ping(host):
    # Function to ping a host
    param = "-n" if platform.system().lower() == "windows" else "-c"
    command = ["ping", param, "1", host]
    return subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

def tcp_connect(rf_classifier, label_encoder, df, host):
    # Function to establish a TCP connection and predict OS version for all open ports
    for port in range(1, 1023):  # Check ports from 1 to 1023
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)

        result = sock.connect_ex((host, port))

        if result == 0:
            window_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            syn_size = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG)

            print("*" * 100)
            print(f"{host}:{port} TCP connection established.")

            predicted_os, major_version, minor_version = predict_os_version(rf_classifier, label_encoder, df, syn_size, window_size)

            print(f'Predicted OS: {predicted_os}, Predicted Major Version: {major_version}, Predicted Minor Version: {minor_version}')

            sock.close()
            return True
        else:
            sock.close()

    return False

def main():
    # Train the Random Forest Classifier and get necessary data
    rf_classifier, label_encoder, df = train_random_forest_classifier()

    # Get the list of IPs from arp-scan
    ip_list = get_ips_from_arp_scan()

    if not ip_list:
        print("No IP addresses found. Exiting.")
        return
    
    print(ip_list)

    for ip in ip_list:
        if ping(ip):
            if tcp_connect(rf_classifier, label_encoder, df, ip):
                # Do something with the extracted information if needed
                pass
        else:
            print("*" * 100)
            print(f"{ip} is not reachable.")

    print("-" * 100)

if __name__ == "__main__":
    main()