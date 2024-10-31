import random
import time
import threading
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx
from matplotlib.animation import FuncAnimation

disconnect=None
disconnect_history=[]
PROB= 0.3
RECONNECT=0.8
class Node:
    def __init__(self, ip):
        self.ip = ip
        self.forwarding_table = {}  # Stores forwarding entries
        self.advertised_table = {}  # Stores advertised entries
        self.sent_packets = []
        self.received_packets = []
        self.control_packets = 0
        
    def send_packet(self, packet, next_hop):
        self.sent_packets.append(packet)
        print(f"Node {self.ip} sent packet {packet.sequence_number} to Node {next_hop}")

    def receive_packet(self, packet):
        self.received_packets.append(packet)
        if not packet.is_data:
            self.control_packets += 1
        print(f"Node {self.ip} received packet {packet.sequence_number}")
    def __repr__(self):
        return f"Node({self.ip})"

class Packet:
    def __init__(self, source, destination, sequence_number, is_data=True):
        self.source = source
        #self.update_type = update_type  # 1 for incremental, 0 for full dump
        self.sequence_number = sequence_number
        self.destination= destination
        self.is_data = is_data
        self.timestamp = time.time()
        #self.dest_count = dest_count
        #self.dest_ips = []
        #self.hops = []
        #self.route_seq_nos = []

    """def add_destination(self, new_dest_ip, hop, route_seq_no):
        self.dest_ips.append(new_dest_ip)
        self.hops.append(hop)
        self.route_seq_nos.append(route_seq_no)"""


class NetworkSimulator:
    def __init__(self, nodes, node_dict):
        self.nodes = nodes # here nodes is the list of all node objects
        self.sequence_number = 0
        self.total_packets = 0
        self.delivered_packets = 0
        self.total_latency = 0
        self.node_dict = node_dict
    def find_route(self, source, destination):
        # A placeholder routing logic for DSDV
        if source in self.node_dict:
            source_obj= self.node_dict.get(source) 
        else:
            return None
        # since here source is an ip
        if destination in source_obj.forwarding_table:
            return source_obj.forwarding_table[destination]["next_hop"]
        else:
            return None  # No route found

    def send_packet(self, source, destination):
        next_hop = self.find_route(source, destination)
        if next_hop is not None:
            packet = Packet(source, destination, self.sequence_number)
            self.sequence_number += 1
            source_obj= self.node_dict.get(source)
            source_obj.send_packet(packet, next_hop)
            self.deliver_packet(packet)
        else:
            print("No route found. Packet dropped.")

    def deliver_packet(self, packet):
        # Simulate the packet reaching the destination

        destination_node = self.node_dict.get(packet.destination)
        destination_node.receive_packet(packet)
        latency = time.time() - packet.timestamp
        self.total_latency += latency
        self.delivered_packets += 1

    def simulate(self, num_packets, source, destination):
        for _ in range(num_packets):
            self.send_packet(source, destination)
            time.sleep(0.01)  # Simulate slight delay

    def calculate_metrics(self):
        # Throughput: Packets delivered / total time taken (in packets per second)
        throughput = self.delivered_packets / (self.total_latency if self.total_latency > 0 else 1)

        # Latency: Average latency of delivered packets
        average_latency = self.total_latency / self.delivered_packets if self.delivered_packets > 0 else 0

        # Packet Delivery Ratio (PDR): Delivered packets / Total sent packets
        pdr = self.delivered_packets / self.total_packets if self.total_packets > 0 else 0

        # Control Overhead: Control packets / Total packets
        control_overhead = sum(node.control_packets for node in self.nodes) / (self.total_packets if self.total_packets > 0 else 1)

        # Routing Load: Number of control packets / Delivered packets
        routing_load = sum(node.control_packets for node in self.nodes) / (self.delivered_packets if self.delivered_packets > 0 else 1)

        return {
            "Throughput": throughput,
            "Average Latency": average_latency,
            "Packet Delivery Ratio (PDR)": pdr,
            "Control Overhead": control_overhead,
            "Routing Load": routing_load
        }
 













node_ips = [f"192.168.1.{i}" for i in range(1, 13)]
nodes = [Node(ip) for ip in node_ips]

network_topology = defaultdict(list)
for node in nodes:
    # Choose 2-4 random other nodes for each connection
    neighbors = random.sample([n for n in nodes if n != node], random.randint(2, 4))
    for neighbor in neighbors:
        if neighbor not in network_topology[node]:  # Avoid duplicate entries
            network_topology[node].append(neighbor)
            network_topology[neighbor].append(node)  # Add reverse connection
# Initialize graph
G = nx.Graph()

# Function to update the graph dynamically
def update_graph():
    G.clear()  # Clear previous graph
    for node, neighbors in network_topology.items():
        for neighbor in neighbors:
            G.add_edge(node.ip, neighbor.ip)

def generate_prob():
    return random.random() < PROB   
def generate_reconnect_prob():
    return random.random() < RECONNECT   

# Function to disconnect a node
def disconnect_graph(node):
    if node in disconnect_history:
        return
    if node in network_topology:
        for neighbor in network_topology[node]:
            if node in network_topology[neighbor]:
                network_topology[neighbor].remove(node)
        network_topology[node].clear()
        disconnect_history.append(node)
        print(f"Node {node.ip} disconnected.")

# Function to reconnect a node to new neighbors
def reconnect_graph(node, new_neighbors):
    for neighbor_ip in new_neighbors:
        neighbor = next((n for n in nodes if n.ip == neighbor_ip), None)
        if neighbor and neighbor != node:
            network_topology[node].append(neighbor)
            network_topology[neighbor].append(node)
    print(f"Node {node.ip} reconnected to {[n for n in new_neighbors]}.")

# Function to perform network changes and visualize them

# Print the network topology
for node, connections in network_topology.items():
    print(f"{node}: {connections}")

def initialize_routing_tables():
    for node in nodes:
        node.forwarding_table[node.ip]={
        "next_hop": node.ip,
        "metric": 0,
        "sequence_number": 0
        }
        for neighbor in network_topology[node]:
            node.forwarding_table[neighbor.ip] = {
                "next_hop": neighbor.ip,
                "metric": 1,
                "sequence_number": 0
            }
            node.advertised_table[neighbor.ip] = node.forwarding_table[neighbor.ip].copy()

initialize_routing_tables()
node_dict = {node.ip: node for node in nodes}
print("\nInitial Forwarding:")
for node in nodes:
    print(f"\nNode {node.ip} Forwarding Table:")
    for dest, route in node.forwarding_table.items():
        print(f"  {dest} -> Next Hop: {route['next_hop']}, Metric: {route['metric']}, Seq#: {route['sequence_number']}")
    #print(f"\nNode {node.ip} Advertised Tables:")
    #for dest, route in node.advertised_table.items():
        #print(f"  {dest} -> Next Hop: {route['next_hop']}, Metric: {route['metric']}, Seq#: {route['sequence_number']}")

def advertise_routes1(disconnect):
    
    for node in nodes:
        #start time
        for neighbor in network_topology[node]:
            if(disconnect!=None and neighbor == disconnect):
                continue
            for dest, route in node.forwarding_table.items():
                #print((dest == neighbor.ip))
                #print(dest)
                if(disconnect!=None and dest == disconnect):
                    continue
                if dest in neighbor.forwarding_table and dest == neighbor.ip :
                    continue
                if (dest  not in neighbor.forwarding_table and dest != neighbor.ip) or \
                   (dest in neighbor.forwarding_table and neighbor.forwarding_table[dest]["sequence_number"] < route["sequence_number"] and route["metric"]!= float("inf")) or \
                   (dest in neighbor.forwarding_table and neighbor.forwarding_table[dest]["next_hop"]=="#" ) or \
                   ( dest in neighbor.forwarding_table and neighbor.forwarding_table[dest]["sequence_number"] == route["sequence_number"] and neighbor.forwarding_table[dest]["metric"] > route["metric"] + 1):
                     
                    if(dest in neighbor.forwarding_table and route["metric"] != float("inf") and neighbor.forwarding_table[dest]["metric"] == float("inf")): #code for reconnection
                        neighbor.forwarding_table[dest] = {
                            "next_hop": node.ip,
                            "metric": route["metric"] +1,
                            "sequence_number": route["sequence_number"]
                        }
                    else:
                        neighbor.forwarding_table[dest] = {
                            "next_hop": node.ip,
                            "metric": route["metric"] +1,
                            "sequence_number": route["sequence_number"]
                        }
                    neighbor.advertised_table[dest] = neighbor.forwarding_table[dest].copy()
               # elif(dest in neighbor.forwarding_table and neighbor.forwarding_table[dest]["sequence_number"] == route["sequence_number"] and neighbor.forwarding_table[dest]["metric"] == route["metric"] + 1):
                  #  neighbor.forwarding_table[dest]["next_hop"]=node.ip
                   # neighbor.forwarding_table[dest]["sequence_number"]+=2
        #end time
def bfs(neighbor, incremental_dump, visited1):
    visited1.append(neighbor)
    visited= visited1.copy()
    #print(incremental_dump)
    for direct in network_topology[neighbor]:
        # now we got that neighboring node
        flag=0
        if direct in visited:
            continue
        for eachip in incremental_dump:
            if direct.forwarding_table[eachip]["metric"] > neighbor.forwarding_table[eachip]["metric"] +1 :
                direct.forwarding_table[eachip] = {
                    "next_hop": neighbor.ip,
                    "metric": neighbor.forwarding_table[eachip]["metric"] +1,
                    "sequence_number": neighbor.forwarding_table[eachip]["sequence_number"]
                }
                flag=1
        if flag==1 and direct not in visited:
            bfs(direct, incremental_dump, visited)            
def advertise_routes(disconnect):
    
    for node in nodes:
        #start time
        if disconnect!=None and node == disconnect :
            continue
        for neighbor in network_topology[node]:
            incremental_dump=[]
            if(disconnect!=None and neighbor.ip == disconnect):
                continue
            for dest, route in node.forwarding_table.items():
                #print((dest == neighbor.ip))
                #print(dest)
                
                if(disconnect!=None and dest == disconnect):
                    continue
                if dest in neighbor.forwarding_table and dest == neighbor.ip :
                    continue
                if dest in neighbor.forwarding_table and dest == node.ip:
                    continue
                if (dest  not in neighbor.forwarding_table and dest != neighbor.ip) or \
                   (dest in neighbor.forwarding_table and neighbor.forwarding_table[dest]["sequence_number"] < route["sequence_number"] and route["metric"]!= float("inf")) or \
                   (dest in neighbor.forwarding_table and neighbor.forwarding_table[dest]["next_hop"]=="#" ) or \
                   ( dest in neighbor.forwarding_table and neighbor.forwarding_table[dest]["sequence_number"] == route["sequence_number"] and route["metric"] !=float("inf") and neighbor.forwarding_table[dest]["metric"] > route["metric"] + 1):
                     
                    if(dest in neighbor.forwarding_table and route["metric"] != float("inf") and neighbor.forwarding_table[dest]["metric"] == float("inf")): #code for reconnection
                        neighbor.forwarding_table[dest] = {
                            "next_hop": node.ip,
                            "metric": route["metric"] +1,
                            "sequence_number": route["sequence_number"]
                        }
                    else:
                        print("neighbor", neighbor.ip , "and for node", node.ip)
                        neighbor.forwarding_table[dest] = {
                            "next_hop": node.ip,
                            "metric": route["metric"] +1,
                            "sequence_number": route["sequence_number"]
                        }
                    if disconnect!=None:    
                        incremental_dump.append(dest)
                        print(dest, "for", node.ip , "and neighbor", neighbor.ip)
                    neighbor.advertised_table[dest] = neighbor.forwarding_table[dest].copy()
            if disconnect!=None:
                disconnectednode=node_dict.get(disconnect)
                visited=[node, disconnectednode]
            if len(incremental_dump)!=0:
                #print("here")
               # print("incremental_dump",incremental_dump)
               # print("visited", visited)
                bfs(neighbor, incremental_dump, visited) # here incremental_dump consists of all the changes dest ips    
               # elif(dest in neighbor.forwarding_table and neighbor.forwarding_table[dest]["sequence_number"] == route["sequence_number"] and neighbor.forwarding_table[dest]["metric"] == route["metric"] + 1):
                  #  neighbor.forwarding_table[dest]["next_hop"]=node.ip
                   # neighbor.forwarding_table[dest]["sequence_number"]+=2
        #end time        
def disconnect_node1(moving_node):
    visited={}
    queue=[]
    target=[]
    node_dict = {node.ip: node for node in nodes}
    visited[moving_node]=True
    for dest, route in moving_node.forwarding_table.items():
        
        if route["metric"]== 1 or route["metric"]==0:
            #if route["metric"]==1:
                #network_topology[node_dict.get(dest)].remove(moving_node)
            queue.append(dest)
            visited[dest]=True
            target.append(moving_node.ip)
        moving_node.forwarding_table[dest]["metric"]= float("inf")
    #print(moving_node.forwarding_table)
    print(queue)
    print(target)
    while(len(queue)!=0):
        node_ip=queue[0]
        visited[node_ip]=True
        targeting=target[0]
        target.pop(0)
        queue.pop(0)
        
        node= node_dict.get(node_ip) # to be completed
        prev= node_dict.get(targeting) # to be completed
        if node.ip == moving_node.ip:
            node_ip= queue[0]
            queue.pop(0)
            targeting=target[0]
            target.pop(0)
        node= node_dict.get(node_ip)
        prev= node_dict.get(targeting)
        print(node.ip, prev.ip)
        for dest, route in node.forwarding_table.items():
            if route["next_hop"]== targeting  and prev.forwarding_table[dest]["metric"]==float("inf"):
                node.forwarding_table[dest]["metric"]=float("inf")
        for each in network_topology[node]:

            if each.ip not in visited or visited[each.ip]==False:
                target.append(node.ip)
                queue.append(each.ip)
                
    moving_node.forwarding_table.clear()
    moving_node.advertised_table.clear()
    for node in nodes:
        print(f"\nNode {node.ip} Forwarding Table:")
        for dest, route in node.forwarding_table.items():
            print(f"  {dest} -> Next Hop: {route['next_hop']}, Metric: {route['metric']}, Seq#: {route['sequence_number']}")               
def disconnect_node2(moving_node): 
    for node in nodes:
        if moving_node.ip in node.forwarding_table:
            node.forwarding_table[moving_node.ip]["metric"] = float("inf")
            #node.forwarding_table[moving_node.ip]["next_hop"] = "#"
            #node.forwarding_table[moving_node.ip]["sequence_number"] = node.forwarding_table[moving_node.ip]["sequence_number"] + 1
            #node.advertised_table[moving_node.ip]["sequence_number"] = node.advertised_table[moving_node.ip]["sequence_number"] + 1
            #node.advertised_table[moving_node.ip]["metric"] = float("inf")
           # node.advertised_table[moving_node.ip]["next_hop"] = "#"
        for dest, route in node.forwarding_table.items():
            if node.forwarding_table[dest]["next_hop"]== moving_node.ip :
               #node.forwarding_table[dest]["next_hop"]= "#"
                node.forwarding_table[dest]["metric"]= float("inf")
                
    
    moving_node.forwarding_table.clear()
    moving_node.advertised_table.clear()
    for node in nodes:
        print(f"\nNode {node.ip} Forwarding Table:")
        for dest, route in node.forwarding_table.items():
            print(f"  {dest} -> Next Hop: {route['next_hop']}, Metric: {route['metric']}, Seq#: {route['sequence_number']}")

def reconnect_node(moving_node, new_neighbors_ips):
    moving_node.forwarding_table[moving_node.ip] = {
                "next_hop": moving_node.ip,
                "metric": 0,
                "sequence_number": 0 # look after this
            }
    network_topology[moving_node] = []  
    for new_neighbor_ip in new_neighbors_ips:
        new_neighbor = next((node for node in nodes if node.ip == new_neighbor_ip), None)
        if new_neighbor:
            network_topology[moving_node].append(new_neighbor)
            network_topology[new_neighbor].append(moving_node)  
            
            moving_node.forwarding_table[new_neighbor.ip] = {
                "next_hop": new_neighbor.ip,
                "metric": 1,
                "sequence_number": 0 # look after this
            }
            new_neighbor.forwarding_table[moving_node.ip]={
                "next_hop": moving_node.ip,
                "metric": 1,
                "sequence_number": 0 # look after this
            }
            moving_node.advertised_table[new_neighbor.ip] = moving_node.forwarding_table[new_neighbor.ip].copy()
    disconnect=None
    for node in nodes:
        print(f"\nNode {node.ip} Forwarding Table:")
        for dest, route in node.forwarding_table.items():
            print(f"  {dest} -> Next Hop: {route['next_hop']}, Metric: {route['metric']}, Seq#: {route['sequence_number']}")
    #advertise_routes(disconnect)
"""advertise_event = threading.Event()
advertise_event.set() """ # Initially set the event to allow advertising
def periodic_advertise():
    while True:
        advertise_event.wait()  # Wait here until the event is set
        advertise_routes(disconnect)
        print("\nRouting Tables after Periodic Advertisement:")
        for node in nodes:
            print(f"\nNode {node.ip} Forwarding Table:")
            for dest, route in node.forwarding_table.items():
                print(f"  {dest} -> Next Hop: {route['next_hop']}, Metric: {route['metric']}, Seq#: {route['sequence_number']}")
        time.sleep(3)

# Start periodic advertising in a separate thread
"""advertising_thread = threading.Thread(target=periodic_advertise)
advertising_thread.daemon = True
advertising_thread.start() """


            
                    
"""while True:
    action = input("Enter a number (1-255) to disconnect the node 192.168.1.<number> or 'r <node_ip> <new_neighbors>' to reconnect (or 'exit' to stop): ")

    if action.isdigit() and 1 <= int(action) <= 255:
        node_ip = f"192.168.1.{action}"
        node = next((node for node in nodes if node.ip == node_ip), None)
        if node:
            disconnect_node1(node)
            disconnect= node.ip
            #advertise_event.clear()
            print(f"\nNode {node_ip} disconnected.")
        else:
            print("Node not found.")
    elif action.startswith('r'):
        _, node_ip, *new_neighbors = action.split()
        node = next((node for node in nodes if node.ip == node_ip), None)
        if node:
            reconnect_node(node, new_neighbors)
            disconnect=None
            #advertise_event.set()
            print(f"\nNode {node_ip} reconnected to {new_neighbors}.")
        else:
            print("Node not found.")
    elif action == 'exit':
        break
    else:
        print("Invalid input. Please enter a number between 1 and 255, or 'r <node_ip> <new_neighbors>'.")"""
def advertise_method(disconnect):
    advertise_routes(disconnect)
    print("\nRouting Tables after Periodic Advertisement:")
    for node in nodes:
        print(f"\nNode {node.ip} Forwarding Table:")
        for dest, route in node.forwarding_table.items():
            print(f"  {dest} -> Next Hop: {route['next_hop']}, Metric: {route['metric']}, Seq#: {route['sequence_number']}")
    time.sleep(2)
def run_network_changes(num_changes):
    for _ in range(num_changes):
        # Randomly disconnect a node
        node = random.choice(nodes)
        if generate_prob():
            
        
            disconnect_graph(node)
            disconnect_node1(node)
            update_graph()
            advertise_method(disconnect)
            advertise_method(disconnect)
            time.sleep(1)  # Wait before next action
        if generate_reconnect_prob():
            
        # Randomly reconnect the node to 2-4 new neighbors
            new_neighbors = random.sample([n.ip for n in nodes if n != node and n not in disconnect_history], random.randint(2, 4))
            reconnect_graph(node, new_neighbors)
            reconnect_node(node, new_neighbors)
            update_graph()
            if node in disconnect_history:
                disconnect_history.remove(node)
            advertise_method(disconnect)
            advertise_method(disconnect)
            time.sleep(1)

def draw(_):
    plt.clf()
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10, font_weight='bold')
    plt.title("Random Bidirectional Network Topology")

# Create a figure for animation
fig = plt.figure(figsize=(10, 8))
update_graph()  # Initial graph setup
for _ in range(5):    
    advertise_method(disconnect) 
thread = threading.Thread(target=run_network_changes, args=(5,))
thread.start()
  
# Run the animation and network changes
ani = FuncAnimation(fig, draw, interval=1000)
#run_network_changes(num_changes=5)
plt.show()

network_sim = NetworkSimulator(nodes, node_dict)
network_sim.simulate(num_packets=10, source='192.168.1.1', destination='192.168.1.12')
metrics = network_sim.calculate_metrics()

print("DSDV Protocol Analysis Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value}")
# r 192.168.1.1 192.168.1.8 192.168.1.5
# def send_packet(src_node, dest_ip, data):
#     packet = Packet(src_node.ip, dest_ip, data)
#     current_node = src_node
#     start_time = time.time()

#     while current_node.ip != dest_ip:
#         if dest_ip not in current_node.forwarding_table:
#             print(f"Packet from {packet.src_ip} to {packet.dest_ip} dropped at {current_node.ip}")
#             return
#         next_hop_ip = current_node.forwarding_table[dest_ip]["next_hop"]
#         packet.hops.append(next_hop_ip)
#         current_node = next(node for node in nodes if node.ip == next_hop_ip)

#     end_time = time.time()
#     latency = end_time - start_time
#     print(f"Packet from {packet.src_ip} to {packet.dest_ip} delivered with hops: {packet.hops}, Latency: {latency:.4f} seconds")



# Metrics Tracking
throughput = []  # Throughput (packets per second)
latency_list = []  # Latency for each packet
pdr = 0  # Packet Delivery Ratio
control_overhead = []  # Control messages sent
routing_load = []  # Routing entries used

# A simple simulation of node movement and packet sending over time
def dynamic_simulation(steps=10):
    for step in range(steps):
        print(f"\nStep {step + 1}:")
        # Randomly move nodes
        moving_node_ip = random.choice(node_ips)
        simulate_node_movement(moving_node_ip)

        # Send packets in this step
        for node in nodes:
            dest_node = random.choice(nodes)
            if node != dest_node:  # Avoid sending to itself
                send_packet(node, dest_node.ip, "Hello!")

        # Update metrics (example values)
        

# Run dynamic simulation
#dynamic_simulation()

# Plotting the metrics
plt.figure(figsize=(15, 10))

# Throughput
# plt.subplot(3, 2, 1)
# plt.plot(throughput, marker='o')
# plt.title('Throughput Over Time')
# plt.xlabel('Simulation Steps')
# plt.ylabel('Packets per Second')

# Latency
# plt.subplot(3, 2, 2)
# plt.plot(latency_list, marker='o', color='orange')
# plt.title('Latency Over Time')
# plt.xlabel('Simulation Steps')
# plt.ylabel('Latency (seconds)')

# Packet Delivery Ratio
# plt.subplot(3, 2, 3)
# plt.plot([pdr] * steps, marker='o', color='green')
# plt.title('Packet Delivery Ratio Over Time')
# plt.xlabel('Simulation Steps')
# plt.ylabel('PDR')

# Control Overhead
# plt.subplot(3, 2, 4)
# plt.plot(control_overhead, marker='o', color='red')
# plt.title('Control Overhead Over Time')
# plt.xlabel('Simulation Steps')
# plt.ylabel('Control Messages Sent')

# Routing Load
# plt.subplot(3, 2, 5)
# plt.plot(routing_load, marker='o', color='purple')
# plt.title('Routing Load Over Time')
# plt.xlabel('Simulation Steps')
# plt.ylabel('Routing Entries Used')

# plt.tight_layout()
# plt.show()



#olsr edge weight 1 or not 

