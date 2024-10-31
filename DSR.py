import random
import matplotlib.pyplot as plt
import time
import itertools
import networkx as nx


class Node:
    """
    Represents a node in a network. Responsible for handling route requests (RREQ),
    route replies (RREP), and maintaining a routing table.
    """
    def __init__(self, node_id, network, ip_address):
        self.node_id = node_id
        self.network = network
        self.ip_address = ip_address
        self.routing_table = {}

    def send_rreq(self, destination):
        """
        Initiates a Route Request (RREQ) to find a path to the destination node.

        Args:
            destination (int): The ID of the destination node.
        """
        path = [self.node_id]
        print(f"Source: Node {self.node_id} ({self.ip_address})  --> Destination: Node {destination}")
        print()
        print(f"Node {self.node_id} initiating RREQ to Node {destination}")
        print()
        self.network.broadcast_rreq(self, destination, path)
        
        
    def receive_rreq(self, source, destination, path):
        """
        Handles an incoming RREQ. If this node is the destination, it replies with an RREP.
        Otherwise, it forwards the RREQ if it hasn't already been in the path.

        Args:
            source (int): The ID of the source node.
            destination (int): The ID of the destination node.
            path (list of int): The path taken by the RREQ so far.
        """
        path.append(self.node_id)
        print(f"Node {self.node_id} received RREQ from Node {source} for destination Node {destination}, Packet: {path}")
        print()
        
        if self.node_id == destination:
            print(f"Node {self.node_id} is the destination.")
            self.network.found_paths.append(path.copy())  # Save path for evaluation
        elif self.node_id not in path[:-1]:  # Avoid cycles
            print(f"Node {self.node_id} broadcasting packet to its neighbours : {self.network.connections[self.node_id]}")
            self.network.broadcast_rreq(self, destination, path.copy())

    def send_rrep(self, source, path):
        """
        Sends a Route Reply (RREP) back to the source node with the discovered path.

        Args:
            source (int): The ID of the source node.
            path (list of int): The path from the source to the destination.
        """
        print(f"Node {self.node_id} sending RREP to Node {source} with path: {path}")
        self.network.send_rrep(self, source, path)

    def receive_rrep(self, path):
        """
        Receives an RREP and updates the routing table with the path to the destination.

        Args:
            path (list of int): The path from the source to the destination.
        """
        destination = path[-1]
        self.routing_table[destination] = path
        print(f"Node {self.node_id} updated routing table for destination Node {destination}: {path}")


    def maintain_routes(self):
        """
        Periodically checks routes and initiates new RREQ if needed.
        """
        for destination, path in list(self.routing_table.items()):
            if not self.network.is_route_active(path):
                print(f"Node {self.node_id} detected broken route to Node {destination}. Initiating new RREQ.")
                self.send_rreq(destination)
                
                
class Network:
    """
    Represents a network of nodes with a randomly generated topology.
    Handles broadcasting of RREQs and sending of RREPs.
    """
    def __init__(self, num_nodes):
        self.nodes = []
        self.connections = {node_id: set() for node_id in range(num_nodes)}
        self.packet_delivered = 0
        self.total_packets = 0
        self.total_latency = 0
        # self.connections = {0: {2, 6}, 1: {3, 5}, 2: {0, 3, 6}, 3: {1, 2},
        #                     4: {8, 9, 7}, 5: {1, 6}, 6: {0, 2, 5, 7, 9},
        #                     7: {4, 6}, 8: {4}, 9: {4, 6}}
        
        self.found_paths = []
        
        # Assign IPs to each node after initializing the network
        self.assign_ips(num_nodes)
        self.create_random_topology(num_nodes)
        
    def assign_ips(self, num_nodes):
        """
        Assigns IPs to each node.
        """
        base_ip = itertools.count(start=1)
        for node_id in range(num_nodes):
            ip_address = f"192.168.1.{next(base_ip)}"
            node = Node(node_id, self, ip_address)
            self.nodes.append(node)
        print("Assigned IPs:")
        for node in self.nodes:
            print(f"Node {node.node_id}: IP Address {node.ip_address}")
        print()

    def create_random_topology(self, num_nodes):
        """
        Creates a random network topology by connecting nodes with a given probability.

        Args:
            num_nodes (int): The number of nodes in the network.
        """
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < 0.3:  # 30% chance of connection
                    self.connections[i].add(j)
                    self.connections[j].add(i)
        print("Network topology created with connections:")
        for node_id, neighbors in self.connections.items():
            print(f"Node {node_id}: Connected to {', '.join(map(str, neighbors))}")
        print()

    def broadcast_rreq(self, sender, destination, path):
        """
        Broadcasts an RREQ from the sender to all its connected neighbors.

        Args:
            sender (Node): The node sending the RREQ.
            destination (int): The ID of the destination node.
            path (list of int): The path taken by the RREQ so far.
        """
        
        self.total_packets += 1
        start_time = time.time()

        # for neighbor_id in self.connections[sender.node_id]:
        #     neighbor = self.nodes[neighbor_id]
        #     if neighbor.node_id not in path:  # Avoid cycles
        #         neighbor.receive_rreq(sender.node_id, destination, path.copy())
        
         # Measure latency for each successful packet delivery
        for neighbor_id in self.connections[sender.node_id]:
            neighbor = self.nodes[neighbor_id]
            if neighbor.node_id not in path:
                neighbor.receive_rreq(sender.node_id, destination, path.copy())
        
        # If the last node in the path is the destination, update metrics
        if path[-1] == destination:
            self.packet_delivered += 1
            end_time = time.time()
            latency = end_time - start_time
            self.total_latency += latency
    
    
    def send_rrep(self, sender, source, path):
        """
        Sends an RREP back along the discovered path to the source node.

        Args:
            sender (Node): The node sending the RREP.
            source (int): The ID of the source node.
            path (list of int): The discovered path from source to destination.
        """
        if len(path) > 1:
            next_hop_id = path[-2]
            path.pop()  # Move to the next hop in the reverse path
            print(f"RREP forwarded from Node {sender.node_id} to Node {next_hop_id} with path: {path}")
            self.nodes[next_hop_id].receive_rrep(path)
            
    def calculate_metrics(self):
        pdr = (self.packet_delivered / self.total_packets) * 100 if self.total_packets > 0 else 0
        avg_latency = self.total_latency / self.packet_delivered if self.packet_delivered > 0 else 0
        throughput = self.packet_delivered / self.total_latency if self.total_latency > 0 else 0
        return pdr, avg_latency, throughput
        
    def calculate_shortest_path(self):
        if self.found_paths:
            shortest_path = min(self.found_paths, key=len)
            print(f"\nShortest path selected for RREP: {shortest_path}")
            self.nodes[shortest_path[-1]].send_rrep(shortest_path[0], shortest_path)
            
    def plot_network(self):
        """
        Plots the network topology using NetworkX and Matplotlib.
        """
        G = nx.Graph()  # Create an empty graph

        # Add nodes and edges to the graph based on network connections
        for node_id in self.connections:
            G.add_node(node_id)
            for neighbor in self.connections[node_id]:
                G.add_edge(node_id, neighbor)

        # Draw the network graph
        pos = nx.spring_layout(G)  # Layout for visual spacing of nodes
        plt.figure(figsize=(10, 8))
        
        # Draw nodes and edges with labels
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue", edgecolors="black")
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color="gray")
        nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")

        # Add title and display
        plt.title("Network Topology")
        plt.axis("off")  # Hide axes
        plt.show()
            
    def is_route_active(self, path):
        """
        Simulates random link failures by checking if a route is active.
        """
        active_status = all(random.random() > 0.1 for _ in path)  # 10% chance each link fails
        return active_status

    def initiate_route_maintenance(self):
        """
        Triggers route maintenance on each node.
        """
        for node in self.nodes:
            node.maintain_routes()

# Simulation
print("-------------------DYNAMIC SOURCE ROUTING PROTOCOL---------------------------------")
print()

network = Network(num_nodes=10)


# Initiate RREQ from Node 0 to Node 9
network.nodes[0].send_rreq(destination=9)

# Display all paths found
print("\nAll possible paths from Node 0 to Node 9:")
for i, path in enumerate(network.found_paths, start=1):
    print(f"Path {i} : {path}")

# Calculate and send RREP along the shortest path
network.calculate_shortest_path()


# Route Maintenance
print("\n--- Route Maintenance ---")
network.initiate_route_maintenance()
network.plot_network()  # Display the network graph

# # Simulation and Visualization
# network = Network(num_nodes=10)
# num_trials = 1  # Running for a fixed number of trials for simplicity
# pdr_results, latency_results, throughput_results = [], [], []

# num_trials = 50
# for i in range(num_trials):
#     # Reset metrics for each trial
#     network.packet_delivered = 0
#     network.total_packets = 0
#     network.total_latency = 0

#     print(f"\n--- Trial {i + 1} ---")
#     network.nodes[0].send_rreq(destination=9)

#     # Gather metrics
#     pdr, avg_latency, throughput = network.calculate_metrics()
#     pdr_results.append(pdr)
#     latency_results.append(avg_latency)
#     throughput_results.append(throughput)
#     print(pdr_results)
#     print(latency_results)
#     print(throughput_results)

# # Plotting the metrics
# plt.figure(figsize=(12, 8))

# # Packet Delivery Ratio (PDR) Plot
# plt.subplot(3, 1, 1)
# plt.plot(pdr_results, color='blue', marker='o', linestyle='-')
# plt.title('Packet Delivery Ratio (PDR) Over Trials')
# plt.xlabel('Trial')
# plt.ylabel('PDR (%)')

# # Latency Plot
# plt.subplot(3, 1, 2)
# plt.plot(latency_results, color='red', marker='o', linestyle='-')
# plt.title('Average Latency Over Trials')
# plt.xlabel('Trial')
# plt.ylabel('Latency (seconds)')

# # Throughput Plot
# plt.subplot(3, 1, 3)
# plt.plot(throughput_results, color='green', marker='o', linestyle='-')
# plt.title('Throughput Over Trials')
# plt.xlabel('Trial')
# plt.ylabel('Throughput (packets/second)')

# plt.tight_layout()
# plt.show()
