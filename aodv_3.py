import networkx as nx
import matplotlib.pyplot as plt
import time
import random
import matplotlib.animation as animation
from collections import deque
# Define the network graph
graph = {
    'A': ['B', 'E'],
    'B': ['A', 'C', 'E', 'K'],
    'C': ['B', 'D', 'F'],
    'D': ['C', 'I'],
    'E': ['A', 'B', 'L'],
    'F': ['C', 'J', 'K'],
    'I': ['D', 'J'],
    'J': ['I', 'F'],
    'K': ['B', 'F', 'L'],
    'L': ['E', 'K']
}

# Global variable to track if a path has been found
path_found = False
final_path = [] 

class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.sequence_num = 0
        self.broadcast_id = 0
        self.routing_table = {}  # Format: {destination_id: [next_hop, hop_count, sequence_num]}
    
    def increment_sequence_num(self):
        self.sequence_num += 1
    
    def increment_broadcast_id(self):
        self.broadcast_id += 1
    
    def update_routing_table(self, destination, next_hop, hop_count, seq_num):
        if (destination not in self.routing_table or
            self.routing_table[destination][1] > hop_count or
            (self.routing_table[destination][1] == hop_count and 
             self.routing_table[destination][2] < seq_num)):
            self.routing_table[destination] = [next_hop, hop_count, seq_num]
            print(f"{self.node_id}: Updated routing table with entry for {destination}: {self.routing_table[destination]}")

class RREQ:
    def __init__(self, source_id, source_seq_num, broadcast_id, dest_addr, dest_seq_num, hop_count=0, latest_sender=None):
        self.source_id = source_id
        self.source_seq_num = source_seq_num
        self.broadcast_id = broadcast_id
        self.dest_addr = dest_addr
        self.dest_seq_num = dest_seq_num
        self.hop_count = hop_count
        self.latest_sender = latest_sender

class RREP:
    def __init__(self, source_id, dest_id, dest_seq_num, hop_count,latest_sender=None):
        self.source_id = source_id
        self.dest_id = dest_id
        self.dest_seq_num = dest_seq_num
        self.hop_count = hop_count
        self.latest_sender=latest_sender

def send_rrep(node, rreq):
    global path_found  
    global final_path
    final_path.insert(0,node.node_id)
    rrep_packet = RREP(
        source_id=node.node_id,  
        dest_id=rreq.source_id,
        dest_seq_num=node.sequence_num,  
        hop_count=0,
        latest_sender=node.node_id
    )
    print(f"{node.node_id} sends RREP to {rreq.source_id}")

    next_hop = node.routing_table[rreq.source_id][0]

    if next_hop is not None:
        receive_rrep(nodes[next_hop], rrep_packet) 
    path_found = True  

def send_rrep2(node, rreq):

    global path_found  
    global final_path

    final_path.append(node.node_id)
    next_hop_node_id = node.routing_table[rreq.dest_addr][0]

    while next_hop_node_id != rreq.dest_addr:
        final_path.append(next_hop_node_id)
        next_hop_node_id = nodes[next_hop_node_id].routing_table[rreq.dest_addr][0]
    final_path.append(rreq.dest_addr)

    rrep_packet = RREP(
        source_id=rreq.dest_addr,  
        dest_id=rreq.source_id,
        dest_seq_num=node.sequence_num,  
        hop_count=node.routing_table[rreq.dest_addr][1],
        latest_sender=node.node_id
    )

    print(f"{node.node_id} sends RREP to {rreq.source_id}")

    next_hop = node.routing_table[rreq.source_id][0]

    if next_hop is not None:
        receive_rrep(nodes[next_hop], rrep_packet) 
    path_found = True

def receive_rrep(node, rrep):
    global path_found  
    global final_path  
    
    print(f"{node.node_id} received RREP from {rrep.source_id} to {rrep.dest_id}, seq_num: {rrep.dest_seq_num}, hop_count: {rrep.hop_count}")
    final_path.insert(0,node.node_id)
    
    if rrep.dest_id == node.node_id:
        node.update_routing_table(rrep.source_id, rrep.latest_sender, rrep.hop_count + 1, rrep.dest_seq_num)
        print("Path found between source and destination")
        return
    
    node.update_routing_table(rrep.source_id, rrep.latest_sender, rrep.hop_count + 1, rrep.dest_seq_num)
    next_hop = node.routing_table[rrep.dest_id][0]
    if next_hop is not None:
        receive_rrep(nodes[next_hop], RREP(
            source_id=rrep.source_id,  
            dest_id=rrep.dest_id,
            dest_seq_num=rrep.dest_seq_num,  
            hop_count=rrep.hop_count + 1,
            latest_sender=node.node_id
        ))

def receive_rreq(node, rreq):
    global path_found
    if path_found:
        return

    # Initialize the queue for BFS and add the starting node
    queue = deque([(node, rreq)])  # Each element is a tuple (current_node, current_rreq)

    while queue and not path_found:
        current_node, current_rreq = queue.popleft()

        # If the current node is the destination, send RREP and stop further propagation
        if current_rreq.dest_addr == current_node.node_id:
            current_node.update_routing_table(current_rreq.source_id, current_rreq.latest_sender, current_rreq.hop_count, current_rreq.source_seq_num)
            send_rrep(current_node, current_rreq)
            path_found = True
            return

        # Update routing table for the source node if it's not the source itself
        if current_node.node_id != current_rreq.source_id:
            current_node.update_routing_table(current_rreq.source_id, current_rreq.latest_sender, current_rreq.hop_count, current_rreq.source_seq_num)
        else:
            current_rreq.hop_count = 0

        # Forward RREQ to neighbors if the destination is not in the routing table
        if current_rreq.dest_addr not in current_node.routing_table:
            for neighbor_id in graph[current_node.node_id]:
                if neighbor_id != current_rreq.latest_sender and neighbor_id != current_rreq.source_id:
                    # Create a new RREQ with incremented hop count
                    rreq_forward = RREQ(
                        current_rreq.source_id,
                        current_rreq.source_seq_num,
                        current_rreq.broadcast_id,
                        current_rreq.dest_addr,
                        current_rreq.dest_seq_num,
                        current_rreq.hop_count + 1,
                        latest_sender=current_node.node_id
                    )
                    print(f"{current_node.node_id} forwards RREQ to {neighbor_id}")
                    queue.append((nodes[neighbor_id], rreq_forward))
        else:
            # If destination is in routing table, send RREP along the path
            send_rrep2(current_node, current_rreq)
            next_hop_node_id = current_node.routing_table[current_rreq.dest_addr][0]
            
            # Traverse the path to update routing tables for intermediate nodes
            while next_hop_node_id != current_rreq.dest_addr:
                current_rreq.hop_count += 1
                nodes[next_hop_node_id].update_routing_table(current_rreq.source_id, current_node.node_id, current_rreq.hop_count, current_rreq.source_seq_num)
                current_node = nodes[next_hop_node_id]
                next_hop_node_id = current_node.routing_table[current_rreq.dest_addr][0]
            
            if next_hop_node_id == current_rreq.dest_addr:
                current_rreq.hop_count += 1
                nodes[next_hop_node_id].update_routing_table(current_rreq.source_id, current_node.node_id, current_rreq.hop_count, current_rreq.source_seq_num)
        


# RERR handling
def send_rerr(node, dest):
    print(f"Node {node.node_id} detected broken route to {dest}. Sending RERR messages.")

    for neighbor_id in graph[node.node_id]:
        neighbor = nodes[neighbor_id]
        for path_dest, path_info in list(neighbor.routing_table.items()):
            if path_info[0] == node.node_id and path_dest == dest:
                remove_from_routing_table(neighbor, path_dest)

                print(f"{neighbor_id} removes route to {dest} due to broken link with {node.node_id}")
                
                send_rerr(neighbor,dest)

def remove_from_routing_table(node, dest):
    if dest in node.routing_table:
        del node.routing_table[dest]

def delete_node(node_id):
    if node_id in nodes:
        node = nodes[node_id]
        
        
        for dest in list(node.routing_table.keys()):
            send_rerr(node, dest)
        
        
        if node_id in graph:
            
            for neighbor in graph[node_id]:
                graph[neighbor].remove(node_id)
          
            del graph[node_id]

        
        del nodes[node_id]
        
        print(f"Node {node_id} has been removed from the network.")

def reconnect_node(node_id,connections):
    connections.sort()
    if node_id not in nodes:
        graph[node_id] = connections
        for conn in connections:
            graph[conn].insert(0,node_id)  
        
        nodes[node_id] = Node(node_id)
        print(f"Node {node_id} has been reconnected to the network with connections to: {connections}")
    else:
        print(f"Node {node_id} is already in the network.")


# Initialize nodes
nodes = {node_id: Node(node_id) for node_id in graph.keys()}

def find_path(start, destination):
    global path_found  
    global final_path  
    path_found = False  
    final_path = []  
    
    source_node = nodes[start]
    source_node.increment_sequence_num()
    source_node.increment_broadcast_id()
   
    rreq_packet = RREQ(start, source_node.sequence_num, source_node.broadcast_id, destination, nodes[destination].sequence_num,0, latest_sender=start)
    
    print(f"\nStarting RREQ from {start} to {destination}")
    receive_rreq(source_node, rreq_packet)
    for node_id, node in nodes.items():
        print(f"\nRouting table for node {node_id}:")
        for dest, info in node.routing_table.items():
            print(f"  Destination: {dest}, Next hop: {info[0]}, Hop count: {info[1]}, Sequence number: {info[2]}")

# Simulate AODV routing request from A to I
# find_path('A', 'I')
# print("\nFinal path from A to I:", ' -> '.join(final_path))

# Display the routing table of each node
# for node_id, node in nodes.items():
#     print(f"\nRouting table for node {node_id}:")
#     for dest, info in node.routing_table.items():
#         print(f"  Destination: {dest}, Next hop: {info[0]}, Hop count: {info[1]}, Sequence number: {info[2]}")

# Ask user for node to delete and simulate RERR
# node_to_delete = input("Enter the node to remove (e.g., 'A', 'B'): ").strip().upper()

# delete_node(node_to_delete)

# for node_id, node in nodes.items():
#     print(f"\nRouting table for node {node_id}:")
#     for dest, info in node.routing_table.items():
#         print(f"  Destination: {dest}, Next hop: {info[0]}, Hop count: {info[1]}, Sequence number: {info[2]}")


# reconnect = input(f"Do you want to reconnect node {node_to_delete} to the network? (yes/no): ").strip().lower()
# if reconnect=="yes":
#     connections = input(f"Enter the nodes you want to connect {node_id} with, separated by commas: ").strip().split(',')
#     connections = [conn.strip() for conn in connections if conn.strip() in graph]
#     reconnect_node(node_to_delete,connections)

# print("---------------------graph-----------------------")
# print(graph)

# print(nodes)

# find_path('A', 'I')
# print("\nFinal path from A to I:", ' -> '.join(final_path))

# Display the routing table of each node
for node_id, node in nodes.items():
    print(f"\nRouting table for node {node_id}:")
    for dest, info in node.routing_table.items():
        print(f"  Destination: {dest}, Next hop: {info[0]}, Hop count: {info[1]}, Sequence number: {info[2]}")


def animate_graph(graph, highlighted_path):
    # Define the graph structure
    G = nx.Graph(graph)  # Assuming 'graph' is your adjacency dictionary
    pos = nx.spring_layout(G)  # Position for nodes

    # Create figure and axis
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', ax=ax)
    plt.title("AODV Network Graph")

    # Function to update the graph for each step in the path
    def update(num):
        ax.clear()  # Clear previous frame

        # Draw the main graph
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', ax=ax)

        # Draw the highlighted part of the path up to the current step
        current_path = highlighted_path[:num+1]
        edges_in_path = [(current_path[i], current_path[i+1]) for i in range(len(current_path) - 1)]
        
        # Highlight the current nodes and edges in the path
        nx.draw_networkx_nodes(G, pos, nodelist=current_path, node_color='orange', node_size=500, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color='red', width=2, ax=ax)

        # Add packet progression info
        ax.set_title(f"AODV Network Graph - Packet Progress to {highlighted_path[num] if num < len(highlighted_path) else 'Destination'}")

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(highlighted_path), interval=10, repeat=False)

    # Display the animation
    plt.show()



def plot_metrics(packet_latencies, delivered_packets, total_packets, simulation_duration):
    # Calculate average latency
    avg_latency = sum(packet_latencies) / len(packet_latencies) if packet_latencies else 0
    
    # Calculate packet delivery ratio (PDR)
    pdr = (delivered_packets / total_packets) * 100 if total_packets > 0 else 0
    
    # Calculate throughput
    throughput = delivered_packets / simulation_duration if simulation_duration > 0 else 0

    # Plotting metrics
    plt.figure(figsize=(15, 5))
    
    # Latency plot
    plt.subplot(1, 3, 1)
    plt.plot(packet_latencies, marker='o', color='b', label='Latency per packet (ms)')
    plt.axhline(y=avg_latency, color='r', linestyle='--', label=f'Avg Latency: {avg_latency:.2f} ms')
    plt.xlabel('Packet Number')
    plt.ylabel('Latency (ms)')
    plt.title('Packet Latency')
    plt.legend()

    # PDR plot
    plt.subplot(1, 3, 2)
    plt.bar(['Packet Delivery Ratio'], [pdr], color='g')
    plt.ylabel('PDR (%)')
    plt.ylim(0, 100)
    plt.title(f'Packet Delivery Ratio: {pdr:.2f}%')

    # Throughput plot
    plt.subplot(1, 3, 3)
    plt.bar(['Throughput'], [throughput], color='purple')
    plt.ylabel('Throughput (packets/sec)')
    plt.title(f'Throughput: {throughput:.2f} packets/sec')

    plt.tight_layout()
    plt.show()




# Define the send_packets function with node deletion/reconnection functionality
def send_packets(source, destination, total_packets):
    # First, find the path from source to destination
    find_path(source, destination)
    if not final_path:
        print("No valid path from", source, "to", destination)
        return
    animate_graph(graph, final_path)
    
    packets_sent = 0
    packets_delivered = 0  # Count of successfully delivered packets
      # Start time for throughput calculation
    total_latency = 0  # Total latency for all packets
    delete_probability = 0.2  # Probability of node deletion
    packet_latencies = []
    
    while packets_sent < total_packets:
        # Simulate sending packet along the path
        print(f"Sending packet {packets_sent + 1} from {source} to {destination} via path:", ' -> '.join(final_path))
        packets_sent += 1
        extra_time = 0
        start_time = time.time()
        # Randomly simulate node deletion during transmission
        if random.random() < delete_probability:
            delete_time = time.time()
            delete_decision = input("Do you want to delete any node during packet transmission? (yes/no): ").strip().lower()
            
            if delete_decision == "yes":
                node_to_delete = input("Enter the node to remove (e.g., 'A', 'B'): ").strip().upper()
                
                if node_to_delete in nodes:
                    delete_node(node_to_delete)

                    # Show updated routing tables after deletion
                    for node_id, node in nodes.items():
                        print(f"\nUpdated routing table for node {node_id}:")
                        for dest, info in node.routing_table.items():
                            print(f"  Destination: {dest}, Next hop: {info[0]}, Hop count: {info[1]}, Sequence number: {info[2]}")

                    # Ask if the user wants to reconnect the deleted node
                    reconnect = input(f"Do you want to reconnect node {node_to_delete} to the network? (yes/no): ").strip().lower()
                    if reconnect == "yes":
                        connections = input(f"Enter the nodes you want to connect {node_to_delete} with, separated by commas: ").strip().split(',')
                        connections = [conn.strip().upper() for conn in connections if conn.strip().upper() in graph]
                        reconnect_node(node_to_delete, connections)
                    
                    reconnect_time = time.time()
                    extra_time = reconnect_time - delete_time  # Extra time spent in re-routing
                    
                    # Recompute path after network change
                    print("\nRecomputing path due to network changes...")
                    find_path(source, destination)
                    animate_graph(graph, final_path)
                    if not final_path:
                        print("No valid path from", source, "to", destination, "after node deletion.")
                        break  # Stop sending packets if path is broken



        # Calculate latency for this packet
        current_time = time.time()
        latency = current_time - start_time - extra_time
        packet_latencies.append(latency)
        start_time = current_time  # Reset start time for next packet
        total_latency += latency

        # Check if packet successfully delivered
        if final_path:
            packets_delivered += 1
            print(f"Packet {packets_sent} delivered successfully with latency: {latency:.4f} seconds")
        else:
            print(f"Packet {packets_sent} failed to deliver due to network changes.")

        # Delay to simulate packet transmission time
        time.sleep(0.5)  # Simulated delay for packet transmission

    # Throughput, PDR, and average latency calculations
    total_time = time.time() - start_time
    avg_latency = total_latency / packets_delivered if packets_delivered > 0 else 0
    throughput = packets_delivered / total_time if total_time > 0 else 0
    packet_delivery_ratio = packets_delivered / packets_sent if packets_sent > 0 else 0

    print("\nPacket transmission summary:")
    print(f"Total packets sent: {packets_sent}/{total_packets}")
    print(f"Packets successfully delivered: {packets_delivered}")
    print(f"Packet Delivery Ratio (PDR): {packet_delivery_ratio:.2f}")
    print(f"Average latency per packet: {avg_latency:.4f} seconds")
    print(f"Throughput: {throughput:.2f} packets/second")

    plot_metrics(packet_latencies, packets_delivered, total_packets, total_time)


# Example usage:
send_packets('A', 'I', total_packets=10)



# Plotting the graph
# G = nx.Graph(graph)
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue')
# plt.title("AODV Network Graph")
# plt.show()
