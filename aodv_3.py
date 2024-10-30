import networkx as nx
import matplotlib.pyplot as plt

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
    
    if rreq.dest_addr == node.node_id:
        
        node.update_routing_table(rreq.source_id, rreq.latest_sender, rreq.hop_count , rreq.source_seq_num)
        send_rrep(node, rreq)
        return  

    
    if (node.node_id!=rreq.source_id):
        node.update_routing_table(rreq.source_id, rreq.latest_sender, rreq.hop_count, rreq.source_seq_num)
    else:
        rreq.hop_count = 0
    
    if rreq.dest_addr not in node.routing_table:
        for neighbor_id in graph[node.node_id]:
            if neighbor_id != rreq.latest_sender and neighbor_id != rreq.source_id:  
                rreq_forward = RREQ(
                    rreq.source_id, 
                    rreq.source_seq_num, 
                    rreq.broadcast_id, 
                    rreq.dest_addr, 
                    rreq.dest_seq_num, 
                    rreq.hop_count + 1, 
                    latest_sender=node.node_id
                )
                print(f"{node.node_id} forwards RREQ to {neighbor_id}")
                receive_rreq(nodes[neighbor_id], rreq_forward)


nodes = {node_id: Node(node_id) for node_id in graph.keys()}


def simulate_aodv(start, destination):
    global path_found  
    global final_path  
    path_found = False  
    final_path = []  
    
    source_node = nodes[start]
    source_node.increment_sequence_num()
    source_node.increment_broadcast_id()
    
   
    rreq_packet = RREQ(start, source_node.sequence_num, source_node.broadcast_id, destination, nodes[destination].sequence_num,0, latest_sender=start)
    
    print(f"\nStarting RREQ from {start} to {destination}")
    # Start the RREQ process
    receive_rreq(source_node, rreq_packet)

# Simulate AODV routing request from A to I
simulate_aodv('A', 'I')
print("\nFinal path from A to I:", ' -> '.join(final_path))

# Display the routing table of each node
for node_id, node in nodes.items():
    print(f"\nRouting table for node {node_id}:")
    for dest, info in node.routing_table.items():
        print(f"  Destination: {dest}, Next hop: {info[0]}, Hop count: {info[1]}, Sequence number: {info[2]}")


# Plotting the graph
G = nx.Graph(graph)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue')
plt.title("AODV Network Graph")
plt.show()

