import networkx as nx
import matplotlib.pyplot as plt

def parse_dag_edges(text):
    edges = set()
    for line in text:
        if ':' in line:
            parts = line.split(':')
            if len(parts) == 3:
                # Sources are connected to the intermediate, which in turn is connected to the targets
                sources = parts[0].split(',')
                intermediate = parts[1]
                targets = parts[2].split(',')

                for src in sources:
                    if src.strip():
                        edges.add((src.strip(), intermediate.strip()))  # Connect source to intermediate

                for tgt in targets:
                    if tgt.strip():
                        edges.add((intermediate.strip(), tgt.strip()))  # Connect intermediate to targets
            elif len(parts) == 2:
                sources = parts[0].split(',')
                targets = parts[1].split(',')

                for src in sources:
                    for tgt in targets:
                        if src.strip() and tgt.strip():
                            edges.add((src.strip(), tgt.strip()))  # Connect source to target directly
    return list(edges)


# draw graph
if __name__ == '__main__':
    filename = 'DAG.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()

    parsed_edges = parse_dag_edges(lines)
    sorted_edges = sorted(parsed_edges, key=lambda edge: int(edge[0]))
    sorted_edges = sorted(sorted_edges, key=lambda edge: int(edge[1]))

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    G.add_edges_from(sorted_edges)

    # Draw the DAG
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=15, font_weight='bold',
            arrows=True)
    plt.title("Directed Acyclic Graph (DAG)", size=15)
    plt.show()
