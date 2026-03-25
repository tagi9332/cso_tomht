import numpy as np

def generate_positions(n_objects, size, edge_buff, sep_buff):
    positions = []
    while len(positions) < n_objects:
        # Random candidate
        x = np.random.uniform(edge_buff, size - edge_buff)
        y = np.random.uniform(edge_buff, size - edge_buff)
        
        # Check separation
        valid = True
        for px, py in positions:
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < sep_buff:
                valid = False
                break
        
        if valid:
            positions.append((x, y))
    return positions