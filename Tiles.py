from collections import deque
import sys
import heapq

goal_node = ((0, 1, 2), (3, 4, 5), (6, 7, 8))  # Goal state
expanded_nodes = 0  # Counter for the number of expanded nodes


def get_zero_position(state):
    """Find the position of the zero (empty tile) in the given state."""
    for i, row in enumerate(state):
        for j, val in enumerate(row):
            if val == 0:
                return i, j


def get_neighbors(state):
    """Find the neighboring states of the current state."""
    row, col = get_zero_position(state)
    neighbors = []
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
    for dr, dc in moves:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_state = [list(r) for r in state]  # Create a mutable copy of the state
            new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
            new_state_tuple = tuple(tuple(r) for r in new_state)  # Convert back to tuple
            neighbors.append(new_state_tuple)
    return neighbors


def bfs_search(start_node, goal_node):
    global expanded_nodes
    queue = deque([start_node])  # Queue for BFS
    visited = {start_node}  # Set of visited nodes
    path = {}  # Path to reconstruct the solution

    while queue:
        current_node = queue.popleft()
        # Here we count the node expansion as soon as we pop the node from the queue
        expanded_nodes += 1

        if current_node == goal_node:
            construct_path(path, start_node, current_node)
            break

        neighbors = get_neighbors(current_node)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)  # Mark the neighbor as visited
                queue.append(neighbor)  # Add to the queue for future expansion
                path[neighbor] = current_node


def construct_path(path, start_node, goal_node):
    """Reconstruct the path from the start state to the goal state."""
    current_node = goal_node
    path_to_solution = []

    if start_node == goal_node:
        print(f"   1. Number of nodes expanded: 0")
        print("   2. Path to solution: ")
        return

    while current_node != start_node:
        previous_node = path[current_node]
        zero_pos_curr = get_zero_position(current_node)
        moved_tile = previous_node[zero_pos_curr[0]][zero_pos_curr[1]]
        path_to_solution.append(moved_tile)
        current_node = previous_node

    path_to_solution.reverse()
    print(f"1. Number of nodes expanded: {expanded_nodes}")
    print("2. Path to solution: " + " -> ".join(map(str, path_to_solution)))


def read_input_from_args():
    global start_node
    start_node_str = sys.argv[1]  # Read input from command line arguments
    start_node = tuple(
        tuple(map(int, start_node_str.split()[i:i + 3])) for i in range(0, len(start_node_str.split()), 3)
    )


# Run the BFS algorithm with input from the command line
read_input_from_args()
print("Algorithm: BFS")
bfs_search(start_node, goal_node)

# IDDFS Algorithm
print("\nIDDFS *****************************************")

goal_node = ((0, 1, 2), (3, 4, 5), (6, 7, 8))  # Goal state with 0 at the start of the matrix
total_nodes_checked = 0  # Total nodes checked
expanded_nodes = 0  # Number of expanded nodes
path = []  # Solution path

def get_zero_position(state):
    """Find the position of the zero in the given state."""
    for i, row in enumerate(state):
        for j, val in enumerate(row):
            if val == 0:
                return i, j

def get_neighbors(state):
    """Find the neighboring states of the current state."""
    row, col = get_zero_position(state)
    neighbors = []
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
    for dr, dc in moves:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_state = [list(r) for r in state]  # Create a mutable copy
            new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
            neighbors.append(tuple(tuple(r) for r in new_state))  # Convert back to tuple

    return neighbors

def dfs_search(start_node, goal_node, depth_limit):
    """Perform Depth-First Search (DFS) with depth limit."""
    global total_nodes_checked, nodes_not_added_to_queue, expanded_nodes
    stack = [(start_node, 0, None)]  # Stack containing (current state, depth, previous state)
    visited = set(start_node)
    path = {}

    while stack:
        current_node, depth, previous_node = stack.pop()
        total_nodes_checked += 1

        if current_node == goal_node:
            construct_path(path, start_node, current_node)
            return True

        if depth < depth_limit:
            visited.add(current_node)
            neighbors = get_neighbors(current_node)

            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append((neighbor, depth + 1, current_node))
                    path[neighbor] = current_node  # Keep track of the previous state for path reconstruction
        expanded_nodes += 1  # Increment the expanded nodes counter after checking the current node

    return False

def iddfs_search(start_node, goal_node, max_depth):
    """Perform Iterative Deepening DFS (IDDFS) with a given max depth."""
    global expanded_nodes
    for depth_limit in range(1, max_depth + 1):
        if dfs_search(start_node, goal_node, depth_limit):
            break

def construct_path(path, start_node, goal_node):
    """Reconstruct the path from the start state to the goal state."""
    current_node = goal_node
    path_to_solution = []

    while current_node != start_node:
        previous_node = path[current_node]
        zero_pos_curr = get_zero_position(current_node)
        moved_tile = previous_node[zero_pos_curr[0]][zero_pos_curr[1]]
        path_to_solution.append(moved_tile)
        current_node = previous_node

    path_to_solution.reverse()
    print(f"   1. Number of nodes expanded: {expanded_nodes}")
    print("   2. Path to solution: " + " -> ".join(map(str, path_to_solution)))

def read_input_from_args():
    global start_node
    start_node_str = sys.argv[1]  # Read input from command line
    start_node = tuple(tuple(map(int, start_node_str.split()[i:i + 3])) for i in range(0, len(start_node_str.split()), 3))
read_input_from_args()
print("   Algorithm: IDDFS")
iddfs_search(start_node, goal_node, 30)  # Adjust the depth limit as needed

# GBFS Algorithm
print("\nGBFS****************************************")

goal_node = ((0, 1, 2), (3, 4, 5), (6, 7, 8))  # Goal state
expanded_nodes = 0  # Number of expanded nodes


def get_zero_position(state):
    """Find the position of the zero in the given state."""
    for i, row in enumerate(state):
        for j, val in enumerate(row):
            if val == 0:
                return i, j


def get_neighbors(state):
    """Find the neighboring states of the current state."""
    row, col = get_zero_position(state)
    neighbors = []
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down

    for dr, dc in moves:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_state = [list(r) for r in state]  # Create a mutable copy
            new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
            neighbors.append(tuple(tuple(r) for r in new_state))  # Convert back to tuple
    return neighbors


def count_linear_conflicts(state):
    """Count the number of linear conflicts in rows and columns."""
    conflicts = 0

    # Check rows
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                continue
            target_row = state[i][j] // 3
            if target_row == i:  # Tile belongs in this row
                for k in range(j + 1, 3):
                    if state[i][k] == 0:
                        continue
                    if state[i][k] // 3 == i and state[i][j] > state[i][k]:
                        conflicts += 2

    # Check columns
    for j in range(3):
        for i in range(3):
            if state[i][j] == 0:
                continue
            target_col = state[i][j] % 3
            if target_col == j:  # Tile belongs in this column
                for k in range(i + 1, 3):
                    if state[k][j] == 0:
                        continue
                    if state[k][j] % 3 == j and state[i][j] > state[k][j]:
                        conflicts += 2

    return conflicts


def heuristic(state, goal_node):
    """Use the number of linear conflicts as heuristic."""
    return count_linear_conflicts(state)


def get_moved_number(current_state, next_state):
    """Find the number that was moved between two states."""
    current_flat = [num for row in current_state for num in row]
    next_flat = [num for row in next_state for num in row]
    for i in range(len(current_flat)):
        if current_flat[i] != next_flat[i] and next_flat[i] == 0:
            return current_flat[i]
    return None


def gbfs_search(start_node, goal_node):
    """Perform Greedy Best-First Search (GBFS) to find the solution."""
    global expanded_nodes
    pq = []
    heapq.heappush(pq, (heuristic(start_node, goal_node), 0, start_node))  # Added counter for tiebreaking
    visited = {start_node}
    came_from = {}
    counter = 1  # Counter for tiebreaking

    if start_node == goal_node:
        print(f"   1. Number of nodes expanded: 0")
        print("   2. Path to solution: ")
        return

    while pq:
        _, _, current_node = heapq.heappop(pq)  # Increment expanded_nodes when popping
        expanded_nodes += 1

        if current_node == goal_node:
            path = construct_path(came_from, start_node, current_node)
            moves = []
            for i in range(len(path) - 1):
                moved_num = get_moved_number(path[i], path[i + 1])
                if moved_num is not None:
                    moves.append(str(moved_num))
            print("Algorithm: GBFS")
            print(f"   1. Number of nodes expanded: {expanded_nodes}")
            print(f"   2. Path to solution: {' -> '.join(moves)}")
            return

        neighbors = get_neighbors(current_node)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                h_value = heuristic(neighbor, goal_node)
                heapq.heappush(pq, (h_value, counter, neighbor))
                counter += 1
                came_from[neighbor] = current_node


def construct_path(came_from, start, goal):
    """Reconstruct the solution path from start to goal."""
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


def read_input_from_args():
    global start_node
    start_node_str = sys.argv[1]  # Read input from command line
    start_node = tuple(
        tuple(map(int, start_node_str.split()[i:i + 3])) for i in range(0, len(start_node_str.split()), 3))


if __name__ == "__main__":
    read_input_from_args()
    gbfs_search(start_node, goal_node)

# A* Algorithm
print("\nA* ****************************************")


goal_node = ((0, 1, 2), (3, 4, 5), (6, 7, 8))  # Goal state with 0 in the top-left corner
total_nodes_checked = 0  # Total number of nodes checked
expanded_nodes = 0  # Number of nodes expanded
path = []  # Solution path


def get_zero_position(state):
    """Finds the position of the zero in the given state."""
    for i, row in enumerate(state):
        for j, val in enumerate(row):
            if val == 0:
                return i, j


def get_neighbors(state):
    """Finds the neighboring states by moving the zero tile."""
    row, col = get_zero_position(state)
    neighbors = []
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down

    for dr, dc in moves:
        new_row, new_col = row + dr, col + dc

        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_state = [list(r) for r in state]  # Create a mutable copy of the state
            new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
            neighbors.append(tuple(tuple(r) for r in new_state))  # Return as tuple

    return neighbors


def heuristic(state, goal_node):
    """Highly selective heuristic function to minimize node expansion."""
    score = 0

    # Get zero position in current and goal state
    current_zero = get_zero_position(state)
    goal_zero = get_zero_position(goal_node)

    # Heavy penalty for zero tile being far from goal position
    zero_distance = abs(current_zero[0] - goal_zero[0]) + abs(current_zero[1] - goal_zero[1])
    score += zero_distance * 5

    # Check each number's position
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:
                expected_row, expected_col = divmod(state[i][j], 3)

                # Distance from expected position
                distance = abs(i - expected_row) + abs(j - expected_col)

                # Very heavy penalty for tiles that should be in the corners
                if state[i][j] in [0, 2, 6, 8] and distance > 0:
                    score += distance * 8

                # Heavy penalty for center piece being out of place
                elif state[i][j] == 4 and (i != 1 or j != 1):
                    score += 10

                # Medium penalty for edge pieces
                elif state[i][j] in [1, 3, 5, 7]:
                    score += distance * 4

    # Add extra penalty for inversions
    for i in range(9):
        row1, col1 = divmod(i, 3)
        val1 = state[row1][col1]
        if val1 != 0:
            for j in range(i + 1, 9):
                row2, col2 = divmod(j, 3)
                val2 = state[row2][col2]
                if val2 != 0 and val1 > val2:
                    score += 2

    return score


def construct_path(path, start_node, goal_node):
    """Reconstructs the path from the start state to the goal state."""
    current_node = goal_node
    path_to_solution = []
    if start_node == goal_node:
        print(f"   1. Number of nodes expanded: 0")
        print("   2. Path to solution: ")
        return

    while current_node != start_node:
        previous_node = path[current_node]
        zero_pos_curr = get_zero_position(current_node)
        zero_pos_prev = get_zero_position(previous_node)
        moved_tile = previous_node[zero_pos_curr[0]][zero_pos_curr[1]]
        path_to_solution.append(moved_tile)
        current_node = previous_node

    path_to_solution.reverse()
    print(f"   1. Number of nodes explored but not added to the queue: {expanded_nodes}")
    print("   2. Path to solution: " + " -> ".join(map(str, path_to_solution)))


def a_star_search(start_node, goal_node):
    """Performs A* search to reach the goal node."""
    global total_nodes_checked, expanded_nodes
    queue = []
    heapq.heappush(queue, (0 + heuristic(start_node, goal_node), start_node))

    g_values = {start_node: 0}
    visited = set()
    path = {}

    while queue:
        current_f, current_node = heapq.heappop(queue)
        if current_node in visited:
            continue

        expanded_nodes += 1

        if current_node == goal_node:
            construct_path(path, start_node, current_node)
            break

        visited.add(current_node)
        neighbors = get_neighbors(current_node)

        for neighbor in neighbors:
            if neighbor in visited:
                continue

            new_g = g_values[current_node] + 1
            if new_g < g_values.get(neighbor, float('inf')):
                g_values[neighbor] = new_g
                f_value = new_g + heuristic(neighbor, goal_node)
                heapq.heappush(queue, (f_value, neighbor))
                path[neighbor] = current_node


def read_input_from_args():
    global start_node
    start_node_str = sys.argv[1]
    start_node = tuple(
        tuple(map(int, start_node_str.split()[i:i + 3])) for i in range(0, len(start_node_str.split()), 3))


if __name__ == "__main__":
    read_input_from_args()
    print("   Algorithm: A*")
    a_star_search(start_node, goal_node)
