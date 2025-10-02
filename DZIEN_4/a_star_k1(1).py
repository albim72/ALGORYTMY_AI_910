import heapq

def heuristic(a,b):
    """Heurystyka - odległosc Manhatann"""
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def a_star(grid,start,goal):
    rows,cols = len(grid),len(grid[0])
    open_set = []
    heapq.heappush(open_set,(0,start))

    came_from = {}
    g_score = {start:0}
    f_score = {start:heuristic(start,goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1] #odwrócenie ścieżki
        neighbours = [
            (current[0]+dx,current[1] + dy)
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]
        ]

        for neighbour in neighbours:
            if 0<=neighbour[0]<rows and 0<=neighbour[1]<cols and grid[neighbour[0]][neighbour[1]] == 0:
                tentative_g_score = g_score[current] + 1

                if tentative_g_score<g_score.get(neighbour,float('inf')):
                    # The line below was updated to assign tentative_g_score to g_score[neighbour]
                    g_score[neighbour] = tentative_g_score
                    came_from[neighbour] = current # This line was also updated. 'came_from' should store the previous node
                    f_score[neighbour] = tentative_g_score + heuristic(neighbour,goal)

                    if neighbour not in [item[1] for item in open_set]:
                        heapq.heappush(open_set,(f_score[neighbour],neighbour))

    return None

#użycie

grid = [
    [0,1,0,0,0],
    [0,1,0,1,0],
    [0,0,0,1,0],
    [1,1,0,1,0],
    [0,0,0,0,0],
]

start = (0,0)
goal = (4,4)

path = a_star(grid,start,goal)
if path:
    print(f"najkrótsza ścieżka: {path}")
else:
    print("brak dostępnej ścieżki!")
