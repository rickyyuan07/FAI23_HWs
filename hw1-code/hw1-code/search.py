# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import maze
from collections import deque
from queue import PriorityQueue
import numpy as np

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def bfs(maze: maze.Maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    
    fringe = deque() # queue
    startLocation = maze.getStart()
    # (location, last position)
    startNode = (startLocation, None)
    fringe.append(startNode)
    visitedLocation = {startLocation}

    endNode = None
    while len(fringe) != 0:
        node = fringe.popleft()
        if maze.isObjective(row=node[0][0], col=node[0][1]):
            endNode = node
            break
        
        neighbors = maze.getNeighbors(row=node[0][0], col=node[0][1])
        for neighbor in neighbors:
            if neighbor not in visitedLocation:
                visitedLocation.add(neighbor)
                fringe.append((neighbor, node))

    path = []
    while endNode is not None:
        path.append(endNode[0])
        endNode = endNode[1]
    
    path.reverse()
    return path


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    fringe = deque() # queue
    startLocation = maze.getStart()
    # (location, last position)
    startNode = (startLocation, None)
    fringe.append(startNode)
    visitedLocation = set()
    visitedLocation.add(startLocation)

    endNode = None
    while len(fringe) != 0:
        node = fringe.popleft()
        if maze.isObjective(row=node[0][0], col=node[0][1]):
            endNode = node
            break
        
        neighbors = maze.getNeighbors(row=node[0][0], col=node[0][1])
        for neighbor in neighbors:
            if neighbor not in visitedLocation:
                visitedLocation.add(neighbor)
                fringe.append((neighbor, node))

    path = []
    while endNode is not None:
        path.append(endNode[0])
        endNode = endNode[1]
    
    path.reverse()
    return path

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    return astar_multi(maze)

def manhattanDistance(start, end):
    return abs(start[0] - end[0]) + abs(start[1] - end[1])

def find(p, x):
    if not p[x] == x:
        p[x] = find(p, p[x])
    return p[x]


def kruskal(nodes: list, edges: list):
    """
    Kruskal's algorithm for finding the minimum spanning tree of a graph.
    
    @param nodes: list of nodes IDs
    @param edges: list of edges, tuple of (cost, ID1, ID2)

    @return cost: the cost of the minimum spanning tree
    """

    edges = sorted(edges, key=lambda x: x[0])
    
    p = {node: node for node in nodes}
    cost = 0
    for edge in edges:
        if find(p, edge[1]) != find(p, edge[2]):
            p[find(p, edge[1])] = find(p, edge[2])
            cost += edge[0]
    
    return cost


def mst_heuristic(current: tuple, objectivesID: list, dist: np.ndarray, mst_dp: dict, ID2objectives: list):
    """
    Heuristic function for A* search.

    @param current: current position
    @param objectivesID: list of dots that need to be visited
    @param dist: adjacency matrix of the graph
    @param mst_dp: heuristic dictionary for dynamic programming
    @param ID2objectives: ID to objectives location mapping

    @return cost: the cost of the minimum spanning tree and the distance from current to the nearest dot
    """
    if len(objectivesID) == 0:
        return 0
    
    l = []
    for ID in objectivesID:
        l.append(manhattanDistance(current, ID2objectives[ID]))
    pos_cost = min(l)

    if tuple(objectivesID) in mst_dp:
        # print("DP:", objectivesID, mst_dp[tuple(objectivesID)]) # DEBUG
        return pos_cost + mst_dp[tuple(objectivesID)]
    
    edges = [] # (cost, ID1, ID2)
    for i in range(len(objectivesID)):
        for j in range(i+1, len(objectivesID)):
            edges.append((dist[objectivesID[i]][objectivesID[j]], objectivesID[i], objectivesID[j]))

    mst_cost = kruskal(objectivesID, edges)
    mst_dp[tuple(objectivesID)] = mst_cost
    return pos_cost + mst_cost


def astar_multi(maze: maze.Maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    all_objectives = maze.getObjectives()
    Objective2ID = {dots: id for id, dots in enumerate(all_objectives)}
    # generate edges, adjacency matrix
    dist = np.zeros((len(all_objectives), len(all_objectives)), dtype=int)
    for i in range(len(all_objectives)):
        for j in range(len(all_objectives)):
            dist[i][j] = manhattanDistance(all_objectives[i], all_objectives[j])
    # print(dist)
    '''
    print(ID2Objective)
    input()
    '''
    # tuple(range(len(all_objectives)))
    mst_dp = {} # record the cost of previously calculated MST cost (heuristic), key: tuple of IDs of dots that haven't been visited
    # (f=(g+heuristic), current cost (g), current location, last node, IDs of dots that haven't been visited (tuple))
    startLocation = maze.getStart()
    startNode = (mst_heuristic(startLocation, list(range(len(all_objectives))), dist, mst_dp, all_objectives)+0,
                 0, startLocation, None, tuple(range(len(all_objectives))))
    # print(mst_dp)
    visitedLocation = {tuple(range(len(all_objectives))): {startLocation}} # key: IDs of dots that haven't been visited, value: set of visited locations
    fringe = PriorityQueue()
    fringe.put(startNode)
    '''
    print(fringe.empty(), fringe.qsize())
    tt = fringe.get()
    print(tt)
    print(fringe.empty(), fringe.qsize())
    input()
    '''
    endNode = None
    while not fringe.empty():
        node = fringe.get()
        # print(node)
        # print(visitedLocation)
        # input()
        if len(node[4]) == 0: # all objectives have been visited
            endNode = node
            break
        
        neighbors = maze.getNeighbors(row=node[2][0], col=node[2][1])
        for neighbor in neighbors:
            if neighbor in Objective2ID and Objective2ID[neighbor] in node[4]: # neighbor is an objective and hasn't been visited
                new_objectives = list(node[4])
                new_objectives.remove(Objective2ID[neighbor])
                new_objectives = tuple(new_objectives)

                if new_objectives in visitedLocation and neighbor in visitedLocation[new_objectives]: # neighbor with the objective state has been visited
                    # print("visited", neighbor, new_objectives)
                    continue

                new_node = (node[1]+1+mst_heuristic(neighbor, new_objectives, dist, mst_dp, all_objectives), 
                            node[1]+1, neighbor, node, new_objectives)
                if new_objectives in visitedLocation:
                    visitedLocation[new_objectives].add(neighbor)
                else:
                    visitedLocation[new_objectives] = {neighbor}
            else:
                if neighbor in visitedLocation[node[4]]:
                    # print("visited", neighbor, node[4])
                    continue
                
                new_node = (node[1]+1+mst_heuristic(neighbor, node[4], dist, mst_dp, all_objectives), 
                            node[1]+1, neighbor, node, node[4])
                visitedLocation[node[4]].add(neighbor)

            fringe.put(new_node)



    path = []
    while endNode is not None:
        path.append(endNode[2])
        endNode = endNode[3]
    
    path.reverse()
    print(path)
    return path


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []
