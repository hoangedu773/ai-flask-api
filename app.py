"""
Author: hoangedu773
GitHub: https://github.com/hoangedu773
Date: 2025-12-14
Description: Flask API Backend cho AI Demo App

API endpoints:
- POST /api/puzzle/generate - Tao puzzle ngau nhien
- POST /api/puzzle/solve - Giai puzzle bang A*
- POST /api/graph/generate - Tao do thi ngau nhien
- POST /api/graph/color - To mau do thi
- POST /api/tictactoe/move - Tim nuoc di AI
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import heapq
import math
import random

app = Flask(__name__)
CORS(app)

# ============================================================================
# A* ALGORITHM (15-PUZZLE)
# ============================================================================

class PuzzleState:
    """Trang thai puzzle cho thuat toan A*"""
    
    def __init__(self, board, g=0, parent=None, move="Start"):
        self.board = [row[:] for row in board]
        self.g = g
        self.h = self.manhattan()
        self.f = self.g + self.h
        self.parent = parent
        self.move = move
    
    def manhattan(self):
        """Tinh khoang cach Manhattan (heuristic)"""
        n = len(self.board)
        dist = 0
        for i in range(n):
            for j in range(n):
                val = self.board[i][j]
                if val != 0:
                    dist += abs(i - (val-1)//n) + abs(j - (val-1)%n)
        return dist
    
    def neighbors(self):
        """Lay cac trang thai ke"""
        n = len(self.board)
        bi, bj = next((i,j) for i in range(n) for j in range(n) if self.board[i][j]==0)
        moves = [(-1,0,"Up"), (1,0,"Down"), (0,-1,"Left"), (0,1,"Right")]
        result = []
        for di, dj, name in moves:
            ni, nj = bi+di, bj+dj
            if 0 <= ni < n and 0 <= nj < n:
                new = [r[:] for r in self.board]
                new[bi][bj], new[ni][nj] = new[ni][nj], new[bi][bj]
                result.append(PuzzleState(new, self.g+1, self, name))
        return result
    
    def is_goal(self):
        n = len(self.board)
        exp = 1
        for i in range(n):
            for j in range(n):
                if i==n-1 and j==n-1:
                    if self.board[i][j] != 0: return False
                else:
                    if self.board[i][j] != exp: return False
                    exp += 1
        return True
    
    def __lt__(self, o): return self.f < o.f


def a_star(board):
    """Thuat toan A* giai puzzle"""
    init = PuzzleState(board)
    heap = [init]
    seen = set()
    nodes = 0
    
    while heap:
        cur = heapq.heappop(heap)
        key = str(cur.board)
        if key in seen: continue
        seen.add(key)
        nodes += 1
        
        if cur.is_goal():
            path = []
            while cur.parent:
                path.append({"move": cur.move, "board": cur.board})
                cur = cur.parent
            return {"success": True, "path": path[::-1], "steps": len(path), "nodes": nodes}
        
        for nb in cur.neighbors():
            if str(nb.board) not in seen:
                heapq.heappush(heap, nb)
        
        if nodes > 50000:
            return {"success": False, "error": "Qua nhieu nodes", "nodes": nodes}
    
    return {"success": False, "error": "Khong tim thay", "nodes": nodes}


def gen_puzzle(n, moves=15):
    """Tao puzzle de giai"""
    board = [[(i*n+j+1) if not (i==n-1 and j==n-1) else 0 for j in range(n)] for i in range(n)]
    bi, bj = n-1, n-1
    for _ in range(moves):
        opts = [(bi+di, bj+dj) for di,dj in [(-1,0),(1,0),(0,-1),(0,1)] if 0<=bi+di<n and 0<=bj+dj<n]
        ni, nj = random.choice(opts)
        board[bi][bj], board[ni][nj] = board[ni][nj], board[bi][bj]
        bi, bj = ni, nj
    return board


# ============================================================================
# GRAPH COLORING
# ============================================================================

def greedy_color(graph):
    """Thuat toan Greedy to mau"""
    n = len(graph)
    colors = [-1] * n
    colors[0] = 0
    for v in range(1, n):
        used = {colors[u] for u in range(n) if graph[v][u] and colors[u]>=0}
        c = 0
        while c in used: c += 1
        colors[v] = c
    return [c+1 for c in colors]


def backtrack_color(graph, m):
    """Thuat toan Backtracking to mau"""
    n = len(graph)
    colors = [0] * n
    
    def safe(v, c):
        return all(not graph[v][u] or colors[u]!=c for u in range(n))
    
    def solve(v):
        if v == n: return True
        for c in range(1, m+1):
            if safe(v, c):
                colors[v] = c
                if solve(v+1): return True
                colors[v] = 0
        return False
    
    return colors if solve(0) else None


def gen_graph(n, density=0.4):
    """Tao do thi ngau nhien"""
    graph = [[0]*n for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < density:
                graph[i][j] = graph[j][i] = 1
                edges.append([i, j])
    return graph, edges


# ============================================================================
# MINIMAX (TICTACTOE)
# ============================================================================

def check_win(board, w):
    """Kiem tra nguoi thang"""
    n = len(board)
    for i in range(n):
        for j in range(n-w+1):
            if board[i][j] and all(board[i][j+k]==board[i][j] for k in range(w)): return board[i][j]
    for j in range(n):
        for i in range(n-w+1):
            if board[i][j] and all(board[i+k][j]==board[i][j] for k in range(w)): return board[i][j]
    for i in range(n-w+1):
        for j in range(n-w+1):
            if board[i][j] and all(board[i+k][j+k]==board[i][j] for k in range(w)): return board[i][j]
            if board[i+w-1][j] and all(board[i+w-1-k][j+k]==board[i+w-1][j] for k in range(w)): return board[i+w-1][j]
    if all(board[i][j] for i in range(n) for j in range(n)): return "Draw"
    return None


def minimax(board, w, is_max, depth, max_d, a, b):
    """Minimax voi Alpha-Beta"""
    win = check_win(board, w)
    if win == "X": return 10 - depth
    if win == "O": return depth - 10
    if win == "Draw" or depth >= max_d: return 0
    
    n = len(board)
    if is_max:
        best = -999
        for i in range(n):
            for j in range(n):
                if not board[i][j]:
                    board[i][j] = "X"
                    best = max(best, minimax(board, w, False, depth+1, max_d, a, b))
                    board[i][j] = None
                    a = max(a, best)
                    if b <= a: break
        return best
    else:
        best = 999
        for i in range(n):
            for j in range(n):
                if not board[i][j]:
                    board[i][j] = "O"
                    best = min(best, minimax(board, w, True, depth+1, max_d, a, b))
                    board[i][j] = None
                    b = min(b, best)
                    if b <= a: break
        return best


def best_move(board, w, player, max_d=9):
    """Tim nuoc di tot nhat"""
    n = len(board)
    is_max = player == "X"
    best = (-999 if is_max else 999, None)
    
    for i in range(n):
        for j in range(n):
            if not board[i][j]:
                board[i][j] = player
                s = minimax(board, w, not is_max, 0, max_d, -999, 999)
                board[i][j] = None
                if (is_max and s > best[0]) or (not is_max and s < best[0]):
                    best = (s, (i, j))
    return best[1]


# ============================================================================
# API ROUTES
# ============================================================================

@app.route("/")
def home():
    return jsonify({"message": "AI Demo API", "author": "hoangedu773"})


@app.route("/api/puzzle/generate", methods=["POST"])
def api_gen_puzzle():
    size = request.json.get("size", 3)
    return jsonify({"puzzle": gen_puzzle(size, 15 if size==3 else 10)})


@app.route("/api/puzzle/solve", methods=["POST"])
def api_solve():
    return jsonify(a_star(request.json.get("puzzle")))


@app.route("/api/graph/generate", methods=["POST"])
def api_gen_graph():
    n = request.json.get("vertices", 6)
    g, e = gen_graph(n)
    return jsonify({"graph": g, "edges": e})


@app.route("/api/graph/color", methods=["POST"])
def api_color():
    g = request.json.get("graph")
    algo = request.json.get("algorithm", "greedy")
    m = request.json.get("maxColors", 3)
    
    if algo == "greedy":
        c = greedy_color(g)
        return jsonify({"success": True, "colors": c, "numColors": max(c)})
    else:
        c = backtrack_color(g, m)
        if c:
            return jsonify({"success": True, "colors": c, "numColors": max(c)})
        return jsonify({"success": False, "error": f"Khong the to voi {m} mau"})


@app.route("/api/tictactoe/move", methods=["POST"])
def api_move():
    board = request.json.get("board")
    ai = request.json.get("aiPlayer", "O")
    n = len(board)
    w = 3 if n == 3 else 4
    move = best_move(board, w, ai, 9 if n==3 else 5)
    if move:
        return jsonify({"row": move[0], "col": move[1]})
    return jsonify({"error": "No move"})


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
