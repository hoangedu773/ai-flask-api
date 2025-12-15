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
CORS(app)  # Cho phep React frontend goi API

# ============================================================================
# THUAT TOAN A* (15-PUZZLE)
# ============================================================================
"""
A* Algorithm la thuat toan tim kiem co informed (co thong tin).
Su dung ham danh gia f(n) = g(n) + h(n):
- g(n): chi phi thuc te tu trang thai dau den n (so buoc da di)
- h(n): uoc luong chi phi tu n den dich (Manhattan Distance)

A* dam bao tim duong di ngan nhat neu h(n) la admissible
(khong bao gio danh gia qua cao chi phi thuc te).
"""


class PuzzleState:
    """
    Lop dai dien cho trang thai cua puzzle trong thuat toan A*.
    
    Attributes:
        board: Ma tran 2D dai dien cho ban puzzle (0 la o trong)
        g: Chi phi tu trang thai bat dau den hien tai (so buoc da di)
        h: Uoc luong chi phi den dich (Manhattan Distance)
        f: Tong chi phi f = g + h (dung de so sanh trong priority queue)
        parent: Trang thai cha (de truy vet duong di khi tim thay dich)
        move: Nuoc di de den trang thai nay ("Up", "Down", "Left", "Right")
    """
    
    def __init__(self, board, g=0, parent=None, move="Start"):
        """
        Khoi tao trang thai puzzle.
        
        Args:
            board: Ma tran 2D ban puzzle
            g: Chi phi tu dau (mac dinh 0)
            parent: Trang thai cha (mac dinh None)
            move: Nuoc di (mac dinh "Start")
        """
        self.board = [row[:] for row in board]  # Deep copy de tranh thay doi board goc
        self.g = g
        self.h = self.manhattan()  # Tinh heuristic
        self.f = self.g + self.h  # Tong chi phi
        self.parent = parent
        self.move = move
    
    def manhattan(self):
        """
        Tinh khoang cach Manhattan (heuristic function).
        
        Manhattan Distance = tong khoang cach tu moi o den vi tri dung cua no.
        Day la admissible heuristic vi khong bao gio uoc luong qua cao.
        
        Vi du: O co gia tri 5 dang o vi tri (1,1) nhung phai o (1,0)
               => khoang cach = |1-1| + |1-0| = 1
        
        Returns:
            Tong khoang cach Manhattan cua tat ca cac o
        """
        n = len(self.board)
        dist = 0
        for i in range(n):
            for j in range(n):
                val = self.board[i][j]
                if val != 0:  # Bo qua o trong
                    # Vi tri dung cua val la ((val-1)//n, (val-1)%n)
                    target_row = (val - 1) // n
                    target_col = (val - 1) % n
                    dist += abs(i - target_row) + abs(j - target_col)
        return dist
    
    def neighbors(self):
        """
        Tao danh sach cac trang thai ke (co the di chuyen den).
        
        Tu o trong, co the di chuyen: Len, Xuong, Trai, Phai
        Moi huong di chuyen tao ra mot trang thai moi.
        
        Returns:
            List cac PuzzleState co the den duoc tu trang thai hien tai
        """
        n = len(self.board)
        # Tim vi tri o trong (gia tri 0)
        bi, bj = next((i, j) for i in range(n) for j in range(n) if self.board[i][j] == 0)
        
        # 4 huong di chuyen: (delta_row, delta_col, ten_nuoc_di)
        moves = [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]
        result = []
        
        for di, dj, name in moves:
            ni, nj = bi + di, bj + dj
            # Kiem tra vi tri moi hop le (trong ban co)
            if 0 <= ni < n and 0 <= nj < n:
                # Tao ban co moi bang cach hoan doi o trong va o ke
                new = [r[:] for r in self.board]
                new[bi][bj], new[ni][nj] = new[ni][nj], new[bi][bj]
                # Tao trang thai moi voi g tang them 1
                result.append(PuzzleState(new, self.g + 1, self, name))
        return result
    
    def is_goal(self):
        """
        Kiem tra xem da den trang thai dich chua.
        
        Trang thai dich: [1, 2, 3]
                         [4, 5, 6]
                         [7, 8, 0]  (0 o goc duoi phai)
        
        Returns:
            True neu da den dich, False neu chua
        """
        n = len(self.board)
        exp = 1  # Gia tri ky vong
        for i in range(n):
            for j in range(n):
                if i == n - 1 and j == n - 1:
                    # O cuoi cung phai la 0
                    if self.board[i][j] != 0:
                        return False
                else:
                    if self.board[i][j] != exp:
                        return False
                    exp += 1
        return True
    
    def __lt__(self, o):
        """
        So sanh 2 trang thai theo gia tri f (cho priority queue).
        Trang thai nao co f nho hon se duoc uu tien xet truoc.
        """
        return self.f < o.f


def a_star(board):
    """
    Thuat toan A* giai bai toan N-puzzle.
    
    Quy trinh:
    1. Khoi tao open list (priority queue) voi trang thai ban dau
    2. Lap:
       a. Lay trang thai co f nho nhat tu open list
       b. Neu la dich -> truy vet va tra ve duong di
       c. Neu khong -> mo rong cac trang thai ke
    3. Dung khi tim thay dich hoac het trang thai
    
    Args:
        board: Ma tran puzzle ban dau
    
    Returns:
        Dict chua:
        - success: True/False
        - path: Danh sach cac buoc di (neu thanh cong)
        - steps: So buoc de giai
        - nodes: So trang thai da xet
        - error: Thong bao loi (neu that bai)
    """
    init = PuzzleState(board)
    heap = [init]  # Priority queue (min-heap theo f)
    seen = set()  # Tap cac trang thai da xet (tranh lap lai)
    nodes = 0  # Dem so trang thai da mo rong
    
    while heap:
        # Lay trang thai co f nho nhat
        cur = heapq.heappop(heap)
        key = str(cur.board)  # Chuyen board thanh string de hash
        
        # Bo qua neu da xet roi
        if key in seen:
            continue
        seen.add(key)
        nodes += 1
        
        # Kiem tra dich
        if cur.is_goal():
            # Truy vet duong di tu dich ve dau
            path = []
            while cur.parent:
                path.append({"move": cur.move, "board": cur.board})
                cur = cur.parent
            # Dao nguoc lai de co duong di tu dau den dich
            return {"success": True, "path": path[::-1], "steps": len(path), "nodes": nodes}
        
        # Mo rong cac trang thai ke
        for nb in cur.neighbors():
            if str(nb.board) not in seen:
                heapq.heappush(heap, nb)
        
        # Gioi han de tranh chay qua lau
        if nodes > 50000:
            return {"success": False, "error": "Qua nhieu nodes", "nodes": nodes}
    
    return {"success": False, "error": "Khong tim thay", "nodes": nodes}


def gen_puzzle(n, moves=15):
    """
    Tao puzzle de giai bang cach shuffle tu trang thai dich.
    
    Cach nay dam bao puzzle luon giai duoc (solvable).
    Bat dau tu trang thai dich, thuc hien 'moves' nuoc di ngau nhien.
    
    Args:
        n: Kich thuoc puzzle (n x n)
        moves: So nuoc di ngau nhien (cang lon cang kho)
    
    Returns:
        Ma tran puzzle
    """
    # Tao trang thai dich: [1,2,3], [4,5,6], [7,8,0]
    board = [[(i * n + j + 1) if not (i == n - 1 and j == n - 1) else 0 
              for j in range(n)] for i in range(n)]
    
    # Vi tri o trong (goc duoi phai)
    bi, bj = n - 1, n - 1
    
    # Thuc hien 'moves' nuoc di ngau nhien
    for _ in range(moves):
        # Tim cac vi tri co the di chuyen den
        opts = [(bi + di, bj + dj) for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)] 
                if 0 <= bi + di < n and 0 <= bj + dj < n]
        # Chon ngau nhien mot huong
        ni, nj = random.choice(opts)
        # Hoan doi
        board[bi][bj], board[ni][nj] = board[ni][nj], board[bi][bj]
        bi, bj = ni, nj
    
    return board


# ============================================================================
# THUAT TOAN GRAPH COLORING (TO MAU DO THI)
# ============================================================================
"""
Graph Coloring la bai toan gan mau cho cac dinh cua do thi sao cho
khong co 2 dinh ke nhau cung mau.

Ung dung thuc te:
- Lap lich (thoi khoa bieu, lich thi)
- Phan bo tan so vo tuyen
- To mau ban do
"""


def greedy_color(graph):
    """
    Thuat toan Greedy de to mau do thi.
    
    Y tuong:
    1. Duyet tung dinh theo thu tu
    2. Gan mau nho nhat chua duoc su dung boi cac dinh ke
    
    Do phuc tap: O(V^2) voi V la so dinh
    Uu diem: Nhanh, don gian
    Nhuoc diem: Khong dam bao so mau toi thieu
    
    Args:
        graph: Ma tran ke (graph[i][j] = 1 neu i ke j)
    
    Returns:
        List mau cua tung dinh (1-indexed)
    """
    n = len(graph)
    colors = [-1] * n  # -1 nghia la chua to mau
    colors[0] = 0  # Dinh dau tien to mau 0
    
    for v in range(1, n):
        # Lay tap mau da duoc su dung boi cac dinh ke
        used = {colors[u] for u in range(n) if graph[v][u] and colors[u] >= 0}
        # Tim mau nho nhat chua duoc su dung
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    
    # Chuyen sang 1-indexed (mau 1, 2, 3,...)
    return [c + 1 for c in colors]


def backtrack_color(graph, m):
    """
    Thuat toan Backtracking de to mau do thi voi m mau.
    
    Y tuong:
    1. Thu tung mau cho moi dinh tu 1 den m
    2. Neu mau hop le (khong trung voi dinh ke) -> to va tiep tuc
    3. Neu khong mau nao hop le -> quay lui (backtrack)
    
    Do phuc tap: O(m^V) voi m la so mau, V la so dinh
    Uu diem: Tim duoc loi giai neu ton tai
    Nhuoc diem: Cham voi do thi lon
    
    Args:
        graph: Ma tran ke
        m: So mau toi da duoc phep dung
    
    Returns:
        List mau neu thanh cong, None neu khong the to voi m mau
    """
    n = len(graph)
    colors = [0] * n  # 0 nghia la chua to
    
    def safe(v, c):
        """
        Kiem tra xem co the gan mau c cho dinh v khong.
        Hop le neu khong co dinh ke nao cung mau c.
        """
        return all(not graph[v][u] or colors[u] != c for u in range(n))
    
    def solve(v):
        """
        Ham de quy giai bai toan tu dinh v.
        
        Args:
            v: Chi so dinh can to mau
        
        Returns:
            True neu to thanh cong tu dinh v tro di
        """
        # Neu da to het tat ca dinh -> thanh cong
        if v == n:
            return True
        
        # Thu tung mau tu 1 den m
        for c in range(1, m + 1):
            if safe(v, c):  # Neu mau c hop le
                colors[v] = c  # To mau
                if solve(v + 1):  # Tiep tuc voi dinh tiep theo
                    return True
                colors[v] = 0  # Backtrack: xoa mau da to
        
        return False  # Khong co mau nao hop le
    
    return colors if solve(0) else None


def gen_graph(n, density=0.4):
    """
    Tao do thi ngau nhien.
    
    Args:
        n: So dinh
        density: Mat do canh (0.0 - 1.0). Cang cao cang nhieu canh.
    
    Returns:
        Tuple (graph, edges):
        - graph: Ma tran ke
        - edges: Danh sach cac canh [[i, j], ...]
    """
    graph = [[0] * n for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            # Xac suat tao canh giua i va j
            if random.random() < density:
                graph[i][j] = graph[j][i] = 1  # Do thi vo huong
                edges.append([i, j])
    
    return graph, edges


# ============================================================================
# THUAT TOAN MINIMAX (TICTACTOE)
# ============================================================================
"""
Minimax la thuat toan tim kiem trong cay game doi khang.

Y tuong chinh:
- MAX player (X) muon maximize diem
- MIN player (O) muon minimize diem
- Gia su doi thu luon choi toi uu (worst-case scenario)

Alpha-Beta Pruning:
- Cat tia cac nhanh khong can xet
- Giam do phuc tap tu O(b^d) xuong O(b^(d/2))
"""


def check_win(board, w):
    """
    Kiem tra nguoi thang trong game TicTacToe.
    
    Args:
        board: Ma tran ban co (X, O, hoac None)
        w: So quan lien tiep can de thang (thuong la 3 hoac 4)
    
    Returns:
        "X" neu X thang
        "O" neu O thang
        "Draw" neu hoa
        None neu chua ket thuc
    """
    n = len(board)
    
    # Kiem tra hang ngang
    for i in range(n):
        for j in range(n - w + 1):
            if board[i][j] and all(board[i][j + k] == board[i][j] for k in range(w)):
                return board[i][j]
    
    # Kiem tra cot doc
    for j in range(n):
        for i in range(n - w + 1):
            if board[i][j] and all(board[i + k][j] == board[i][j] for k in range(w)):
                return board[i][j]
    
    # Kiem tra duong cheo xuong
    for i in range(n - w + 1):
        for j in range(n - w + 1):
            if board[i][j] and all(board[i + k][j + k] == board[i][j] for k in range(w)):
                return board[i][j]
            # Kiem tra duong cheo len
            if board[i + w - 1][j] and all(board[i + w - 1 - k][j + k] == board[i + w - 1][j] for k in range(w)):
                return board[i + w - 1][j]
    
    # Kiem tra hoa (tat ca o da danh)
    if all(board[i][j] for i in range(n) for j in range(n)):
        return "Draw"
    
    return None  # Chua ket thuc


def minimax(board, w, is_max, depth, max_d, a, b):
    """
    Thuat toan Minimax voi Alpha-Beta Pruning.
    
    Args:
        board: Trang thai ban co hien tai
        w: So quan lien tiep can de thang
        is_max: True neu la luot cua MAX player (X)
        depth: Do sau hien tai
        max_d: Do sau toi da (gioi han de tang toc)
        a: Alpha - gia tri tot nhat cua MAX
        b: Beta - gia tri tot nhat cua MIN
    
    Returns:
        Diem so cua trang thai (duong neu X thang, am neu O thang)
    """
    # Kiem tra trang thai ket thuc
    win = check_win(board, w)
    if win == "X":
        return 10 - depth  # X thang, uu tien thang nhanh
    if win == "O":
        return depth - 10  # O thang
    if win == "Draw" or depth >= max_d:
        return 0  # Hoa hoac dat do sau toi da
    
    n = len(board)
    
    if is_max:  # Luot cua MAX (X)
        best = -999
        for i in range(n):
            for j in range(n):
                if not board[i][j]:  # O trong
                    board[i][j] = "X"  # Thu danh
                    best = max(best, minimax(board, w, False, depth + 1, max_d, a, b))
                    board[i][j] = None  # Huy bo (undo)
                    a = max(a, best)
                    if b <= a:  # Cat tia Beta
                        break
        return best
    else:  # Luot cua MIN (O)
        best = 999
        for i in range(n):
            for j in range(n):
                if not board[i][j]:
                    board[i][j] = "O"
                    best = min(best, minimax(board, w, True, depth + 1, max_d, a, b))
                    board[i][j] = None
                    b = min(b, best)
                    if b <= a:  # Cat tia Alpha
                        break
        return best


def best_move(board, w, player, max_d=9):
    """
    Tim nuoc di tot nhat cho AI.
    
    Duyet tat ca cac o trong, thu tung nuoc di va danh gia
    bang Minimax. Tra ve nuoc di co diem cao nhat (neu la MAX)
    hoac thap nhat (neu la MIN).
    
    Args:
        board: Trang thai ban co hien tai
        w: So quan lien tiep can de thang
        player: "X" hoac "O"
        max_d: Do sau toi da cho Minimax
    
    Returns:
        Tuple (row, col) cua nuoc di tot nhat
    """
    n = len(board)
    is_max = player == "X"
    best = (-999 if is_max else 999, None)  # (diem, vi tri)
    
    for i in range(n):
        for j in range(n):
            if not board[i][j]:  # O trong
                board[i][j] = player  # Thu danh
                s = minimax(board, w, not is_max, 0, max_d, -999, 999)
                board[i][j] = None  # Huy bo
                
                # Cap nhat nuoc di tot nhat
                if (is_max and s > best[0]) or (not is_max and s < best[0]):
                    best = (s, (i, j))
    
    return best[1]


# ============================================================================
# API ROUTES (CAC ENDPOINT)
# ============================================================================

@app.route("/")
def home():
    """
    Trang chu API.
    Tra ve thong tin co ban ve API.
    """
    return jsonify({
        "message": "AI Demo API",
        "author": "hoangedu773",
        "endpoints": [
            "POST /api/puzzle/generate",
            "POST /api/puzzle/solve",
            "POST /api/graph/generate",
            "POST /api/graph/color",
            "POST /api/tictactoe/move"
        ]
    })


@app.route("/api/puzzle/generate", methods=["POST"])
def api_gen_puzzle():
    """
    Tao puzzle ngau nhien.
    
    Request body: {"size": 3}  (3 hoac 4)
    Response: {"puzzle": [[1,2,3], [4,5,0], [7,8,6]]}
    """
    size = request.json.get("size", 3)
    moves = 15 if size == 3 else 10  # 3x3 thi 15 buoc, 4x4 thi 10 buoc
    return jsonify({"puzzle": gen_puzzle(size, moves)})


@app.route("/api/puzzle/solve", methods=["POST"])
def api_solve():
    """
    Giai puzzle bang thuat toan A*.
    
    Request body: {"puzzle": [[1,2,3], [4,5,0], [7,8,6]]}
    Response: {"success": true, "path": [...], "steps": 5, "nodes": 12}
    """
    return jsonify(a_star(request.json.get("puzzle")))


@app.route("/api/graph/generate", methods=["POST"])
def api_gen_graph():
    """
    Tao do thi ngau nhien.
    
    Request body: {"vertices": 6}
    Response: {"graph": [[0,1,0,...], ...], "edges": [[0,1], [1,2], ...]}
    """
    n = request.json.get("vertices", 6)
    g, e = gen_graph(n)
    return jsonify({"graph": g, "edges": e})


@app.route("/api/graph/color", methods=["POST"])
def api_color():
    """
    To mau do thi.
    
    Request body: {
        "graph": [[0,1,0,...], ...],
        "algorithm": "greedy" hoac "backtrack",
        "maxColors": 3 (chi dung cho backtrack)
    }
    Response: {"success": true, "colors": [1,2,1,3,...], "numColors": 3}
    """
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
    """
    Tim nuoc di tot nhat cho AI.
    
    Request body: {
        "board": [["X", null, "O"], ...],
        "aiPlayer": "O"
    }
    Response: {"row": 1, "col": 1}
    """
    board = request.json.get("board")
    ai = request.json.get("aiPlayer", "O")
    n = len(board)
    w = 3 if n == 3 else 4  # 3x3 thi can 3, 4x4 tro len can 4
    max_depth = 9 if n == 3 else 5  # Gioi han do sau de tang toc
    
    move = best_move(board, w, ai, max_depth)
    if move:
        return jsonify({"row": move[0], "col": move[1]})
    return jsonify({"error": "No move"})


# ============================================================================
# CHAY APP
# ============================================================================
if __name__ == "__main__":
    # Debug=True de tu dong reload khi thay doi code
    # host="0.0.0.0" de cho phep truy cap tu mang ngoai
    app.run(debug=True, port=5000, host="0.0.0.0")
