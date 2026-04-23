"""
Puzzle-8 -- Tres Estrategias de Busca
======================================
1. Busca em Largura (BFS) pura
2. Busca Gulosa  f(x) = g(x)   onde g(x) = pecas fora do lugar
3. A*            f(x) = g(x) + h(x)
                   g(x) = pecas fora do lugar
                   h(x) = distancia de Manhattan
"""

import heapq
import sys
import time
import random
from collections import deque

# Garante saida UTF-8 no terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

GOAL = (1, 2, 3,
        4, 5, 6,
        7, 8, 0)   # 0 representa o espaço vazio

MOVES = {           # índice do espaço → índices vizinhos válidos
    0: [1, 3],
    1: [0, 2, 4],
    2: [1, 5],
    3: [0, 4, 6],
    4: [1, 3, 5, 7],
    5: [2, 4, 8],
    6: [3, 7],
    7: [4, 6, 8],
    8: [5, 7],
}


def get_neighbors(state: tuple) -> list[tuple]:
    neighbors = []
    blank = state.index(0)
    for target in MOVES[blank]:
        lst = list(state)
        lst[blank], lst[target] = lst[target], lst[blank]
        neighbors.append(tuple(lst))
    return neighbors


# ─────────────────────────────────────────────
# Heurísticas
# ─────────────────────────────────────────────

def misplaced(state: tuple) -> int:
    return sum(1 for i, v in enumerate(state) if v != 0 and v != GOAL[i])


def manhattan(state: tuple) -> int:
    dist = 0
    for i, v in enumerate(state):
        if v == 0:
            continue
        goal_idx = GOAL.index(v)
        row_cur, col_cur = divmod(i, 3)
        row_goal, col_goal = divmod(goal_idx, 3)
        dist += abs(row_cur - row_goal) + abs(col_cur - col_goal)
    return dist


# ─────────────────────────────────────────────
# Algoritmos de Busca
# ─────────────────────────────────────────────

def bfs(start: tuple) -> dict:
    """
    Busca em Largura (BFS).
    Retorna dicionário com solução e métricas.
    """
    if start == GOAL:
        return {"path": [start], "expanded": 0, "time": 0.0}

    t0 = time.perf_counter()
    frontier = deque()
    frontier.append((start, [start]))
    visited = {start}
    expanded = 0

    while frontier:
        state, path = frontier.popleft()
        expanded += 1

        for nb in get_neighbors(state):
            if nb == GOAL:
                elapsed = time.perf_counter() - t0
                return {
                    "path": path + [nb],
                    "expanded": expanded,
                    "time": elapsed,
                }
            if nb not in visited:
                visited.add(nb)
                frontier.append((nb, path + [nb]))

    return {"path": None, "expanded": expanded, "time": time.perf_counter() - t0}


def greedy(start: tuple) -> dict:
    """
    Busca Gulosa com f(x) = g(x) = número de peças fora do lugar.
    Usa fila de prioridade (min-heap) ordenada por g(x).
    """
    if start == GOAL:
        return {"path": [start], "expanded": 0, "time": 0.0}

    t0 = time.perf_counter()
    # heap: (prioridade, contador, estado, caminho)
    counter = 0
    heap = [(misplaced(start), counter, start, [start])]
    visited = set()
    expanded = 0

    while heap:
        _, _, state, path = heapq.heappop(heap)

        if state in visited:
            continue
        visited.add(state)
        expanded += 1

        if state == GOAL:
            elapsed = time.perf_counter() - t0
            return {"path": path, "expanded": expanded, "time": elapsed}

        for nb in get_neighbors(state):
            if nb not in visited:
                counter += 1
                heapq.heappush(heap, (misplaced(nb), counter, nb, path + [nb]))

    return {"path": None, "expanded": expanded, "time": time.perf_counter() - t0}


def astar(start: tuple) -> dict:
    """
    A* com f(x) = g(x) + h(x).
      g(x) = número de peças fora do lugar
      h(x) = distância de Manhattan
    """
    if start == GOAL:
        return {"path": [start], "expanded": 0, "time": 0.0}

    t0 = time.perf_counter()
    counter = 0
    g0 = misplaced(start)
    h0 = manhattan(start)
    heap = [(g0 + h0, counter, start, [start])]
    # best_f[state] = menor f já registrado para aquele estado
    best_f = {start: g0 + h0}
    expanded = 0

    while heap:
        f, _, state, path = heapq.heappop(heap)

        if f > best_f.get(state, float("inf")):
            continue

        expanded += 1

        if state == GOAL:
            elapsed = time.perf_counter() - t0
            return {"path": path, "expanded": expanded, "time": elapsed}

        for nb in get_neighbors(state):
            g_nb = misplaced(nb)
            h_nb = manhattan(nb)
            f_nb = g_nb + h_nb
            if f_nb < best_f.get(nb, float("inf")):
                best_f[nb] = f_nb
                counter += 1
                heapq.heappush(heap, (f_nb, counter, nb, path + [nb]))

    return {"path": None, "expanded": expanded, "time": time.perf_counter() - t0}



def print_board(state: tuple):
    """Imprime o tabuleiro 3×3."""
    for i in range(0, 9, 3):
        row = []
        for v in state[i:i+3]:
            row.append(" " if v == 0 else str(v))
        print(" | ".join(row))
        if i < 6:
            print("---------")


def print_result(name: str, result: dict, show_path: bool = False):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    if result["path"] is None:
        print("  Solução não encontrada.")
    else:
        print(f"  Solução encontrada!")
        print(f"  Passos na solução  : {len(result['path']) - 1}")
        print(f"  Estados expandidos : {result['expanded']}")
        print(f"  Tempo de execução  : {result['time']:.6f} s")
        if show_path:
            print("\n  Caminho da solução:")
            for step, s in enumerate(result["path"]):
                print(f"\n  Passo {step}:")
                print_board(s)


# Menu
def is_solvable(state: tuple) -> bool:
    """
    Verifica se o puzzle é solúvel contando inversões.
    Para tabuleiro 3×3: solucionável se o número de inversões for par.
    """
    tiles = [v for v in state if v != 0]
    inv = sum(
        1
        for i in range(len(tiles))
        for j in range(i + 1, len(tiles))
        if tiles[i] > tiles[j]
    )
    return inv % 2 == 0



def generate_random_board() -> tuple:
    """Gera um estado inicial aleatório que seja obrigatoriamente solúvel e diferente do objetivo."""
    state = list(GOAL)
    while True:
        random.shuffle(state)
        t_state = tuple(state)
        if is_solvable(t_state) and t_state != GOAL:
            return t_state


def main():
    print("PUZZLE-8")

    state = generate_random_board()

    print("\nEstado inicial:")
    print_board(state)

    if not is_solvable(state):
        print("\n⚠  Este estado NÃO tem solução (número de inversões ímpar).")
        return

    print(f"\nObjetivo:")
    print_board(GOAL)

    print("\nEscolha o(s) algoritmo(s):")
    print("  [1] BFS — Busca em Largura")
    print("  [2] Gulosa — f(x) = g(x)  (peças fora do lugar)")
    print("  [3] A*    — f(x) = g(x) + h(x)  (peças fora + Manhattan)")
    print("  [4] Executar todos os três")
    choice = input("\nOpção: ").strip()

    show = input("\nMostrar caminho passo a passo? (s/N): ").strip().lower() == "s"

    algorithms = {
        "1": [("BFS (Busca em Largura)", bfs)],
        "2": [("Gulosa  f(x)=g(x) — Peças fora do lugar", greedy)],
        "3": [("A*  f(x)=g(x)+h(x)", astar)],
        "4": [
            ("BFS (Busca em Largura)", bfs),
            ("Gulosa  f(x)=g(x) — Peças fora do lugar", greedy),
            ("A*  f(x)=g(x)+h(x)", astar),
        ],
    }

    selected = algorithms.get(choice, algorithms["4"])

    for name, fn in selected:
        print(f"\n>> Executando {name}...")
        result = fn(state)
        print_result(name, result, show_path=show)

    print("\n")


main()
