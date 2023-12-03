import matplotlib.pyplot as plt
import numpy as np
import heapq

# Создаем поле из клеток и определяем начальную и конечную точки
field = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])
start = (6, 2)
end = (8, 4)

# Функция для нахождения минимального пути на поле с помощью алгоритма Дейкстры
def find_path(field, start, end):
    rows, cols = field.shape
    distances = np.inf * np.ones((rows, cols)) # Расстояния от начальной точки до каждой клетки
    distances[start] = 0
    visited = np.zeros((rows, cols), dtype=bool) # Посещенные клетки
    previous = np.zeros((rows, cols, 2), dtype=int) # Предыдущие клетки в пути
    heap = [(0, start)] # Приоритетная очередь для выбора клеток с минимальным расстоянием

    while heap:
        current_dist, current_pos = heapq.heappop(heap)
        if current_pos == end:
            break
        if visited[current_pos]:
            continue
        visited[current_pos] = True

        # Перебираем соседние клетки
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            next_pos = current_pos[0] + dx, current_pos[1] + dy
            if next_pos[0] >= 0 and next_pos[0] < rows and next_pos[1] >= 0 and next_pos[1] < cols and not visited[next_pos] and field[next_pos] == 0:
                if (dx, dy) in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    new_dist = current_dist + 1.4
                else:
                    new_dist = current_dist + 1
                if new_dist < distances[next_pos]:
                    distances[next_pos] = new_dist
                    previous[next_pos] = current_pos
                    heapq.heappush(heap, (new_dist, next_pos))
    
    # Восстанавливаем путь
    path = []
    current_pos = end
    while current_pos != start:
        path.append(current_pos)
        current_pos = tuple(previous[tuple(current_pos)])

    return [start] + path[::-1]

# Визуализация поля и найденного пути
def visualize_path(field, path):
    plt.imshow(field, cmap='binary')
    plt.xticks(np.arange(len(field[0])), np.arange(len(field[0])))
    plt.yticks(np.arange(len(field)), np.arange(len(field)))
    plt.grid(color='grey', linestyle='-', linewidth=0.5)
    plt.plot(*start[::-1], 'go', markersize=10) # Начальная точка зеленым цветом
    plt.plot(*end[::-1], 'ro', markersize=10) # Конечная точка красным цветом
    if len(path) > 0:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, 'b-', linewidth=2) # Путь синим цветом
    plt.show()

# Находим минимальный путь и визуализируем его
path = find_path(field, start, end)
visualize_path(field, path)
