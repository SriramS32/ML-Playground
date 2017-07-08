#Python 3
def min_squares(r, c):
    if r==c:
        return 1
    if min_cost[r][c] is not None:
        return min_cost[r][c]
    cost = r*c
    for i in range(1, int(r/2)+1):
        cost = min(min_squares(r-i, c) + min_squares(i, c), cost)
    
    for i in range(1, int(c/2)+1):
        cost = min(min_squares(r, c-i) + min_squares(r, i), cost)
    
    min_cost[r][c] = cost
    return cost

# Num of input cases
t = int(input())
for i in range(0, t):
    row, col = map(int, input().split())
    # print(row, col)
    min_cost = [[None]*(col+1) for _ in range(row+1)]
    print(min_squares(row, col))








    