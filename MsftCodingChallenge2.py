import numpy as np
from sys import stdout

def num_infected_neighbors(grid, n, x, y):
	infected_count = 0
	# Above
	if not (x-1 < 0) and grid[x-1][y]==1:
		infected_count += 1
	# Right
	if not (y+1 >= n) and grid[x][y+1]==1:
		infected_count += 1
	# Left
	if not (y-1 < 0) and grid[x][y-1]==1:
		infected_count += 1
	# Bottom
	if not (x+1 >= n) and grid[x+1][y]==1:
		infected_count += 1
	return infected_count

def infect_neighbors(grid):
	n = len(grid)
	new_infection = False
	for x in range(n):
		for y in range(n):
			if num_infected_neighbors(grid, n, x, y) >= 2:
				if(grid[x][y] == 0):
					new_infection = True
					grid[x][y] = 1 # Infect it :)
	# print new_infection
	return new_infection


with open('input.txt', 'r') as inFile:
	grid = []
	for line in inFile.readlines():
		row = []
		for char in line[:-1]:
			row.append(int(char))
		grid.append(row)

while (infect_neighbors(grid)):
	continue

for row in grid:
	for col in row:
		stdout.write(str(col))
	print ''