

import random
import math
class Cell:
  def __init__(self, x, y, value, id):
    self.alien_id = id
    self.x = x
    self.y = y
    self.value = value
    self.parent = None
    self.left = None
    self.right = None
    self.top = None
    self.bottom = None
    self.visited = False
    self.neighbors = []
    self.open = False
    self.loc = [x, y]
    self.alien = False
    self.alien_probability = 0
    self.cell_in_detector = False
    self.crewmate_probability = .5
    self.prob_alien_moved_from = 0
    self.prob_alien_moved_to = 0
    self.move_value = 0

def initialize_cell_neighbors(grid):
  rows=len(grid)
  cols=len(grid)
  size=len(grid)

  for i in range(rows):
    for j in range(cols):
      left=j-1
      right=j+1
      top=i-1
      bottom=i+1
      #Testing for valid neighbor coordinates
      if (top<0):
        top=-1
      if (bottom==size):
        bottom=-1
      if (left<0):
        left=-1
      if (right==size):
        right=-1

      if (left!=-1):
        grid[i][j].left=grid[i][left]
      else:
        grid[i][j].left = None
      if (right!=-1):
        grid[i][j].right=grid[i][right]
      else:
        grid[i][j].right = None
      if (top!=-1):
        grid[i][j].top=grid[top][j]
      else:
        grid[i][j].top = None
      if (bottom!=-1):
        grid[i][j].bottom=grid[bottom][j]
      else:
        grid[i][j].bottom = None

  return grid

def create_grid(rows, cols):
    grid = [[Cell(i, j, 0, 3) for j in range(cols)] for i in range(rows)]
    return grid

def is_valid(cell, grid):
  if(0 <= cell.x < len(grid)):
    if(0 <= cell.y < len(grid[0])):
      return True
  return False

def get_neighbors(cell, grid):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    neighbors = []

    for dx, dy in directions:
        new_x, new_y = cell.x + dx, cell.y + dy
        if is_valid(Cell(new_x, new_y, '', 3), grid):
            neighbors.append(grid[new_x][new_y])

    return neighbors

def get_open_neighbors(cell, grid):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    open_neighbors = []

    for dx, dy in directions:
        new_x, new_y = cell.x + dx, cell.y + dy
        if is_valid(Cell(new_x, new_y, '', 3), grid):
          if (grid[new_x][new_y].open):
            open_neighbors.append(grid[new_x][new_y])

    return open_neighbors

def detect_if_cell_open(cell):
  if (cell.value == 1):
    return True
  if (cell.value == 'A'):
    return True
  if (cell.value == 'C'):
    return True
  if (cell.value == 'B'):
    return True
  if (cell.value == 0):
    return False

def detect_open_cells(grid):
  open_cells = []
  for row in grid:
    for cell in row:
      if (cell.open):
        open_cells.append(cell)
  return open_cells

def initialize_detected_cells(grid, bot_cell, detector_size):
  count=0
  for row in grid:
    for cell in row:
      if (bot_cell.x - detector_size) <= cell.x <= (bot_cell.x + detector_size) :
        if (bot_cell.y - detector_size) <= cell.y <= (bot_cell.y + detector_size):
          cell.cell_in_detector = True
  return grid

def initialize_aliens(num_aliens, grid, bot_cell, detector_size):
  aliens = []                                                # initialize initialize_ship function                                                            # initialize matrix
  current_alien_start_cells = []
  open_cells = []
  grid=initialize_detected_cells(grid, bot_cell, detector_size)
  for row in grid:
    for cell in row:
      if (cell.open):
        if not cell.cell_in_detector:
          open_cells.append(cell)

  for i in range(num_aliens):
    alien_start_cell = random.choice(open_cells)
    while (alien_start_cell in current_alien_start_cells):
      alien_start_cell = random.choice(open_cells)
    current_alien_start_cells.append(alien_start_cell)
    alien_start_cell.id = i
    alien_start_cell.alien = True
    alien_start_cell.value = 'A'
    aliens.append(alien_start_cell)

  for row in grid:
    for cell in row:
      if (cell.open):
        if not cell.cell_in_detector:
          cell.alien_probability = 1/len(open_cells)
        else:
          cell.alien_probability=0
  return aliens

def initialize_prob_crewmate(grid, num_crewmate):
  num_open_cells = len(detect_open_cells(grid))
  for row in grid:
    for cell in row:
      if cell.open and cell.value != 'B':
        cell.crewmate_probability = 1/(num_open_cells-1)
      elif cell.open and cell.value == 'B':
        cell.crewmate_probability = 0
      else:
        cell.crewmate_probability = 0

def initialize_prob_alien_moved(grid):
  for row in grid:
    for cell in row:
      if cell.open:
        if cell.neighbors:
          cell.prob_alien_moved_from = 1/len(cell.neighbors)
        else:
          cell.prob_alien_moved_from = 0

import heapq
def dijkstra(grid,source):
  distances = {}
  predecessors={}
  for i in range(len(grid)):
    for j in range(len(grid[0])):
        distances[(i,j)]=float('inf')
        predecessors[(i,j)]=None
  distances[source] = 0
  priority_queue=[[0,source]]
  while priority_queue:
    current_distance, current_vertex = heapq.heappop(priority_queue)
        # Check if the current path is shorter than the known shortest path
    if current_distance > distances[current_vertex]:
        continue

    for neighbor in get_open_neighbors(grid[current_vertex[0]][current_vertex[1]],grid):
        neighbor=(neighbor.loc[0],neighbor.loc[1])
        distance = current_distance + 1

              # If a shorter path is found, update distances and predecessors
        if distance < distances[neighbor]:
          distances[neighbor] = distance
          predecessors[neighbor] = current_vertex
          heapq.heappush(priority_queue, (distance, neighbor))

  return distances, predecessors

import pandas as pd

def draw_ship_df(grid):


  columns_123=[]
  rows_123=[]
  for i in range(len(grid)):
    columns_123.append(i)
    rows_123.append(i)

  values_matrix=[]

  for row in grid:
    temp_row=[]
    for cell in row:
      edit_val=""
      if (cell.value==1 and cell.crewmate_probability != 0):
        edit_val="o"
      elif (cell.value==1 and cell.crewmate_probability == 0):
        edit_val = "O"
      elif (cell.value == 'C'):
        edit_val="C"
      elif (cell.value == 'A'):
        edit_val="A"
      elif (cell.value == 'C*'):
        edit_val="c"
      elif (cell.value == 'B'):
        edit_val="B"
      elif (cell.value == 'D'):
        edit_val="D"
      else:
        edit_val="x"
      temp_row.append(edit_val)
    values_matrix.append(temp_row)

  df = pd.DataFrame(values_matrix, columns=columns_123, index=rows_123)
  print(df.to_string())
  print("\n")

def detect_possible_future_cells(open_cell, possible_future_cells, n):
  for neighbor in open_cell.neighbors:

    counter=0

    open_neighbor_count=0

    for neighbor_neighbor in neighbor.neighbors:

      if (neighbor_neighbor.value==1):
        counter=counter+1

      if (neighbor_neighbor.value==1):
        open_neighbor_count=open_neighbor_count+1

    if (counter<2 and open_neighbor_count>=1):
      possible_future_cells.append(neighbor)
  return possible_future_cells

def back_check_possible_future_cells(possible_future_cells):
  for cell in possible_future_cells:
    counter=0
    for neighbor in cell.neighbors:
      if (neighbor.value==1):
        counter=counter+1
    if (counter>1):
      possible_future_cells.remove(cell)
  return possible_future_cells

def find_dead_ends(grid, n):
  dead_ends=[]
  for row in grid:
    for cell in row:
      if cell.value==1:
        open_count=0
        for neighbor in cell.neighbors:
          if neighbor.value==1:
            open_count=open_count+1
        if (open_count==1):
          dead_ends.append(cell)
  return dead_ends

def open_half_of_dead_ends(grid, dead_ends, n):
  num_dead_ends=len(dead_ends)
  nodes_to_open=int(num_dead_ends/2+1)
  for i in range(nodes_to_open):
    try:
      dead_end_to_open = random.choice(dead_ends)
    except:
      continue
    valid_neighbors_to_open=[]
    for neighbor in dead_end_to_open.neighbors:
      if (neighbor.value==1):
        continue
      else:
        valid_neighbors_to_open.append(neighbor)
    if len(valid_neighbors_to_open)!=0:
      dead_end_neighbor_to_open = random.choice(valid_neighbors_to_open)
    dead_end_neighbor_to_open.value=1
    dead_end_neighbor_to_open.open=True
    dead_ends.remove(dead_end_to_open)
  return grid, dead_ends

import random
def explore_grid(n, grid, num_aliens, num_crew, detector_size):                                                    # initialize function
  # Generate two random values within the range [1, n]
  random_value1 = random.randint(1, n-2)                                        # generate random y value
  random_value2 = random.randint(1, n-2)                                        # generate random x value

  starting_cell=grid[random_value1][random_value2]                            # initialize open_cell
  # Open First Cell
  starting_cell.value = 1         # Open First Cell
  current_open_cell=starting_cell                                               # initialize current_open_cell as open_cell
  possible_future_cells=[]                                                      # initialize possible_future_cells

  while True:
    if possible_future_cells:                                                 # if possible_future_cells not empty
      new_chosen_cell = random.choice(possible_future_cells)                  # initialize new_chosen_cell as random cell from possible_future_cells
      new_chosen_cell.value = 1                                               #open the new_choesn_cell
      new_chosen_cell.open = True
      current_open_cell=new_chosen_cell                                       # set current_open_cell to be the newest open cell
      possible_future_cells.remove(new_chosen_cell)                           # remove new_chosen_cell from possible_future_cells
    possible_future_cells = detect_possible_future_cells(current_open_cell, possible_future_cells, n) # add valid neighbors of current_open_cell to possible_future_cells
    if (starting_cell in possible_future_cells):
        possible_future_cells.remove(starting_cell)
    possible_future_cells = back_check_possible_future_cells(possible_future_cells) # remove invalid cells from possible_future_cells
    if (starting_cell in possible_future_cells):
        possible_future_cells.remove(starting_cell)

    if (len(possible_future_cells)==0):                                       # if no more possible_future_cells to add end while loop
      break

  dead_ends=find_dead_ends(grid, n)
  grid, des = open_half_of_dead_ends(grid, dead_ends, n)

  bot_cell = create_bot(grid)
  aliens = initialize_aliens(num_aliens, grid, bot_cell, detector_size)


  for x in range(num_crew):
    possible_crewmate_cells = detect_open_cells(grid)
    possible_crewmate_cells.remove(bot_cell)
    crewmate_cell = random.choice(possible_crewmate_cells)
    crewmates.append(crewmate_cell)
    crewmate_cell.value = 'C'


  return aliens, crewmates, bot_cell                                                               # return completed matrix

def create_bot(grid):
  open_cells=detect_open_cells(grid)
  bot_start_cell=random.choice(open_cells)
  bot_start_cell.value = 'B'
  bot_start_cell.crewmate_probability = 0
  return bot_start_cell

def take_step(grid, aliens, crewmates, bot_cell,time_steps):

  import random
  import copy
  time_steps+=1

  if bot_id == 1 or bot_id == 2 or bot_id == 3 or bot_id == 4 or bot_id == 5 or bot_id == 6:
    alien_beep = alien_detector(grid, bot_cell, detector_size)

    detect_alien_probabilities(grid)

  if bot_id == 7 or bot_id == 8:
    alien_beep = alien_detector(grid, bot_cell, detector_size)

    detect_alien_pairs_probs(grid, alien_beep)


  new_aliens=[]

  random.shuffle(aliens)
  for current_alien in aliens:
    for crew_cell in crewmates:

      for other_alien in aliens:
        if current_alien.id != other_alien.id:
          if (current_alien.loc == other_alien.loc):
            grid[current_alien.x][current_alien.y].value = 'A'
      if (current_alien.loc == bot_cell.loc):
        grid[current_alien.x][current_alien.y].value = 'B'

      if (current_alien.loc == crew_cell.loc):
        grid[current_alien.x][current_alien.y].value = 'C'
      else:
        grid[current_alien.x][current_alien.y].value = 1

    current_alien.open = True
    current_alien.id = 3
    current_alien.alien = False
    grid[current_alien.x][current_alien.y].open = True
    grid[current_alien.x][current_alien.y].alien = False
    x_old=current_alien.x
    y_old=current_alien.y
    current_alien.neighbors = get_open_neighbors(current_alien, grid)

    try:
      alien_move = random.choice(current_alien.neighbors)
    except:
      for alien in aliens:
        alien.neighbors = get_open_neighbors(alien, grid)
      alien_move = random.choice(current_alien.neighbors)

    current_alien=grid[alien_move.x][alien_move.y]
    current_alien.neighbors = get_open_neighbors(alien_move, grid)
    current_alien.x = alien_move.x
    current_alien.y = alien_move.y
    current_alien.loc = [alien_move.x, alien_move.y]

    x1=x_old
    y1=y_old

    grid[current_alien.x][current_alien.y].alien = True
    # Updating the values for current alien to new choice
    current_alien.id = 0
    current_alien.alien = True

        # Updating values of old cell
    grid[x_old][y_old].alien = False
    grid[x_old][y_old].id = 0

    grid[current_alien.x][current_alien.y].alien = True
    grid[current_alien.x][current_alien.y].id = 0

    for crew_cell in crewmates:
      if current_alien == crew_cell:
        grid[current_alien.x][current_alien.y].value = 'C*'
        current_alien.value = 'C*'

    else:
      grid[current_alien.x][current_alien.y].value = 'A'
      current_alien.value = 'A'

    for alien_cell in aliens:
      if bot_cell == alien_cell:
        draw_ship_df(grid)
        new_aliens.append(current_alien)
        return "The bot was caught by the aliens!", crewmates, new_aliens,time_steps, alien_beep

    new_aliens.append(current_alien)


  aliens=new_aliens

  if bot_id == 1 or bot_id == 2 or bot_id == 3  or bot_id == 4 or bot_id == 5 or bot_id == 6:
    alien_move_prob(grid)
    for row in grid:
      for cell in row:
        cell.alien_probability = cell.prob_alien_moved_to

  if bot_id == 7 or bot_id == 8:
    alien_pairs_move_probs(grid)

  if bot_id == 1 or bot_id == 2 or bot_id == 3  or bot_id == 6:
    crewmate_beep = detect_crewmate_beep(grid, crewmates, bot_cell, alpha,time_steps)
    detect_crewmate_prob(grid, bot_cell, crewmate_beep, time_steps, crewmates)

  if bot_id == 4 or bot_id == 5 or bot_id == 7 or bot_id == 8:
    crewmate_beep = detect_crewmate_beep(grid, crewmates, bot_cell, alpha,time_steps)
    detect_crewmate_pairs_probs(grid, bot_cell, crewmate_beep, time_steps)

  return "Status normal", crewmates, new_aliens,time_steps, alien_beep

def detect_if_crew_memeber_blocked(grid, crew_start_cell) :
  if (crew_start_cell.neighbors):
    print("Not Blocked")
  else:
    print("Blocked")

def alien_detector(grid, bot_cell, detector_size) :
  alien_beep = False
  for row in grid:
    for cell in row:
      cell.cell_in_detector = False
      if (bot_cell.x - detector_size) <= cell.x <= (bot_cell.x + detector_size) :
        if (bot_cell.y - detector_size) <= cell.y <= (bot_cell.y + detector_size):
          cell.cell_in_detector = True

          if cell.alien == True:
            alien_beep = True

  return alien_beep

def detect_alien_probabilities(grid):
  if alien_beep:
    prob_alien_detected = 1
    cell_probs = []
    for row in grid:
      for cell in row:
        if cell.open:
          if not cell.cell_in_detector:
            cell.alien_probability = 0
    for row in grid:
      for cell in row:
        if cell.open:
          if cell.cell_in_detector:
            if cell.alien_probability==0:
              cell.alien_probability=0.00000001
            cell_probs.append(cell.alien_probability*prob_alien_detected)

    sum_of_probs = sum(cell_probs)
    for row in grid:
      for cell in row:
        if cell.open:
          if cell.cell_in_detector:
            cell.alien_probability = (cell.alien_probability*prob_alien_detected)/(sum_of_probs*prob_alien_detected)
  else:
    prob_alien_detected = 1
    cell_probs = []
    for row in grid:
      for cell in row:
        if cell.open:
          if cell.cell_in_detector:
            cell.alien_probability = 0
    for row in grid:
      for cell in row:
        if cell.open:
          if cell.alien_probability==0:
              cell.alien_probability=0.00000001
          cell_probs.append(cell.alien_probability*prob_alien_detected)
    sum_of_probs = sum(cell_probs)
    for row in grid:
      for cell in row:
        if cell.open:
          if not cell.cell_in_detector:
            cell.alien_probability = (cell.alien_probability*prob_alien_detected)/(sum_of_probs*prob_alien_detected)

def alien_move_prob(grid):
  for row in grid:
    for cell in row:
      if cell.open:
        denom_values = []
        x_prime_values = []
        for row in grid:
          for current_cell in row:
            if current_cell.open:
              for neighbor in current_cell.neighbors:
                # Neighbor.alien_probability is the prob of how likely the alien was present in that neighbor
                # neighbor.prob_alien_moved_from is how likely the current cell is the alien location given that this neighbor was the prev alien location
                denom_values.append(neighbor.alien_probability*neighbor.prob_alien_moved_from) #*probability_of_collecting_final_observation
        beta_t = 1/sum(denom_values)
        for neighbor in cell.neighbors:
            x_prime_values.append(neighbor.alien_probability*neighbor.prob_alien_moved_from) #*probability_of_collecting_final_observation
        x_prime = sum(x_prime_values)

        cell.prob_alien_moved_to = beta_t*x_prime
      else:
        cell.prob_alien_moved_to = 0

import numpy as np
def detect_crewmate_beep(grid, crewmates, bot_cell, alpha,time_steps):
  for crew_cell in crewmates:
    d=distance_map[(bot_cell.x,bot_cell.y)][(crew_cell.x,crew_cell.y)]
    p_crew = math.exp((-1)*alpha*(d-1)) #e−α(d−1)
    if p_crew>1:
      p_crew=0
    crewmate_beep_value = np.random.binomial(1, p_crew, 1)
    if crewmate_beep_value[0] == 1:
      return True
  else:
    return False

def detect_crewmate_prob(grid,bot_cell, crewmate_beep, time_steps, crewmates):
  if crewmate_beep:
    prob_crewmate_detected = 1
    cell_probs = []
    for row in grid:
      for cell in row:
        if cell.open:
          d=distance_map[(bot_cell.x,bot_cell.y)][(cell.x,cell.y)]
          p_beep = math.exp((-1)*alpha*(d)) #e−α(d−1)
          cell.crewmate_probability = p_beep*cell.crewmate_probability
          cell_probs.append(cell.crewmate_probability)
    sum_of_probs = sum(cell_probs)
    for row in grid:
      for cell in row:
        if cell.open:
          cell.crewmate_probability = (cell.crewmate_probability)/(sum_of_probs)
  if not crewmate_beep:
    prob_crewmate_detected = 1
    cell_probs = []
    for row in grid:
      for cell in row:
        if cell.open:
          d=distance_map[(bot_cell.x,bot_cell.y)][(cell.x,cell.y)]
          p_beep = (1 - math.exp((-1)*alpha*(d))) #e−α(d−1)
          cell.crewmate_probability = p_beep*cell.crewmate_probability
          cell_probs.append(cell.crewmate_probability)
    sum_of_probs = sum(cell_probs)
    for row in grid:
      for cell in row:
        if cell.open:
          cell.crewmate_probability = (cell.crewmate_probability)/(sum_of_probs)

from decimal import Decimal, getcontext
getcontext().prec = 20
# Function to add a key-value pair to the hashmap
# The key is a pair of coordinate pairs ((x, y), (a, b))
def add_coordinates_crewmate_pairs(x, y, a, b, value):
    crewmate_pairs_probs[((x, y), (a, b))] = value

# Function to get a value from the hashmap given a key
def get_value_crewmate_pairs(x, y, a, b):
    return crewmate_pairs_probs.get(((x, y), (a, b)), Decimal(1/((num_open_cells-1)*(num_open_cells-1))))

from decimal import Decimal, getcontext
getcontext().prec = 20
def initialize_crewmate_pairs_probs(crewmate_pairs_probs):
  num_open_cells = len(detect_open_cells(grid))
  prob = Decimal(1/((num_open_cells-2)*(num_open_cells-2)))
  for row in grid:
    for cellj in row:
      for row in grid:
        for cellk in row:
          if cellk != cellj:
            add_coordinates_crewmate_pairs(cellj.x, cellj.y, cellk.x, cellk.y, prob)

from decimal import Decimal, getcontext
getcontext().prec = 20
def detect_crewmate_pairs_probs(grid, bot_cell, crewmate_beep, time_steps):
  #Finding the sum of probablities i.e. normalization factor
  if crewmate_beep:
    denom_values = []
    for rowj in grid:
      for cellj in rowj:
        if cellj.open:
          if cellj != bot_cell:
            dj = distance_map[(bot_cell.x,bot_cell.y)][(cellj.x, cellj.y)]
            prob_beep_from_cellj = Decimal(math.exp((-1)*alpha*(dj-1)))
            for rowk in grid:
              for cellk in rowk:
                if cellk.open:
                  if cellk != cellj:
                    if cellk != bot_cell:
                      dk = distance_map[(bot_cell.x,bot_cell.y)][(cellk.x, cellk.y)]
                      prob_beep_from_cellk = Decimal(math.exp((-1)*alpha*(dk-1)))
                      prob_beep_from_cellj_and_cellk = Decimal(1-(1-prob_beep_from_cellj)*(1-prob_beep_from_cellk))

                      denom_values.append(Decimal(get_value_crewmate_pairs(cellj.x, cellj.y, cellk.x, cellk.y)*prob_beep_from_cellj_and_cellk))

    sum_denom = sum(denom_values)

    for rowj in grid:
      for cellj in rowj:
        if cellj.open:
          if cellj != bot_cell:
            d = distance_map[(bot_cell.x,bot_cell.y)][(cellj.x, cellj.y)]
            prob_beep_from_cellj = Decimal(math.exp((-1)*alpha*(d-1)))
            for rowk in grid:
              for cellk in rowk:
                if cellk.open:
                  if cellk != cellj:
                    if cellk != bot_cell:
                      d = distance_map[(bot_cell.x,bot_cell.y)][(cellk.x, cellk.y)]
                      prob_beep_from_cellk = Decimal(math.exp((-1)*alpha*(d-1)))
                      prob_beep_from_cellj_and_cellk = Decimal(1-(1-prob_beep_from_cellj)*(1-prob_beep_from_cellk))
                      add_coordinates_crewmate_pairs(cellj.x, cellj.y, cellk.x, cellk.y, Decimal((get_value_crewmate_pairs(cellj.x, cellj.y, cellk.x, cellk.y)*prob_beep_from_cellj_and_cellk)/sum_denom))
    # finding probability for cell j
    for rowj in grid:
      for cellj in rowj:
        if cellj.open:
          if cellj != bot_cell:
            cellj_crewmate_probs = []
            for rowk in grid:
              for cellk in rowk:
                if cellk.open:
                  if cellk != cellj:
                    if cellk != bot_cell:
                      cellj_crewmate_probs.append(get_value_crewmate_pairs(cellj.x, cellj.y, cellk.x, cellk.y))
            cellj.crewmate_probability = sum(cellj_crewmate_probs)

  if not crewmate_beep:
    denom_values = []
    for row in grid:
      for cellj in row:
        if cellj.open:
          if cellj != bot_cell:
            dj = distance_map[(bot_cell.x,bot_cell.y)][(cellj.x, cellj.y)]
            prob_beep_from_cellj = Decimal((math.exp((-1)*alpha*(dj-1))))
            for row in grid:
              for cellk in row:
                if cellk.open:
                  if cellk != cellj:
                    if cellk != bot_cell:
                      dk = distance_map[(bot_cell.x,bot_cell.y)][(cellk.x, cellk.y)]
                      prob_beep_from_cellk = Decimal((math.exp((-1)*alpha*(dk-1))))

                      prob_beep_from_cellj_and_cellk = Decimal((1-prob_beep_from_cellj)*(1-prob_beep_from_cellk))

                      denom_values.append(Decimal(get_value_crewmate_pairs(cellj.x, cellj.y, cellk.x, cellk.y)*prob_beep_from_cellj_and_cellk))

    sum_denom = sum(denom_values)


    for row in grid:
      for cellj in row:
        if cellj.open:
          if cellj != bot_cell:
            d = distance_map[(bot_cell.x,bot_cell.y)][(cellj.x, cellj.y)]
            prob_beep_from_cellj = Decimal((math.exp((-1)*alpha*(d-1))))
            for row in grid:
              for cellk in row:
                if cellk.open:
                  if cellk != cellj:
                    if cellk != bot_cell:
                      d = distance_map[(bot_cell.x,bot_cell.y)][(cellk.x, cellk.y)]
                      prob_beep_from_cellk = Decimal((math.exp((-1)*alpha*(d-1))))

                      prob_beep_from_cellj_and_cellk = Decimal((1-prob_beep_from_cellj)*(1-prob_beep_from_cellk))

                      add_coordinates_crewmate_pairs(cellj.x, cellj.y, cellk.x, cellk.y, Decimal((get_value_crewmate_pairs(cellj.x, cellj.y, cellk.x, cellk.y)*prob_beep_from_cellj_and_cellk)/sum_denom))

    for row in grid:
      for cellj in row:
        if cellj.open:
          if cellj != bot_cell:
            cellj_crewmate_probs = []
            for row in grid:
              for cellk in row:
                if cellk != cellj:
                  if cellk.open:
                    if cellk != bot_cell:
                      cellj_crewmate_probs.append(get_value_crewmate_pairs(cellj.x, cellj.y, cellk.x, cellk.y))
            cellj.crewmate_probability = sum(cellj_crewmate_probs)

# Function to add a key-value pair to the hashmap
# The key is a pair of coordinate pairs ((x, y), (a, b))
def add_coordinates_alien_pairs(x, y, a, b, value):
    alien_pairs_probs[((x, y), (a, b))] = value

# Function to get a value from the hashmap given a key
def get_value_alien_pairs(x, y, a, b):
    return alien_pairs_probs.get(((x, y), (a, b)), 1/((num_open_cells-1)*(num_open_cells-1)))

def initialize_alien_pairs_probs(alien_pairs_probs, grid):
  grid = initialize_detected_cells(grid, bot_cell, detector_size)
  open_cells = []
  for row in grid:
    for cell in row:
      if (cell.open):
        if not cell.cell_in_detector:
          open_cells.append(cell)

  num_open_cells = len(open_cells)
  prob = 1/((num_open_cells-1)*(num_open_cells-1))
  for row in grid:
    for cellj in row:
      if cellj.open:
        if not cellj.cell_in_detector:
          for row in grid:
            for cellk in row:
              if cellk.open:
                if not cellk.cell_in_detector:
                  if cellk != cellj:
                    add_coordinates_alien_pairs(cellj.x, cellj.y, cellk.x, cellk.y, prob)
  for row in grid:
    for cellj in row:
      if cellj.open:
        if cellj.cell_in_detector:
          for row in grid:
            for cellk in row:
              if cellk.open:
                if cellk.cell_in_detector:
                  if cellk != cellj:
                    add_coordinates_alien_pairs(cellj.x, cellj.y, cellk.x, cellk.y, 0)

def prob_alien_detected(bot_cell, detector_size, cellj, cellk) :
  prob_beep = 0
  if (bot_cell.x - detector_size) <= cellj.x and cellj.x <= (bot_cell.x + detector_size) :
    if (bot_cell.y - detector_size) <= cellj.y and cellj.y <= (bot_cell.y + detector_size):
      prob_beep = 1
  if (bot_cell.x - detector_size) <= cellk.x and cellk.x <= (bot_cell.x + detector_size) :
    if (bot_cell.y - detector_size) <= cellk.y and cellk.y <= (bot_cell.y + detector_size):
      prob_beep = 1


  return prob_beep

def detect_alien_pairs_probs(grid, alien_beep):
  if alien_beep:

    denom_values = []
    for row in grid:
      for cellj in row:
        if cellj.open:
          if cellj != bot_cell:
            for row in grid:
              for cellk in row:
                if cellk.open:
                  if cellj != cellk:
                    if cellk != bot_cell:
                      prob_alien_detected_given_cellj_and_cellk = prob_alien_detected(bot_cell, detector_size, cellj, cellk)
                      if get_value_alien_pairs(cellj.x, cellj.y, cellk.x, cellk.y)*prob_alien_detected_given_cellj_and_cellk==0:
                        cell.alien_probability=0.00000001
                      denom_values.append(get_value_alien_pairs(cellj.x, cellj.y, cellk.x, cellk.y)*prob_alien_detected_given_cellj_and_cellk)
    sum_denom_values = sum(denom_values)

    for row in grid:
      for cellj in row:
        if cellj.open:
          if cellj != bot_cell:
            for row in grid:
              for cellk in row:
                if cellk.open:
                  if cellj != cellk:
                    if cellk != bot_cell:
                      prob_alien_detected_given_cellj_and_cellk = prob_alien_detected(bot_cell, detector_size, cellj, cellk)
                      add_coordinates_alien_pairs(cellj.x, cellj.y, cellk.x, cellk.y, ((get_value_alien_pairs(cellj.x, cellj.y, cellk.x, cellk.y))*prob_alien_detected_given_cellj_and_cellk)/sum_denom_values)

    for row in grid:
      for cellj in row:
        if cellj.open:
          if cellj != bot_cell:
            cellj_alien_probs = []
            for row in grid:
              for cellk in row:
                if cellk.open:
                  if cellj != cellk:
                    if cellk != bot_cell:
                      cellj_alien_probs.append(get_value_alien_pairs(cellj.x, cellj.y, cellk.x, cellk.y))
            cellj.alien_probability = sum(cellj_alien_probs)

  if not alien_beep:

    denom_values = []
    for row in grid:
      for cellj in row:
        if cellj.open:
          if cellj != bot_cell:
            for row in grid:
              for cellk in row:
                if cellk.open:
                  if cellj != cellk:
                    if cellk != bot_cell:
                      prob_alien_detected_given_cellj_and_cellk = prob_alien_detected(bot_cell, detector_size, cellj, cellk)
                      if get_value_alien_pairs(cellj.x, cellj.y, cellk.x, cellk.y)*prob_alien_detected_given_cellj_and_cellk==0:
                        cell.alien_probability=0.00000001
                      denom_values.append(get_value_alien_pairs(cellj.x, cellj.y, cellk.x, cellk.y)*prob_alien_detected_given_cellj_and_cellk)
    sum_denom_values = sum(denom_values)

    for row in grid:
      for cellj in row:
        if cellj.open:
          if cellj != bot_cell:
            for row in grid:
              for cellk in row:
                if cellk.open:
                  if cellj != cellk:
                    if cellk != bot_cell:
                      prob_alien_detected_given_cellj_and_cellk = prob_alien_detected(bot_cell, detector_size, cellj, cellk)
                      add_coordinates_alien_pairs(cellj.x, cellj.y, cellk.x, cellk.y, ((get_value_alien_pairs(cellj.x, cellj.y, cellk.x, cellk.y))*prob_alien_detected_given_cellj_and_cellk)/sum_denom_values)

    for row in grid:
      for cellj in row:
        if cellj.open:
          if cellj != bot_cell:
            cellj_alien_probs = []
            for row in grid:
              for cellk in row:
                if cellk.open:
                  if cellj != cellk:
                    if cellk != bot_cell:
                      cellj_alien_probs.append(get_value_alien_pairs(cellj.x, cellj.y, cellk.x, cellk.y))
            cellj.alien_probability = sum(cellj_alien_probs)

def alien_pairs_move_probs(grid):
  denom_values = []
  for row in grid:
    for cellj in row:
      if cellj.open:
        if cellj != bot_cell:
          for row in grid:
            for cellk in row:
              if cellk.open:
                if cellj != cellk:
                  if cellj != bot_cell:
                    for neighborj in cellj.neighbors:
                      for neighbork in cellk.neighbors:
                        if neighborj != neighbork:
                          denom_values.append(get_value_alien_pairs(neighborj.x, neighborj.y, neighbork.x, neighbork.y)*(neighborj.prob_alien_moved_from*neighbork.prob_alien_moved_from))
  sum_denom_values = sum(denom_values)
  beta_t = 1/sum_denom_values

  for row in grid:
    for cellj in row:
      if cellj.open:
        if cellj != bot_cell:
          x_prime_values = []
          for row in grid:
            for cellk in row:
              if cellk.open:
                if cellj != cellk:
                  if cellk != bot_cell:
                    for neighborj in cellj.neighbors:
                      for neighbork in cellk.neighbors:
                        if neighborj != neighbork:
                          neighborj_conditional_prob_alien_moved_from = neighborj.prob_alien_moved_from + neighbork.prob_alien_moved_from - neighborj.prob_alien_moved_from*neighbork.prob_alien_moved_from
                          x_prime_values.append(get_value_alien_pairs(neighborj.x, neighborj.y, neighbork.x, neighbork.y)*(neighborj_conditional_prob_alien_moved_from*neighbork.prob_alien_moved_from))
                        else:
                          x_prime_values.append(get_value_alien_pairs(neighborj.x, neighborj.y, neighbork.x, neighbork.y)*(neighborj.prob_alien_moved_from*neighbork.prob_alien_moved_from))
          x_prime = sum(x_prime_values)
          cellj.alien_probability = beta_t*x_prime

import pandas as pd
def draw_alien_probabilities_matrix(grid):
  columns_123=[]
  rows_123=[]
  for i in range(len(grid)):
    columns_123.append(i)
    rows_123.append(i)

  values_matrix=[]

  for row in grid:
    temp_row=[]
    for cell in row:
      if cell.value == 'B':
        temp_row.append("B")
      elif cell.alien_probability==0 and cell.open!=True:
        temp_row.append('X')
      else:
        temp_row.append(cell.alien_probability)
    values_matrix.append(temp_row)

  df = pd.DataFrame(values_matrix, columns=columns_123, index=rows_123)
  print(df.to_string())
  print("\n")

import pandas as pd
def draw_crewmate_probabilities_matrix(grid):
    pd.set_option('display.float_format', '{:.15f}'.format)

    columns_123 = []
    rows_123 = []
    for i in range(len(grid)):
        columns_123.append(i)
        rows_123.append(i)

    values_matrix = []

    for row in grid:
        temp_row = []
        for cell in row:
            if cell.value == 'B':
                temp_row.append("B")
            elif not cell.open:
                temp_row.append('X')
            else:
                temp_row.append(cell.crewmate_probability)

        values_matrix.append(temp_row)

    df = pd.DataFrame(values_matrix, columns=columns_123, index=rows_123)
    print(df.to_string())
    print("\n")

    pd.reset_option('display.float_format')

def sum_probabilities_alien(grid):
  probabilities = []
  for row in grid:
    for cell in row:
      probabilities.append(cell.alien_probability)
  return sum(probabilities)

def sum_probabilities_crewmate(grid):
  probabilities = []
  for row in grid:
    for cell in row:
      probabilities.append(cell.crewmate_probability)
  return sum(probabilities)

def detect_open_neighbors_bot_1(current_cell, matrix_size):
  open_neighbors = []
  arr=[current_cell.left,current_cell.right,current_cell.top,current_cell.bottom]
  for i in arr:
    if (i==None):
      continue
    if (i.x==-1 or i.y==-1 or i.x==matrix_size or i.y==matrix_size):
      continue
    if i.value==0:
      continue
    else:
      open_neighbors.append(i)
  return open_neighbors

def sort_open_neighbors(grid, open_neighs):
  open_neighs.sort(key=lambda cell: cell.crewmate_probability, reverse=True)
  on_copy=open_neighs
  open_neighbors_with_alien_prob_0 = [cell for cell in open_neighs if cell.alien_probability < 0.2]
  if (len(open_neighbors_with_alien_prob_0)==0):
    open_neighs=on_copy
    sorted_open_neighbors = sorted(open_neighs, key=lambda obj: (-obj.crewmate_probability, obj.alien_probability))
  else:
    return open_neighbors_with_alien_prob_0
  return sorted_open_neighbors

def find_path(source,dest):
  src=(source[0],source[1])
  dst=(dest[0],dest[1])
  tempo=pred_map[src][dst]
  if tempo is None:
    print(f"The destination {dst} is a blocked cell")
    print(pred_map[src])
    return []
  found_path=[[tempo[0],tempo[1]],[dest[0],dest[1]]]
  while tempo!=src:
    tempo=pred_map[src][tempo]
    found_path.insert(0,[tempo[0],tempo[1]])
  return found_path

def move_bot1(grid, bot_cell, path, aliens, crewmates, time_steps):
  if bot_cell.value=='A' or bot_cell.value=='C*':
    return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
  while True:
    next=path[1]
    next_cell = grid[next[0]][next[1]]
    if next_cell.alien_probability>0.00001:
      status, crewmates, aliens,time_steps, alien_beep = take_step(grid, aliens, crewmates, bot_cell,time_steps)
      max_prob_val=max(c.crewmate_probability for r in grid for c in r)
      max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
      max_prob_cell=random.choice(max_prob_index)
      path=find_path(bot_cell.loc,max_prob_cell)
      continue
    bot_cell.value=1
    grid[bot_cell.x][bot_cell.y].value=1
    if (grid[next[0]][next[1]].value!='C' and grid[next[0]][next[1]].value!='C*' and grid[next[0]][next[1]].value!='A'):
      grid[next[0]][next[1]].value='B'
      next_cell.value='B'
      next_cell.open = True

    bot_cell=next_cell
    if bot_cell.value=='A' or bot_cell.value=='C*':
      return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
    if bot_cell.value=='C':
      bot_cell.value='B'
      return "The bot has rescued a crewmate", bot_cell, crewmates, grid, aliens,time_steps

    bot_cell.crewmate_probability=0
    print("PATH  RETURNED BY BFS IS",path)

    for row in grid:
      for cell in row:
        cell.visited=False
        cell.parent=None

    grid[next[0]][next[1]].value='B'
    next_cell.value='B'

    status, crewmates, aliens,time_steps, alien_beep = take_step(grid, aliens, crewmates, bot_cell,time_steps)
    draw_ship_df(grid)
    max_prob_val=max(c.crewmate_probability for r in grid for c in r)
    max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
    max_prob_cell=random.choice(max_prob_index)
    path=find_path(bot_cell.loc,max_prob_cell)
    if status == "The bot was caught by the aliens!":
      return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
    if bot_cell.value=='A' or bot_cell.value=='C*':
      return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps

  return "The bot has rescued a crewmate", bot_cell, crewmates, grid, aliens,time_steps

"""Bot 1 Driver Code"""

bot_id = 1
size = 30
num_aliens = 1
num_crew = 1
alphas = [ -2*math.log(.1)/(size/2), -2*math.log(.25)/(size/2), -2*math.log(.5)/(size/2) ] # Calculating alpha based on ship size
detector_sizes=[1,5,7,10,15]
crewmates = []
num_time_steps=[]
sim=0
successes=0
alien_caught=0
num_sims=30
for alpha in alphas:
  for detector_size in detector_sizes:
    for _ in range(num_sims):
      alien_beep = False
      time_steps=1
      grid = create_grid(size, size)
      for row in grid:
        for cell in row:
          cell.neighbors = get_neighbors(cell, grid)
      aliens, crewmates, bot_cell = explore_grid(size, grid, num_aliens, num_crew, detector_size)

      grid=initialize_cell_neighbors(grid)
      num_open_cells = len(detect_open_cells(grid))
      for row in grid:
        for cell in row:
          cell.open = detect_if_cell_open(cell)
          cell.neighbors = get_open_neighbors(cell, grid)
      distance_map={}
      pred_map={}
      for i in range(len(grid)):
        for j in range(len(grid[0])):
          s=(i,j)
          dlist,f=dijkstra(grid,s)
          distance_map[s]=dlist
          pred_map[s]=f

      initialize_prob_crewmate(grid, num_crew)
      initialize_prob_alien_moved(grid)

      max_prob_val=max(c.crewmate_probability for r in grid for c in r)
      max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
      max_prob_cell=random.choice(max_prob_index)
      path=find_path(bot_cell.loc,max_prob_cell)

      msg, bot_cell, crewmates, grid, aliens,time_steps = move_bot1(grid, bot_cell, path, aliens, crewmates, time_steps)
      print(msg)

      sim+=1
      print("Simmulation Number: ", sim)
      if (msg=="The bot was caught by the aliens!"):
        alien_caught+=1
      elif (msg=="The bot has rescued a crewmate"):
        successes+=1
        num_time_steps.append(time_steps)
      print("SIMULATION NUMBER",sim)
      print("BOT STATUS",msg)
      print("CURRENT STATUS OF NUM TIME STEPS IS",num_time_steps)

      file_name_1 = f"interim_output_sim_{sim}_alpha_{alpha}_k_{detector_size}.txt"

      with open(file_name_1, "a") as file1:
        file1.write("SIMULATION NUMBER"+str(sim)+"\n")
        file1.write("BOT STATUS"+str(msg)+"\n")
        file1.write("CURRENT STATUS OF NUM TIME STEPS IS"+str(num_time_steps)+"\n")



    print("FOR ALPHA VALUE",alpha,"AND K VALUE",detector_size, "FOR BOT1")
    print("OUT OF",num_sims,"SIMULATIONS")
    print("BOT RESCUED CREWMATE",successes,"TIMES")
    print("BOT WAS CAUGHT BY ALIENS",alien_caught,"TIMES")

    print("AVERAGE MOVES REQUIRED TO RESCUE CREWMATE IS",sum(num_time_steps)/successes)

    file_name = f"output_alpha_{alpha}_k_{detector_size}.txt"

    with open(file_name, "a") as file:
      file.write("FOR ALPHA VALUE"+str(alpha)+"AND K VALUE"+str(detector_size)+ "FOR BOT1"+"\n")
      file.write("OUT OF"+str(num_sims)+"SIMULATIONS"+"\n")
      file.write("BOT RESCUED CREWMATE"+str(successes)+"TIMES"+"\n")
      file.write("BOT WAS CAUGHT BY ALIENS"+str(alien_caught)+"TIMES"+"\n")
      file.write("AVERAGE MOVES REQUIRED TO RESCUE CREWMATE IS"+str(sum(num_time_steps)/successes)+"\n")
      file.write("PROBABILITY OF BOT SUCCESSFULLY AVOIDING ALIENS IS"+str(sum(num_time_steps)/successes)+"\n")

"""Bot 2 Code"""

def find_path_bot2(source,dest_list):
  len_list=[]
  pth=[]
  for i,ele in enumerate(dest_list):
    pth.insert(i,find_path(source,ele))
    if len(pth[i])==0:
      len_list.insert(i,float('inf'))
      continue
    len_list.insert(i,len(pth[i]))
  min_len=min(len_list)
  print("min length is: ",min_len)
  print("len list is: ",len_list)
  for i,p in enumerate(pth):
    if len_list[i]==min_len:
      return pth[i]

def move_bot2(grid, bot_cell, path, aliens, crewmates, time_steps, alien_beep):
  while True:
    bot_cell.crewmate_probability=0

    if alien_beep:
      print("ALIEN AVOIDANCE ACTIVATED")
      alien_avoidance(grid)
      print(path)
    for crew_cell in crewmates:
      if bot_cell == crew_cell:
          bot_cell.value='B'
          return "The bot has rescued a crewmate", bot_cell, crewmates, grid, aliens,time_steps
    for alien_cell in aliens:
      if bot_cell.value== 'A':
        return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
    curr_v=0
    while path[curr_v]!=bot_cell.loc:
      curr_v+=1
    curr_v+=1
    if curr_v==len(path):
      print("end of path reached")
      bot_cell.crewmate_probability=0
      max_prob_val=max(c.crewmate_probability for r in grid for c in r)             # Find max crew prob
      max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]   #Find cell of max prob
      print("closest_max_prob_cell: ", max_prob_index)
      draw_crewmate_probabilities_matrix(grid)
      path = find_path_bot2(bot_cell.loc, max_prob_index)                  # Find closest cell with max prob
      print("bot_cell: ",bot_cell.loc)
      print(path)
      msg, bot_cell, crewmates, grid, aliens, time_steps = move_bot2(grid, bot_cell, path, aliens, crewmates, time_steps, alien_beep)  #Moving on path

    for alien_cell in aliens:
      if bot_cell == alien_cell:
        return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps

    for crew_cell in crewmates:
      if bot_cell == crew_cell:

        bot_cell.value='B'
        return "The bot has rescued a crewmate", bot_cell, crewmates, grid, aliens,time_steps
    next=path[curr_v]
    next_cell = grid[next[0]][next[1]]
    bot_cell.value=1
    grid[bot_cell.x][bot_cell.y].value=1

    for crew_cell in crewmates:
      if (grid[next[0]][next[1]] != crew_cell):
        for alien_cell in aliens:
          if (grid[next[0]][next[1]] != alien_cell):
            grid[next[0]][next[1]].value='B'
            next_cell.value='B'
            next_cell.open = True

    bot_cell=next_cell

    for alien_cell in aliens:
      if bot_cell == alien_cell:
        return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps

    for crew_cell in crewmates:
      if bot_cell == crew_cell:

        bot_cell.value='B'
        return "The bot has rescued a crewmate", bot_cell, crewmates, grid, aliens,time_steps


    for row in grid:
      for cell in row:
        cell.visited=False
        cell.parent=None

    grid[next[0]][next[1]].value='B'
    next_cell.value='B'

    status, crewmates, aliens, time_steps, alien_beep = take_step(grid, aliens, crewmates, bot_cell,time_steps)

    draw_ship_df(grid)

    if curr_v != len(path):
      max_crew_prob_cell = random.choice(detect_open_cells(grid))
      for row in grid:
        for cell in row:
          if cell.crewmate_probability > max_crew_prob_cell.crewmate_probability:
            max_crew_prob_cell = cell
      if max_crew_prob_cell.crewmate_probability != path[len(path)-1] and max_crew_prob_cell.crewmate_probability > 0.5:                                                # Exiting path to center if a cells crewmate prob goes over .5
        print("detected high prob crewmate")
        bot_cell.crewmate_probability=0
        max_prob_val=max(c.crewmate_probability for r in grid for c in r)             # Find max crew prob
        max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]   #Find cell of max prob
        print("closest_max_prob_cell: ", max_prob_index)
        draw_crewmate_probabilities_matrix(grid)
        path = find_path_bot2(bot_cell.loc, max_prob_index)                  # Find closest cell with max prob
        print("bot_cell: ",bot_cell.loc)
        print(path)
        msg, bot_cell, crewmates, grid, aliens,time_steps = move_bot2(grid, bot_cell, path, aliens, crewmates, time_steps, alien_beep)  #Moving on path

    if status == "The bot was caught by the aliens!":
      return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
    for alien_cell in aliens:
      if bot_cell == alien_cell:
        return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps

  print("End of move_bot2")
  return "The bot has rescued a crewmate", bot_cell, crewmates, grid, aliens,time_steps

def alien_avoidance(grid):
  left_sector = 0
  right_sector = 0
  top_sector = 0
  bottom_sector = 0
  for x in range(len(grid)):
    for y in range(len(grid[0])):
      if x < y and x + y < len(grid):
        right_sector = right_sector + grid[x][y].alien_probability
      if x <= y and x + y >= len(grid):
        top_sector = top_sector + grid[x][y].alien_probability
      if x >= y and x + y <= len(grid):
        bottom_sector = bottom_sector + grid[x][y].alien_probability
      if x >= y and x + y >= len(grid):
        left_sector = left_sector + grid[x][y].alien_probability
  max_sector = max(left_sector, right_sector, top_sector, bottom_sector)
  if max_sector == left_sector:
    print("Max alien prob is left sector")
    new_x, new_y = bot_cell.x + 1, bot_cell.y
    if is_valid(Cell(new_x, new_y, '', 3), grid) and grid[bot_cell.x+1][bot_cell.y].open:
      path = [bot_cell.loc, grid[bot_cell.x+1][bot_cell.y].loc]
    else:
      if min(top_sector, bottom_sector) == top_sector:
        new_x, new_y = bot_cell.x, bot_cell.y + 1
        if is_valid(Cell(new_x, new_y, '', 3), grid) and grid[bot_cell.x][bot_cell.y+1].open:
          path = [bot_cell.loc, grid[bot_cell.x][bot_cell.y+1].loc]
      else:
        new_x, new_y = bot_cell.x, bot_cell.y - 1
        if is_valid(Cell(new_x, new_y, '', 3), grid) and grid[bot_cell.x][bot_cell.y-1].open:
          path = [bot_cell.loc,grid[bot_cell.x][bot_cell.y-1].loc]
  if max_sector == right_sector:
    print("Max alien prob is right sector")
    new_x, new_y = bot_cell.x - 1, bot_cell.y
    if is_valid(Cell(new_x, new_y, '', 3), grid) and grid[bot_cell.x-1][bot_cell.y].open:
      path = [bot_cell.loc,grid[bot_cell.x-1][bot_cell.y].loc]
    else:
      if min(top_sector, bottom_sector) == top_sector:
        new_x, new_y = bot_cell.x, bot_cell.y + 1
        if is_valid(Cell(new_x, new_y, '', 3), grid) and grid[bot_cell.x][bot_cell.y+1].open:
          path = [bot_cell.loc,grid[bot_cell.x][bot_cell.y+1].loc]
      else:
        new_x, new_y = bot_cell.x, bot_cell.y - 1
        if is_valid(Cell(new_x, new_y, '', 3), grid) and grid[bot_cell.x][bot_cell.y-1].open:
          path = [bot_cell.loc,grid[bot_cell.x][bot_cell.y-1].loc]
  if max_sector == top_sector:
    print("Max alien prob is top sector")
    new_x, new_y = bot_cell.x, bot_cell.y-1
    if is_valid(Cell(new_x, new_y, '', 3), grid) and grid[bot_cell.x][bot_cell.y-1].open:
      path = [bot_cell.loc,grid[bot_cell.x][bot_cell.y-1].loc]
    else:
      if min(right_sector, left_sector) == left_sector:
        new_x, new_y = bot_cell.x-1, bot_cell.y
        if is_valid(Cell(new_x, new_y, '', 3), grid) and grid[bot_cell.x-1][bot_cell.y].open:
          path = [bot_cell.loc,grid[bot_cell.x-1][bot_cell.y].loc]
      else:
        new_x, new_y = bot_cell.x+1, bot_cell.y
        if is_valid(Cell(new_x, new_y, '', 3), grid) and grid[bot_cell.x+1][bot_cell.y].open:
          path = [bot_cell.loc,grid[bot_cell.x+1][bot_cell.y].loc]
  if max_sector == bottom_sector:
    print("Max alien prob is bottom sector")
    new_x, new_y = bot_cell.x, bot_cell.y + 1
    if is_valid(Cell(new_x, new_y, '', 3), grid) and grid[bot_cell.x][bot_cell.y+1].open:
      path = [bot_cell.loc,grid[bot_cell.x][bot_cell.y+1].loc]
    else:
      if min(right_sector, left_sector) == left_sector:
        new_x, new_y = bot_cell.x-1, bot_cell.y
        if is_valid(Cell(new_x, new_y, '', 3), grid) and grid[bot_cell.x-1][bot_cell.y].open:
          path = [bot_cell.loc,grid[bot_cell.x-1][bot_cell.y].loc]
      else:
        new_x, new_y = bot_cell.x+1, bot_cell.y
        if is_valid(Cell(new_x, new_y, '', 3), grid) and grid[bot_cell.x+1][bot_cell.y].open:
          path = [bot_cell.loc,grid[bot_cell.x+1][bot_cell.y].loc]

"""BOT 2 DRIVER CODE"""

bot_id=2
size=30
num_aliens = 1
num_crew = 1
num_time_steps=[]
sim=0
successes=0
alien_caught=0
alphas = [ -2*math.log(.1)/(size/2), -2*math.log(.25)/(size/2), -2*math.log(.5)/(size/2) ]
detector_sizes=[5,7,10]

num_sims=30

for alpha in alphas:
  for detector_size in detector_sizes:
    num_time_steps=[]
    sim=0
    successes=0
    alien_caught=0
    for _ in range(num_sims):
      alien_beep = False
      crewmates = []
      time_steps=1
      grid = create_grid(size, size)
      for row in grid:
        for cell in row:
          cell.neighbors = get_neighbors(cell, grid)
      aliens, crewmates, bot_cell = explore_grid(size, grid, num_aliens, num_crew, detector_size)
      print("Crewmates location: ")
      print(crewmates[0].loc)
      print("Aliens location: ")
      print(aliens[0].loc)
      grid=initialize_cell_neighbors(grid)
      num_open_cells = len(detect_open_cells(grid))
      for row in grid:
        for cell in row:
          cell.open = detect_if_cell_open(cell)
          cell.neighbors = get_open_neighbors(cell, grid)
      distance_map={}
      pred_map={}
      for i in range(len(grid)):
        for j in range(len(grid[0])):
          s=(i,j)
          dlist,f=dijkstra(grid,s)
          distance_map[s]=dlist
          pred_map[s]=f
      initialize_prob_crewmate(grid, num_crew)
      initialize_prob_alien_moved(grid)
      print("INITIAL SHIP")
      draw_ship_df(grid)
      print("INITIAL CREW PROBS MATRIX")
      draw_crewmate_probabilities_matrix(grid)
      print("initial alien prob matrix")
      draw_alien_probabilities_matrix(grid)
      print("Sum of Alien Probabilities")
      print(sum_probabilities_alien(grid))
      print("Sum of Crewmate Probabilities")
      print(sum_probabilities_crewmate(grid))
      #------------------------------
      centre_cells=[]
      if len(grid)%2==1:
        centre_cells.append([len(grid)//2,len(grid)//2])                             # Finding Center Cell
      else:
        centre_cells.append([len(grid)//2,len(grid)//2])
        centre_cells.append([len(grid)//2-1,len(grid)//2])
        centre_cells.append([len(grid)//2,len(grid)//2-1])
        centre_cells.append([len(grid)//2-1,len(grid)//2-1])
      if len(centre_cells)==1 and grid[centre_cells[0][0]][centre_cells[0][1]].open==False:     #Corner case closed center cell
        centre_cells_mod=grid[centre_cells[0][0]][centre_cells[0][1]].neighbors
        centre_cells=[ce.loc for ce in centre_cells_mod]
      print("BOT CELL ", bot_cell.loc)
      print("CENTER CELL ", centre_cells[0])
      if bot_cell.loc == centre_cells[0]:                                                 # Corner case where bot spawns on center cell
        print("centre cells: ",centre_cells)
        print("bot_cell: ",bot_cell.loc)
        print("bot_cell spawned at center cell")
        max_prob_val=max(c.crewmate_probability for r in grid for c in r)             # Find max crew prob
        max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]   #Find cell of max prob
        print("closest_max_prob_cell: ", max_prob_index)
        draw_crewmate_probabilities_matrix(grid)
        path = find_path_bot2(bot_cell.loc, max_prob_index)                  # Find closest cell with max prob
        print("bot_cell: ",bot_cell.loc)
        print(path)
        msg, bot_cell, crewmates, grid, aliens,time_steps = move_bot2(grid, bot_cell, path, aliens, crewmates, time_steps, alien_beep)  #Moving on path

      path = find_path_bot2(bot_cell.loc, centre_cells)                                 # Path to center cell
      print("centre cells: ",centre_cells)
      print("bot_cell: ",bot_cell.loc)
      print(path)
      msg, bot_cell, crewmates, grid, aliens,time_steps = move_bot2(grid, bot_cell, path, aliens, crewmates, time_steps, alien_beep)  #Moving on path
      if msg=="reached centre":
        print("REACHED THE CENTRE")
      elif msg=="entered a high prob region":
        print("REACHED AN AREA WITH PROB >0.25")

      print(msg)
      sim+=1
      print("Simmulation Number: ", sim)
      if (msg=="The bot was caught by the aliens!"):
          alien_caught+=1
      elif (msg=="The bot has rescued a crewmate"):
          successes+=1
          num_time_steps.append(time_steps)
      print("SIMULATION NUMBER",sim)
      print("BOT STATUS",msg)
      print("CURRENT STATUS OF NUM TIME STEPS IS",num_time_steps)

      file_name_1 = f"bot2_interim_output_sim_{sim}_alpha_{alpha}_k_{detector_size}.txt"

      with open(file_name_1, "a") as file1:
        file1.write("SIMULATION NUMBER"+str(sim)+"\n")
        file1.write("BOT STATUS"+str(msg)+"\n")
        file1.write("CURRENT STATUS OF NUM TIME STEPS IS"+str(num_time_steps)+"\n")

    print("OUT OF 30 SIMULATIONS")
    print("BOT RESCUED CREWMATE",successes,"TIMES")
    print("BOT WAS CAUGHT BY ALIENS",alien_caught,"TIMES")

    print("AVERAGE MOVES REQUIRED TO RESCUE CREWMATE IS",sum(num_time_steps)/successes)
    print("AVERAGE NUMBER OF CREWMATES RESCUED PER SIMULATION",successes/num_sims)
    print("The time steps taken are: ",time_steps)


    file_name = f"bot2_output_alpha_{alpha}_k_{detector_size}.txt"

    with open(file_name, "a") as file:
      file.write("FOR ALPHA VALUE"+str(alpha)+"AND K VALUE"+str(detector_size)+ "FOR BOT2"+"\n")
      file.write("OUT OF"+str(num_sims)+"SIMULATIONS"+"\n")
      file.write("BOT RESCUED CREWMATE"+str(successes)+"TIMES"+"\n")
      file.write("BOT WAS CAUGHT BY ALIENS"+str(alien_caught)+"TIMES"+"\n")
      file.write("AVERAGE MOVES REQUIRED TO RESCUE CREWMATE IS"+str(sum(num_time_steps)/successes)+"\n")
      file.write("PROBABILITY OF BOT SUCCESSFULLY AVOIDING ALIENS IS"+str(sum(num_time_steps)/successes)+"\n")

"""BOT 3 CODE"""

def move_bot3(grid, bot_cell, path, aliens, crewmates, time_steps):
  for alien_cell in aliens:
    if bot_cell == alien_cell:
      return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
  while len(path) == 0:
    print("NO PATH NOW SO ALIENS MOVE")
    status,crewmates, aliens,time_steps = take_step(grid, aliens, crewmates, bot_cell,time_steps)
    res, path, time_steps = breadth_first_search_bot_1(grid,bot_cell, bot_cell, aliens, time_steps, crewmates)
  while True:
    next=path[1]
    print("MOVING THE BOT TO",next)
    next_cell = grid[next[0]][next[1]]
        if next_cell.alien_probability>0.00001:
          #print("condition entered")
          status, crewmates, aliens,time_steps, alien_beep, grid = take_step(grid, aliens, crewmates, bot_cell,time_steps)
          max_prob_val=max(c.crewmate_probability for r in grid for c in r)
          max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
          max_prob_cell=random.choice(max_prob_index)
          path=find_path(bot_cell.loc,max_prob_cell)
          continue
    bot_cell.value=1
    grid[bot_cell.x][bot_cell.y].value=1
    if (grid[next[0]][next[1]].value!='C' and grid[next[0]][next[1]].value!='C*' and grid[next[0]][next[1]].value!='A'):
      grid[next[0]][next[1]].value='B'
      next_cell.value='B'
      next_cell.open = True

    bot_cell=next_cell
    for alien_cell in aliens:
      if bot_cell == alien_cell:
        print("ALIENS CAUGHT BOT AT",bot_cell.loc)
        draw_ship_df(grid)
        return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
    for crew_cell in crewmates:
      if bot_cell == crew_cell:

        print("BOT FOUND CREWMATE AT",bot_cell.loc)
        draw_ship_df(grid)
        bot_cell.value='B'
        return "The bot has rescued a crewmate", bot_cell, crewmates, grid, aliens,time_steps

    bot_cell.crewmate_probability=0

    max_prob_val=max(c.crewmate_probability for r in grid for c in r)
    max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
    max_prob_cell=random.choice(max_prob_index)
    path=find_path(bot_cell.loc,max_prob_cell)

    for row in grid:
      for cell in row:
        cell.visited=False
        cell.parent=None

    grid[next[0]][next[1]].value='B'
    next_cell.value='B'

    print("BOT MOVED")
    draw_ship_df(grid)

    status, crewmates, aliens,time_steps, alien_beep = take_step(grid, aliens, crewmates, bot_cell,time_steps)

    print("ALIEN MOVED")
    draw_ship_df(grid)

    if status == "The bot was caught by the aliens!":
      return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
    for alien_cell in aliens:
      if bot_cell == alien_cell:
        return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
    print("PATH GOING TO BE FOLLOWED BY BOT3 IS",path)

  return "The bot has rescued a crewmate", bot_cell, crewmates, grid, aliens,time_steps

# BOT 3 DRIVER CODE

bot_id = 3
size = 30
num_aliens = 1
num_crew = 2
alphas = [ -2*math.log(.1)/(size/2), -2*math.log(.25)/(size/2), -2*math.log(.5)/(size/2) ]
crewmate_pairs_probs = {}
crewmates = []
num_time_steps=[]
sim=0
crew1_res=0
crew_res_both=0
alien_caught=0

detector_sizes=[1,5,7,10,15]

num_sims=30

for alpha in alphas:
  for detector_size in detector_sizes:
    num_time_steps=[]
    sim=0
    crew1_res=0
    crew_res_both=0
    alien_caught=0
    for _ in range(num_sims):
      alien_beep = False
      time_steps=1
      crewmate_pairs_probs = {}
      crewmates = []
      grid = create_grid(size, size)
      for row in grid:
        for cell in row:
          cell.neighbors = get_neighbors(cell, grid)
      aliens, crewmates, bot_cell = explore_grid(size, grid, num_aliens, num_crew, detector_size)


      grid=initialize_cell_neighbors(grid)
      num_open_cells = len(detect_open_cells(grid))
      for row in grid:
        for cell in row:
          cell.open = detect_if_cell_open(cell)
          cell.neighbors = get_open_neighbors(cell, grid)

      distance_map={}
      pred_map={}
      for i in range(len(grid)):
        for j in range(len(grid[0])):
          s=(i,j)
          dlist,f=dijkstra(grid,s)
          distance_map[s]=dlist
          pred_map[s]=f

      initialize_prob_crewmate(grid, num_crew)
      initialize_prob_alien_moved(grid)
      draw_ship_df(grid)
      print("INITIAL CREW PROBS MATRIX")
      draw_crewmate_probabilities_matrix(grid)
      print("initial alien prob matrix")
      draw_alien_probabilities_matrix(grid)
      print("Sum of Alien Probabilities")
      print(sum_probabilities_alien(grid))
      print("Sum of Crewmate Probabilities")
      print(sum_probabilities_crewmate(grid))


      max_prob_val=max(c.crewmate_probability for r in grid for c in r)
      max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
      max_prob_cell=random.choice(max_prob_index)
      path=find_path(bot_cell.loc,max_prob_cell)
      for row in grid:
        for cell in row:
          cell.visited=False
          cell.parent=None

      print("-----------------------------------------------------------------------------------------------------")
      print("crew member was found via path",path)
      if len(path)!=0:
        print("Now moving Bot to rescue Crewmate...")

      msg1, bot_cell, crewmates, grid, aliens,time_steps = move_bot3(grid, bot_cell, path, aliens, crewmates, time_steps)
      print(msg1)

      if (msg1=="The bot has rescued a crewmate"):
        time_steps_1=time_steps
        print("NEW STARTING POSITION OF BOT IS",bot_cell.loc)

        bot_cell.crewmate_probability=0

        for row in grid:
          for cell in row:
            cell.visited=False
            cell.parent=None

        grid[bot_cell.x][bot_cell.y].value='B'
        bot_cell.value='B'

        draw_ship_df(grid)

        max_prob_val=max(c.crewmate_probability for r in grid for c in r)
        max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
        max_prob_cell=random.choice(max_prob_index)
        path=find_path(bot_cell.loc,max_prob_cell)
        print("-----------------------------------------------------------------------------------------------------")
        print(" Second crew member was found via path",path)

        for row in grid:
          for cell in row:
            cell.visited=False
            cell.parent=None

        msg2, bot_cell, crewmates, grid, aliens,time_steps = move_bot3(grid, bot_cell, path, aliens, crewmates, time_steps)
        print(msg2)

      print("SIMULATION NUMBER",sim)
      print("CURRENT STATUS OF NUM TIME STEPS IS",num_time_steps)

      file_name_1 = f"bot3_interim_output_sim_{sim}_alpha_{alpha}_k_{detector_size}.txt"

      with open(file_name_1, "a") as file1:
        file1.write("SIMULATION NUMBER "+str(sim)+"\n")
        file1.write("CURRENT STATUS OF NUM TIME STEPS IS "+str(num_time_steps)+"\n")

      if (msg1=="The bot was caught by the aliens!"):
        alien_caught+=1
      if (msg1=="The bot has rescued a crewmate" and msg2=="The bot was caught by the aliens!"):
        crew1_res+=1
        alien_caught+=1
        num_time_steps.append(time_steps_1)
      if (msg1=="The bot has rescued a crewmate" and msg2=="The bot has rescued a crewmate"):
        crew_res_both+=1
        num_time_steps.append(time_steps)

      sim+=1

      print("END OF SIMULATION. RELOAD GRID.")
      print("\n")
      print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
      print("\n")

    print("THE BOT RESCUED",crew1_res+crew_res_both*2,"CREWMATES")

    print("AVERAGE NUMBER OF CREWMATES RESCUED PER SIMULATION",(crew1_res+crew_res_both*2)/num_sims)

    if crew_res_both != 0:
      print("AVERAGE NUMBER OF MOVES NEEDED TO RESCUE ALL CREWMATES IS",sum(num_time_steps)/crew_res_both)
    else:
      print("AVERAGE NUMBER OF MOVES NEEDED TO RESCUE ALL CREWMATES IS NONE RESQUED")

    file_name = f"bot3_output_alpha_{alpha}_k_{detector_size}.txt"

    with open(file_name, "a") as file:
      file.write("FOR ALPHA VALUE"+str(alpha)+"AND K VALUE "+str(detector_size)+ "FOR BOT3"+"\n")
      file.write("OUT OF "+str(num_sims)+"SIMULATIONS"+"\n")
      file.write("BOT RESCUED CREWMATE "+str(crew1_res+crew_res_both*2)+" TIMES"+"\n")
      file.write("BOT WAS CAUGHT BY ALIENS "+str(alien_caught)+" TIMES"+"\n")
      file.write("AVERAGE MOVES REQUIRED TO RESCUE CREWMATE IS "+str(sum(num_time_steps)/(crew1_res+crew_res_both*2))+"\n")

"""BOT 4 Code"""

def move_bot4(grid, bot_cell, path, aliens, crewmates, time_steps):
  for alien_cell in aliens:
    if bot_cell == alien_cell:
      return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
  while len(path) == 0:
    print("NO PATH NOW SO ALIENS MOVE")
    status,crewmates, aliens,time_steps, alien_beep = take_step(grid, aliens, crewmates, bot_cell,time_steps)
    max_prob_val=max(c.crewmate_probability for r in grid for c in r)
    max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
    max_prob_cell=random.choice(max_prob_index)
    path=find_path(bot_cell.loc,max_prob_cell)
  while True:
    next=path[1]
    print("MOVING THE BOT TO",next)
        if next_cell.alien_probability>0.00001:
          #print("condition entered")
          status, crewmates, aliens,time_steps, alien_beep, grid = take_step(grid, aliens, crewmates, bot_cell,time_steps)
          max_prob_val=max(c.crewmate_probability for r in grid for c in r)
          max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
          max_prob_cell=random.choice(max_prob_index)
          path=find_path(bot_cell.loc,max_prob_cell)
          continue
    next_cell = grid[next[0]][next[1]]
    bot_cell.value=1
    grid[bot_cell.x][bot_cell.y].value=1
    if (grid[next[0]][next[1]].value!='C' and grid[next[0]][next[1]].value!='C*' and grid[next[0]][next[1]].value!='A'):
      grid[next[0]][next[1]].value='B'
      next_cell.value='B'
      next_cell.open = True

    bot_cell=next_cell
    for alien_cell in aliens:
      if bot_cell == alien_cell:
        print("ALIENS CAUGHT BOT AT",bot_cell.loc)
        draw_ship_df(grid)
        return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
    for crew_cell in crewmates:
      if bot_cell == crew_cell:
        print("BOT FOUND CREWMATE AT",bot_cell.loc)
        draw_ship_df(grid)
        bot_cell.value='B'
        return "The bot has rescued a crewmate", bot_cell, crewmates, grid, aliens,time_steps


    bot_cell.crewmate_probability=0

    max_prob_val=max(c.crewmate_probability for r in grid for c in r)
    max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
    max_prob_cell=random.choice(max_prob_index)
    path=find_path(bot_cell.loc,max_prob_cell)

    for row in grid:
      for cell in row:
        cell.visited=False
        cell.parent=None

    grid[next[0]][next[1]].value='B'
    next_cell.value='B'

    status, crewmates, aliens,time_steps, alien_beep = take_step(grid, aliens, crewmates, bot_cell,time_steps)


    if status == "The bot was caught by the aliens!":
      return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
    for alien_cell in aliens:
      if bot_cell == alien_cell:
        return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
    print("PATH GOING TO BE FOLLOWED BY BOT4 IS",path)

  return "The bot has rescued a crewmate", bot_cell, crewmates, grid, aliens,time_steps

# BOT 4 DRIVER CODE
bot_id = 4
size = 50
num_aliens = 1
num_crew = 2
detector_size = 5
alpha = -2*math.log(.1)/size                                                                  # Calculating alpha based on ship size
print("Alpha: ", alpha)
crewmate_pairs_probs = {}
crewmates = []
num_time_steps=[]
sim=0
crew1_res=0
crew_res_both=0
alien_caught=0

num_sims=10

for _ in range(10):
  alien_beep = False
  time_steps=1
  crewmate_pairs_probs = {}
  crewmates = []
  grid = create_grid(size, size)
  for row in grid:
    for cell in row:
      cell.neighbors = get_neighbors(cell, grid)

  aliens, crewmates, bot_cell = explore_grid(size, grid, num_aliens, num_crew, detector_size)


  grid=initialize_cell_neighbors(grid)
  num_open_cells = len(detect_open_cells(grid))
  for row in grid:
    for cell in row:
      cell.open = detect_if_cell_open(cell)
      cell.neighbors = get_open_neighbors(cell, grid)

  distance_map={}
  pred_map={}
  for i in range(len(grid)):
    for j in range(len(grid[0])):
      s=(i,j)
      dlist,f=dijkstra(grid,s)
      distance_map[s]=dlist
      pred_map[s]=f

  initialize_prob_crewmate(grid, num_crew)
  initialize_crewmate_pairs_probs(crewmate_pairs_probs)
  initialize_prob_alien_moved(grid)
  draw_ship_df(grid)
  print("INITIAL CREW PROBS MATRIX")
  draw_crewmate_probabilities_matrix(grid)
  print("initial alien prob matrix")
  draw_alien_probabilities_matrix(grid)
  print("Sum of Alien Probabilities")
  print(sum_probabilities_alien(grid))
  print("Sum of Crewmate Probabilities")
  print(sum_probabilities_crewmate(grid))

  max_prob_val=max(c.crewmate_probability for r in grid for c in r)
  max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
  max_prob_cell=random.choice(max_prob_index)
  path=find_path(bot_cell.loc,max_prob_cell)
  for row in grid:
    for cell in row:
      cell.visited=False
      cell.parent=None

  print("-----------------------------------------------------------------------------------------------------")
  print("crew member was found via path",path)
  if len(path)!=0:
    print("Now moving Bot to rescue Crewmate...")

  msg1, bot_cell, crewmates, grid, aliens,time_steps = move_bot4(grid, bot_cell, path, aliens, crewmates, time_steps)
  print(msg1)

  if (msg1=="The bot has rescued a crewmate"):
    print("NEW STARTING POSITION OF BOT IS",bot_cell.loc)

    bot_cell.crewmate_probability=0

    for row in grid:
      for cell in row:
        cell.visited=False
        cell.parent=None

    grid[bot_cell.x][bot_cell.y].value='B'
    bot_cell.value='B'

    max_prob_val=max(c.crewmate_probability for r in grid for c in r)
    max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
    max_prob_cell=random.choice(max_prob_index)
    path=find_path(bot_cell.loc,max_prob_cell)
    print("-----------------------------------------------------------------------------------------------------")
    print(" Second crew member was found via path",path)

    for row in grid:
      for cell in row:
        cell.visited=False
        cell.parent=None

    msg2, bot_cell, crewmates, grid, aliens,time_steps = move_bot4(grid, bot_cell, path, aliens, crewmates, time_steps)
    print(msg2)

  if (msg1=="The bot was caught by the aliens!" or msg2=="The bot was caught by the aliens!"):
    alien_caught+=1
  if (msg1=="The bot has rescued a crewmate" and msg2=="The bot was caught by the aliens!"):
    crew1_res+=1
  if (msg1=="The bot has rescued a crewmate" and msg2=="The bot has rescued a crewmate"):
    crew_res_both+=1
    num_time_steps.append(time_steps)

  print("END OF SIMULATION. RELOAD GRID.")

print("OUT OF 30 SIMULATIONS AND 60 CREWMATES")
print("THE BOT RESCUED",crew1_res+crew_res_both*2,"CREWMATES")
print("AVERAGE NUMBER OF CREWMATES RESCUED PER SIMULATION",(crew1_res+crew_res_both*2)/30)
if crew_res_both != 0:
  print("AVERAGE NUMBER OF MOVES NEEDED TO RESCUE ALL CREWMATES IS",sum(num_time_steps)/crew_res_both)
else:
  print("AVERAGE NUMBER OF MOVES NEEDED TO RESCUE ALL CREWMATES IS NONE RESQUED")
print("PROB OF RESCUING BOTH CREWMATES IS ",crew_res_both/num_sims,"\n")

"""Bot 5 Code"""

bot_id=5
size=10
num_aliens = 1
num_crew = 2
detector_size = 1
alpha = -2*math.log(.1)/(size/2)                                                                  # Calculating alpha based on ship size
print("Alpha: ", alpha)
num_time_steps=[]
sim=0

num_sims=1

alien_caught=0
crew1_res=0
crew_res_both=0

for _ in range(num_sims):
  alien_beep = False
  crewmates = []
  crewmate_pairs_probs = {}
  alien_pairs_probs = {}
  time_steps=1
  grid = create_grid(size, size)
  for row in grid:
    for cell in row:
      cell.neighbors = get_neighbors(cell, grid)
  aliens, crewmates, bot_cell = explore_grid(size, grid, num_aliens, num_crew, detector_size)
  print("Crewmates location: ")
  print(crewmates[0].loc)
  print("Aliens location: ")
  print(aliens[0].loc)
  grid=initialize_cell_neighbors(grid)
  num_open_cells = len(detect_open_cells(grid))
  for row in grid:
    for cell in row:
      cell.open = detect_if_cell_open(cell)
      cell.neighbors = get_open_neighbors(cell, grid)
  distance_map={}
  pred_map={}
  for i in range(len(grid)):
    for j in range(len(grid[0])):
      s=(i,j)
      dlist,f=dijkstra(grid,s)
      distance_map[s]=dlist
      pred_map[s]=f
  initialize_prob_crewmate(grid, num_crew)
  initialize_crewmate_pairs_probs(crewmate_pairs_probs)
  initialize_prob_alien_moved(grid)
  print("INITIAL SHIP")
  draw_ship_df(grid)
  print("INITIAL CREW PROBS MATRIX")
  draw_crewmate_probabilities_matrix(grid)
  print("initial alien prob matrix")
  draw_alien_probabilities_matrix(grid)
  print("Sum of Alien Probabilities")
  print(sum_probabilities_alien(grid))
  print("Sum of Crewmate Probabilities")
  print(sum_probabilities_crewmate(grid))
  #------------------------------
  centre_cells=[]
  if len(grid)%2==1:
    centre_cells.append([len(grid)//2,len(grid)//2])                             # Finding Center Cell
  else:
    centre_cells.append([len(grid)//2,len(grid)//2])
    centre_cells.append([len(grid)//2-1,len(grid)//2])
    centre_cells.append([len(grid)//2,len(grid)//2-1])
    centre_cells.append([len(grid)//2-1,len(grid)//2-1])
  if len(centre_cells)==1 and grid[centre_cells[0][0]][centre_cells[0][1]].open==False:     #Corner case closed center cell
    centre_cells_mod=grid[centre_cells[0][0]][centre_cells[0][1]].neighbors
    centre_cells=[ce.loc for ce in centre_cells_mod]
  print("BOT CELL ", bot_cell.loc)
  print("CENTER CELL ", centre_cells[0])
  if bot_cell.loc == centre_cells[0]:                                                 # Corner case where bot spawns on center cell
    print("centre cells: ",centre_cells)
    print("bot_cell: ",bot_cell.loc)
    print("bot_cell spawned at center cell")
    max_prob_val=max(c.crewmate_probability for r in grid for c in r)             # Find max crew prob
    max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]   #Find cell of max prob
    print("closest_max_prob_cell: ", max_prob_index)
    draw_crewmate_probabilities_matrix(grid)
    path = find_path_bot2(bot_cell.loc, max_prob_index)                  # Find closest cell with max prob
    print("bot_cell: ",bot_cell.loc)
    print(path)
    msg, bot_cell, crewmates, grid, aliens,time_steps = move_bot2(grid, bot_cell, path, aliens, crewmates, time_steps, alien_beep)  #Moving on path

  path = find_path_bot2(bot_cell.loc, centre_cells)                                 # Path to center cell
  print("centre cells: ",centre_cells)
  print("bot_cell: ",bot_cell.loc)
  print(path)
  msg1, bot_cell, crewmates, grid, aliens,time_steps = move_bot2(grid, bot_cell, path, aliens, crewmates, time_steps, alien_beep)  #Moving on path


  print(msg1)
  msg2=""

  if (msg1=="The bot has rescued a crewmate"):
    crewmates.remove(bot_cell) # Removing the rescued crewmate cell from the crewmates array

    print("NEW STARTING POSITION OF BOT IS",bot_cell.loc)

    bot_cell.crewmate_probability=0

    for row in grid:
      for cell in row:
        cell.visited=False
        cell.parent=None

    grid[bot_cell.x][bot_cell.y].value='B'
    bot_cell.value='B'

    max_prob_val=max(c.crewmate_probability for r in grid for c in r)
    max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
    max_prob_cell=random.choice(max_prob_index)
    path=find_path_bot2(bot_cell.loc,[[max_prob_cell[0],max_prob_cell[1]]])
    print("PATH TO SECOND CREW IS",path)
    print("-----------------------------------------------------------------------------------------------------")
    print(" Second crew member was found via path",path)

    for row in grid:
      for cell in row:
        cell.visited=False
        cell.parent=None

    for cr in crewmates:
      print(cr.loc)
    msg2, bot_cell, crewmates, grid, aliens,time_steps = move_bot2(grid, bot_cell, path, aliens, crewmates, time_steps, alien_beep)
    draw_ship_df(grid)
    print(msg2)

  if (msg1=="The bot was caught by the aliens!" or msg2=="The bot was caught by the aliens!"):
    alien_caught+=1
  if (msg1=="The bot has rescued a crewmate" and msg2=="The bot was caught by the aliens!"):
    crew1_res+=1
  if (msg1=="The bot has rescued a crewmate" and msg2=="The bot has rescued a crewmate"):
    crew_res_both+=1
    num_time_steps.append(time_steps)
  sim+=1
  print("SIMULATION",sim,"TOOK",time_steps)
  print("CURRENT NUM TIME STEPS ARRAY IS",num_time_steps)

  print("END OF SIMULATION",sim,". RELOAD GRID.")

print("OUT OF",num_sims,"SIMULATIONS AND",num_sims*2,"CREWMATES")
print("THE BOT RESCUED",crew1_res+crew_res_both*2,"CREWMATES")
print("AVERAGE NUMBER OF CREWMATES RESCUED PER SIMULATION",(crew1_res+crew_res_both*2)/num_sims)
if crew_res_both != 0:
  print("AVERAGE NUMBER OF MOVES NEEDED TO RESCUE ALL CREWMATES IS",sum(num_time_steps)/crew_res_both)
else:
  print("AVERAGE NUMBER OF MOVES NEEDED TO RESCUE ALL CREWMATES IS NONE RESCUED")
print("PROB OF RESCUING BOTH CREWMATES IS ",crew_res_both/num_sims,"\n")

"""BOT 6 Code"""

def move_bot6(grid, bot_cell, path, aliens, crewmates, time_steps):
  if bot_cell.value=='A' or bot_cell.value=='C*':
    return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
  while len(path) == 0:
    print("NO PATH NOW SO ALIENS MOVE")
    status,crewmates, aliens,time_steps, alien_beep = take_step(grid, aliens, crewmates, bot_cell,time_steps)
    max_prob_val=max(c.crewmate_probability for r in grid for c in r)
    max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
    max_prob_cell=random.choice(max_prob_index)
    path=find_path(bot_cell.loc,max_prob_cell)
  while True:
    next=path[1]
    print("MOVING THE BOT TO",next)
        if next_cell.alien_probability>0.00001:
          #print("condition entered")
          status, crewmates, aliens,time_steps, alien_beep, grid = take_step(grid, aliens, crewmates, bot_cell,time_steps)
          max_prob_val=max(c.crewmate_probability for r in grid for c in r)
          max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
          max_prob_cell=random.choice(max_prob_index)
          path=find_path(bot_cell.loc,max_prob_cell)
          continue
    next_cell = grid[next[0]][next[1]]
    bot_cell.value=1
    grid[bot_cell.x][bot_cell.y].value=1
    if (grid[next[0]][next[1]].value!='C' and grid[next[0]][next[1]].value!='C*' and grid[next[0]][next[1]].value!='A'):
      grid[next[0]][next[1]].value='B'
      next_cell.value='B'
      next_cell.open = True

    bot_cell=next_cell
    if bot_cell.value=='A' or bot_cell.value=='C*':
      print("ALIENS CAUGHT BOT AT",bot_cell.loc)
      draw_ship_df(grid)
      return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
    if bot_cell.value=='C':

      print("BOT FOUND CREWMATE AT",bot_cell.loc)
      draw_ship_df(grid)
      bot_cell.value='B'
      return "The bot has rescued a crewmate", bot_cell, crewmates, grid, aliens,time_steps

    bot_cell.crewmate_probability=0

    max_prob_val=max(c.crewmate_probability for r in grid for c in r)
    max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
    max_prob_cell=random.choice(max_prob_index)
    path=find_path(bot_cell.loc,max_prob_cell)

    for row in grid:
      for cell in row:
        cell.visited=False
        cell.parent=None

    grid[next[0]][next[1]].value='B'
    next_cell.value='B'

    print("BOT MOVED")
    draw_ship_df(grid)

    status, crewmates, aliens,time_steps, alien_beep = take_step(grid, aliens, crewmates, bot_cell,time_steps)

    print("ALIEN MOVED")
    draw_ship_df(grid)

    if status == "The bot was caught by the aliens!":
      return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
    if bot_cell.value=='A' or bot_cell.value=='C*':
      return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
    print("PATH GOING TO BE FOLLOWED BY BOT3 IS",path)

  return "The bot has rescued a crewmate", bot_cell, crewmates, grid, aliens,time_steps

# BOT 6 DRIVER CODE

bot_id = 6
size = 30
num_aliens = 2
num_crew = 2
detector_sizes=[ 5,7,10 ]
alphas = [ -2*math.log(.1)/(size/2), -2*math.log(.25)/(size/2), -2*math.log(.5)/(size/2) ]
num_time_steps=[]
sim=0
crew1_res=0
crew_res_both=0
alien_caught=0
num_sims=10

for alpha in alphas:
  for detector_size in detector_sizes:
    sim=0
    crew1_res=0
    crew_res_both=0
    alien_caught=0
    num_time_steps=[]
    for _ in range(num_sims):
      alien_beep = False
      time_steps=1
      crewmates = []
      grid = create_grid(size, size)
      for row in grid:
        for cell in row:
          cell.neighbors = get_neighbors(cell, grid)
      aliens, crewmates, bot_cell = explore_grid(size, grid, num_aliens, num_crew, detector_size)

      grid=initialize_cell_neighbors(grid)
      num_open_cells = len(detect_open_cells(grid))
      for row in grid:
        for cell in row:
          cell.open = detect_if_cell_open(cell)
          cell.neighbors = get_open_neighbors(cell, grid)

      distance_map={}
      pred_map={}
      for i in range(len(grid)):
        for j in range(len(grid[0])):
          s=(i,j)
          dlist,f=dijkstra(grid,s)
          distance_map[s]=dlist
          pred_map[s]=f

      initialize_prob_crewmate(grid, num_crew)
      initialize_prob_alien_moved(grid)
      draw_ship_df(grid)
      print("INITIAL CREW PROBS MATRIX")
      draw_crewmate_probabilities_matrix(grid)
      print("initial alien prob matrix")
      draw_alien_probabilities_matrix(grid)
      print("Sum of Alien Probabilities")
      print(sum_probabilities_alien(grid))
      print("Sum of Crewmate Probabilities")
      print(sum_probabilities_crewmate(grid))


      max_prob_val=max(c.crewmate_probability for r in grid for c in r)
      max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
      max_prob_cell=random.choice(max_prob_index)
      path=find_path(bot_cell.loc,max_prob_cell)

      for row in grid:
        for cell in row:
          cell.visited=False
          cell.parent=None

      print("-----------------------------------------------------------------------------------------------------")
      print("crew member was found via path",path)
      if len(path)!=0:
        print("Now moving Bot to rescue Crewmate...")

      msg1, bot_cell, crewmates, grid, aliens,time_steps = move_bot6(grid, bot_cell, path, aliens, crewmates, time_steps)
      print(msg1)

      if (msg1=="The bot has rescued a crewmate"):
        print("NEW STARTING POSITION OF BOT IS",bot_cell)
        max_prob_val=max(c.crewmate_probability for r in grid for c in r)
        max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
        max_prob_cell=random.choice(max_prob_index)
        path=find_path(bot_cell.loc,max_prob_cell)

        print("-----------------------------------------------------------------------------------------------------")
        print(" Second crew member was found via path",path)

        for row in grid:
          for cell in row:
            cell.visited=False
            cell.parent=None

        msg2, bot_cell, crewmates, grid, aliens,time_steps = move_bot6(grid, bot_cell, path, aliens, crewmates, time_steps)
        print(msg2)

      if (msg1=="The bot was caught by the aliens!" or msg2=="The bot was caught by the aliens!"):
        alien_caught+=1
      if (msg1=="The bot has rescued a crewmate" and msg2=="The bot was caught by the aliens!"):
        crew1_res+=1
      if (msg1=="The bot has rescued a crewmate" and msg2=="The bot has rescued a crewmate"):
        crew_res_both+=1
        num_time_steps.append(time_steps)

      file_name_1 = f"bot6_interim_output_sim_{sim}_alpha_{alpha}_k_{detector_size}.txt"

      with open(file_name_1, "a") as file1:
        file1.write("SIMULATION NUMBER "+str(sim+1)+"\n")
        file1.write("CURRENT STATUS OF NUM TIME STEPS IS "+str(num_time_steps)+"\n")
        file1.write("CURRENT STATUS: BOT RESCUED CREWMATE "+str(crew1_res+crew_res_both*2)+" TIMES"+"\n")

      print("CURRENT STATUS OF NUM TIME STEPS IS",num_time_steps)
      sim+=1

      print("END OF SIMULATION. RELOAD GRID.")

    print("THE BOT RESCUED",crew1_res+crew_res_both*2,"CREWMATES")
    if crew_res_both != 0:
      print("AVERAGE NUMBER OF MOVES NEEDED TO RESCUE ALL CREWMATES IS",sum(num_time_steps)/crew_res_both)
    else:
      print("AVERAGE NUMBER OF MOVES NEEDED TO RESCUE ALL CREWMATES IS NONE RESQUED")

    file_name = f"bot6_output_alpha_{alpha}_k_{detector_size}.txt"

    with open(file_name, "a") as file:
      file.write("FOR ALPHA VALUE"+str(alpha)+"AND K VALUE "+str(detector_size)+ "FOR BOT6"+"\n")
      file.write("OUT OF "+str(num_sims)+"SIMULATIONS"+"\n")
      file.write("BOT RESCUED CREWMATE "+str(crew1_res+crew_res_both*2)+" TIMES"+"\n")
      file.write("BOT WAS CAUGHT BY ALIENS "+str(alien_caught)+" TIMES"+"\n")
      if crew_res_both != 0:
        file.write("AVERAGE MOVES REQUIRED TO RESCUE ALL CREWMATES IS "+str(sum(num_time_steps)/(crew_res_both*2))+"\n")
      else:
        file.write("THE BOT WASNT ABLE TO RESCUE BOTH CREWMATES IN ANY CASE"+"\n")

      file.write("PROB OF RESCUING BOTH CREWMATES IS "+str(crew_res_both/num_sims)+"\n")

def move_bot7(grid, bot_cell, path, aliens, crewmates, time_steps):
  for alien_cell in aliens:
    if bot_cell == alien_cell:
      return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
  while len(path) == 0:
    print("NO PATH NOW SO ALIENS MOVE")
    status,crewmates, aliens,time_steps, alien_beep = take_step(grid, aliens, crewmates, bot_cell,time_steps)
    max_prob_val=max(c.crewmate_probability for r in grid for c in r)
    max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
    max_prob_cell=random.choice(max_prob_index)
    path=find_path(bot_cell.loc,max_prob_cell)
  while True:
    next=path[1]
    print("MOVING THE BOT TO",next)
        if next_cell.alien_probability>0.00001:
          #print("condition entered")
          status, crewmates, aliens,time_steps, alien_beep, grid = take_step(grid, aliens, crewmates, bot_cell,time_steps)
          max_prob_val=max(c.crewmate_probability for r in grid for c in r)
          max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
          max_prob_cell=random.choice(max_prob_index)
          path=find_path(bot_cell.loc,max_prob_cell)
          continue
    next_cell = grid[next[0]][next[1]]
    bot_cell.value=1
    grid[bot_cell.x][bot_cell.y].value=1
    for alien_cell in aliens:
      if (grid[next[0]][next[1]] != alien_cell):
        for crew_cell in crewmates:
          if (grid[next[0]][next[1]] != crew_cell):
            grid[next[0]][next[1]].value='B'
            next_cell.value='B'
            next_cell.open = True

    bot_cell=next_cell
    for alien_cell in aliens:
      if bot_cell == alien_cell:
        print("ALIENS CAUGHT BOT AT",bot_cell.loc)
        draw_ship_df(grid)
        return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
    for crew_cell in crewmates:
      if bot_cell == crew_cell:
        bot_cell.crewmate_probability = 0
        print("BOT FOUND CREWMATE AT",bot_cell.loc)
        draw_ship_df(grid)
        bot_cell.value='B'
        return "The bot has rescued a crewmate", bot_cell, crewmates, grid, aliens,time_steps

    bot_cell.crewmate_probability=0

    max_prob_val=max(c.crewmate_probability for r in grid for c in r)
    max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
    max_prob_cell=random.choice(max_prob_index)
    path=find_path(bot_cell.loc,max_prob_cell)

    for row in grid:
      for cell in row:
        cell.visited=False
        cell.parent=None

    grid[next[0]][next[1]].value='B'
    next_cell.value='B'

    print("BOT MOVED")
    draw_ship_df(grid)

    status, crewmates, aliens,time_steps, alien_beep = take_step(grid, aliens, crewmates, bot_cell,time_steps)

    print("ALIEN MOVED")
    draw_ship_df(grid)

    if status == "The bot was caught by the aliens!":
      return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
    for alien_cell in aliens:
      if bot_cell == alien_cell:
        return "The bot was caught by the aliens!", bot_cell, crewmates, grid, aliens,time_steps
    print("PATH GOING TO BE FOLLOWED BY BOT7 IS",path)

  return "The bot has rescued a crewmate", bot_cell, crewmates, grid, aliens,time_steps

# BOT 7 DRIVER CODE
bot_id = 7
size = 30
num_aliens = 2
num_crew = 2

detector_sizes=[ 5, 10. 15 ]
alphas = [ -2*math.log(.1)/(size/2), -2*math.log(.25)/(size/2), -2*math.log(.5)/(size/2) ]

crewmate_pairs_probs = {}
crewmates = []
num_time_steps=[]
sim=0
crew1_res=0
crew_res_both=0
alien_caught=0
num_sims=10
for alpha in alphas:
  for detector_size in detector_sizes:
    sim=0
    crew1_res=0
    crew_res_both=0
    alien_caught=0
    num_time_steps=[]
    for _ in range(num_sims):
      alien_beep = False
      crewmates = []
      crewmate_pairs_probs = []
      alien_pairs_probs = {}
      crewmate_pairs_probs = {}
      time_steps=1
      grid = create_grid(size, size)
      for row in grid:
        for cell in row:
          cell.neighbors = get_neighbors(cell, grid)
      aliens, crewmates, bot_cell = explore_grid(size, grid, num_aliens, num_crew, detector_size)

      grid=initialize_cell_neighbors(grid)
      num_open_cells = len(detect_open_cells(grid))
      for row in grid:
        for cell in row:
          cell.open = detect_if_cell_open(cell)
          cell.neighbors = get_open_neighbors(cell, grid)
      distance_map={}
      pred_map={}
      for i in range(len(grid)):
        for j in range(len(grid[0])):
          s=(i,j)
          dlist,f=dijkstra(grid,s)
          distance_map[s]=dlist
          pred_map[s]=f

      print(distance_map)
      initialize_prob_crewmate(grid, num_crew)
      initialize_crewmate_pairs_probs(crewmate_pairs_probs)
      initialize_alien_pairs_probs(alien_pairs_probs, grid)
      initialize_prob_alien_moved(grid)
      draw_ship_df(grid)
      print("INITIAL CREW PROBS MATRIX")
      draw_crewmate_probabilities_matrix(grid)
      print("initial alien prob matrix")
      draw_alien_probabilities_matrix(grid)
      print("Sum of Alien Probabilities")
      print(sum_probabilities_alien(grid))
      print("Sum of Crewmate Probabilities")
      print(sum_probabilities_crewmate(grid))


      max_prob_val=max(c.crewmate_probability for r in grid for c in r)
      max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
      max_prob_cell=random.choice(max_prob_index)
      path=find_path(bot_cell.loc,max_prob_cell)

      for row in grid:
        for cell in row:
          cell.visited=False
          cell.parent=None


      if len(path)!=0:
        print("Now moving Bot to rescue Crewmate...")

      msg1, bot_cell, crewmates, grid, aliens,time_steps = move_bot7(grid, bot_cell, path, aliens, crewmates, time_steps)
      print(msg1)

      if (msg1=="The bot has rescued a crewmate"):
        time_steps_1=time_steps
        print("NEW STARTING POSITION OF BOT IS",bot_cell)
        max_prob_val=max(c.crewmate_probability for r in grid for c in r)
        max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
        max_prob_cell=random.choice(max_prob_index)
        path=find_path(bot_cell.loc,max_prob_cell)


        for row in grid:
          for cell in row:
            cell.visited=False
            cell.parent=None

        msg2, bot_cell, crewmates, grid, aliens,time_steps = move_bot7(grid, bot_cell, path, aliens, crewmates, time_steps)
        print(msg2)
        print("Final crewmate probs matrix:")
        draw_crewmate_probabilities_matrix(grid)

      print("SIMULATION NUMBER",sim)


      if (msg1=="The bot was caught by the aliens!"):
        alien_caught+=1
      if (msg1=="The bot has rescued a crewmate" and msg2=="The bot was caught by the aliens!"):
        crew1_res+=1
        alien_caught+=1
        num_time_steps.append(time_steps_1)
      if (msg1=="The bot has rescued a crewmate" and msg2=="The bot has rescued a crewmate"):
        crew_res_both+=1
        num_time_steps.append(time_steps)

      file_name_1 = f"bot7_interim_output_sim_{sim}_alpha_{alpha}_k_{detector_size}.txt"

      with open(file_name_1, "a") as file1:
        file1.write("SIMULATION NUMBER "+str(sim)+"\n")
        file1.write("CURRENT STATUS OF NUM TIME STEPS IS "+str(num_time_steps)+"\n")

      print("CURRENT STATUS OF NUM TIME STEPS IS",num_time_steps)
      sim+=1



      print("END OF SIMULATION. RELOAD GRID.")

    print("THE BOT RESCUED",crew1_res+crew_res_both*2,"CREWMATES")
    if crew_res_both != 0:
      print("AVERAGE NUMBER OF MOVES NEEDED TO RESCUE ALL CREWMATES IS",sum(num_time_steps)/crew_res_both)
    else:
      print("AVERAGE NUMBER OF MOVES NEEDED TO RESCUE ALL CREWMATES IS NONE RESCUED")
    file_name = f"bot7_output_alpha_{alpha}_k_{detector_size}.txt"

    with open(file_name, "a") as file:
      file.write("FOR ALPHA VALUE"+str(alpha)+"AND K VALUE "+str(detector_size)+ "FOR BOT7"+"\n")
      file.write("OUT OF "+str(num_sims)+"SIMULATIONS"+"\n")
      file.write("BOT RESCUED CREWMATE "+str(crew1_res+crew_res_both*2)+" TIMES"+"\n")
      file.write("BOT WAS CAUGHT BY ALIENS "+str(alien_caught)+" TIMES"+"\n")
      file.write("AVERAGE MOVES REQUIRED TO RESCUE CREWMATE IS "+str(sum(num_time_steps)/(crew1_res+crew_res_both*2))+"\n")
      file.write("PROB OF RESCUING BOTH CREWMATES IS "+str(crew_res_both/num_sims)+"\n")

"""Bot 8 Code

"""

bot_id=8
size=30
num_aliens = 2
num_crew = 2
alphas = [ -2*math.log(.1)/(size/2), -2*math.log(.25)/(size/2), -2*math.log(.5)/(size/2) ]
detector_sizes=[ 5,7,10 ]
num_time_steps=[]
sim=0
crew1_res=0
crew_res_both=0
alien_caught=0
num_sims=10
for alpha in alphas:
  for detector_size in detector_sizes:
    sim=0
    num_time_steps=[]
    crew1_res=0
    crew_res_both=0
    alien_caught=0
    for _ in range(num_sims):
      beep1=False
      crewmates = []
      crewmate_pairs_probs = {}
      alien_pairs_probs = {}
      time_steps=1
      grid = create_grid(size, size)
      for row in grid:
        for cell in row:
          cell.neighbors = get_neighbors(cell, grid)
      aliens, crewmates, bot_cell = explore_grid(size, grid, num_aliens, num_crew, detector_size)
      print("Crewmate 1 location: ")
      print(crewmates[0].loc)
      print("Crewmate 2 location: ")
      print(crewmates[1].loc)
      print("Aliens location: ")
      print(aliens[0].loc)
      grid=initialize_cell_neighbors(grid)
      num_open_cells = len(detect_open_cells(grid))
      for row in grid:
        for cell in row:
          cell.open = detect_if_cell_open(cell)
          cell.neighbors = get_open_neighbors(cell, grid)
      distance_map={}
      pred_map={}
      for i in range(len(grid)):
        for j in range(len(grid[0])):
          s=(i,j)
          dlist,f=dijkstra(grid,s)
          distance_map[s]=dlist
          pred_map[s]=f
      initialize_prob_crewmate(grid, num_crew)
      initialize_prob_alien_moved(grid)
      print("INITIAL SHIP")
      draw_ship_df(grid)
      print("INITIAL CREW PROBS MATRIX")
      draw_crewmate_probabilities_matrix(grid)
      print("initial alien prob matrix")
      draw_alien_probabilities_matrix(grid)
      print("Sum of Alien Probabilities")
      print(sum_probabilities_alien(grid))
      print("Sum of Crewmate Probabilities")
      print(sum_probabilities_crewmate(grid))
      #------------------------------
      centre_cells=[]
      if len(grid)%2==1:
        centre_cells.append([len(grid)//2,len(grid)//2])                             # Finding Center Cell
      else:
        centre_cells.append([len(grid)//2,len(grid)//2])
        centre_cells.append([len(grid)//2-1,len(grid)//2])
        centre_cells.append([len(grid)//2,len(grid)//2-1])
        centre_cells.append([len(grid)//2-1,len(grid)//2-1])
      if len(centre_cells)==1 and grid[centre_cells[0][0]][centre_cells[0][1]].open==False:     #Corner case closed center cell
        centre_cells_mod=grid[centre_cells[0][0]][centre_cells[0][1]].neighbors
        centre_cells=[ce.loc for ce in centre_cells_mod]
      print("BOT CELL ", bot_cell.loc)
      print("CENTER CELL ", centre_cells[0])
      if bot_cell.loc == centre_cells[0]:                                                 # Corner case where bot spawns on center cell
        print("centre cells: ",centre_cells)
        print("bot_cell: ",bot_cell.loc)
        print("bot_cell spawned at center cell")
        max_prob_val=max(c.crewmate_probability for r in grid for c in r)             # Find max crew prob
        max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]   #Find cell of max prob
        print("closest_max_prob_cell: ", max_prob_index)
        draw_crewmate_probabilities_matrix(grid)
        path = find_path_bot2(bot_cell.loc, max_prob_index)                  # Find closest cell with max prob
        print("bot_cell: ",bot_cell.loc)
        print(path)
        msg, bot_cell, crewmates, grid, aliens,time_steps = move_bot2(grid, bot_cell, path, aliens, crewmates, time_steps, beep1)  #Moving on path

      path = find_path_bot2(bot_cell.loc, centre_cells)                                 # Path to center cell
      print("centre cells: ",centre_cells)
      print("bot_cell: ",bot_cell.loc)
      print(path)


      msg, bot_cell, crewmates, grid, aliens,time_steps = move_bot2(grid, bot_cell, path, aliens, crewmates, time_steps, beep1)  #Moving on path
      if msg=="reached centre":
        print("REACHED THE CENTRE")
      elif msg=="entered a high prob region":
        print("REACHED AN AREA WITH PROB >0.25")

      print(msg)
      sim+=1
      print("Simmulation Number: ", sim)
      if (msg=="The bot was caught by the aliens!"):
        alien_caught+=1
      elif (msg=="The bot has rescued a crewmate"):
        print("NEW STARTING POSITION OF BOT IS",bot_cell)
        time_steps_1=time_steps
        max_prob_val=max(c.crewmate_probability for r in grid for c in r)
        max_prob_index=[(i,j) for i,row in enumerate(grid) for j,val in enumerate(row) if val.crewmate_probability==max_prob_val]
        max_prob_cell=random.choice(max_prob_index)
        path=find_path(bot_cell.loc,max_prob_cell)


        for row in grid:
          for cell in row:
            cell.visited=False
            cell.parent=None

        msg2, bot_cell, crewmates, grid, aliens,time_steps = move_bot2(grid, bot_cell, path, aliens, crewmates, time_steps, beep1)
        print(msg2)
        print("Final crewmate probs matrix:")
        draw_crewmate_probabilities_matrix(grid)

      if (msg=="The bot was caught by the aliens!"):
        alien_caught+=1
      if (msg=="The bot has rescued a crewmate" and msg2=="The bot was caught by the aliens!"):
        crew1_res+=1
        alien_caught+=1
        num_time_steps.append(time_steps_1)
      if (msg=="The bot has rescued a crewmate" and msg2=="The bot has rescued a crewmate"):
        crew_res_both+=1
        num_time_steps.append(time_steps)

      file_name_1 = f"bot8_interim_output_sim_{sim}_alpha_{alpha}_k_{detector_size}.txt"

      with open(file_name_1, "a") as file1:
        file1.write("SIMULATION NUMBER "+str(sim)+"\n")
        file1.write("CURRENT STATUS OF NUM TIME STEPS IS "+str(num_time_steps)+"\n")

      print("CURRENT STATUS OF NUM TIME STEPS IS",num_time_steps)

      sim+=1
      print("END OF SIMULATION. RELOAD GRID.")

print("THE BOT RESCUED",crew1_res+crew_res_both*2,"CREWMATES")
if crew_res_both != 0:
  print("AVERAGE NUMBER OF MOVES NEEDED TO RESCUE ALL CREWMATES IS",sum(num_time_steps)/crew_res_both)
else:
  print("AVERAGE NUMBER OF MOVES NEEDED TO RESCUE ALL CREWMATES IS NONE RESCUED")

file_name = f"bot8_output_alpha_{alpha}_k_{detector_size}.txt"

with open(file_name, "a") as file:
  file.write("CREW RES 1"+str(crew1_res))
  file.write("CREW RES BOTH"+str(crew_res_both))
  file.write("FOR ALPHA VALUE"+str(alpha)+"AND K VALUE "+str(detector_size)+ "FOR BOT8"+"\n")
  file.write("OUT OF "+str(num_sims)+"SIMULATIONS"+"\n")
  file.write("BOT RESCUED CREWMATE "+str(crew1_res+crew_res_both*2)+" TIMES"+"\n")
  file.write("BOT WAS CAUGHT BY ALIENS "+str(alien_caught)+" TIMES"+"\n")
  file.write("AVERAGE MOVES REQUIRED TO RESCUE CREWMATE IS "+str(sum(num_time_steps)/(crew1_res+crew_res_both*2))+"\n")
  file.write("PROB OF RESCUING BOTH CREWMATES IS "+str(crew_res_both/num_sims)+"\n")
