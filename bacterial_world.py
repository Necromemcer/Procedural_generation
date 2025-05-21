import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import re
import string
import math
from collections import defaultdict

class Grid:
    def __init__(self, width, height, amount_food, radius_food, amount_radiation, radius_radiation):
        ### 0 - empty, 1 - cell
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height), dtype=int)
        ### 0 - empty, float - food concentration
        self.food_grid = np.zeros((width, height), dtype=float)
        for seed in range(amount_food):
            seed_coord = (random.randrange(self.width), random.randrange(self.height))
            for i in range(max(0, seed_coord[0] - radius_food), min(width, seed_coord[0] + radius_food + 1)):
                for j in range(max(0, seed_coord[1] - radius_food), min(height, seed_coord[1] + radius_food + 1)):
                    distance = np.sqrt((i - seed_coord[0])**2 + (j - seed_coord[1])**2)
                    if distance <= radius_food:
                        self.food_grid[i, j] += 1 - (distance / radius_food)
        ### 0 - empty, float - radiation concentration
        self.radiation_grid = np.zeros((width, height), dtype=float)
        for seed in range(amount_radiation):
            seed_coord = (random.randrange(self.width), random.randrange(self.height))
            for i in range(max(0, seed_coord[0] - radius_radiation), min(width, seed_coord[0] + radius_radiation + 1)):
                for j in range(max(0, seed_coord[1] - radius_radiation), min(height, seed_coord[1] + radius_radiation + 1)):
                    distance = np.sqrt((i - seed_coord[0])**2 + (j - seed_coord[1])**2)
                    if distance <= radius_radiation:
                        self.radiation_grid[i, j] += 1 - (distance / radius_radiation)**2

    def food_gradient(self, coords):
        return self.food_grid[coords]
    
    def radiation_gradient(self, coords):
        return self.radiation_grid[coords]

    def is_position_occupied(self, coords):
        return self.grid[coords] == 1

    def count_live_neighbors(self, coords):
        total = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                x_edge = coords[0] + i
                y_edge = coords[1] + j
                if ((0 <= x_edge <= self.width - 1) and (0 <= y_edge <= self.height - 1)):
                    total += self.grid[x_edge, y_edge]
                else:
                    total += 1
        return total

    def cell_density(self, coords):
        """
        Compute the density of cells in a 3x3 area around the given position (x, y).
        """
        density = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                x_edge = coords[0] + i
                y_edge = coords[1] + j
                if ((0 <= x_edge <= self.width - 1) and (0 <= y_edge <= self.height - 1)):
                    density += self.grid[x_edge, y_edge]
                else: 
                    density += 1
        return density

    def choose_division_direction(self, coords, overpopulation_constant):
        """
        Choose a direction for cell division based on the density of the surrounding population in a 5x5 area.
        """
        min_density = float('inf')
        best_direction = None

        # Check all 8 neighboring cells
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                x_edge = coords[0] + i
                y_edge = coords[1] + j
                if((0 <= x_edge <= self.width - 1) and (0 <= y_edge <= self.height - 1)):
                    if (not self.is_position_occupied((x_edge, y_edge))) and (self.count_live_neighbors((x_edge, y_edge)) <= overpopulation_constant):
                        current_density = self.cell_density((x_edge, y_edge))
                        if current_density < min_density:
                            min_density = current_density
                            best_direction = (x_edge, y_edge)
        return best_direction
    
    def choose_movement_direction(self, coords, speed):
        """
        Compute the best direction of movement based on food and radiation gradients.
        """
        best_score = self.food_gradient(coords) #- self.radiation_gradient(coords)
        best_direction = None

        # Iterate over the speed + 2 radius around the cell
        for i in range(-(2 + speed), (3 + speed)):
            for j in range(-(2 + speed), (3 + speed)):
                if (i == 0 and j == 0) or (math.sqrt(i**2 + j**2) > speed):
                    continue  # Skip the cell itself and constrain the radius
                # Calculate the new coordinates, considering grid boundaries
                new_x = coords[0] + i
                new_y = coords[1] + j
                if ((0 <= new_x <= self.width - 1) and (0 <= new_y <= self.height - 1)):
                    if (not self.is_position_occupied((new_x, new_y))):
                        # Assign a score based on food and radiation
                        score = self.food_gradient((new_x, new_y)) #- self.radiation_gradient((new_x, new_y))
                        # Update the best direction if the current score is higher
                        if score > best_score:
                            best_score = score
                            best_direction = (new_x, new_y)
        return best_direction
    
    def display(self):
        plt.imshow(self.grid, cmap='binary')
        plt.show()



class Bacteria:
    def __init__(self, initial_population, initial_genomes, initial_phenotypes, initial_positions, initial_energy, genotype_to_phenotype):
        self.population = initial_population
        self.genomes = initial_genomes
        self.phenotypes = initial_phenotypes 
        self.positions = initial_positions
        self.energy = initial_energy
        self.genotype_to_phenotype = genotype_to_phenotype
    
    def current_population(self):
        return self.population

    def current_genome(self, ID):
        return self.genomes[ID]

    def current_phenotype(self, ID):
        return self.phenotypes[ID]

    def current_position(self, ID):
        return self.positions[ID]
    
    def current_energy(self, ID):
        return self.energy[ID]
    
    def add_individuum(self, ID, genome, position, energy):
        self.population.add(ID)
        self.genomes[ID] = genome
        self.phenotypes[ID] = self.genotype_to_phenotype_conversion(genome)
        self.positions[ID] = position
        self.energy[ID] = energy

    def remove_individuum(self, ID):
        self.population.remove(ID)
        self.genomes.pop(ID)
        self.phenotypes.pop(ID)
        self.positions.pop(ID)
        self.energy.pop(ID)

    def genotype_to_phenotype_conversion(self, genome):
        return [trait for trait in list(self.genotype_to_phenotype.keys()) for occurence in re.findall(f'(?=({self.genotype_to_phenotype[trait]}))', genome)]

    def mutate(self, ID, gene_pool, mutation_rates, radiation):
        genome = list(self.genomes[ID])
        for i in range(len(self.genomes[ID])):
            mutation_chance = mutation_rates['passive'] + mutation_rates['radiation'] * radiation
            if random.random() <= mutation_chance:
                genome[i] = random.choice(gene_pool)
        self.genomes[ID] = ''.join(genome)
        if not ID.count('_spore'):
            self.phenotypes[ID] = self.genotype_to_phenotype_conversion(self.genomes[ID])
        
    def metabolize(self, ID, energy_rates, food):
        if ID.count('_spore'):
            self.energy[ID] += energy_rates['life_spore']
        else:
            self.energy[ID] += energy_rates['life_cell'] + energy_rates['food'] * food
    
    def photosynthesize(self, ID, energy_rates):
        self.energy[ID] += energy_rates['photosynthesis'] * self.phenotypes[ID].count('photosynthesis')

    def form_spore(self, ID):
        spore_number = self.phenotypes[ID].count('spore_formation')
        spore_names = []
        for i in range(spore_number):
            spore_ID = ID + f'_spore.{i+1}'
            spore_names.append(spore_ID)
            self.add_individuum(spore_ID, self.genomes[ID], self.positions[ID], (self.energy[ID]/spore_number))
            self.phenotypes[spore_ID] = ['spore_movement', 'germination']
        position = self.positions[ID]
        self.remove_individuum(ID)
        return spore_names, position

    def germinate(self, ID):
        new_ID = '_'.join(('_'.join(ID.split('_')[:-1]), ID.split('.')[-1]))
        self.add_individuum(new_ID, self.genomes[ID], self.positions[ID], self.energy[ID])
        position = self.positions[ID]
        self.remove_individuum(ID)
        return new_ID, position

class Simulation:
    def __init__(self, width, height, amount_food, radius_food, amount_radiation, radius_radiation, gene_pool, mutation_rates, energy_rates, movement_rates, genotype_to_phenotype, phenotype_to_color, overpopulation_constant, seed):
        ### Simulation parameters
        self.grid = Grid(width, height, amount_food, radius_food, amount_radiation, radius_radiation)
        self.mutation_rates = mutation_rates
        self.energy_rates = energy_rates
        self.movement_rates = movement_rates
        self.gene_pool = gene_pool
        self.genotype_to_phenotype = genotype_to_phenotype
        self.phenotype_to_color = phenotype_to_color
        self.overpopulation_constant = overpopulation_constant
        self.log = defaultdict(list)
        self.seed_population = defaultdict(list)
        ### Generate initial population
        initial_population = set()
        initial_genomes = dict()
        initial_phenotypes = dict()
        initial_positions = dict()
        initial_energy = dict()
        for genome in range(len(seed.values())):
            for number in range(list(seed.values())[genome]):
                ID = ''.join(random.choice(string.ascii_letters) for _ in range(int(round(math.log(sum(list(seed.values())), 10) + 1, 0))))
                initial_population.add(ID)
                initial_genomes[ID] = list(seed.keys())[genome]
                self.seed_population[f'Progenitor population {list(seed.keys())[genome]}'].append(ID)
                initial_phenotypes[ID] = [trait for trait in list(self.genotype_to_phenotype.keys()) for occurence in re.findall(f'(?=({self.genotype_to_phenotype[trait]}))', list(seed.keys())[genome])]
                initial_positions[ID] = (random.randrange(self.grid.width), random.randrange(self.grid.height))
                self.grid.grid[initial_positions[ID]] = 1
                initial_energy[ID] = self.energy_rates['initial']
        self.bacteria = Bacteria(initial_population, initial_genomes, initial_phenotypes, initial_positions, initial_energy, genotype_to_phenotype)

    def genotype_to_phenotype_conversion(self, genome):
        return [trait for trait in list(self.genotype_to_phenotype.keys()) for occurence in re.findall(f'(?=({self.genotype_to_phenotype[trait]}))', genome)]

    def get_bacterium_color(self, ID):
        if ID.count('_spore'):
            return (0.5, 0.25, 0)
        colors = [self.phenotype_to_color[trait] for trait in self.bacteria.phenotypes[ID] if trait in self.phenotype_to_color]
        if not colors:
            return (0, 0, 0)
        # Mix colors by averaging
        mixed_color = np.mean(colors, axis=0)
        return mixed_color

    def division_checkpoint(self, ID):
        scenario_1 = (self.bacteria.energy[ID] + self.energy_rates['division'] >= 2) and self.grid.choose_division_direction(self.bacteria.positions[ID], self.overpopulation_constant)
        scenario_2 = ((self.bacteria.energy[ID] + self.energy_rates['division'] + self.energy_rates['life_cell'] + self.energy_rates['food'] * self.grid.food_gradient(self.bacteria.positions[ID])) >= 1) and self.grid.choose_division_direction(self.bacteria.positions[ID], self.overpopulation_constant)
        return (scenario_1 or scenario_2) and (not ID.count('_spore'))

    def division(self, ID):
        self.bacteria.energy[ID] += self.energy_rates['division']
        ### Generate child ID 
        matches = re.findall(f"{re.escape(ID)}\\.(\\d+)(?=\\.|;|$)", ";".join([ID for ID in self.bacteria.population]))
        # Convert the matched numbers to integers and find the largest number
        if matches:
            largest_number = max(map(int, matches))
        else:
            largest_number = 0
        new_ID = f"{ID}.{largest_number + 1}"
        new_genome = list(self.bacteria.genomes[ID])
        for i in range(len(new_genome)):
            mutation_chance = self.mutation_rates['division'] + self.mutation_rates['radiation'] * self.grid.radiation_gradient(self.bacteria.positions[ID])
            if random.random() <= mutation_chance:
                new_genome[i] = random.choice(self.gene_pool)
        new_genome = ''.join(new_genome)
        new_position = self.grid.choose_division_direction(self.bacteria.positions[ID], self.overpopulation_constant)
        ### Let photosynthetics be at a disadvantage
        if self.genotype_to_phenotype_conversion(new_genome).count('photosynthesis'):
            new_energy = -self.energy_rates['division'] * 0.3
        else:    
            new_energy = -self.energy_rates['division'] * 0.5
        self.bacteria.add_individuum(new_ID, new_genome, new_position, new_energy)
        return new_ID, new_position

    def towards_better_life(self, ID, best_direction):
        """
        Move the cell in the best direction, considering the available speed.
        """

        best_x, best_y = best_direction
        x, y = self.bacteria.positions[ID]

        self.grid.grid[(x, y)] = 0

        dx = best_x - x
        dy = best_y - y

        # Calculate the distance to the best direction
        distance = math.sqrt(dx**2 + dy**2)        
        speed = self.bacteria.phenotypes[ID].count('motility') * self.movement_rates['cell']

        # If the distance is greater than the speed, move partially in that direction
        if distance > speed:
            dx = int(round(dx * speed / distance))
            dy = int(round(dy * speed / distance))

            # Calculate the new position
            new_x = x + dx
            new_y = y + dy

            # Ensure the new position is within grid boundaries
            new_x = max(0, min(self.grid.width - 1, new_x))
            new_y = max(0, min(self.grid.height - 1, new_y))

            if self.grid.is_position_occupied((new_x, new_y)):
                min_distance = float('inf')
                min_distance_point = None
                for i in range(0, dx + 1):
                    for j in range(0, dy + 1):
                        # Calculate the new coordinates, considering grid boundaries
                        alt_x = max(0, min(self.grid.width - 1, x + i))
                        alt_y = max(0, min(self.grid.height - 1, y + j))
                        # Check if the position is unoccupied
                        if not self.grid.is_position_occupied((alt_x, alt_y)):
                            distance = math.sqrt((alt_x - new_x)**2 + (alt_y - new_y)**2)
                            if distance < min_distance:
                                min_distance_point = (alt_x, alt_y)
                                min_distance = distance
                self.grid.grid[min_distance_point] = 1
                self.bacteria.positions[ID] = min_distance_point
                self.bacteria.energy[ID] += self.energy_rates['movement'] * math.sqrt((min_distance_point[0] - x)**2 + (min_distance_point[1] - y)**2)
                return min_distance_point
            else:
                self.grid.grid[(new_x, new_y)] = 1
                self.bacteria.positions[ID] = (new_x, new_y)
                self.bacteria.energy[ID] += self.energy_rates['movement'] * math.sqrt((new_x - x)**2 + (new_y - y)**2)
                return (new_x, new_y)
        else:
            self.grid.grid[(best_x, best_y)] = 1
            self.bacteria.positions[ID] = (best_x, best_y)
            self.bacteria.energy[ID] += self.energy_rates['movement'] * math.sqrt((best_x - x)**2 + (best_y - y)**2)
            return (best_x, best_y)

    def random_cell_movement(self, ID):
        '''
        Move in random direction according to the available speed
        '''
        speed = self.bacteria.phenotypes[ID].count('motility') * self.movement_rates['cell']

        x, y = self.bacteria.positions[ID]
        self.grid.grid[(x, y)] = 0

        max_distance = 0
        max_distance_points = []
        for i in range(-speed, speed):
            for j in range(-speed, speed):
                if (i == 0 and j == 0) or (math.sqrt(i**2 + j**2) > speed):
                    continue  # Skip the current position and constrain the radius
                # Calculate the new coordinates, considering grid boundaries
                alt_x = max(0, min(self.grid.width - 1, x + i))
                alt_y = max(0, min(self.grid.height - 1, y + j))
                # Check if the position is unoccupied
                if not self.grid.is_position_occupied((alt_x, alt_y)):
                    distance = math.sqrt((alt_x - x)**2 + (alt_y - y)**2)
                    if distance == max_distance:
                        max_distance_points.append((alt_x, alt_y))
                    if distance > max_distance:
                        max_distance_points.clear()
                        max_distance_points.append((alt_x, alt_y))
                        max_distance = distance
        if max_distance_points:
            choice = random.choice(max_distance_points)
            self.grid.grid[choice] = 1
            self.bacteria.positions[ID] = choice
            self.bacteria.energy[ID] += self.energy_rates['movement'] * math.sqrt((choice[0] - x)**2 + (choice[1] - y)**2) 
        else:
            self.grid.grid[(x, y)] = 1

    def random_spore_movement(self, ID):
        '''
        Move in random direction according to the available speed
        '''
        speed = self.movement_rates['spore']

        x, y = self.bacteria.positions[ID]

        # Calculate the new position
        new_x = x + random.choice([-speed, speed])
        new_y = y + random.choice([-speed, speed])

        # Ensure the new position is within grid boundaries
        new_x = max(0, min(self.grid.width - 1, new_x))
        new_y = max(0, min(self.grid.height - 1, new_y))

        self.bacteria.positions[ID] = (new_x, new_y)

    def death(self, ID):
        if not ID.count('_spore'):
            self.grid.grid[self.bacteria.positions[ID]] = 0
        self.bacteria.remove_individuum(ID)

    def check_positions(self):
        return bool(len([bacterium for bacterium in self.bacteria.population if not bacterium.count('_spore')]) == sum(sum(self.grid.grid)))

    def check_lethal_mutation(self, ID):
        if not ID.count('_spore'):
            if self.bacteria.phenotypes[ID].count('lethal'):
                self.death(ID)
                return 'dead'

    def turn(self):
        population_slice = self.bacteria.population.copy()
        queue = list(population_slice)
        #to_be_added = set()
        to_be_removed_metabolism = set()
        to_be_removed_mutation = set()
        to_be_removed_turn = set()
        random.shuffle(queue)
        for ID in queue:
            ### Metabolism occurs               
            self.bacteria.metabolize(ID, self.energy_rates, self.grid.food_gradient(self.bacteria.positions[ID]))
            ### Death due starvation (energy depletion)
            if self.bacteria.energy[ID] <= 0:
                self.death(ID)
                to_be_removed_metabolism.add(ID)
                self.log[ID].append(f"{ID} died due to starvation in Energy Phase")
            # if not self.check_positions():
            #     self.log.append('Checkpoint_Metabolism')
        if to_be_removed_metabolism:
            for ID in to_be_removed_metabolism:
                queue.remove(ID)
        random.shuffle(queue)
        for ID in queue:
            ### Mutations occur
            self.bacteria.mutate(ID, self.gene_pool, self.mutation_rates, self.grid.radiation_gradient(self.bacteria.positions[ID]))
            if self.check_lethal_mutation(ID):
                to_be_removed_mutation.add(ID)
                self.log[ID].append(f"{ID} died due to acquiring a lethal mutation in Mutation Phase")
        if to_be_removed_mutation:
            for ID in to_be_removed_mutation:
                queue.remove(ID)
        random.shuffle(queue)
        for ID in queue:
            ### If bacteria passes the division conditions - it divides. Otherwise skip to action
            if self.division_checkpoint(ID):
                newborn_ID, newborn_position = self.division(ID)
                if not self.check_lethal_mutation(newborn_ID):
                    self.grid.grid[newborn_position] = 1
                    self.log[ID].append(f"{ID} gave birth to {newborn_ID} at position {newborn_position}")
                    self.log[newborn_ID].append(f"{newborn_ID} was born at position {newborn_position}")
                # if not self.check_positions():
                #     self.log.append('Checkpoint_Division')
                else:
                    self.log[newborn_ID].append(f"{newborn_ID} died due to acquiring a lethal mutation during birth")
            ### Actions are taken if no division occured
            else:
                ### If it is a spore: germinate if conditions are grateful, move otherwise
                if ID.count('_spore'):
                    if (self.grid.food_gradient(self.bacteria.positions[ID]) * self.energy_rates['food'] >= -self.energy_rates['life_cell']) and (not self.grid.is_position_occupied(self.bacteria.positions[ID])) and (self.grid.count_live_neighbors(self.bacteria.positions[ID]) <= self.overpopulation_constant):
                        germinated_ID, germinated_position = self.bacteria.germinate(ID)
                        to_be_removed_turn.add(ID)
                        if not self.check_lethal_mutation(germinated_ID):
                            self.grid.grid[germinated_position] = 1
                            self.log[ID].append(f"{ID} germinated into {germinated_ID} at {germinated_position}")
                        # if not self.check_positions():
                        #     self.log.append('Checkpoint_Germination')
                        else:
                            self.log[germinated_ID].append(f"{germinated_ID} died due to experiencing a lethal mutation upon germination")
                    else:
                        self.random_spore_movement(ID)
                        # if not self.check_positions():
                        #     self.log.append('Checkpoint_Spore_Movement')
                ### If it is a cell: try to photosynthesize | move if you can | try to form a spore if things are bad | stay in place
                else:
                    options = set(self.bacteria.phenotypes[ID])
                    if options:
                        if ('photosynthesis' in options) and ('motility' in options):
                            if self.grid.choose_movement_direction(self.bacteria.positions[ID], self.bacteria.phenotypes[ID].count('motility')) and (self.bacteria.energy[ID] > self.bacteria.phenotypes[ID].count('motility') * self.energy_rates['movement']):
                                chosen_direction = self.grid.choose_movement_direction(self.bacteria.positions[ID], self.bacteria.phenotypes[ID].count('motility'))
                                actual_direction = self.towards_better_life(ID, self.grid.choose_movement_direction(self.bacteria.positions[ID], self.bacteria.phenotypes[ID].count('motility')))
                                self.log[ID].append(f"{ID} aimed for {chosen_direction} and moved to {actual_direction}")
                                # if not self.check_positions():
                                #     self.log.append('Checkpoint_P&M_Cell_Movement')
                            else:
                                self.bacteria.photosynthesize(ID, self.energy_rates)
                        elif 'photosynthesis' in options:    
                            self.bacteria.photosynthesize(ID, self.energy_rates)
                        elif (self.bacteria.energy[ID] > self.bacteria.phenotypes[ID].count('motility') * self.energy_rates['movement']) and ('motility' in options):
                            if self.grid.choose_movement_direction(self.bacteria.positions[ID], self.bacteria.phenotypes[ID].count('motility')):
                                chosen_direction = self.grid.choose_movement_direction(self.bacteria.positions[ID], self.bacteria.phenotypes[ID].count('motility'))
                                actual_direction = self.towards_better_life(ID, self.grid.choose_movement_direction(self.bacteria.positions[ID], self.bacteria.phenotypes[ID].count('motility')))
                                self.log[ID].append(f"{ID} aimed for {chosen_direction} and moved to {actual_direction}")
                                # if not self.check_positions():
                                #     self.log.append('Checkpoint_M_Cell_Movement')
                            else:
                                self.random_cell_movement(ID)
                                self.log[ID].append(f'{ID} moved in random direction')
                                # if not self.check_positions():
                                #     self.log.append('Checkpoint_M_Random_Cell_Movement')
                        elif (self.grid.food_gradient(self.bacteria.positions[ID]) == 0) and (self.bacteria.energy[ID]) < 1 and ('spore_formation' in options):
                            self.grid.grid[self.bacteria.positions[ID]] = 0
                            spore_names, position = self.bacteria.form_spore(ID)
                            to_be_removed_turn.add(ID)
                            self.log[ID].append(f"{ID} formed spores {spore_names} at {position}")
                            # if not self.check_positions():
                            #     self.log.append('Checkpoint_Spore_Formation')
        if to_be_removed_turn:
            for ID in to_be_removed_turn:
                try:
                    queue.remove(ID)
                except:
                    print(ID)
        # if not self.check_positions():
        #         self.log.append('Checkpoint_Before_Death')
        for ID in queue:
            ### Death due starvation (energy depletion)
            if self.bacteria.energy[ID] <= 0:
                self.death(ID)
                self.log[ID].append(f"{ID} died due to starvation in Death Phase")
                # if not self.check_positions():
                #     self.log.append('Checkpoint_Death_1')
            ### Death due overpopulation
            elif not ID.count('_spore'):
                live_neighbors = self.grid.count_live_neighbors(self.bacteria.positions[ID])
                if live_neighbors > self.overpopulation_constant:
                    self.death(ID)
                    self.log[ID].append(f"{ID} died due overpopulation in Death Phase")
                    # if not self.check_positions():
                    #     self.log.append('Checkpoint_Death_2')

    def count_children(self, ID):
        return len(re.findall(f"{re.escape(ID)}", ";".join([ID for ID in self.bacteria.population])))

    def position_to_name(self, coords):
        return list(self.bacteria.positions.keys())[list(self.bacteria.positions.values()).index(coords)]

    def run(self, steps):
        for _ in range(steps):
            self.turn()

    def animate(self, steps, offspring_stats, save_to_gif):
        fig, ax = plt.subplots()

        # Initialize the grid for display
        display_grid = np.zeros((self.grid.width, self.grid.height, 3))

        # Initialize imshow plots for the grid, food, and radiation gradients
        img = ax.imshow(display_grid)
        food_gradient = ax.imshow(self.grid.food_grid, cmap='Reds', alpha=0.3)
        radiation_gradient = ax.imshow(self.grid.radiation_grid, cmap='Purples', alpha=0.3)

        # Initialize text for statistics
        stats_text = ax.text(1.02, 0.95, '', transform=ax.transAxes, verticalalignment='top', fontsize=12)

        def update_frame(frame):
            self.turn()

            # Reset the display grid
            display_grid.fill(1)

            # Update the display grid with bacteria colors
            for ID in self.bacteria.population:
                x, y = self.bacteria.positions[ID]
                color = self.get_bacterium_color(ID)
                display_grid[x, y] = color

            # Update the grid display
            img.set_array(display_grid)

            # Update food and radiation gradients
            food_gradient.set_array(self.grid.food_grid)
            radiation_gradient.set_array(self.grid.radiation_grid)

            if offspring_stats:
                # Update the statistics text
                population_stats = {}
                progenitor_stats = {}
                for population in range(len(list(self.seed_population.values()))):
                    population_stats[list(self.seed_population.keys())[population]] = 0
                    for progenitor in list(self.seed_population.values())[population]:
                        population_stats[list(self.seed_population.keys())[population]] += self.count_children(progenitor)
                        progenitor_stats[progenitor] = self.count_children(progenitor)

                stats_text_content = f"{'\n'.join([f'{key}: {value}' for key, value in progenitor_stats.items()])}\n\n\n{'\n'.join([f'{key}: {value}' for key, value in population_stats.items()])}"
                stats_text.set_text(stats_text_content)
                return img, food_gradient, radiation_gradient, stats_text

            return img, food_gradient, radiation_gradient

        ani = animation.FuncAnimation(fig, update_frame, frames=steps, interval=300, blit=False)

        if not save_to_gif:
            plt.show()
        else:
            ani.save(f"BW_animation_{steps}_steps.gif")
    
    def show(self, offspring_stats):
        fig, ax = plt.subplots()

        # Initialize the grid for display
        display_grid = np.zeros((self.grid.width, self.grid.height, 3))

        # Initialize imshow plots for the grid, food, and radiation gradients
        img = ax.imshow(display_grid)
        food_gradient = ax.imshow(self.grid.food_grid, cmap='Reds', alpha = 0.3)
        radiation_gradient = ax.imshow(self.grid.radiation_grid, cmap='Purples', alpha = 0.3)

        # Initialize text for statistics
        stats_text = ax.text(1.02, 0.95, '', transform=ax.transAxes, verticalalignment='top', fontsize=12)

        # Reset the display grid
        display_grid.fill(1) # empty color = white

        # Update the display grid with bacteria colors
        for ID in self.bacteria.population:
            x, y = self.bacteria.positions[ID]
            color = self.get_bacterium_color(ID)
            display_grid[x, y] = color

        # Update the grid display
        img.set_array(display_grid)

        # Update food and radiation gradients
        food_gradient.set_array(self.grid.food_grid)
        radiation_gradient.set_array(self.grid.radiation_grid)

        if offspring_stats:
                # Update the statistics text
                population_stats = {}
                progenitor_stats = {}
                for population in range(len(list(self.seed_population.values()))):
                    population_stats[list(self.seed_population.keys())[population]] = 0
                    for progenitor in list(self.seed_population.values())[population]:
                        population_stats[list(self.seed_population.keys())[population]] += self.count_children(progenitor)
                        progenitor_stats[progenitor] = self.count_children(progenitor)

                stats_text_content = f"{'\n'.join([f'{key}: {value}' for key, value in progenitor_stats.items()])}\n\n\n{'\n'.join([f'{key}: {value}' for key, value in population_stats.items()])}"
                stats_text.set_text(stats_text_content)

        plt.show()


seed = {'0011100333': 10, '0022220033': 10}
gene_pool = ['0', '1', '2', '3']
mutation_rates = {'passive': 0.0005, 'division': 0.01, 'radiation': 0.07}
energy_rates = {'initial': 7, 'life_cell': -0.1, 'life_spore': -0.01, 'food': 3, 'division': -5, 'photosynthesis': 0.2, 'movement': -0.05}
movement_rates = {'cell': 2, 'spore': 5}
genotype_to_phenotype = {'photosynthesis': '11', 'motility': '22', 'spore_formation': '33', 'lethal': '000'}
phenotype_to_color = {
    'photosynthesis': (0, 1, 0),  # Green
    'motility': (0, 0, 1), # Blue
}
overpopulation_constant = 5

simulation = Simulation(150, 150, 5, 30, 5, 30, gene_pool, mutation_rates, energy_rates, movement_rates, genotype_to_phenotype, phenotype_to_color, overpopulation_constant, seed)
simulation.animate(300, offspring_stats=True, save_to_gif=False)
