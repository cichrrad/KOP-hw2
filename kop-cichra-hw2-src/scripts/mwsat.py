import random, math , csv

class WeightedSATParser:
    def __init__(self, file_path):
        """
        Initialize the WeightedSATParser with the given file path.
        """
        self.file_path = file_path
        self.num_variables = 0
        self.num_clauses = 0
        self.weights = []
        self.clauses = []
        self.weight_ceil = 0  # Maximum possible weight (sum of all weights)

    def parse_file(self):
        """
        Parse the MWCNF file to extract the number of variables, clauses, weights, and clauses.
        """
        with open(self.file_path, 'r') as file:
            for line in file:
                if line.startswith('c'):
                    continue
                if line.startswith('p mwcnf'):
                    _, _, num_vars, num_clauses = line.split()
                    self.num_variables = int(num_vars)
                    self.num_clauses = int(num_clauses)
                elif line.startswith('w'):
                    self.weights = list(map(int, line.split()[1:-1]))  # Parse weights and ignore trailing 0
                    self.weight_ceil = sum(self.weights)  # Compute the weight ceiling
                else:
                    clause = list(map(int, line.split()[:-1]))  # Parse clause and ignore trailing 0
                    self.clauses.append(clause)

    def is_solution(self, solution):
        """
        Evaluate a solution by checking if all clauses are satisfied and calculating weight.
        :param solution: A list of booleans where True means a variable is non-negated, and False means it is negated.
        :return: A tuple (all_satisfied, satisfied_clauses, sum_weight):
                 - all_satisfied: True if all clauses are satisfied, False otherwise.
                 - satisfied_clauses: Number of satisfied clauses.
                 - sum_weight: Sum of weights for non-negated variables.
        """
        satisfied_clauses = 0
        sum_weight = sum(self.weights[i] for i, val in enumerate(solution) if val)

        for clause in self.clauses:
            satisfied = False
            for var in clause:
                var_index = abs(var) - 1  # Variables are 1-indexed in the file
                if (var > 0 and solution[var_index]) or (var < 0 and not solution[var_index]):
                    satisfied = True
                    break
            if satisfied:
                satisfied_clauses += 1

        all_satisfied = (satisfied_clauses == len(self.clauses))  # Ensure all clauses are satisfied
        return all_satisfied, satisfied_clauses, sum_weight


#class SATSolverBasic:
#    def __init__(self, problem):
#        """
#        Initialize the SAT solver with a given SAT problem.
#        """
#        self.problem = problem
#        self.best_params = None  # Track the best parameter set
#        self.best_weight = 0  # Track the weight of the best solution
#        self.temperature = None  # Current temperature for simulated annealing
#
#    def solve(self, max_flips=1000, p1=0.2, p2=0.2):
#        """
#        Solve the SAT problem using the given parameters.
#        :param max_flips: Maximum number of variable flips to explore.
#        :param p1: Probability of keeping a flip that worsens the weight.
#        :param p2: Probability of keeping a flip that decreases the number of satisfied clauses.
#        :return: The first valid solution found and its weight, or (None, 0) if no valid solution is found.
#        """
#        current_solution = [random.choice([True, False]) for _ in range(self.problem.num_variables)]
#        all_satisfied, satisfied_clauses, current_weight = self.problem.is_solution(current_solution)
#
#        # If the initial solution is valid, return it immediately
#        if all_satisfied:
#            return current_solution, current_weight
#
#        # Initialize the best solution found (must be valid)
#        best_solution = None
#        best_weight = 0
#
#        for _ in range(max_flips):
#            # Randomly pick a variable to flip
#            flip_index = random.randint(0, self.problem.num_variables - 1)
#            current_solution[flip_index] = not current_solution[flip_index]
#
#            # Evaluate the new solution
#            all_satisfied, new_satisfied_clauses, new_weight = self.problem.is_solution(current_solution)
#
#            # Only update the best solution if it's valid and better
#            if all_satisfied and new_weight > best_weight:
#                best_solution = current_solution[:]
#                best_weight = new_weight
#
#            # Decide whether to keep the flip (even if it's invalid for now)
#            if new_satisfied_clauses >= satisfied_clauses:
#                if new_weight >= current_weight:
#                    # Improvement in both weight and satisfied clauses
#                    current_weight = new_weight
#                    satisfied_clauses = new_satisfied_clauses
#                elif random.random() < p1:
#                    # Accept worsening weight
#                    current_weight = new_weight
#            elif random.random() < p2:
#                # Accept worsening clause satisfaction
#                satisfied_clauses = new_satisfied_clauses
#                current_weight = new_weight
#            else:
#                # Revert the flip if not accepted
#                current_solution[flip_index] = not current_solution[flip_index]
#
#        # Return the best valid solution found, or None if no valid solution was found
#        if best_solution is None:
#            print("no solution.")
#            return [],-1
#        #print(best_weight)
#        return best_solution, best_weight
#
#    def anneal(self, initial_max_flips=1000, initial_p1=0.2, initial_p2=0.2,
#               T0=1.0, cooling_rate=0.95, max_iterations=100, trials=5, log_file="annealing_log.csv"):
#        """
#        Perform simulated annealing to optimize the solver parameters and log results.
#        """
#        self.best_params = {'max_flips': initial_max_flips, 'p1': initial_p1, 'p2': initial_p2}
#        self.best_weight = 0
#        self.best_configuration = None
#        self.temperature = T0
#
#        current_params = self.best_params.copy()
#        current_weight = 0
#
#        # Open the log file and write headers
#        with open(log_file, mode='w', newline='') as file:
#            writer = csv.DictWriter(file, fieldnames=[
#                'iteration', 'temperature', 'max_flips', 'p1', 'p2',
#                'avg_weight', 'best_weight', 'best_configuration', 'improved'
#            ])
#            writer.writeheader()
#
#            for iteration in range(max_iterations):
#                weights = []
#                configurations = []
#
#                # Run the solver with the current parameters
#                for _ in range(trials):  # Run multiple trials to compute average weight
#                    solution, weight = self.solve(
#                        max_flips=current_params['max_flips'],
#                        p1=current_params['p1'],
#                        p2=current_params['p2']
#                    )
#                    weights.append(weight)
#                    configurations.append(solution)
#
#                avg_weight = sum(weights) / len(weights)
#                best_trial_index = weights.index(max(weights))
#                best_trial_weight = weights[best_trial_index]
#                best_trial_configuration = configurations[best_trial_index]
#
#                # Check if we found a better solution
#                improved = False
#                if best_trial_weight > self.best_weight:
#                    self.best_weight = best_trial_weight
#                    self.best_params = current_params.copy()
#                    self.best_configuration = best_trial_configuration
#                    improved = True
#
#                # Format the best configuration as a string using numbers
#                best_config_str = self.format_configuration(self.best_configuration)
#
#                # Log the iteration data
#                writer.writerow({
#                    'iteration': iteration,
#                    'temperature': self.temperature,
#                    'max_flips': current_params['max_flips'],
#                    'p1': current_params['p1'],
#                    'p2': current_params['p2'],
#                    'avg_weight': avg_weight,
#                    'best_weight': self.best_weight,
#                    'best_configuration': best_config_str,
#                    'improved': improved
#                })
#
#                # Generate a new parameter set
#                new_params = {
#                    'max_flips': max(1, current_params['max_flips'] + random.randint(-50, 50)),
#                    'p1': min(max(0, current_params['p1'] + random.uniform(-0.05, 0.05)), 1),
#                    'p2': min(max(0, current_params['p2'] + random.uniform(-0.05, 0.05)), 1)
#                }
#
#                # Run the solver with the new parameters
#                _, new_weight = self.solve(
#                    max_flips=new_params['max_flips'],
#                    p1=new_params['p1'],
#                    p2=new_params['p2']
#                )
#
#                # Decide whether to accept the new parameters
#                delta_weight = new_weight - current_weight
#                if delta_weight > 0 or random.random() < self._acceptance_probability(delta_weight):
#                    current_params = new_params
#                    current_weight = new_weight
#
#                # Cool down the temperature
#                self.temperature *= cooling_rate
#
#        # Print the best configuration
#        print(f"Best Configuration: {self.format_configuration(self.best_configuration)}")
#        print(f"Best Weight: {self.best_weight}")
#
#    def format_configuration(self, configuration):
#        """
#        Format the configuration as a string of numbers: '1 -2 -3 ...'
#        :param configuration: The boolean list representing the variable configuration.
#        :return: A formatted string representation of the configuration.
#        """
#        return " ".join(
#            [str(i + 1) if configuration[i] else f"-{i + 1}" for i in range(len(configuration))]
#        )
#
#    def _acceptance_probability(self, delta_weight):
#        """
#        Calculate the probability of accepting a worse solution.
#        :param delta_weight: Difference in weight between new and current solutions.
#        :return: Probability of accepting the worse solution.
#        """
#        return min(1, math.exp(delta_weight / self.temperature))
#
#import random
#
#class SATSolverScored:
#    def __init__(self, problem):
#        """
#        Initialize the SAT solver with a given SAT problem.
#        """
#        self.problem = problem
#        self.scores = []  # Store precomputed variable scores
#        self.ordered_variables = []  # Store variables ordered by score
#        self.best_solution = None  # Track the best solution found
#        self.best_weight = 0  # Track the weight of the best solution
#
#    def precompute_scores(self, alpha=1.0, beta=1.0):
#        """
#        Precompute variable scores based on their relevance in the formula.
#        :param alpha: Coefficient for weighting the variable's weight.
#        :param beta: Coefficient for the clause relevance impact.
#        """
#        self.scores = []
#        for i in range(self.problem.num_variables):
#            non_negated_clauses = 0
#            negated_clauses = 0
#            for clause in self.problem.clauses:
#                if (i + 1) in clause:  # Non-negated form
#                    non_negated_clauses += 1
#                if -(i + 1) in clause:  # Negated form
#                    negated_clauses += 1
#
#            # Compute the relevance score for the variable
#            score = (
#                alpha * self.problem.weights[i] +
#                beta * (non_negated_clauses - negated_clauses)
#            )
#            self.scores.append(score)
#
#        # Sort variables by score in descending order
#        indexed_scores = list(enumerate(self.scores))
#        self.ordered_variables = [
#            index for index, score in sorted(indexed_scores, key=lambda x: x[1], reverse=True)
#        ]
#
#    def initialize_configuration(self):
#        """
#        Initialize the configuration as all False (negated).
#        """
#        self.initial_configuration = [False] * self.problem.num_variables
#
#    def solve(self, max_flips=1000, alpha=1.0, beta=1.0, fallback_probability=0.5):
#        """
#        Solve the SAT problem using the ordered variables and flipping phase.
#        :param max_flips: Maximum number of variable flips to explore.
#        :param alpha: Coefficient for weighting the variable's weight.
#        :param beta: Coefficient for the clause relevance impact.
#        :param fallback_probability: Initial probability to relax scoring-based flipping.
#        :return: The best solution found and its weight.
#        """
#        # Step 1: Precompute scores and initialize configuration
#        self.precompute_scores(alpha=alpha, beta=beta)
#        self.initialize_configuration()
#
#        # Use the initial configuration as the starting point
#        current_solution = self.initial_configuration[:]
#        _, satisfied_clauses, current_weight = self.problem.is_solution(current_solution)
#
#        # Step 2: Iterative flipping phase
#        for flip in range(max_flips):
#            unsatisfied_clauses = [
#                clause for clause in self.problem.clauses if not any(
#                    (var > 0 and current_solution[abs(var) - 1]) or
#                    (var < 0 and not current_solution[abs(var) - 1])
#                    for var in clause
#                )
#            ]
#
#            # If all clauses are satisfied, update the best solution and return
#            if not unsatisfied_clauses:
#                if current_weight > self.best_weight:
#                    self.best_solution = current_solution[:]
#                    self.best_weight = current_weight
#                return self.best_solution, self.best_weight
#
#            # Randomly pick an unsatisfied clause
#            clause = random.choice(unsatisfied_clauses)
#
#            # Choose the best variable to flip based on the ordered variables
#            best_flip = None
#            best_flip_weight = -float('inf')
#
#            for var in sorted(clause, key=lambda v: self.ordered_variables.index(abs(v) - 1)):
#                index = abs(var) - 1
#                current_solution[index] = not current_solution[index]  # Flip the variable
#
#                # Evaluate the new solution
#                all_satisfied, new_satisfied_clauses, new_weight = self.problem.is_solution(current_solution)
#
#                # Prioritize flips that satisfy more clauses or improve weight
#                if new_satisfied_clauses >= satisfied_clauses and new_weight > best_flip_weight:
#                    best_flip = index
#                    best_flip_weight = new_weight
#                elif random.random() < fallback_probability:
#                    best_flip = index
#
#                current_solution[index] = not current_solution[index]  # Revert the flip
#
#            # Apply the best flip
#            if best_flip is not None:
#                current_solution[best_flip] = not current_solution[best_flip]
#                _, satisfied_clauses, current_weight = self.problem.is_solution(current_solution)
#
#                # Update the best solution if all clauses are satisfied
#                if satisfied_clauses == len(self.problem.clauses) and current_weight > self.best_weight:
#                    self.best_solution = current_solution[:]
#                    self.best_weight = current_weight
#                    return self.best_solution, self.best_weight
#
#            # Gradually reduce the fallback probability
#            fallback_probability *= 0.95
#
#        # If no valid solution better than the initial one is found
#        return self.best_solution, self.best_weight

import random
import math
import csv

class SATSolverScored2:
    def __init__(self, problem):
        self.problem = problem
        self.scores = []
        self.ordered_variables = []
        self.best_solution = None
        self.best_weight = 0
        self.csv_log_path = "annealing_log.csv"  # Path for CSV log

    def precompute_scores(self, alpha=1.0, beta=4.0):
        self.scores = []
        for i in range(self.problem.num_variables):
            non_negated_clauses = 0
            negated_clauses = 0
            for clause in self.problem.clauses:
                if (i + 1) in clause:
                    non_negated_clauses += 1
                if -(i + 1) in clause:
                    negated_clauses += 1
            score = (
                alpha * self.problem.weights[i]
                + beta * (non_negated_clauses - negated_clauses)
            )
            self.scores.append(score)
        indexed_scores = list(enumerate(self.scores))
        self.ordered_variables = [
            index for index, score in sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        ]

    def initialize_configuration(self):
        self.initial_configuration = [False] * self.problem.num_variables

    def solve(self, T0=17.5, alpha=0.98, max_iterations=1000, T_min=0.01, max_stagnation=30):
        self.precompute_scores()
        self.initialize_configuration()
        current_solution = self.initial_configuration[:]
        _, satisfied_clauses, current_weight = self.problem.is_solution(current_solution)
        self.best_solution = current_solution[:]
        self.best_weight = current_weight

        T = T0
        stagnation_counter = 0
        last_best_weight = self.best_weight

        with open(self.csv_log_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Iteration", "Temperature", "Satisfied Clauses", "Current Weight", "Best Weight", "Configuration"])

            for iteration in range(max_iterations):
                unsatisfied_clauses = [
                    clause for clause in self.problem.clauses if not any(
                        (var > 0 and current_solution[abs(var) - 1]) or
                        (var < 0 and not current_solution[abs(var) - 1])
                        for var in clause
                    )
                ]

                if not unsatisfied_clauses:
                    if current_weight > self.best_weight:
                        self.best_solution = current_solution[:]
                        self.best_weight = current_weight
                    break

                # Select a random unsatisfied clause and a variable to flip
                clause = random.choice(unsatisfied_clauses)
                flip_var = random.choice(clause)
                index = abs(flip_var) - 1

                current_solution[index] = not current_solution[index]
                all_satisfied, new_satisfied_clauses, new_weight = self.problem.is_solution(current_solution)

                delta = (new_satisfied_clauses - satisfied_clauses) + (new_weight - current_weight) / self.problem.weight_ceil
                accept_prob = math.exp(delta / T) if delta < 0 else 1.0

                if random.random() < accept_prob:
                    satisfied_clauses = new_satisfied_clauses
                    current_weight = new_weight
                    if all_satisfied and current_weight > self.best_weight:
                        self.best_solution = current_solution[:]
                        self.best_weight = current_weight
                    stagnation_counter = 0  # Reset stagnation counter
                else:
                    current_solution[index] = not current_solution[index]

                # Track stagnation
                if self.best_weight == last_best_weight:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                    last_best_weight = self.best_weight

                # Reheat temperature if stagnation is detected
                if stagnation_counter >= max_stagnation:
                    T = T0 * 0.5  # Reheat temperature
                    stagnation_counter = 0

                # Cooling schedule
                T = max(T * alpha, T_min)

                # Periodic perturbation to escape local optima
                if iteration % 100 == 0:
                    for _ in range(random.randint(1, 3)):  # Flip 1-3 random variables
                        rand_var = random.randint(0, self.problem.num_variables - 1)
                        current_solution[rand_var] = not current_solution[rand_var]

                # Log progress
                csvwriter.writerow([
                    iteration,
                    T,
                    satisfied_clauses,
                    current_weight,
                    self.best_weight,
                    self.best_solution  # Append the configuration of the best solution
                ])

                if iteration % 100 == 0 or iteration == max_iterations - 1:
                    print(f"Iteration {iteration}: T={T:.4f}, Satisfied={satisfied_clauses}, Best Weight={self.best_weight}")

        return self.best_solution, self.best_weight


# Step 1: Parse the SAT problem
# file_path = "input.txt"  # Replace with the path to your MWCNF file
# parser = WeightedSATParser(file_path)
# parser.parse_file()

# # Step 2: Initialize the SAT Solver
# solver = SATSolverScored2(problem=parser)

# Step 3: Solve the SAT problem using simulated annealing with anti-stagnation
# Parameters:
# T0 = Initial temperature
# alpha = Cooling rate
# max_iterations = Maximum number of iterations
# T_min = Minimum temperature
# max_stagnation = Maximum iterations allowed without improvement before reheating
# T0 = 17.5
# alpha = 0.98
# max_iterations = 100000
# T_min = 0.0
# max_stagnation = 35

# best_solution, best_weight = solver.solve(
#     T0=T0,
#     alpha=alpha,
#     max_iterations=max_iterations,
#     T_min=T_min,
#     max_stagnation=max_stagnation
# )

# # Step 4: Print the results
# print("\n=== Simulated Annealing Results with Anti-Stagnation ===")
# print(f"Best Weight: {best_weight}")
# print(f"Best Solution (Variable Assignments): {best_solution}")
# print(f"CONFIRMATION -- IS CORRECT?: {parser.is_solution(best_solution)[0]}")

# # Optional: Analyze the log
# print(f"Annealing log saved to: {solver.csv_log_path}")


# Example usage:
# m_flips = 4000
# a = 1.0
# b = 10.0
# fp = 0.3
# trials = 100
# s=0

# parser = WeightedSATParser("path/to/file")
# parser.parse_file()
# for i in range(trials):
#     solver = SATSolverScored(parser)    
#     solution,weight = solver.solve(max_flips=m_flips, alpha=a, beta=b, fallback_probability=fp)
#     if solution == None:
#         print("===No valid solution found.===\n")
#     else:
#         s+=1
#         sol_str = ""
#         for i in range(1,len(solution)):
#             if solution[i]:
#                 sol_str += str(i) + " "
#             else:
#                 sol_str += "-"+str(i) + " "
#         print(f"Solution: {sol_str}")
#         check,_,w = parser.is_solution(solution)
#         print(f"{check} - {w}")
#         if not check:
#             print("\n\n:((((((\n\n")
#             exit()
# print("Parameters: \nmax_flips: " + str(m_flips) + "\na: " + str(a) + "\nb: " + str(b) + "\nfp: " + str(fp))
# print("Success rate: " + str(s/trials * 100) + "%")
