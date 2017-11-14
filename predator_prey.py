import matplotlib.pyplot as plt
import numpy as np


class PredatorPreyModel(object):
    def __init__(self,  economic_profit, predators=5, prey=10, prey_growth_rate=4.0, prey_death_rate=1.0, predator_death_rate=2.0, predator_growth_rate=0.075, harvesting_reward=1.0, harvesting_cost=1.0, harvesting_effort=0.999):
        """
        Sets default values for the following instance variables:
        Lotka-Volterra equation coefficients:
            self.predators - Predator population at time 0
            self.prey - Prey population at time 0
            self.prey_growth_rate - Prey growth rate in the absence of predators
            self.prey_death_rate - Prey death rate due to predations
            self.predator_growth_rate - Predator growth rate per eaten prey
            self.predator_death_rate - Predator decay rate due to absence of prey
            self.delta_time - Delta time for the differential equation calculation
        """
        self.predators = predators
        self.prey = prey
        self.prey_growth_rate = prey_growth_rate
        self.prey_death_rate = prey_death_rate
        self.predator_death_rate = predator_death_rate
        self.predator_growth_rate = predator_growth_rate
        self.delta_time=0.02
        self.harvesting_reward = harvesting_reward
        self.harvesting_cost = harvesting_cost
        self.economic_profit = economic_profit
        self.harvesting_effort = harvesting_effort

    def prey_change(self, prey, predators, harvesting_effort):
        """
        Calculates the change in prey population size using the Lotka-Volterra
        equation for prey and the time delta defined in "self.dt"
        """

        # Calculate the rate of population change
        return prey * (self.prey_growth_rate - prey * self.prey_death_rate - predators - harvesting_effort)


    def predator_change(self, prey, predators):
        """
        Calculates the change in predator population size using the
        Lotka-Volterra equation for predators and the time delta defined in
        "self.dt"
        """

        # Calculate the rate of population change
        return predators * ( - self.predator_death_rate + prey)


    def harvesting_effort_change(self, prey):
        # p - harvesting reward
        # c - harvesting cost
        # v - economic profit
        return self.harvesting_effort * (self.harvesting_reward * prey - self.harvesting_cost) - self.economic_profit


    def calculate_improved_euler(self, delta_time=0.02, iterations=10000):
        """
        Calculates the predator/prey population growth for the given parameters
        (Defined in the __init__ docstring). Returns the following dictionary:
        {'predator': [predator population history as a list],
         'prey': [prey population history as a list]}
        """
        predator_history = []
        prey_history = []
        harvesting_effort_history = []

        for i in range(iterations):
            xk_1 = self.prey_change(self.prey, self.predators, self.harvesting_effort) * delta_time
            yk_1 = self.predator_change(self.prey, self.predators) * delta_time
            ek_1 = self.harvesting_effort_change(self.prey) * delta_time
            xk_2 = self.prey_change(self.prey + xk_1, self.predators + yk_1, self.harvesting_effort + ek_1) * delta_time
            yk_2 = self.predator_change(self.prey + xk_1, self.predators + yk_1) * delta_time
            ek_2 = self.harvesting_effort_change(self.prey + xk_1) * delta_time


            self.prey = self.prey + (xk_1 + xk_2) / 2
            self.predators = self.predators + (yk_1 + yk_2) / 2
            self.harvesting_effort = self.harvesting_effort + (ek_1 + ek_2) / 2

            predator_history.append(self.predators)
            prey_history.append(self.prey)
            harvesting_effort_history.append(self.harvesting_effort)

        return {'predator': predator_history, 'prey': prey_history, 'harvesting_effort': harvesting_effort_history}

    def calculate_runge_kutta(self, delta_time=0.02, iterations=1000):
        """
        Calculates the predator/prey population growth for the given parameters
        (Defined in the __init__ docstring). Returns the following dictionary:
        {'predator': [predator population history as a list],
         'prey': [prey population history as a list]}
        """
        predator_history = []
        prey_history = []

        for i in range(iterations):
            xk_1 = self.prey_change_interaction(self.prey, self.predators) * delta_time
            yk_1 = self.predator_change_interaction(self.prey, self.predators) * delta_time
            xk_2 = self.prey_change_interaction(self.prey + 0.5 * xk_1, self.predators + 0.5 * yk_1) * delta_time
            yk_2 = self.predator_change_interaction(self.prey + 0.5 * xk_1, self.predators + 0.5 * yk_1) * delta_time
            xk_3 = self.prey_change_interaction(self.prey + 0.5 * xk_2, self.predators + 0.5 * yk_2) * delta_time
            yk_3 = self.predator_change_interaction(self.prey + 0.5 * xk_2, self.predators + 0.5 * yk_2) * delta_time
            xk_4 = self.prey_change_interaction(self.prey + xk_3, self.predators + yk_3) * delta_time
            yk_4 = self.predator_change_interaction(self.prey + xk_3, self.predators + yk_3) * delta_time

            self.prey = self.prey + (xk_1 + 2*xk_2 + 2*xk_3 + xk_4) / 6
            self.predators = self.predators + (yk_1 + 2 * yk_2 + 2 * yk_3 + yk_4) / 6

            predator_history.append(self.predators)
            prey_history.append(self.prey)

        return {'predator': predator_history, 'prey': prey_history}


def main():
    gc = PredatorPreyModel(prey = 1.997, predators = 0.998, economic_profit=0.989)
    dynamics = gc.calculate_improved_euler()
    prey_populations = dynamics['prey']
    predator_populations = dynamics['predator']

    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5, hspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(predator_populations, 'r-', label='predator')
    ax1.plot(prey_populations, 'b-', label='prey')

    # ax1.plot(z, 'g-', label='prey')
    ax1.set_title("Dynamics in time")
    ax1.set_xlabel("time")
    ax1.grid()
    ax1.legend(loc='best')

    ax2.plot(prey_populations, predator_populations, color="blue")
    ax2.set_xlabel("prey")
    ax2.set_ylabel("predator")
    ax2.set_title("Phase space")
    ax2.grid()
    plt.show()

def main2():
    population_sizes = np.arange(0.9, 1.8, 0.1)
    alpha = 2.0/3.0
    beta = 4.0/3.0
    gamma = 1.0
    delta = 1.0
    fig = plt.figure(figsize=(15, 5))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel("prey")
    ax2.set_ylabel("predator")
    ax2.set_title("Phase space")
    ax2.grid()
    for population_size in population_sizes:
        print population_size
        gc = PredatorPreyModel(population_size, population_size, alpha, beta, gamma, delta)
        populations2 = gc.calculate_runge_kutta()
        populations = gc.calculate_improved_euler()
        prey_populations = populations['prey']
        predator_populations = populations['predator']
        prey_populations2 = populations2['prey']
        predator_populations2 = populations2['predator']
        ax2.plot(prey_populations2, predator_populations2)
    plt.show()


if __name__ == "__main__":
    main()
