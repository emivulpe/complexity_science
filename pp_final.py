import matplotlib.pyplot as plt

class PredatorPreyModel(object):

    def __init__(self, predators=5, prey=10,
                 prey_growth_rate=1.0, prey_death_rate=0.1,
                 predator_death_rate=1.0, predator_growth_rate=0.075,
                 eta=0.1, B=4):
        """
        Sets default values for the following instance variables:
        Lotka-Volterra equation coefficients:
            self.predators - Predator population at time 0
            self.prey - Prey population at time 0
            self.prey_growth_rate - Prey growth rate in the absence of predators
            self.prey_death_rate - Prey death rate due to predations
            self.predator_growth_rate - Predator growth rate per eaten prey
            self.predator_death_rate - Predator decay rate due to absence of prey
            self.B - Allee effect constant
            self.eta - predator vs prey proportionality constant
        """

        self.predators = predators
        self.prey = prey
        self.prey_growth_rate = prey_growth_rate
        self.prey_death_rate = prey_death_rate
        self.predator_death_rate = predator_death_rate
        self.predator_growth_rate = predator_growth_rate
        self.B = B
        self.eta = eta

    def prey_change_alley_competition(self, prey, predators):
        """
        Calculates the change in prey population size
        """

        return prey * self.prey_growth_rate - self.prey_death_rate * predators * prey

    def predator_change_alley_competition(self, prey, predators):
        """
        Calculates the change in predator population with terms for intraspecific competition and Allee effect
        """

        return (predators * self.predator_growth_rate * prey * (
        predators / (predators + self.B) * (1 - predators / (self.eta * prey)))) - self.predator_death_rate * predators

    def improved_euler(self, prey_change_f, predator_change_f,
                       delta_time=0.02, iterations=100):
        predator_history = []
        prey_history = []

        for i in range(iterations):
            xk_1 = prey_change_f(self.prey, self.predators) * delta_time
            yk_1 = predator_change_f(self.prey, self.predators) * delta_time
            xk_2 = prey_change_f(self.prey + xk_1, self.predators + yk_1) * delta_time
            yk_2 = predator_change_f(self.prey + xk_1, self.predators + yk_1) * delta_time

            self.prey = self.prey + (xk_1 + xk_2) / 2
            self.predators = self.predators + (yk_1 + yk_2) / 2

            predator_history.append(self.predators)
            prey_history.append(self.prey)

        return {'predator': predator_history, 'prey': prey_history}


    def runge_kutta(self, prey_change_f, predator_change_f,
                    delta_time=0.02, iterations=100):

        predator_history = []
        prey_history = []

        for i in range(iterations):
            xk_1 = prey_change_f(self.prey, self.predators) * delta_time
            yk_1 = predator_change_f(self.prey, self.predators) * delta_time
            xk_2 = prey_change_f(self.prey + 0.5 * xk_1, self.predators + 0.5 * yk_1) * delta_time
            yk_2 = predator_change_f(self.prey + 0.5 * xk_1, self.predators + 0.5 * yk_1) * delta_time
            xk_3 = prey_change_f(self.prey + 0.5 * xk_2, self.predators + 0.5 * yk_2) * delta_time
            yk_3 = predator_change_f(self.prey + 0.5 * xk_2, self.predators + 0.5 * yk_2) * delta_time
            xk_4 = prey_change_f(self.prey + xk_3, self.predators + yk_3) * delta_time
            yk_4 = predator_change_f(self.prey + xk_3, self.predators + yk_3) * delta_time

            self.prey = self.prey + (xk_1 + 2 * xk_2 + 2 * xk_3 + xk_4) / 6
            self.predators = self.predators + (yk_1 + 2 * yk_2 + 2 * yk_3 + yk_4) / 6

            predator_history.append(self.predators)
            prey_history.append(self.prey)

        return {'predator': predator_history, 'prey': prey_history}


def main():
    gc = PredatorPreyModel(prey=25.0, predators=20.0, prey_growth_rate=0.1, prey_death_rate=0.005,
                           predator_growth_rate=0.05, predator_death_rate=0.4, B=25.5, eta=2.05)

    populations_alley_competition = gc.runge_kutta(gc.prey_change_alley_competition, gc.predator_change_alley_competition, 0.03, 15000)

    prey_populations_alley_competition = populations_alley_competition['prey']
    predator_populations_alley_competition = populations_alley_competition['predator']
    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5, hspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(predator_populations_alley_competition, 'r-', label='predator')
    ax1.plot(prey_populations_alley_competition, 'b-', label='prey')
    ax1.set_title("Dynamics in time")
    ax1.set_xlabel("time")
    ax1.grid()
    ax1.legend(loc='best')

    ax2.plot(prey_populations_alley_competition, predator_populations_alley_competition, color="blue")
    ax2.set_xlabel("prey")
    ax2.set_ylabel("predator")
    ax2.set_title("Phase space")
    ax2.grid()

    plt.show()


if __name__ == "__main__":
    main()




# add function that decides which function for the fixed points to execute depending on the type of the model
# add same functions for computing the population dynamics
