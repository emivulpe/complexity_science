import matplotlib.pyplot as plt
import numpy as np


# http://www.tandfonline.com/doi/full/10.1080/23737867.2017.1282843

class PredatorPreyModel(object):

    def __init__(self, predators=5, prey=10,
                 prey_growth_rate=1.0, prey_death_rate=0.1,
                 predator_death_rate=1.0, predator_growth_rate=0.075,
                 nu=0.1, B=4, K= 230.0): #add a parameter for the model
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
        self.B = B
        self.nu = nu
        self.K = K

    def prey_change_lotka_volterra(self, prey, predators):
        """
        Calculates the change in prey population size using the Lotka-Volterra
        equation for prey and the time delta defined in "self.dt"
        """

        # Calculate the rate of population change
        return prey * (self.prey_growth_rate - self.prey_death_rate * predators)

    def predator_change_lotka_volterra(self, prey, predators):
        """
        Calculates the change in predator population size using the
        Lotka-Volterra equation for predators and the time delta defined in
        "self.dt"
        """

        # Calculate the rate of population change
        return predators * (self.predator_growth_rate * prey - self.predator_death_rate)

    def prey_change_alley_competition(self, prey, predators):
        """
        Calculates the change in prey population size using the Lotka-Volterra
        equation for prey and the time delta defined in "self.dt"
        """

        # Calculate the rate of population change
        return prey * self.prey_growth_rate - self.prey_death_rate * predators * prey

    def predator_change_alley_competition(self, prey, predators):
        """
        Calculates the change in predator population size using the
        Lotka-Volterra equation for predators and the time delta defined in
        "self.dt"
        """

        return (predators * self.predator_growth_rate * prey * (
        predators / (predators + self.B) * (1 - predators / (self.nu * prey)))) - self.predator_death_rate * predators

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
        """
        Calculates the predator/prey population growth for the given parameters
        (Defined in the __init__ docstring). Returns the following dictionary:
        {'predator': [predator population history as a list],
         'prey': [prey population history as a list]}
        """
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


    def prey_change_alley_competition_saturation(self, prey, predators):
        """
        Calculates the change in prey population size using the Lotka-Volterra
        equation for prey and the time delta defined in "self.dt"
        """

        # Calculate the rate of population change
        return (prey * self.prey_growth_rate)*(1 - (prey / self.K)) - self.prey_death_rate * predators * prey

def compute_fixed_points(prey_change_f, predator_change_f,
                         r=100):
    fixed_points = []
    fp = {}
    for x in range(1, r):
        for y in range(1, r):

            # if prey >= 0 and predator >= 0 and prey < 1 and predator < 1:
            #    print x, y, prey, predator
            if (prey_change_f(float(x), float(y)) == 0.0) and (
                predator_change_f(float(x), float(y)) < 0.4) and (
                predator_change_f(float(x), float(y)) >= 0.0):
                prey = prey_change_f(float(x), float(y))
                predator = predator_change_f(float(x), float(y))
                if y not in fp.keys():
                    fp[y] = (x, predator)
                else:
                    if fp[y][1] > predator:
                        fp[y] = (x, predator)
                print x, y, prey, predator
    for y in fp.keys():
        fixed_points.append((fp[y][0], y))
    return fixed_points


    def prey_change_alley(self, prey, predators):
        """
        Calculates the change in prey population size using the Lotka-Volterra
        equation for prey and the time delta defined in "self.dt"
        """

        # Calculate the rate of population change
        return prey * self.prey_growth_rate - self.prey_death_rate * predators * prey

    def predator_change_alley(self, prey, predators):
        """
        Calculates the change in predator population size using the
        Lotka-Volterra equation for predators and the time delta defined in
        "self.dt"
        """

        # Calculate the rate of population change
        return (predators * self.predator_growth_rate * prey * (
        predators / (predators + self.B))) - self.predator_death_rate * predators


    def prey_change_competition(self, prey, predators):

        return self.prey_growth_rate * prey * (1 - (prey + a12 * predators) / k1)

    def predator_change_competition(self, prey, predators):

        return self.predator_growth_rate * predators * (1 - (predators + a21 * prey) / k2)


#    def compute_fixed_points_equation(self):

#        return (self.predator_death_rate * (self.prey_growth_rate + (self.prey_death_rate * self.B))) / (self.predator_growth_rate * self.)
def main2():
    gc = PredatorPreyModel()
    populations2 = gc.calculate_runge_kutta()
    populations = gc.calculate_improved_euler()
    prey_populations = populations['prey']
    predator_populations = populations['predator']
    prey_populations2 = populations2['prey']
    predator_populations2 = populations2['predator']

    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5, hspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(predator_populations, 'r-', label='predator')
    ax1.plot(prey_populations, 'b-', label='prey')
    ax1.plot(predator_populations2, 'y-', label='predator_rk')
    ax1.plot(prey_populations2, 'g-', label='prey_rk')
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


def main():
    population_sizes = np.arange(0.9, 1.8, 0.1)
    alpha = 2.0 / 3.0
    beta = 4.0 / 3.0
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


def main_competition():
    gc = PredatorPreyModel(10, 10, prey_growth_rate=0.2, predator_growth_rate=0.5)
    populations = gc.calculate_improved_euler_competition(0.02, 10000)
    print populations
    prey_populations = populations['prey']
    predator_populations = populations['predator']

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


def main_no_interaction():
    gc = PredatorPreyModel(10, 10, prey_growth_rate=0.2, predator_growth_rate=0.5, predator_death_rate=0.2)
    populations = gc.calculate_improved_euler_no_interaction(0.02, 10000)
    print populations
    prey_populations = populations['prey']
    predator_populations = populations['predator']

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


def main_alley():
    gc = PredatorPreyModel(prey=50.0, predators=20.0, prey_growth_rate=0.1, prey_death_rate=0.005,
                           predator_growth_rate=0.05, predator_death_rate=0.4, B=30.0, nu=1.0)
    populations_alley = gc.calculate_improved_euler_alley(0.02, 4000)
    gc2 = PredatorPreyModel(prey=50.0, predators=20.0, prey_growth_rate=0.1, prey_death_rate=0.005,
                            predator_growth_rate=0.05, predator_death_rate=0.4, B=4.0, nu=1.0)
    print gc2.predator_change_alley_competition(30.0, 20.0)
    gc2.find_fixed_points_alley_competition()
    populations_alley_competition = gc2.calculate_improved_euler_alley_competition(0.02, 8000)
    print populations_alley
    prey_populations_alley = populations_alley['prey']
    predator_populations_alley = populations_alley['predator']
    prey_populations_alley_competition = populations_alley_competition['prey']
    predator_populations_alley_competition = populations_alley_competition['predator']
    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5, hspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    """
    ax1.plot(predator_populations_alley, 'r-', label='predator')
    ax1.plot(prey_populations_alley, 'b-', label='prey')
    # ax1.plot(z, 'g-', label='prey')
    ax1.set_title("Dynamics in time")
    ax1.set_xlabel("time")
    ax1.grid()
    ax1.legend(loc='best')
    """
    ax2.plot(prey_populations_alley_competition, predator_populations_alley_competition, color="blue")
    ax2.set_xlabel("prey")
    ax2.set_ylabel("predator")
    ax2.set_title("Phase space")
    ax2.grid()


    ax1.plot(prey_populations_alley, predator_populations_alley, color="blue")
    ax1.set_xlabel("prey")
    ax1.set_ylabel("predator")
    ax1.set_title("Phase space")
    ax1.grid()

    # plt.show()


def main_alley2():
    gc = PredatorPreyModel(prey=50.0, predators=20.0, prey_growth_rate=0.1, prey_death_rate=0.005,
                           predator_growth_rate=0.05, predator_death_rate=0.4, B=30.0, nu=1.0)
    populations_alley = gc.calculate_improved_euler_alley(0.02, 4000)
    # gc2 = PredatorPreyModel(prey = 50.0, predators = 20.0, prey_growth_rate = 0.5, prey_death_rate=0.03, predator_growth_rate = 0.02, predator_death_rate=0.3, B=1.0, nu=1.0)
    # with 7500 gc2 = PredatorPreyModel(prey = 50.0, predators = 20.0, prey_growth_rate = 0.5, prey_death_rate=0.03, predator_growth_rate = 0.02, predator_death_rate=0.3, B=5.0, nu=3.8)
    # gc2 = PredatorPreyModel(prey = 50.0, predators = 20.0, prey_growth_rate = 0.5, prey_death_rate=0.03, predator_growth_rate = 0.02, predator_death_rate=0.3, B=5.0, nu=0.1)
    gc2 = PredatorPreyModel(prey=50.0, predators=20.0, prey_growth_rate=0.5, prey_death_rate=0.03,
                            predator_growth_rate=0.02, predator_death_rate=0.3, B=18.5185, nu=1.0)

    populations_alley_competition = gc2.calculate_improved_euler_alley_competition(0.02, 7500)
    print populations_alley
    prey_populations_alley = populations_alley['prey']
    predator_populations_alley = populations_alley['predator']
    prey_populations_alley_competition = populations_alley_competition['prey']
    predator_populations_alley_competition = populations_alley_competition['predator']
    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5, hspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    """
    ax1.plot(predator_populations_alley, 'r-', label='predator')
    ax1.plot(prey_populations_alley, 'b-', label='prey')
    # ax1.plot(z, 'g-', label='prey')
    ax1.set_title("Dynamics in time")
    ax1.set_xlabel("time")
    ax1.grid()
    ax1.legend(loc='best')
    """
    ax2.plot(prey_populations_alley_competition, predator_populations_alley_competition, color="blue")
    ax2.set_xlabel("prey")
    ax2.set_ylabel("predator")
    ax2.set_title("Phase space")
    ax2.grid()


    ax1.plot(prey_populations_alley, predator_populations_alley, color="blue")
    ax1.set_xlabel("prey")
    ax1.set_ylabel("predator")
    ax1.set_title("Phase space")
    ax1.grid()
    plt.show()


def main_alley3():
    gc = PredatorPreyModel(prey=50.0, predators=20.0, prey_growth_rate=0.03, prey_death_rate=0.002,
                            predator_growth_rate=0.02, predator_death_rate=0.3, B=15.0, nu=1.0)

    # gc2 = PredatorPreyModel(prey = 50.0, predators = 20.0, prey_growth_rate = 0.03, prey_death_rate=0.002, predator_growth_rate = 0.02, predator_death_rate=0.3, B=14.0, nu=1.0)
    # gc2 = PredatorPreyModel(prey = 50.0, predators = 20.0, prey_growth_rate = 0.03, prey_death_rate=0.002, predator_growth_rate = 0.02, predator_death_rate=0.3, B=16.0, nu=1.0)
    # gc2 = PredatorPreyModel(prey = 50.0, predators = 20.0, prey_growth_rate = 0.03, prey_death_rate=0.002, predator_growth_rate = 0.02, predator_death_rate=0.3, B=3.0, nu=0.75)
    # gc2 = PredatorPreyModel(prey = 50.0, predators = 20.0, prey_growth_rate = 0.03, prey_death_rate=0.002, predator_growth_rate = 0.02, predator_death_rate=0.3, B=5.0, nu=2.0) # different
    gc2 = PredatorPreyModel(prey=50.0, predators=20.0, prey_growth_rate=0.03, prey_death_rate=0.002,
                            predator_growth_rate=0.02, predator_death_rate=0.3, B=15.0, nu=1.0)  # different
    populations_alley = gc2.runge_kutta(gc2.prey_change_alley_competition, gc2.predator_change_alley_competition, 0.02, 9500)

    populations_alley_competition = gc.runge_kutta(gc.prey_change_alley_competition, gc.predator_change_alley_competition, 0.02, 9500)
    fixed_points = compute_fixed_points(gc.prey_change_alley_competition, gc.predator_change_alley_competition)
    fixed_points2 = compute_fixed_points(gc2.prey_change_alley_competition, gc2.predator_change_alley_competition)
    print "gc", fixed_points
    print "gc2", fixed_points2

    print populations_alley
    prey_populations_alley = populations_alley['prey']
    predator_populations_alley = populations_alley['predator']
    prey_populations_alley_competition = populations_alley_competition['prey']
    predator_populations_alley_competition = populations_alley_competition['predator']
    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5, hspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    """
    ax1.plot(predator_populations_alley, 'r-', label='predator')
    ax1.plot(prey_populations_alley, 'b-', label='prey')
    # ax1.plot(z, 'g-', label='prey')
    ax1.set_title("Dynamics in time")
    ax1.set_xlabel("time")
    ax1.grid()
    ax1.legend(loc='best')
    """
    ax2.plot(prey_populations_alley_competition, predator_populations_alley_competition, color="blue")
    ax2.set_xlabel("prey")
    ax2.set_ylabel("predator")
    ax2.set_title("Phase space")
    ax2.grid()
    for point in fixed_points:
        ax2.scatter(point[0], point[1], facecolors='none', edgecolors='r', s=55)

    ax1.plot(prey_populations_alley, predator_populations_alley, color="blue")
    ax1.set_xlabel("prey")
    ax1.set_ylabel("predator")
    ax1.set_title("Phase space")
    ax1.grid()
    for point in fixed_points2:
        ax1.scatter(point[0], point[1], facecolors='none', edgecolors='r', s=55)
    plt.show()

def main_3():
    gc1 = PredatorPreyModel(prey=30.0, predators=10.0, prey_growth_rate=0.03, prey_death_rate=0.002,
                            predator_growth_rate=0.02, predator_death_rate=0.3, B=14.0, nu=1.0)  # different
    gc2 = PredatorPreyModel(prey=45.0, predators=15.0, prey_growth_rate=0.03, prey_death_rate=0.002,
                            predator_growth_rate=0.02, predator_death_rate=0.3, B=14.0, nu=1.0)  # different
    gc3 = PredatorPreyModel(prey=60.0, predators=30.0, prey_growth_rate=0.03, prey_death_rate=0.002,
                            predator_growth_rate=0.02, predator_death_rate=0.3, B=16.0, nu=1.0)  # different

    populations_alley_competition_below_fp = gc1.runge_kutta(gc1.prey_change_alley_competition, gc1.predator_change_alley_competition, 0.02, 2000)
    populations_alley_competition_fp = gc2.runge_kutta(gc2.prey_change_alley_competition, gc2.predator_change_alley_competition, 0.02, 2000)
    populations_alley_competition_above_fp = gc3.runge_kutta(gc3.prey_change_alley_competition, gc3.predator_change_alley_competition, 0.02, 2000)

    fixed_points = gc2.find_fixed_points_alley_competition_iterative()

    fixed_points2 = compute_fixed_points(gc2.prey_change_alley_competition, gc2.predator_change_alley_competition)

    print "fp1", fixed_points
    print "fp2", fixed_points2
    #print gc2.predator_change_alley_competition(45.0, 15.0)

    prey_populations_alley_competition_below_fp = populations_alley_competition_below_fp['prey']
    predator_populations_alley_competition_below_fp = populations_alley_competition_below_fp['predator']
    prey_populations_alley_competition_fp = populations_alley_competition_fp['prey']
    predator_populations_alley_competition_fp = populations_alley_competition_fp['predator']
    prey_populations_alley_competition_above_fp = populations_alley_competition_above_fp['prey']
    predator_populations_alley_competition_above_fp = populations_alley_competition_above_fp['predator']
    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5, hspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)


    ax1.plot(predator_populations_alley_competition_below_fp, 'r-', label='predator')
    ax1.plot(prey_populations_alley_competition_below_fp, 'r--', label='prey')
    ax1.plot(predator_populations_alley_competition_fp, 'b-', label='predator')
    ax1.plot(prey_populations_alley_competition_fp, 'b--', label='prey')
    ax1.plot(predator_populations_alley_competition_above_fp, 'g-', label='predator')
    ax1.plot(prey_populations_alley_competition_above_fp, 'g--', label='prey')
    # ax1.plot(z, 'g-', label='prey')
    ax1.set_title("Dynamics in time")
    ax1.set_xlabel("time")
    ax1.grid()
    ax1.legend(loc='best')

    ax2.plot(prey_populations_alley_competition_above_fp, predator_populations_alley_competition_above_fp, color="blue")
    ax2.set_xlabel("prey")
    ax2.set_ylabel("predator")
    ax2.set_title("Phase space")
    ax2.grid()
    for point in fixed_points:
        ax2.scatter(point[0], point[1], facecolors='none', edgecolors='r', s=55)
    ax2.set_ylim(ymin=0)
    ax2.set_xlim(xmin=0)
    handles, labels = ax1.get_legend_handles_labels()
    print handles, labels
    ax1.legend(handles[:2], labels[:2])
    plt.show()


def main_many():

    population_sizes = range(10, 60, 10)
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
        gc = PredatorPreyModel(prey=population_size, predators=population_size, prey_growth_rate=0.03, prey_death_rate=0.002,
                                predator_growth_rate=0.02, predator_death_rate=0.3, B=14.0, nu=1.0)  # different
        populations2 = gc.runge_kutta(gc.prey_change_alley_competition, gc.predator_change_alley_competition, 10000)
        populations = gc.improved_euler(gc.prey_change_alley_competition, gc.predator_change_alley_competition, 10000)
        prey_populations = populations['prey']
        predator_populations = populations['predator']
        prey_populations2 = populations2['prey']
        predator_populations2 = populations2['predator']
        print predator_populations2, "test1"
        print prey_populations2, "test2"
        ax2.plot(prey_populations2, predator_populations2)
    plt.show()




def main_alley4():
    gc = PredatorPreyModel(prey=100.0, predators=20.0, prey_growth_rate=0.03, prey_death_rate=0.002,
                            predator_growth_rate=0.02, predator_death_rate=0.3, B=16.0, nu=1.0)

    # gc2 = PredatorPreyModel(prey = 50.0, predators = 20.0, prey_growth_rate = 0.03, prey_death_rate=0.002, predator_growth_rate = 0.02, predator_death_rate=0.3, B=14.0, nu=1.0)
    # gc2 = PredatorPreyModel(prey = 50.0, predators = 20.0, prey_growth_rate = 0.03, prey_death_rate=0.002, predator_growth_rate = 0.02, predator_death_rate=0.3, B=16.0, nu=1.0)
    # gc2 = PredatorPreyModel(prey = 50.0, predators = 20.0, prey_growth_rate = 0.03, prey_death_rate=0.002, predator_growth_rate = 0.02, predator_death_rate=0.3, B=3.0, nu=0.75)
    # gc2 = PredatorPreyModel(prey = 50.0, predators = 20.0, prey_growth_rate = 0.03, prey_death_rate=0.002, predator_growth_rate = 0.02, predator_death_rate=0.3, B=5.0, nu=2.0) # different
    gc2 = PredatorPreyModel(prey=100.0, predators=20.0, prey_growth_rate=0.03, prey_death_rate=0.002,
                            predator_growth_rate=0.02, predator_death_rate=0.3, B=16.0, nu=1.0)  # different
    populations_alley = gc2.runge_kutta(gc2.prey_change_alley_competition, gc2.predator_change_alley_competition, 0.02, 9500)

    populations_alley_competition = gc.runge_kutta(gc.prey_change_alley_competition_saturation, gc.predator_change_alley_competition, 0.02, 9500)
    fixed_points = compute_fixed_points(gc.prey_change_alley_competition_saturation, gc.predator_change_alley_competition)
    fixed_points2 = compute_fixed_points(gc2.prey_change_alley_competition, gc2.predator_change_alley_competition)
    print "gc", fixed_points
    print "gc2", fixed_points2

    print populations_alley
    prey_populations_alley = populations_alley['prey']
    predator_populations_alley = populations_alley['predator']
    prey_populations_alley_competition = populations_alley_competition['prey']
    predator_populations_alley_competition = populations_alley_competition['predator']
    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5, hspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    """
    ax1.plot(predator_populations_alley, 'r-', label='predator')
    ax1.plot(prey_populations_alley, 'b-', label='prey')
    # ax1.plot(z, 'g-', label='prey')
    ax1.set_title("Dynamics in time")
    ax1.set_xlabel("time")
    ax1.grid()
    ax1.legend(loc='best')
    """
    ax2.plot(prey_populations_alley_competition, predator_populations_alley_competition, color="blue")
    ax2.set_xlabel("prey")
    ax2.set_ylabel("predator")
    ax2.set_title("Phase space")
    ax2.grid()
    for point in fixed_points:
        ax2.scatter(point[0], point[1], facecolors='none', edgecolors='r', s=55)

    ax1.plot(prey_populations_alley, predator_populations_alley, color="blue")
    ax1.set_xlabel("prey")
    ax1.set_ylabel("predator")
    ax1.set_title("Phase space")
    ax1.grid()
    for point in fixed_points2:
        ax1.scatter(point[0], point[1], facecolors='none', edgecolors='r', s=55)
    plt.show()


def main_alley5():
    gc = PredatorPreyModel(prey=25.0, predators=20.0, prey_growth_rate=0.1, prey_death_rate=0.005,
                           predator_growth_rate=0.05, predator_death_rate=0.4, B=25.5, nu=2.05)

    # gc2 = PredatorPreyModel(prey = 50.0, predators = 20.0, prey_growth_rate = 0.03, prey_death_rate=0.002, predator_growth_rate = 0.02, predator_death_rate=0.3, B=14.0, nu=1.0)
    # gc2 = PredatorPreyModel(prey = 50.0, predators = 20.0, prey_growth_rate = 0.03, prey_death_rate=0.002, predator_growth_rate = 0.02, predator_death_rate=0.3, B=16.0, nu=1.0)
    # gc2 = PredatorPreyModel(prey = 50.0, predators = 20.0, prey_growth_rate = 0.03, prey_death_rate=0.002, predator_growth_rate = 0.02, predator_death_rate=0.3, B=3.0, nu=0.75)
    # gc2 = PredatorPreyModel(prey = 50.0, predators = 20.0, prey_growth_rate = 0.03, prey_death_rate=0.002, predator_growth_rate = 0.02, predator_death_rate=0.3, B=5.0, nu=2.0) # different
    gc2 = PredatorPreyModel(prey=50.0, predators=20.0, prey_growth_rate=0.1, prey_death_rate=0.005,
                            predator_growth_rate=0.05, predator_death_rate=0.4, B=50.0, nu=1.0)  # different
    populations_alley = gc2.runge_kutta(gc2.prey_change_alley_competition, gc2.predator_change_alley_competition, 0.1, 595)

    populations_alley_competition = gc.runge_kutta(gc.prey_change_alley_competition, gc.predator_change_alley_competition, 0.03, 15000)
    fixed_points = compute_fixed_points(gc.prey_change_alley_competition, gc.predator_change_alley_competition)
    fixed_points2 = compute_fixed_points(gc2.prey_change_alley_competition, gc2.predator_change_alley_competition)
    print "gc", fixed_points
    print "gc2", fixed_points2

    print populations_alley
    prey_populations_alley = populations_alley['prey']
    predator_populations_alley = populations_alley['predator']
    prey_populations_alley_competition = populations_alley_competition['prey']
    predator_populations_alley_competition = populations_alley_competition['predator']
    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5, hspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    """
    ax1.plot(predator_populations_alley, 'r-', label='predator')
    ax1.plot(prey_populations_alley, 'b-', label='prey')
    # ax1.plot(z, 'g-', label='prey')
    ax1.set_title("Dynamics in time")
    ax1.set_xlabel("time")
    ax1.grid()
    ax1.legend(loc='best')
    """
    ax2.plot(prey_populations_alley_competition, predator_populations_alley_competition, color="blue")
    ax2.set_xlabel("prey")
    ax2.set_ylabel("predator")
    ax2.set_title("Phase space")
    ax2.grid()
    #for point in fixed_points:
    #    ax2.scatter(point[0], point[1], facecolors='none', edgecolors='r', s=55)


    ax1.plot(predator_populations_alley_competition, 'r-', label='predator')
    ax1.plot(prey_populations_alley_competition, 'b-', label='prey')
    # ax1.plot(z, 'g-', label='prey')
    ax1.set_title("Dynamics in time")
    ax1.set_xlabel("time")
    ax1.grid()
    ax1.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    main_alley5()




# add function that decides which function for the fixed points to execute depending on the type of the model
# add same functions for computing the population dynamics
