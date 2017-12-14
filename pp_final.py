import argparse
import matplotlib.pyplot as plt

def setup_parser():
    parser = argparse.ArgumentParser(description="Simulation for predator-prey population dynamics with Allee effect and intraspecific competition.")
    parser.add_argument("-N", dest="prey", type=int, default=25, help="the initial number of prey")
    parser.add_argument("-P", dest="predators", type=int, default=20, help="the initial number of predators")
    parser.add_argument("-r1", dest="prey_growth_rate", type=float, default=0.1, help="the prey growth rate")
    parser.add_argument("-delta", dest="prey_death_rate", type=float, default=0.005, help="the prey death rate")
    parser.add_argument("-theta", dest="predator_growth_rate", type=float, default=0.05, help="the predator growth rate")
    parser.add_argument("-r2", dest="predator_death_rate", type=float, default=0.4, help="the predator death rate")
    parser.add_argument("-B", dest="allee_constant", type=float, default=25.0, help="the Allee constant")
    parser.add_argument("-eta", dest="proportionality_constant", type=float, default=2.006, help="the proportionality constant")
    parser.add_argument("-ts", dest="time_step", type=float, default=0.1, help="the time step for the Runge-Kutta method")
    parser.add_argument("-ti", dest="time_interval", type=int, default=19500, help="the time interval for the Runge-Kutta method")

    return parser.parse_args()


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

        self.predators = float(predators)
        self.prey = float(prey)
        self.prey_growth_rate = float(prey_growth_rate)
        self.prey_death_rate = float(prey_death_rate)
        self.predator_death_rate = float(predator_death_rate)
        self.predator_growth_rate = float(predator_growth_rate)
        self.B = float(B)
        self.eta = float(eta)

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

    def compute_predator_fp(self):
        return (self.prey_growth_rate / (self.prey_death_rate * self.eta)) + ((self.predator_death_rate / self.predator_growth_rate) * ((self.prey_growth_rate + (self.prey_death_rate * self.B)) / self.prey_growth_rate))

    def compute_prey_fp(self):
        return self.prey_growth_rate / self.prey_death_rate

    def compute_fixed_point(self):
        return (self.compute_predator_fp(), self.compute_prey_fp())

    def compute_trace(self):
        t1 = self.B * self.predator_death_rate*self.prey_death_rate**2*self.eta
        t2 = self.prey_growth_rate**2 * self.predator_growth_rate
        t3 = self.prey_growth_rate * self.prey_death_rate * self.eta
        t4 = self.B * self.prey_death_rate**2 * self.eta
        return (t1 - t2) / (t3 + t4)

    def compute_determinant(self):
        t1 = self.prey_growth_rate**2 * self.predator_death_rate * self.prey_death_rate * self.eta
        t2 = self.B * self.prey_growth_rate * self.predator_death_rate * self.prey_death_rate**2 * self.eta
        t3 = self.prey_growth_rate**3 * self.predator_growth_rate
        t4 = self.prey_growth_rate * self.prey_death_rate * self.eta
        t5 = self.B * self.prey_death_rate**2 * self.eta
        return (t1 + t2 + t3) / (t4 + t5)

    def fixed_point_type(self):
        determinant = self.compute_determinant()
        if determinant < 0:
            return "saddle"
        elif determinant == 0:
            return "centre"
        else:
            trace = self.compute_trace()
            if trace < -10**-10:
                return "stable"
            elif trace > 10**-10:
                return "unstable"
            else:
                return "centre"
def draw_dynamics_plots(prey_population_dynamics, predator_population_dynamics, fixed_point, fixed_point_type):

    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5, hspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(predator_population_dynamics, 'r-', label='predator')
    ax1.plot(prey_population_dynamics, 'b-', label='prey')
    ax1.set_title("Dynamics in time")
    ax1.set_xlabel("time")
    ax1.grid()
    ax1.legend(loc='best')

    ax2.plot(prey_population_dynamics, predator_population_dynamics, color="blue")
    ax2.set_xlabel("prey")
    ax2.set_ylabel("predator")
    ax2.set_title("Phase space")
    ax2.grid()

    facecolors, edgecolors = setup_fixed_point(fixed_point_type)

    ax2.scatter(fixed_point[0], fixed_point[1], facecolors=facecolors, edgecolors=edgecolors, s=55)

    plt.show()

def setup_fixed_point(fixed_point_type):
    facecolors = 'g'
    edgecolors = 'g'
    if fixed_point_type is 'stable':
        facecolors = 'r'
        edgecolors = 'r'
    elif fixed_point_type is 'unstable':
        facecolors = 'none'
        edgecolors = 'r'
    elif fixed_point_type is 'center':
        facecolors = 'y'
        edgecolors = 'y'

    return facecolors, edgecolors




def main():
    args = setup_parser()

    model = PredatorPreyModel(prey=args.prey, predators=args.predators, prey_growth_rate=args.prey_growth_rate, prey_death_rate=args.prey_death_rate,
                           predator_growth_rate=args.predator_growth_rate, predator_death_rate=args.predator_death_rate, B=args.allee_constant, eta=args.proportionality_constant)


    populations_alley_competition = model.runge_kutta(model.prey_change_alley_competition, model.predator_change_alley_competition, args.time_step, args.time_interval)

    prey_population_dynamics = populations_alley_competition['prey']
    predator_population_dynamics = populations_alley_competition['predator']
    fixed_point = model.compute_fixed_point()
    fixed_point_type = model.fixed_point_type()
    print ("The fixed point is ({:.2f}, {:.2f}) and it is {}.".format(fixed_point[0], fixed_point[1], fixed_point_type))
    draw_dynamics_plots(prey_population_dynamics, predator_population_dynamics, fixed_point, fixed_point_type)

if __name__ == "__main__":
    main()

