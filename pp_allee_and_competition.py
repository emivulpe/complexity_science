import argparse
import matplotlib.pyplot as plt


def setup_parser():

    """
    A function to setup a command-line arguments parser.

    :return: the parsed command line arguments
    """

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
    parser.add_argument("-m", dest="method", type=str, default="rc", help="the method to calculate the differential equations for the predator and the prey population change", choices=["rc", "ie"])

    return parser.parse_args()


class PredatorPreyModel(object):
    """A class for the predator-prey model with Alee effect and intraspecific competition in the predator population"""


    def __init__(self, predators=5, prey=10,
                 prey_growth_rate=1.0, prey_death_rate=0.1,
                 predator_death_rate=1.0, predator_growth_rate=0.075,
                 eta=0.1, B=4):
        """
        A initializing method for the predator-prey model.

        :param predators: the initial number of predators
        :param prey: the initial number of prey
        :param prey_growth_rate: the prey growth rate (r1)
        :param prey_death_rate: the prey death rate (delta)
        :param predator_death_rate: the predator death rate (r2)
        :param predator_growth_rate: the predator growth rate (theta)
        :param eta: the proportionality constant
        :param B: the Allee effect constant
        """

        self.predators = float(predators)
        self.prey = float(prey)
        self.prey_growth_rate = float(prey_growth_rate)
        self.prey_death_rate = float(prey_death_rate)
        self.predator_death_rate = float(predator_death_rate)
        self.predator_growth_rate = float(predator_growth_rate)
        self.B = float(B)
        self.eta = float(eta)

    def prey_change(self, prey, predators):
        """
        Calculates the change in prey population.

        :param prey: the number of prey
        :param predators: the number of predators

        :return: the change in prey population
        """

        return prey * self.prey_growth_rate - self.prey_death_rate * predators * prey

    def predator_change(self, prey, predators):
        """
        Calculates the change in predator population with terms for intraspecific competition and Allee effect.

        :param prey: the number of prey
        :param predators: the number of predators

        :return: the change in predator population
        """

        return (predators * self.predator_growth_rate * prey * (
        predators / (predators + self.B) * (1 - predators / (self.eta * prey)))) - self.predator_death_rate * predators

    def improved_euler(self,
                       time_step=0.02, time_interval=100):
        """
        Improved Euler method for the change functions of the predator and prey population.

        :param time_step: the time step for the Euler method
        :param time_interval: the time interval for the Euler method

        :return: A dictionary containing the prey and the predator population history, calculated with the Euler method
        for the specified time step and time interval.
        """

        predator_history = []
        prey_history = []

        for i in range(time_interval):
            xk_1 = self.prey_change(self.prey, self.predators) * time_step
            yk_1 = self.predator_change(self.prey, self.predators) * time_step
            xk_2 = self.prey_change(self.prey + xk_1, self.predators + yk_1) * time_step
            yk_2 = self.predator_change(self.prey + xk_1, self.predators + yk_1) * time_step

            self.prey = self.prey + (xk_1 + xk_2) / 2
            self.predators = self.predators + (yk_1 + yk_2) / 2

            predator_history.append(self.predators)
            prey_history.append(self.prey)

        return {'predator': predator_history, 'prey': prey_history}

    def runge_kutta(self, delta_time=0.02, iterations=100):
        """
        Runge-Kutta method for the change functions of the predator and prey population.

        :param time_step: the time step for the Euler method
        :param time_interval: the time interval for the Euler method

        :return: A dictionary containing the prey and the predator population history, calculated with the Runge-Kutta method
        for the specified time step and time interval.
        """

        predator_history = []
        prey_history = []

        for i in range(iterations):
            xk_1 = self.prey_change(self.prey, self.predators) * delta_time
            yk_1 = self.predator_change(self.prey, self.predators) * delta_time
            xk_2 = self.prey_change(self.prey + 0.5 * xk_1, self.predators + 0.5 * yk_1) * delta_time
            yk_2 = self.predator_change(self.prey + 0.5 * xk_1, self.predators + 0.5 * yk_1) * delta_time
            xk_3 = self.prey_change(self.prey + 0.5 * xk_2, self.predators + 0.5 * yk_2) * delta_time
            yk_3 = self.predator_change(self.prey + 0.5 * xk_2, self.predators + 0.5 * yk_2) * delta_time
            xk_4 = self.prey_change(self.prey + xk_3, self.predators + yk_3) * delta_time
            yk_4 = self.predator_change(self.prey + xk_3, self.predators + yk_3) * delta_time

            self.prey = self.prey + (xk_1 + 2 * xk_2 + 2 * xk_3 + xk_4) / 6
            self.predators = self.predators + (yk_1 + 2 * yk_2 + 2 * yk_3 + yk_4) / 6

            predator_history.append(self.predators)
            prey_history.append(self.prey)

        return {'predator': predator_history, 'prey': prey_history}

    def compute_predator_fp(self):
        """
        A method to compute the predator portion of the fixed point (P*)..
        :return: the predator portion of the fixed point
        """

        return (self.prey_growth_rate / (self.prey_death_rate * self.eta)) + ((self.predator_death_rate / self.predator_growth_rate) * ((self.prey_growth_rate + (self.prey_death_rate * self.B)) / self.prey_growth_rate))

    def compute_prey_fp(self):
        """
        A method to compute the prey portion of the fixed point (N*).
        :return: the prey portion of the fixed point
        """

        return self.prey_growth_rate / self.prey_death_rate

    def compute_fixed_point(self):
        """
        A method to compute the fixed point for the model.
        :return: the fixed point (P*, N*)
        """

        return (self.compute_predator_fp(), self.compute_prey_fp())

    def compute_trace(self):
        """
        A method to compute the trace of the Jacobian matrix of the model.

        :return: the trace of the Jacobian matrix
        """

        t1 = self.B * self.predator_death_rate*self.prey_death_rate**2*self.eta
        t2 = self.prey_growth_rate**2 * self.predator_growth_rate
        t3 = self.prey_growth_rate * self.prey_death_rate * self.eta
        t4 = self.B * self.prey_death_rate**2 * self.eta
        return (t1 - t2) / (t3 + t4)

    def compute_determinant(self):
        """
        A method to compute the determinant of the Jacobian matrix of the model.

        :return: the determinant of the Jacobian matrix
        """

        t1 = self.prey_growth_rate**2 * self.predator_death_rate * self.prey_death_rate * self.eta
        t2 = self.B * self.prey_growth_rate * self.predator_death_rate * self.prey_death_rate**2 * self.eta
        t3 = self.prey_growth_rate**3 * self.predator_growth_rate
        t4 = self.prey_growth_rate * self.prey_death_rate * self.eta
        t5 = self.B * self.prey_death_rate**2 * self.eta
        return (t1 + t2 + t3) / (t4 + t5)

    def fixed_point_type(self):
        """
        A method to determine the type of the fixed point based on the determinant and the trace of the Jacobian matrix of the model.

        :return: the type of the fixed point: stable, center or unstable
        """

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
    """
    A method to draw the plots for the dynamics of predator and prey populations.

    :param prey_population_dynamics: the history of the prey population change
    :param predator_population_dynamics: the history of the predator population change
    :param fixed_point: the fixed point
    :param fixed_point_type: the type of the fixed point
    """

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
    """
    A method to set up how the fixed point looks in a plot.

    :param fixed_point_type: the type of the fixed point
    :return: the fill color of the point on the graph and the edgecolor of the point
    Unstable point has red border and no fill color.
    Center points are filled yellow.
    Stable points are filled red.
    """

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
    """
    The entry point for the module.
    """

    # Setup the command line argument parser
    args = setup_parser()

    # Setup the model
    model = PredatorPreyModel(prey=args.prey, predators=args.predators, prey_growth_rate=args.prey_growth_rate, prey_death_rate=args.prey_death_rate,
                           predator_growth_rate=args.predator_growth_rate, predator_death_rate=args.predator_death_rate, B=args.allee_constant, eta=args.proportionality_constant)

    # Determine which method to use for the differential equations
    if args.method is "rc":
        populations_alley_competition = model.runge_kutta(args.time_step, args.time_interval)
        print ("Using Runge-Kutta method for the differential equations with time step {} for time interval [0, {}]!".format(args.time_step, args.time_interval))
    else:
        print ("Using Improved Euler method for the differential equations with time step {} for time interval [0, {}]!".format(args.time_step, args.time_interval))
        populations_alley_competition = model.improved_euler(args.time_step, args.time_interval)

    # Compute the fixed point and determine its type
    fixed_point = model.compute_fixed_point()
    fixed_point_type = model.fixed_point_type()
    print("The fixed point is ({:.2f}, {:.2f}) and it is {}.".format(fixed_point[0], fixed_point[1], fixed_point_type))

    # Plot the results
    prey_population_dynamics = populations_alley_competition['prey']
    predator_population_dynamics = populations_alley_competition['predator']
    draw_dynamics_plots(prey_population_dynamics, predator_population_dynamics, fixed_point, fixed_point_type)


if __name__ == "__main__":
    main()

