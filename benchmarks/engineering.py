"""Real-world engineering design benchmarks from the RE suite."""

import numpy as np

from .base import RowWisePhysicalBenchmark


def _violations(values):
    values = np.asarray(values, dtype=np.float64)
    return np.where(values < 0.0, -values, 0.0)


class FourBarTruss(RowWisePhysicalBenchmark):
    ideal_point = np.array([1237.84142, 0.00276142375], dtype=np.float64)
    nadir_point = np.array([2886.36956, 0.04], dtype=np.float64)

    def __init__(self):
        lower = [1.0, np.sqrt(2.0), np.sqrt(2.0), 1.0]
        upper = [3.0, 3.0, 3.0, 3.0]
        super().__init__("Four Bar Truss Design", 4, 2, lower, upper)

    def _evaluate_physical(self, x):
        x1, x2, x3, x4 = x
        force, elastic_modulus, length = 10.0, 2.0e5, 200.0
        weight = length * (
            2.0 * x1 + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4
        )
        displacement = (force * length / elastic_modulus) * (
            2.0 / x1
            + 2.0 * np.sqrt(2.0) / x2
            - 2.0 * np.sqrt(2.0) / x3
            + 2.0 / x4
        )
        return [weight, displacement]


class PressureVessel(RowWisePhysicalBenchmark):
    ideal_point = np.array([15.9018008, 0.0], dtype=np.float64)
    nadir_point = np.array([5852.05897, 1288669.78], dtype=np.float64)

    def __init__(self):
        super().__init__(
            "Pressure Vessel Design",
            4,
            2,
            [1.0, 1.0, 10.0, 10.0],
            [100.0, 100.0, 200.0, 240.0],
        )

    def _evaluate_physical(self, x):
        x1 = 0.0625 * int(np.round(x[0]))
        x2 = 0.0625 * int(np.round(x[1]))
        x3, x4 = x[2], x[3]
        cost = (
            (0.6224 * x1 * x3 * x4)
            + (1.7781 * x2 * x3 * x3)
            + (3.1661 * x1 * x1 * x4)
            + (19.84 * x1 * x1 * x3)
        )
        constraints = _violations(
            [
                x1 - 0.0193 * x3,
                x2 - 0.00954 * x3,
                np.pi * x3 * x3 * x4
                + (4.0 / 3.0) * (np.pi * x3 * x3 * x3)
                - 1296000.0,
            ]
        )
        return [cost, constraints.sum()]


class HatchCover(RowWisePhysicalBenchmark):
    ideal_point = np.array([60.5, 0.0], dtype=np.float64)
    nadir_point = np.array([481.608089, 44.2819048], dtype=np.float64)

    def __init__(self):
        super().__init__(
            "Hatch Cover Design",
            2,
            2,
            [0.5, 0.5],
            [4.0, 50.0],
        )

    def _evaluate_physical(self, x):
        x1, x2 = x
        elastic_modulus = 700000.0
        sigma_b = 4500.0 / (x1 * x2)
        constraints = _violations(
            [
                1.0 - sigma_b / 700.0,
                1.0 - (1800.0 / x2) / 450.0,
                1.0
                - ((56.2 * 10000.0) / (elastic_modulus * x1 * x2**2)) / 1.5,
                1.0 - sigma_b / (elastic_modulus * x1**2 / 100.0),
            ]
        )
        noise = np.random.normal(0.0, 1e-6)
        return [x1 + 120.0 * x2, constraints.sum() + noise]


class DiscBrake(RowWisePhysicalBenchmark):
    ideal_point = np.array([-0.721525, 1.13907204, 0.0], dtype=np.float64)
    nadir_point = np.array([5.3067, 9.06681054, 4323470580.0], dtype=np.float64)

    def __init__(self):
        super().__init__(
            "Disc Brake Design",
            4,
            3,
            [55.0, 75.0, 1000.0, 11.0],
            [80.0, 110.0, 3000.0, 20.0],
        )

    def _evaluate_physical(self, x):
        x1, x2, x3, x4 = x
        radius_difference = x2**2 - x1**2
        cubic_difference = x2**3 - x1**3
        mass = 4.9e-5 * radius_difference * (x4 - 1.0)
        stopping_time = (
            9.82e6
            * radius_difference
            / (x3 * x4 * cubic_difference)
        )
        constraints = _violations(
            [
                x2 - x1 - 20.0,
                0.4 - x3 / (3.14 * radius_difference),
                1.0
                - 2.22e-3 * x3 * cubic_difference / radius_difference**2,
                2.66e-2
                * x3
                * x4
                * cubic_difference
                / radius_difference
                - 900.0,
            ]
        )
        return [mass, stopping_time, constraints.sum()]


class SpeedReducer(RowWisePhysicalBenchmark):
    ideal_point = np.array([2352.34611, 694.233587, 0.0], dtype=np.float64)
    nadir_point = np.array([7098.54658, 1695.96386, 397.358905], dtype=np.float64)

    def __init__(self):
        super().__init__(
            "Speed Reducer Design",
            7,
            3,
            [2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0],
            [3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5],
        )

    def _evaluate_physical(self, x):
        x1, x2, x3_raw, x4, x5, x6, x7 = x
        x3 = np.round(x3_raw)
        weight = (
            0.7854
            * x1
            * x2**2
            * ((10.0 * x3**2) / 3.0 + 14.933 * x3 - 43.0934)
            - 1.508 * x1 * (x6**2 + x7**2)
            + 7.477 * (x6**3 + x7**3)
            + 0.7854 * (x4 * x6**2 + x5 * x7**2)
        )
        stress = np.sqrt(((745.0 * x4) / (x2 * x3)) ** 2 + 1.69e7) / (
            0.1 * x6**3
        )
        other_stress = -np.sqrt(
            ((745.0 * x5) / (x2 * x3)) ** 2 + 1.575e8
        ) / (0.1 * x7**3) + 1100.0
        constraints = _violations(
            [
                -1.0 / (x1 * x2**2 * x3) + 1.0 / 27.0,
                -1.0 / (x1 * x2**2 * x3**2) + 1.0 / 397.5,
                -x4**3 / (x2 * x3 * x6**4) + 1.0 / 1.93,
                -x5**3 / (x2 * x3 * x7**4) + 1.0 / 1.93,
                -x2 * x3 + 40.0,
                -x1 / x2 + 12.0,
                -5.0 + x1 / x2,
                -1.9 + x4 - 1.5 * x6,
                -1.9 + x5 - 1.1 * x7,
                -stress + 1300.0,
                other_stress,
            ]
        )
        return [weight, stress, constraints.sum()]


class GearTrain(RowWisePhysicalBenchmark):
    ideal_point = np.array([7.89473684e-05, 12.0, 0.0], dtype=np.float64)
    nadir_point = np.array([5.931, 49.0, 0.355720675], dtype=np.float64)

    def __init__(self):
        super().__init__(
            "Gear Train Design",
            4,
            3,
            [12.0] * 4,
            [60.0] * 4,
        )

    def _evaluate_physical(self, x):
        x1, x2, x3, x4 = np.round(x)
        ratio_error = np.abs(6.931 - (x3 / x1) * (x4 / x2))
        maximum_gear = max([x1, x2, x3, x4])
        constraint = _violations([0.5 - ratio_error / 6.931])[0]
        return [ratio_error, maximum_gear, constraint]


class RocketInjector(RowWisePhysicalBenchmark):
    ideal_point = np.array([0.00889341422, 0.00488000019, -0.4315], dtype=np.float64)
    nadir_point = np.array([1.002, 1.09751726, 1.09380596], dtype=np.float64)

    def __init__(self):
        super().__init__("Rocket Injector Design", 4, 3)

    def _evaluate_physical(self, x):
        alpha, height, opening, temperature = x
        first = (
            0.692
            + 0.477 * alpha
            - 0.687 * height
            - 0.080 * opening
            - 0.0650 * temperature
            - 0.167 * alpha**2
            - 0.0129 * height * alpha
            + 0.0796 * height**2
            - 0.0634 * opening * alpha
            - 0.0257 * opening * height
            + 0.0877 * opening**2
            - 0.0521 * temperature * alpha
            + 0.00156 * temperature * height
            + 0.00198 * temperature * opening
            + 0.0184 * temperature**2
        )
        second = (
            0.153
            - 0.322 * alpha
            + 0.396 * height
            + 0.424 * opening
            + 0.0226 * temperature
            + 0.175 * alpha**2
            + 0.0185 * height * alpha
            - 0.0701 * height**2
            - 0.251 * opening * alpha
            + 0.179 * opening * height
            + 0.0150 * opening**2
            + 0.0134 * temperature * alpha
            + 0.0296 * temperature * height
            + 0.0752 * temperature * opening
            + 0.0192 * temperature**2
        )
        third = (
            0.370
            - 0.205 * alpha
            + 0.0307 * height
            + 0.108 * opening
            + 1.019 * temperature
            - 0.135 * alpha**2
            + 0.0141 * height * alpha
            + 0.0998 * height**2
            + 0.208 * opening * alpha
            - 0.0301 * opening * height
            - 0.226 * opening**2
            + 0.353 * temperature * alpha
            - 0.0497 * temperature * opening
            - 0.423 * temperature**2
            + 0.202 * height * alpha**2
            - 0.281 * opening * alpha**2
            - 0.342 * height**2 * alpha
            - 0.245 * height**2 * opening
            + 0.281 * opening**2 * height
            - 0.184 * temperature**2 * alpha
            - 0.281 * height * alpha * opening
        )
        return [first, second, third]


class CarSideImpact(RowWisePhysicalBenchmark):
    ideal_point = np.array([15.5760643, 3.58525, 10.6106444, 0.0], dtype=np.float64)
    nadir_point = np.array([42.7680062, 4.42724878, 13.0913557, 9.44926882], dtype=np.float64)

    def __init__(self):
        super().__init__(
            "Car Side Impact Design",
            7,
            4,
            [0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4],
            [1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2],
        )

    def _evaluate_physical(self, x):
        x1, x2, x3, x4, x5, x6, x7 = x
        weight = (
            1.98
            + 4.9 * x1
            + 6.67 * x2
            + 6.98 * x3
            + 4.01 * x4
            + 1.78 * x5
            + 0.00001 * x6
            + 2.73 * x7
        )
        pubic_force = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        abdomen = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        viscous = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        constraints = _violations(
            [
                1.0 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3),
                0.32
                - (
                    0.261
                    - 0.0159 * x1 * x2
                    - 0.06486 * x1
                    - 0.019 * x2 * x7
                    + 0.0144 * x3 * x5
                    + 0.0154464 * x6
                ),
                0.32
                - (
                    0.214
                    + 0.00817 * x5
                    - 0.045195 * x1
                    - 0.0135168 * x1
                    + 0.03099 * x2 * x6
                    - 0.018 * x2 * x7
                    + 0.007176 * x3
                    + 0.023232 * x3
                    - 0.00364 * x5 * x6
                    - 0.018 * x2**2
                ),
                0.32
                - (0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2**2),
                32.0
                - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7),
                32.0
                - (33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728),
                32.0 - (46.36 - 9.9 * x2 - 4.4505 * x1),
                4.0 - pubic_force,
                9.9 - abdomen,
                15.7 - viscous,
            ]
        )
        return [weight, pubic_force, 0.5 * (abdomen + viscous), constraints.sum()]


class ConceptualMarine(RowWisePhysicalBenchmark):
    ideal_point = np.array([-2756.25904, 3962.55784, 1947.88086, 0.0], dtype=np.float64)
    nadir_point = np.array([-663.415705, 15811.2376, 5195.43503, 13.0265363], dtype=np.float64)

    def __init__(self):
        super().__init__(
            "Conceptual Marine Design",
            6,
            4,
            [150.0, 20.0, 13.0, 10.0, 14.0, 0.63],
            [274.32, 32.31, 25.0, 11.71, 18.0, 0.75],
        )

    def _evaluate_physical(self, x):
        length, breadth, depth, draft, speed, block_coefficient = x
        displacement = (
            1.025 * length * breadth * draft * block_coefficient
        )
        velocity = 0.5144 * speed
        gravity = 9.8065
        froude = velocity / np.power(gravity * length, 0.5)
        a = (
            4977.06 * block_coefficient * block_coefficient
            - 8105.61 * block_coefficient
            + 4456.51
        )
        b = (
            -10847.2 * block_coefficient * block_coefficient
            + 12817.0 * block_coefficient
            - 6960.32
        )
        power = (
            np.power(displacement, 2.0 / 3.0) * np.power(speed, 3.0)
        ) / (a + b * froude)
        outfit = (
            1.0
            * np.power(length, 0.8)
            * np.power(breadth, 0.6)
            * np.power(depth, 0.3)
            * np.power(block_coefficient, 0.1)
        )
        steel = (
            0.034
            * np.power(length, 1.7)
            * np.power(breadth, 0.7)
            * np.power(depth, 0.4)
            * np.power(block_coefficient, 0.5)
        )
        machinery = 0.17 * np.power(power, 0.9)
        light_ship = steel + outfit + machinery
        ship_cost = 1.3 * (
            (2000.0 * np.power(steel, 0.85))
            + (3500.0 * outfit)
            + (2400.0 * np.power(power, 0.8))
        )
        deadweight = displacement - light_ship
        running_cost = 40000.0 * np.power(deadweight, 0.3)
        sea_days = (5000.0 / 24.0) * speed
        daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
        fuel_cost = 1.05 * daily_consumption * sea_days * 100.0
        port_cost = 6.3 * np.power(deadweight, 0.8)
        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous = 2.0 * np.power(deadweight, 0.5)
        cargo_deadweight = deadweight - fuel_carried - miscellaneous
        port_days = 2.0 * ((cargo_deadweight / 8000.0) + 0.5)
        trips_per_year = 350.0 / (sea_days + port_days)
        voyage_cost = (fuel_cost + port_cost) * trips_per_year
        annual_cost = 0.2 * ship_cost + running_cost + voyage_cost
        annual_cargo = cargo_deadweight * trips_per_year
        constraints = _violations(
            [
                length / breadth - 6.0,
                -length / depth + 15.0,
                -length / draft + 19.0,
                0.45 * deadweight**0.31 - draft,
                0.7 * depth + 0.7 - draft,
                500000.0 - deadweight,
                deadweight - 3000.0,
                0.32 - froude,
                (
                    0.53 * draft
                    + (
                        (0.085 * block_coefficient - 0.002)
                        * breadth
                        * breadth
                    )
                    / (draft * block_coefficient)
                    - (1.0 + 0.52 * depth)
                )
                - 0.07 * breadth,
            ]
        )
        return [
            annual_cost / annual_cargo,
            light_ship,
            -annual_cargo,
            constraints.sum(),
        ]


class WaterResource(RowWisePhysicalBenchmark):
    ideal_point = np.array(
        [63840.2774, 30.0, 285346.896, 183749.967, 7.22222222, 0.0],
        dtype=np.float64,
    )
    nadir_point = np.array(
        [76347.3928, 1350.0, 2853468.96, 15725809.9, 346735.0, 93789.3225],
        dtype=np.float64,
    )

    def __init__(self):
        super().__init__(
            "Water Resource Planning",
            3,
            6,
            [0.01, 0.01, 0.01],
            [0.45, 0.10, 0.10],
        )

    def _evaluate_physical(self, x):
        x1, x2, x3 = x
        objectives = np.zeros(6, dtype=np.float64)
        objectives[0] = 106780.37 * (x2 + x3) + 61704.67
        objectives[1] = 3000.0 * x1
        objectives[2] = 305700.0 * 2289.0 * x2 / (0.06 * 2289.0) ** 0.65
        objectives[3] = 250.0 * 2289.0 * np.exp(-39.75 * x2 + 9.9 * x3 + 2.74)
        objectives[4] = 25.0 * (1.39 / (x1 * x2) + 4940.0 * x3 - 80.0)
        constraints = _violations(
            [
                1.0 - (0.00139 / (x1 * x2) + 4.94 * x3 - 0.08),
                1.0 - (0.000306 / (x1 * x2) + 1.082 * x3 - 0.0986),
                50000.0 - (12.307 / (x1 * x2) + 49408.24 * x3 + 4051.02),
                16000.0 - (2.098 / (x1 * x2) + 8046.33 * x3 - 696.71),
                10000.0 - (2.138 / (x1 * x2) + 7883.39 * x3 - 705.04),
                2000.0 - (0.417 * x1 * x2 + 1721.26 * x3 - 136.54),
                550.0 - (0.164 / (x1 * x2) + 631.13 * x3 - 54.48),
            ]
        )
        objectives[5] = constraints.sum() + np.random.normal(0.0, 1e-6)
        return objectives


class CarCabDesign(RowWisePhysicalBenchmark):
    ideal_point = np.array(
        [15.5513439, 0.0, 0.0, 0.0907242296, 0.367459287, 0.527364947,
         0.735415465, 0.618676034, 0.660886967],
        dtype=np.float64,
    )
    nadir_point = np.array(
        [42.6835133, 1.33622652, 342.340087, 1.01496018, 1.54367647,
         1.31245897, 1.2206352, 1.17458862, 1.04376916],
        dtype=np.float64,
    )

    def __init__(self):
        super().__init__(
            "Car Cab Design",
            7,
            9,
            [0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4],
            [1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2],
        )

    def _evaluate_physical(self, x):
        x1, x2, x3, x4, x5, x6, x7 = x
        x8 = 0.006 * np.random.normal(0.0, 1.0) + 0.345
        x9 = 0.006 * np.random.normal(0.0, 1.0) + 0.192
        x10 = 10.0 * np.random.normal(0.0, 1.0)
        x11 = 10.0 * np.random.normal(0.0, 1.0)

        objectives = np.zeros(9, dtype=np.float64)
        objectives[0] = (
            1.98
            + 4.9 * x1
            + 6.67 * x2
            + 6.98 * x3
            + 4.01 * x4
            + 1.75 * x5
            + 0.00001 * x6
            + 2.73 * x7
        )
        objectives[1] = max(
            0.0,
            (
                1.16
                - 0.3717 * x2 * x4
                - 0.00931 * x2 * x10
                - 0.484 * x3 * x9
                + 0.01343 * x6 * x10
            ),
        )
        objectives[2] = max(
            0.0,
            (
                0.261
                - 0.0159 * x1 * x2
                - 0.188 * x1 * x8
                - 0.019 * x2 * x7
                + 0.0144 * x3 * x5
                + 0.87570001 * x5 * x10
                + 0.08045 * x6 * x9
                + 0.00139 * x8 * x11
                + 0.00001575 * x10 * x11
            )
            / 0.32,
        )
        objectives[3] = max(
            0.0,
            (
                0.214
                + 0.00817 * x5
                - 0.131 * x1 * x8
                - 0.0704 * x1 * x9
                + 0.03099 * x2 * x6
                - 0.018 * x2 * x7
                + 0.0208 * x3 * x8
                + 0.121 * x3 * x9
                - 0.00364 * x5 * x6
                + 0.0007715 * x5 * x10
                - 0.0005354 * x6 * x10
                + 0.00121 * x8 * x11
                + 0.00184 * x9 * x10
                - 0.018 * x2 * x2
            )
            / 0.32,
        )
        objectives[4] = max(
            0.0,
            (
                0.74
                - 0.61 * x2
                - 0.163 * x3 * x8
                + 0.001232 * x3 * x10
                - 0.166 * x7 * x9
                + 0.227 * x2 * x2
            )
            / 0.32,
        )
        average_intrusion = (
            (
                28.98
                + 3.818 * x3
                - 4.2 * x1 * x2
                + 0.0207 * x5 * x10
                + 6.63 * x6 * x9
                - 7.77 * x7 * x8
                + 0.32 * x9 * x10
            )
            + (
                33.86
                + 2.95 * x3
                + 0.1792 * x10
                - 5.057 * x1 * x2
                - 11.0 * x2 * x8
                - 0.0215 * x5 * x10
                - 9.98 * x7 * x8
                + 22.0 * x8 * x9
            )
            + (
                46.36
                - 9.9 * x2
                - 12.9 * x1 * x8
                + 0.1107 * x3 * x10
            )
        ) / 3.0
        objectives[5] = max(0.0, average_intrusion / 32.0)
        objectives[6] = max(
            0.0,
            (
                4.72
                - 0.5 * x4
                - 0.19 * x2 * x3
                - 0.0122 * x4 * x10
                + 0.009325 * x6 * x10
                + 0.000191 * x11 * x11
            )
            / 4.0,
        )
        objectives[7] = max(
            0.0,
            (
                10.58
                - 0.674 * x1 * x2
                - 1.95 * x2 * x8
                + 0.02054 * x3 * x10
                - 0.0198 * x4 * x10
                + 0.028 * x6 * x10
            )
            / 9.9,
        )
        objectives[8] = max(
            0.0,
            (
                16.45
                - 0.489 * x3 * x7
                - 0.843 * x5 * x6
                + 0.0432 * x9 * x10
                - 0.0556 * x9 * x11
                - 0.000786 * x11 * x11
            )
            / 15.7,
        )
        return objectives
