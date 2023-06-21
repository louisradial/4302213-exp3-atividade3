import pandas
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

@dataclass
class Field:
    position: np.ndarray
    intensity: np.ndarray

    def interpolate(self, x: float) -> float:
        f = interpolate.interp1d(self.position, self.intensity)
        if x > min(self.position) and x < max(self.position):
            return f(x)
        else:
            return 0

class ElectricField(Field):
    @staticmethod
    def from_csv(filename: str, voltage: float) -> "ElectricField":
        df = pandas.read_csv(filename, sep="\t")
        x = df.iloc[:, 0].to_numpy()*1e-3
        y = df.iloc[:, 1].to_numpy()/voltage
        return ElectricField(x, y)

class MagneticField(Field):
    @staticmethod
    def from_csv(filename: str, current: float, offset: float) -> "MagneticField":
        df = pandas.read_csv(filename, sep=",")
        x = df.iloc[:, 0].to_numpy() - offset
        y = df.iloc[:, 1].to_numpy()/current
        return MagneticField(x, y)

def output(e_field: ElectricField, b_field: MagneticField, positions: np.ndarray) -> pandas.DataFrame:
    outpute = []
    outputb = []
    for x in positions:
        outpute.append(e_field.interpolate(x))
        outputb.append(b_field.interpolate(x))
    plt.plot(positions, outputb)
    plt.show()
    return pandas.DataFrame(data={'x (mm)': positions, 'alfa (1/m)': outpute, 'beta (T/A)': outputb})

if __name__ == "__main__":
    e = ElectricField.from_csv("elétrico", 10.0)
    b = MagneticField.from_csv("magnético-49.7mA", 49.7e-3, 0.975-8.81e-2)
    df = output(e, b, np.arange(0, 280e-3, 1e-3))
    df.to_csv("out.csv")

