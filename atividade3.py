import pandas
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

@dataclass
class Field:
    position: np.ndarray
    intensity: np.ndarray

class ElectricField(Field):
    @staticmethod
    def from_csv(filename: str, voltage: float) -> "ElectricField":
        df = pandas.read_csv(filename, sep=" ")
        x = df.iloc[:, 0].to_numpy()
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
    alfa = interpolate.interp1d(e_field.position, e_field.intensity)
    beta = interpolate.interp1d(b_field.position, b_field.intensity)
    outpute = alfa(positions)
    outputb = beta(positions)
    return pandas.DataFrame(data={'x (mm)': positions, 'alfa (1/m)': outpute, 'beta (T/A)': outputb})
