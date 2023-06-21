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
    def __init__(self, filename, voltage) -> None:
        df = pandas.read_csv(filename, sep=" ")
        x = df.iloc[:, 0].to_numpy()
        y = df.iloc[:, 1].to_numpy()/voltage
        self.position = x
        self.intensity = y

class MagneticField(Field):
    def __init(self, filename, current) -> None:
        df = pandas.read_csv(filename, sep=",")
        x = df.iloc[:, 0].to_numpy()
        y = df.iloc[:, 1].to_numpy()/current
        self.position = x
        self.intensity = y
