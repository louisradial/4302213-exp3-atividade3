from collections.abc import Callable
from typing import Tuple
import pandas
import math
from math import pi
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import interpolate
import scipy.odr as odr

SAVE_FIGURE = True
def save_figure(fig, of):
    if of is not None:
        print("[save] saving figure at " + of)
        if SAVE_FIGURE:
            fig.savefig(of, dpi=500)
            print("[save] figure saved at " + of)

# error bar fmt
kwargs_errorbar = {
    "ecolor":"black" ,
    "capsize": 0.5,
    "elinewidth": 1,
    "ms": 3,
    "marker": "o",
    "markerfacecolor": "black",
    "markeredgecolor": "black",
    "ls": "none",
}

@dataclass
class ModelParameters:
    peak: float = 2.12e-2  # Hz
    center: float = 0.975  # V
    width: float = -2e-2
    offset: float = 0  # V

    def to_numpy(self) -> np.ndarray:
        return np.array([self.peak, self.center, self.width, self.offset])

    @staticmethod
    def from_numpy(params: np.ndarray) -> "ModelParameters":
        peak = params[0]
        center = params[1]
        width = params[2]
        offset = params[3]
        return ModelParameters(peak, center, width, offset)

ModelFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]

def gaussian(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    return params[0]*np.exp(- np.square((x - params[1])/params[2])) + params[3]

@dataclass
class ModelData:
    x: np.ndarray
    sx: np.ndarray
    y: np.ndarray
    sy: np.ndarray

    @staticmethod
    def from_csv(filename: str, dataset: int) -> "ModelData":
        df = pandas.read_csv(filename, header=0, sep=",")
        x = df.iloc[:, 0].to_numpy()
        sx = df.iloc[:, 1].to_numpy()/2
        y = df.iloc[:, 2*dataset].to_numpy()
        sy = np.abs(df.iloc[:, 2*dataset+1].to_numpy())
        return ModelData(x, sx, y, sy)

@dataclass
class ModelFittedCurve:
    x: np.ndarray
    y: np.ndarray
    residues: np.ndarray
    s_residues: np.ndarray

    def plot_fit(self, axes, color):
        axes.plot(self.x, self.y, c=color)

    def plot_residues(self, axes, x:np.ndarray, sx:np.ndarray):
        axes.errorbar(x=x, xerr=sx, y=self.residues, yerr=self.s_residues, **kwargs_errorbar)
        axes.set_xlim(min(x), max(x))

class Model:
    def __init__(self, fn: ModelFunction) -> None:
        self.function = fn

    def fit_curve(self, data: ModelData, estimate: ModelParameters) -> None:
        self.odr_fit(data, estimate)
        residues, s_residues = self.residues(data)
        print(np.shape(residues), np.shape(s_residues))
        x_fit = np.linspace(min(data.x), max(data.x), num=2000)
        y_fit = self.function(self.parameters.to_numpy(), x_fit)
        self.fit = ModelFittedCurve(x_fit, y_fit, residues, s_residues)

    def odr_fit(self, data: ModelData, estimate: ModelParameters) -> None:
        odr_data = odr.RealData(x=data.x, y=data.y, sx=data.sx, sy=data.sy)
        odr_model = odr.Model(self.function)
        odr_output = odr.ODR(odr_data, odr_model, estimate.to_numpy(), maxit=20000).run()
        self.parameters = ModelParameters.from_numpy(odr_output.beta)
        self.s_parameters = ModelParameters.from_numpy(odr_output.sd_beta)
        print("Beta: ", self.parameters)
        print("Beta std error", self.s_parameters)
        print("Reason for Halting:")
        for r in odr_output.stopreason:
            print('  %s' % r)
        print("χ² = ", odr_output.sum_square)
        print("NGL = ", np.size(data.x) - np.size(odr_output.beta))

    def residues(self, data: ModelData):
        x_var = np.array([data.x + data.sx, data.x - data.sx])
        y_prd = self.function(self.parameters.to_numpy(), data.x)
        y_var = self.function(self.parameters.to_numpy(), x_var)
        s_res = np.abs(y_var - y_prd).mean()
        s_res = np.hypot(data.sy, s_res)
        return data.y - y_prd, s_res

@dataclass
class Field:
    position: np.ndarray
    intensity: np.ndarray

class ElectricField(Field):
    @staticmethod
    def from_csv(filename: str, voltage: float) -> "ElectricField":
        df = pandas.read_csv(filename, sep="\t")
        x = df.iloc[:, 0].to_numpy()*1e-3
        y = df.iloc[:, 1].to_numpy()/voltage
        return ElectricField(x, y)

    def interpolate(self, x: float) -> float:
        f = interpolate.interp1d(self.position, self.intensity)
        if x > min(self.position) and x < max(self.position):
            return f(x)
        else:
            return 0

    def output(self, positions: np.ndarray) -> np.ndarray:
        res = []
        for x in positions:
            res.append(self.interpolate(x))
        return np.array(res)

class MagneticField(Field):
    @staticmethod
    def from_csv(filename: str, current: float, offset: float) -> "MagneticField":
        df = pandas.read_csv(filename, sep=",")
        x = df.iloc[:, 0].to_numpy() - offset
        y = df.iloc[:, 1].to_numpy()/current
        return MagneticField(x, y)

def alfa_beta(e_field: ElectricField, b_field_parameters: ModelParameters, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    alfa = e_field.output(positions)
    beta = gaussian(b_field_parameters.to_numpy(), positions)
    return alfa, beta

if __name__ == "__main__":
    # carrega simulação do FEMM
    e = ElectricField.from_csv("elétrico", 10.0)
    # carrega dados do sensor Hall, não usar mais.
    # b = MagneticField.from_csv("magnético-49.7mA", 49.7e-3, 0.975-8.81e-2)

    # ajusta dados do sensor Hall, normalizados pela corrente
    parameters = []
    sparameters = []
    model = Model(gaussian)
    for i in range (1, 5):
        data = ModelData.from_csv("magnético", i)
        estimate = ModelParameters()
        model.fit_curve(data, estimate)
        parameters.append(model.parameters)
        sparameters.append(model.s_parameters)
        name = "Dependência do campo magnético normalizado pela corrente com a posição"
        fig = plt.figure(name, layout="constrained")
        fig.suptitle(name)
        gs = GridSpec(2,1,figure=fig, height_ratios=[3,1])
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax0.errorbar(x=data.x, xerr=data.sx, y=data.y, yerr=np.square(data.sy), **kwargs_errorbar)
        model.fit.plot_fit(ax0, "#cba6f7")
        ax0.set(ylabel=r"$\beta(x)$ (T/A)", )
        ax0.grid(which='major', alpha=0.8)
        ax0.grid(which='minor', alpha=0.2)
        model.fit.plot_residues(ax1, data.x, data.sx)
        ax0.tick_params(labelbottom=False)
        ax1.grid(which='major', alpha=0.8)
        ax1.grid(which='minor', alpha=0.2)
        ax1.set(xlabel="$x$ (m)")
        plt.show()
        save_figure(fig, "beta{0}".format(i))

    # escolher parâmetros
    b_params = parameters[0]
    b_params.center = 8.81e-2 # centro da bobina fica 8.81cm do início do feixe

    # calcular alfa e beta nas posições desejadas
    alfa, beta = alfa_beta(e,b_params, np.arange(0, 280e-3, 1e-3))
