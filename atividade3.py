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

SAVE_FIGURE = False
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

    def plot(self, title: str, color: str, ylabel: str, of=None):
        fig = plt.figure(title, layout="constrained")
        fig.suptitle(title)
        gs = GridSpec(1,1, figure=fig)
        ax = fig.add_subplot(gs[0])
        ax.plot(self.position, self.intensity, c=color)
        ax.set(ylabel=ylabel)
        ax.set(xlabel=r"$x$ (m)")
        ax.grid(which='major', alpha=0.8)
        ax.grid(which='minor', alpha=0.2)
        plt.show()
        save_figure(fig, of)


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

    @staticmethod
    def from_parameters(parameters: ModelParameters) -> "MagneticField":
        k = 4/math.sqrt(2)
        x = np.linspace(parameters.center - k*parameters.width, parameters.center + k*parameters.width, num = 1000)
        y = gaussian(parameters.to_numpy(), x)
        return MagneticField(x, y)

def alfa_beta(e_field: ElectricField, b_field: MagneticField, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    alfa = e_field.output(positions)
    beta = b_field.output(positions)
    return alfa, beta

if __name__ == "__main__":
    # carrega simulação do FEMM
    e = ElectricField.from_csv("elétrico", 10.0) # 9.2 ficaria bom ein...
    e.plot("Dependência do campo elétrico normalizado pela tensão com a posição", "#fab387", r"$\alpha(x)$ (1/m)", of="alfa")
    # carrega dados do sensor Hall, não usar mais.
    # b = MagneticField.from_csv("magnético-49.7mA", 49.7e-3, 0.975-8.81e-2)

    # ajusta dados do sensor Hall, normalizados pela corrente
    model = Model(gaussian)
    estimate = ModelParameters()
    data = []
    for i in range (1, 5):
        data.append(ModelData.from_csv("magnético", i))
        model.fit_curve(data[i-1], estimate)
        name = "Dependência do campo magnético normalizado pela corrente com a posição"
        fig = plt.figure(name, layout="constrained")
        fig.suptitle(name)
        gs = GridSpec(2,1,figure=fig, height_ratios=[3,1])
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax0.errorbar(x=data[i-1].x, xerr=data[i-1].sx, y=data[i-1].y, yerr=np.square(data[i-1].sy), **kwargs_errorbar)
        model.fit.plot_fit(ax0, "#cba6f7")
        ax0.set(ylabel=r"$\beta(x)$ (T/A)", )
        ax0.grid(which='major', alpha=0.8)
        ax0.grid(which='minor', alpha=0.2)
        model.fit.plot_residues(ax1, data[i-1].x, data[i-1].sx)
        ax0.tick_params(labelbottom=False)
        ax1.grid(which='major', alpha=0.8)
        ax1.grid(which='minor', alpha=0.2)
        ax1.set(xlabel="$x$ (m)")
        plt.show()
        save_figure(fig, "beta{0}".format(i))

    x = np.concatenate((data[0].x, data[1].x, data[2].x, data[3].x))
    sx = np.concatenate((data[0].sx, data[1].sx, data[2].sx, data[3].sx))
    y = np.concatenate((data[0].y, data[1].y, data[2].y, data[3].y))
    sy = np.concatenate((data[0].sy, data[1].sy, data[2].sy, data[3].sy))
    data = ModelData(x, sx, y, sy)
    model.fit_curve(data, estimate)
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
    save_figure(fig, "beta")

    # escolher parâmetros
    b_params = model.parameters
    b_params.center = 8.81e-2# + 6e-3 # centro da bobina fica 8.81cm do início do feixe
    # b_params.offset = 0
    b = MagneticField.from_parameters(b_params)
    # b.plot("Dependência do campo magnético normalizado pela corrente com a posição", "#cba6f7", r"$\beta(x)$ (T/A)")

    # calcular alfa e beta nas posições desejadas
    alfa, beta = alfa_beta(e,b, np.arange(0, 280e-3, 1e-3))
    plt.plot(np.arange(0, 280e-3, 1e-3), beta)
    plt.show()

def traj(vp, i, vac):
    q = 1.6*10**(-19)
    m = 9.11*10**(-31)
    B = []
    E = []
    k = 0
    while k <= 279:
        B.append(i*beta[k])
        E.append(vp*alfa[k])
        k += 1
    vx = [math.sqrt(2*vac*q/m)]
    vz = [0]
    ax = [0]
    az = [0]
    Dt = [0]
    z = [0]
    x = [0]
    j = 0
    while j <= 278:
        j += 1
        ax.append((q/m)*(B[j]*vz[j-1]))
        az.append((q/m)*(E[j]-B[j]*vx[j-1]))
        x.append(j*1e-3)
        vx.append(np.sqrt(vx[j-1]**2 + 2e-3*ax[j]))
        Dt.append(2e-3/(vx[j]+vx[j-1]))
        vz.append(vz[j-1]+az[j]*Dt[j])
        z.append(z[j-1]+vz[j-1]*Dt[j-1]+(az[j]*Dt[j]**2)/2)
    data = [x,z]
    return data

def graf(cols, cor, title):
    name  = ["z", "vp", "vac", "i"]
    data = pandas.read_csv("novo.csv", usecols = cols, names = name, header=0)
    if cols == [0,1,2,3] or cols == [4,5,6,7]:
        yz = []
        z = []
        vp = np.linspace(-30, 30, num = 100)
        vac = np.linspace(400, 1200, num = 100)
        for v in vp:
            simu = traj(v, 0, 1200)
            z.append(1e3*simu[1][-1])
            yz.append(v/1200)
        for v in vac:
            simu = traj(8, 0, v)
            z.append(1e3*simu[1][-1])
            yz.append(8/v)
        y = []
        for i in range(len(data.z)):
            y.append(data.vp[i]/data.vac[i])
        plt.scatter(y, data.z, label = "Dados", color = "black", s = 10)
        plt.plot(yz, z, label = "Simulação", color = cor)
        plt.xlabel(r"$\frac{V_p}{V_{AC}}$")
        plt.ylabel("Deflexão (mm)")
        plt.title(title)
        plt.grid()
        plt.legend()
        plt.show()
    if cols == [8,9,10,11] or cols == [12,13,14,15]:
        yz = []
        z = []
        i = np.linspace(-4e-2, 4e-2, num = 100)
        vac = np.linspace(300, 1200, num = 100)
        for j in i:
            simu = traj(0, j, 1000)
            z.append(1e3*simu[1][-1])
            yz.append(j/math.sqrt(1000))
        for v in vac:
            simu = traj(0, 2e-2, v)
            z.append(1e3*simu[1][-1])
            yz.append(2e-2/math.sqrt(v))
        for v in vac:
            simu = traj(0, -2e-2, v)
            z.append(1e3*simu[1][-1])
            yz.append(-2e-2/math.sqrt(v))
        # for i in range(len(data.z)):
        #     simu = traj(data.vp[i], data.i[i], data.vac[i])
        #     z.append(-1e3*simu[1][-1])
        y = []
        z_ = []
        for i in range(len(data.z)):
            z_.append(-10*data.z[i])
            y.append(data.i[i]/np.sqrt(data.vac[i]))
        plt.scatter(y, z_, label = "Dados", color = "black", s = 10)
        plt.plot(yz, z, label = "Simulação", color = cor)
        plt.xlabel(r"$\frac{i}{\sqrt{V_{AC}}}\;\left(\frac{A}{V^{\frac{1}{2}}}\right)$")
        plt.xticks(np.arange(-0.0010,0.0013, 0.0005))
        plt.ylabel("Deflexão (mm)")
        plt.title(title)
        plt.grid()
        plt.legend()
        plt.show()

graf([0,1,2,3], "purple", "Simulação com campo elétrico")
graf([4,5,6,7], "olive", "Simulação com campo elétrico")
graf([8,9,10,11], "cyan", "Simulação com campo magnético")
graf([12,13,14,15], "pink", "Simulação com campo magnético")
