from __future__ import annotations
from typing import Any, Callable, Optional, Union

import functools
import importlib.resources
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.interpolate
import warnings
import unyt

__all__ = [
    "ASTMG173",
    "blackbody",
    "CIE_A",
    "CIE_CMF_X",
    "CIE_CMF_Y",
    "CIE_CMF_Z",
    "CIE_D",
    "CIE_D65",
    "CIE_F",
    "CIE_PHOTOPIC",
    "CIE_S0",
    "CIE_S1",
    "CIE_S2",
    "CIE_TCS",
    "Color",
    "Spectrum",
    "StickSpectrum",
]


def blackbody(
    T: Union[float, unyt.unit_quantity]
) -> Callable[[unyt.unit_array], unyt.unit_array]:
    """Blackbody spectral radiance in units

    Parameters
    ----------
    T : Union[float, unyt.unit_quantity]
        Blackbody temperature.

    Returns
    -------
    Callable[[unyt.unit_array],unyt.unit_array]
        Blackbody spectrum.
    """
    if not isinstance(T, unyt.unyt_array):
        T = T * unyt.K

    c1 = 2 * unyt.h * unyt.c ** 2
    c2 = (unyt.h * unyt.c / (unyt.kb * T)).to(unyt.nm)

    def f(x):
        return (c1 / x ** 5 / (np.exp((c2 / x)) - 1)).to(
            unyt.watt / unyt.m ** 2 / unyt.nm
        )

    return f


def gaussian(
    mean: unyt.unyt_quantity, std: unyt.unyt_quantity
) -> Callable[
    [Union[unyt.unyt_quantity, unyt.unyt_array]],
    Union[unyt.unyt_quantity, unyt.unyt_array],
]:
    def f(x):
        return np.exp(-(((x - mean) / std) ** 2)) / np.sqrt(np.pi) / std

    return f


def lorentzian(mean, gamma):
    def f(x):
        return 1 / gamma * np.pi / (1 + ((x - mean) / gamma) ** 2)

    return f


class Spectrum:
    def __init__(
        self, x: unyt.unyt_array, y: unyt.unyt_array, **interp_kwargs: Any
    ) -> None:
        inds = x.argsort()
        self.x = x[inds]
        self.y = unyt.unyt_array(y)[inds]

        if "ext" not in interp_kwargs:
            interp_kwargs["ext"] = "zeros"

        self.interp_kwargs = interp_kwargs

    @property
    def interp(self) -> scipy.interpolate.UnivariateSpline:
        """scipy.interpolate.UnivariateSpline : Interpolate spectrum."""
        return scipy.interpolate.InterpolatedUnivariateSpline(
            self.x.value, self.y.value, **self.interp_kwargs
        )

    def __call__(self, x: Union[unyt.unyt_array, np.ndarray]) -> unyt.unyt_array:
        """Evaluate spectrum via interpolation.

        Parameters
        ----------
        x : Union[unyt.unyt_array, numpy.ndarray]
            Input points.

        Returns
        -------
        unyt.unyt_array
            Spectrum values.
        """
        if isinstance(x, unyt.unyt_array):
            return self.interp(x.to(self.x.units).value) * self.y.units

        return self.interp(x) * self.y.units

    def integrate(
        self,
        xmin: Optional[unyt.unyt_quantity] = None,
        xmax: Optional[unyt.unyt_quantity] = None,
    ) -> unyt.unyt_quantity:
        """Integrate spectrum.

        Parameters
        ----------
        xmin : unyt.unyt_quantity, optional
            Lower integration limit.
            By default None.
        xmax : unyt.unyt_quantity, optional
            Upper integration limit.
            By default None.

        Returns
        -------
        unyt.unyt_quantity
            Integral of spectrum.
        """
        if xmin is None:
            xmin = self.x.min()
        else:
            xmin = xmin.to(self.x.units)

        if xmax is None:
            xmax = self.x.max()
        else:
            xmax = xmax.to(self.x.units)

        return (
            self.interp.integral(xmin.value, xmax.value) * self.x.units * self.y.units
        )

    def convert(self, unit: unyt.Unit, jacobian: Optional[bool] = True) -> Spectrum:
        """Change variable of x-axis and apply appropriate Jacobian transformation.

        Parameters
        ----------
        unit : unyt.Unit
            New x-axis unit.
        jacobian : bool, optional
            If True, apply Jacobian to ordinate.
            By default True.

        Returns
        -------
        Spectrum
            Converted spectrum.
        """
        x = self.x.to_equivalent(unit, "spectral")
        if jacobian:
            y = self.y * (self.x / x)
        else:
            y = self.y

        return Spectrum(x, y)

    def plot(
        self, x: Optional[unyt.unyt_array] = None, title: Optional[str] = None
    ) -> None:
        """Plot spectrum.

        Parameters
        ----------
        x : unyt.unyt_array, optional
            Plot range.
            By default None.
        title : str, optional
            Plot title.
            By default None.
        """
        if x is None:
            plt.plot(self.x.value, self.y.value)
        else:
            plt.plot(x, self(x))

        plt.xlabel(f"Units: {self.x.units}")
        plt.ylabel(f"Units: {self.y.units}")
        if title is not None:
            plt.title(title)
        plt.show()

    def color(self, illuminant: Optional[Spectrum]) -> Color:
        """Color object for this spectrum.

        Parameters
        ----------
        illuminant : Spectrum, optional
            Illuminant spectrum.
            By default None.

        Returns
        -------
        Color
            Color object for this spectrum.
        """
        return Color(self, illuminant=illuminant)

    def __mul__(self, other) -> Spectrum:
        if isinstance(other, Spectrum):
            x = np.union1d(self.x, other.x) * (self.x.units)
            y = self(x) * other(x)
            return Spectrum(x, y)

        return Spectrum(self.x, self.y * other)

    def __truediv__(self, other) -> Spectrum:
        if isinstance(other, Spectrum):
            x = np.union1d(self.x, other.x) * (self.x.units)
            y = self(x) / other(x)
            return Spectrum(x, y)

        return Spectrum(self.x, self.y / other)


class StickSpectrum(Spectrum):
    """Spectrum consisting of one or more delta functions."""

    def broaden_gaussian(self, x: unyt.unyt_array, std: unyt.unyt_quantity) -> Spectrum:
        """Broaden stick spectrum with Gaussian peaks.

        Parameters
        ----------
        x
            Spectrum range.
        std
            Gaussian scale parameter

        Returns
        -------
        Spectrum
            Broadened spectrum.
        """
        y = 0
        for mean, height in zip(self.x, self.y):
            y += height * gaussian(mean, std)(x)
        return Spectrum(x, y)

    def broaden_lorentzian(
        self, x: unyt.unyt_array, gamma: unyt.unyt_quantity
    ) -> Spectrum:
        """Broaden stick spectrum with Lorentzian peaks.

        Parameters
        ----------
        x
            Spectrum range.
        gamma
            Lorentzian scale parameter

        Returns
        -------
        Spectrum
            Broadened spectrum.
        """
        y = 0
        for mean, height in zip(self.x, self.y):
            y += height * lorentzian(mean, gamma)(x)
        return Spectrum(x, y)


class TimeSeries:
    @property
    def dt(self) -> unyt.unyt_array:
        # FIXME: should we even check for uniform spacing since this method,
        # like most/all methods, is susceptible to floating point errors?
        spacings = np.diff(self.x.value)
        dt, *_ = spacings
        if not np.allclose(spacings, dt):
            raise ValueError(
                "Time array does not have a unique spacing.\
                Consider interpolating to a uniformly spaced time array first."
            )
        else:
            return dt * self.x.units

    @property
    def T(self) -> unyt.unyt_array:
        return self.x[-1] - self.x[0]

    def damp(self) -> None:  # , final_damp_value):
        final_damp_value = 1e-4  # unyt.unyt_array(value=1e-4,unit=self.x.units)
        etasq = -np.log(final_damp_value) / self.T ** 2

        damp = np.exp(-(etasq * self.x ** 2).value)

        self.y *= damp

    def fourier_transform(self, pad_len=None):
        if pad_len is None:
            pad_len = len(self.x)

        fft_value = np.fft.fft(a=self.y.value, n=pad_len) / pad_len
        fft_real = fft_value.real * self.y.units
        fft_imag = fft_value.imag * self.y.units
        fft_freq = np.fft.fftfreq(n=pad_len, d=self.dt().value) / self.x.units

        return (
            Spectrum(x=fft_freq, y=fft_real),
            Spectrum(x=fft_freq, y=fft_imag),
        )


class ASTMG173(Spectrum):
    """ASTM G-173-03 solar spectral irradiance."""

    def __init__(self) -> None:
        with importlib.resources.path("materia.spectra", "ASTMG173.csv") as datafile:
            x, y = np.loadtxt(datafile, skiprows=2, usecols=[0, 2], delimiter=",").T

        x *= unyt.nm
        y *= unyt.watt / unyt.m ** 2 / unyt.nm

        super().__init__(x=x, y=y)


class CIE_CMF_X(Spectrum):
    """CIE color matching function for X tristimulus value."""

    def __init__(self):
        with importlib.resources.path("materia.spectra", "CIE_CMF_X.csv") as datafile:
            x, y = np.loadtxt(datafile, delimiter=",").T

        x *= unyt.nm
        y *= unyt.dimensionless

        super().__init__(x=x, y=y)


class CIE_CMF_Y(Spectrum):
    """CIE color matching function for Y tristimulus value."""

    def __init__(self):
        with importlib.resources.path("materia.spectra", "CIE_CMF_Y.csv") as datafile:
            x, y = np.loadtxt(datafile, delimiter=",").T

        x *= unyt.nm
        y *= unyt.dimensionless

        super().__init__(x=x, y=y)


class CIE_CMF_Z(Spectrum):
    """CIE color matching function for Z tristimulus value."""

    def __init__(self):
        with importlib.resources.path("materia.spectra", "CIE_CMF_Z.csv") as datafile:
            x, y = np.loadtxt(datafile, delimiter=",").T

        x *= unyt.nm
        y *= unyt.dimensionless

        super().__init__(x=x, y=y)


class CIE_A(Spectrum):
    """CIE A illuminant."""

    def __init__(self) -> None:
        with importlib.resources.path("materia.spectra", "CIE_A.csv") as datafile:
            x, y = np.loadtxt(datafile, delimiter=",").T

        x *= unyt.nm
        y *= unyt.dimensionless

        super().__init__(x=x, y=y)


class CIE_D(Spectrum):
    """CIE D illuminant.

    Parameters
    ----------
    T : unyt.unyt_quantity
        Illuminant temperature.
    """

    def __init__(self, T: unyt.unyt_quantity) -> None:
        wavs = np.linspace(300, 830, 107) * unyt.nm
        if T < 4000 * unyt.K or T > 25000 * unyt.K:
            raise ValueError(
                "Illuminant temperature must be between 4000 K and 25000 K."
            )

        # equations taken from http://www.brucelindbloom.com/Eqn_T_to_xy.html
        if T >= 4000 * unyt.K and T <= 7000 * unyt.K:
            x = (
                -4.6070e9 * unyt.K ** 3 / T ** 3
                + 2.9678e6 * unyt.K ** 2 / T ** 2
                + 0.09911e3 * unyt.K / T
                + 0.244063
            )
        elif T > 7000 * unyt.K and T <= 25000 * unyt.K:
            x = (
                -2.0064e9 * unyt.K ** 3 / T ** 3
                + 1.9018e6 * unyt.K ** 2 / T ** 2
                + 0.24748e3 * unyt.K / T
                + 0.237040
            )

        y = -3.000 * x ** 2 + 2.870 * x - 0.275

        # equations taken from
        # http://www.brucelindbloom.com/index.html?Eqn_DIlluminant.html
        M = 0.0241 + 0.2562 * x - 0.7341 * y
        M1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / M
        M2 = (0.0300 - 31.4424 * x + 30.0717 * y) / M
        vals = CIE_S0().y + M1 * CIE_S1().y + M2 * CIE_S2().y

        super().__init__(x=wavs, y=vals)


class CIE_D65(Spectrum):
    """CIE D65 illuminant."""

    def __init__(self) -> None:
        with importlib.resources.path("materia.spectra", "CIE_D65.csv") as datafile:
            x, y = np.loadtxt(datafile, delimiter=",").T

        x *= unyt.nm
        y *= unyt.dimensionless

        super().__init__(x=x, y=y)


class CIE_F(Spectrum):
    """CIE F illuminant.

    Parameters
    ----------
    n : int
        Number between 1 and 12 specifying which F illuminant to use.
    """

    def __init__(self, n: int) -> None:
        with importlib.resources.path("materia.spectra", "CIE_F.csv") as datafile:
            x, y = np.loadtxt(datafile, usecols=[0, n], delimiter=",").T

        x *= unyt.nm
        y *= unyt.dimensionless

        super().__init__(x=x, y=y)


CIE_PHOTOPIC = CIE_CMF_Y


class CIE_S0(Spectrum):
    def __init__(self):
        with importlib.resources.path("materia.spectra", "CIE_S0.csv") as datafile:
            x, y = np.loadtxt(datafile, delimiter=",").T

        x *= unyt.nm
        y *= unyt.dimensionless

        super().__init__(x=x, y=y)


class CIE_S1(Spectrum):
    def __init__(self):
        with importlib.resources.path("materia.spectra", "CIE_S0.csv") as datafile:
            x, y = np.loadtxt(datafile, delimiter=",").T

        x *= unyt.nm
        y *= unyt.dimensionless

        super().__init__(x=x, y=y)


class CIE_S2(Spectrum):
    def __init__(self):
        with importlib.resources.path("materia.spectra", "CIE_S0.csv") as datafile:
            x, y = np.loadtxt(datafile, delimiter=",").T

        x *= unyt.nm
        y *= unyt.dimensionless

        super().__init__(x=x, y=y)


class CIE_TCS(Spectrum):
    def __init__(self, n: int) -> None:
        with importlib.resources.path("materia.spectra", "CIE_TCS.csv") as datafile:
            x, y = np.loadtxt(datafile, skiprows=3, usecols=[0, n], delimiter=",").T

        x *= unyt.nm
        y *= unyt.dimensionless

        super().__init__(x=x, y=y)


class Color:
    def __init__(
        self,
        spectrum: Spectrum,
        illuminant: Optional[Spectrum] = None,
        white_point: Optional[tuple[float, float]] = None,
    ) -> None:
        """Color object.

        Parameters
        ----------
        spectrum
            Spectrum of interest.
        illuminant
            Illuminant spectrum, by default None.
        white_point
            White point expressed in u*, v* 1976 CIELUV coordinates.
        """
        self.spectrum = spectrum
        self.illuminant = illuminant or spectrum
        self.white_point = white_point

    def show(self, ax=None) -> None:
        """Show color on sRGB display."""
        if ax is None:
            fig, ax = plt.subplots()

        return ax.imshow([[self.sRGB]])

    @property
    @functools.cache
    def XYZ(self) -> tuple[float, float, float]:
        """tuple[float, float, float] : Coordinates in CIE 1931 XYZ color space."""
        xbar = CIE_CMF_X()
        ybar = CIE_CMF_Y()
        zbar = CIE_CMF_Z()

        X = (xbar * self.spectrum).integrate()
        Y = (ybar * self.spectrum).integrate()
        Z = (zbar * self.spectrum).integrate()

        N = (ybar * self.illuminant).integrate()

        return (X / N).item(), (Y / N).item(), (Z / N).item()

    @property
    def rgb(self) -> tuple[float, float, float]:
        """tuple[float, float, float] : Coordinates in linear RGB color space."""
        M = np.array(
            [
                [3.2406, -1.5372, -0.4986],
                [-0.9689, 1.8758, 0.0415],
                [0.0557, -0.2040, 1.0570],
            ]
        )
        X, Y, Z = self.XYZ
        XYZ = np.array([X / Y, Y / Y, Z / Y])[:, None]
        R, G, B = (M @ XYZ).squeeze()
        return R, G, B

    @property
    def sRGB(self) -> tuple[float, float, float]:
        """tuple[float, float, float] : Coordinates in sRGB color space."""
        rgb = np.array(self.rgb)
        i = rgb <= 0.0031308
        RGB = np.zeros(
            3,
        )
        RGB[i] += 12.92 * rgb[i]
        RGB[~i] += (1.055 * rgb ** (1 / 2.4) - 0.055)[~i]
        R, G, B = np.maximum(0, np.minimum(1, RGB))
        return R, G, B

    @property
    @functools.cache
    def LMS(self) -> tuple[float, float, float]:
        """tuple[float, float, float] : Coordinates in LMS color space."""
        hpe = hunt_pointer_estevez_transform()

        XYZ = np.array((self.XYZ))[:, None]
        L, M, S = (hpe @ XYZ).squeeze()

        return L, M, S

    @property
    @functools.cache
    def UVW(self) -> tuple[float, float, float]:
        """tuple[float, float, float] : Coordinates in 1964 CIEUVW color space."""
        X, Y, Z = self.XYZ

        if self.white_point is None:
            U = 2 * X / 3
            V = Y
            W = 0.5 * (-X + 3 * Y + Z)
        else:
            u0, v0 = self.white_point
            u, v = self.uv

            W = 25 * np.power(Y, 1 / 3) - 17
            U = 13 * W * (u - u0)
            V = 13 * W * (v - v0)

        return U, V, W

    @property
    @functools.cache
    def xyY(self) -> tuple[float, float, float]:
        """tuple[float, float, float] : Chromaticity coordinates
        based on CIE 1931 XYZ color space."""
        X, Y, Z = self.XYZ
        norm = X + Y + Z

        x = X / norm
        y = Y / norm

        return x, y, Y

    @property
    @functools.cache
    def uv(self) -> tuple[float, float]:
        """tuple[float, float] : Chromaticity coordinates in CIE 1960 color space."""
        x, y, _ = self.xyY

        denom = -2 * x + 12 * y + 3

        u = 4 * x / denom
        v = 6 * y / denom

        return u, v

    @property
    @functools.cache
    def uvprime(self) -> tuple[float, float]:
        """tuple[float, float] : Chromaticity coordinates in 1976 CIELUV color space."""
        x, y, _ = self.xyY

        denom = -2 * x + 12 * y + 3

        u = 4 * x / denom
        v = 9 * y / denom

        return u, v

    @property
    @functools.cache
    def CCT_DC(self) -> tuple[unyt.unyt_quantity, float]:
        """tuple[unyt.unyt_quantity, float] : Correlated color temperature
        and distance to Planckian locus. Computed using the 1960 CIE color space.
        If distance is greater than 0.05, CCT may be unreliable.
        """
        u, v = self.uv
        uvT = planckian_locus_ucs(exact=True)

        def _error(T):
            uT, vT = uvT(T=T * unyt.K)
            return (uT - u) ** 2 + (vT - v) ** 2

        CCT, DC_squared, _, _ = scipy.optimize.fminbound(
            func=_error, x1=1000, x2=15000, full_output=True
        )

        return CCT * unyt.K, np.sqrt(DC_squared)

    @property
    def cri(self) -> float:
        """float : Color rendering index."""
        CCT, DC = self.CCT_DC

        if DC > 0.05:
            warnings.warn(
                """Distance from UCS Planckian locus too high.
                    Illuminant is insufficiently white for accurate
                    CRI determination."""
            )

        if CCT < 5000 * unyt.K:
            x = np.linspace(380, 780, 1001) * unyt.nm
            y = blackbody(CCT)(x)
            reference = Spectrum(x=x, y=y)
        else:
            reference = CIE_D(T=CCT)

        reference /= reference(560 * unyt.nm)

        samples = (CIE_TCS(n) for n in range(1, 9))

        R_scores = tuple(
            special_cri(
                self.spectrum,
                reference,
                sample,
            )
            for sample in samples
        )

        return np.mean(R_scores)

    @property
    def avt(self) -> float:
        """float : Average visible transmittance."""
        weights = ASTMG173() * CIE_PHOTOPIC()
        return ((weights * self.spectrum).integrate() / weights.integrate()).item()


def cd(u, v):
    c = (4 - u - 10 * v) / v
    d = (1.708 * v + 0.404 - 1.481 * u) / v

    return c, d


def UVW(u, v, Y, u0, v0):
    W = 25 * Y ** (1 / 3) - 17
    U = 13 * W * (u - u0)
    V = 13 * W * (v - v0)
    return U, V, W


def adapt_whitepoint(u_t, v_t, u_r, v_r, u_wp, v_wp) -> tuple[float, float]:
    c_t, d_t = cd(u_t, v_t)
    c_r, d_r = cd(u_r, v_r)
    c_wp, d_wp = cd(u_wp, v_wp)

    denom = 16.518 + 1.481 * (c_r / c_t) * c_wp - (d_r / d_t) * d_wp
    u = (10.872 + 0.404 * (c_r / c_t) * c_wp - 4 * (d_r / d_t) * d_wp) / denom
    v = 5.520 / denom

    return u, v


def special_cri(test: Spectrum, reference: Spectrum, sample: Spectrum) -> float:
    u_t, v_t = Color(test).uv
    _, Y_t, _ = Color(test).XYZ
    u_r, v_r = Color(reference).uv
    _, Y_r, _ = Color(reference).XYZ
    u_ts, v_ts = Color(sample * test).uv
    _, Y_ts, _ = Color(sample * test).XYZ
    u_rs, v_rs = Color(sample * reference).uv
    _, Y_rs, _ = Color(sample * reference).XYZ

    c_t, d_t = cd(u_t, v_t)
    c_r, d_r = cd(u_r, v_r)
    c_ts, d_ts = cd(u_ts, v_ts)

    denom = 16.518 + 1.481 * (c_r / c_t) * c_ts - (d_r / d_t) * d_ts
    u = (10.872 + 0.404 * (c_r / c_t) * c_ts - 4 * (d_r / d_t) * d_ts) / denom
    v = 5.520 / denom

    U_ts, V_ts, W_ts = UVW(u, v, 100 * Y_ts / Y_t, u_r, v_r)
    U_rs, V_rs, W_rs = UVW(u_rs, v_rs, 100 * Y_rs / Y_r, u_r, v_r)
    dist = np.linalg.norm([U_ts - U_rs, V_ts - V_rs, W_ts - W_rs])
    return 100 - 4.6 * dist


def hunt_pointer_estevez_transform() -> np.ndarray:
    """Hunt-Pontier-Estevez transformation matrix (normalized to D65 illuminant).

    Returns
    -------
    np.ndarray
        Hunt-Pontier-Estevez transformation matrix.
    """
    return np.array(
        [
            [0.4002400, 0.7076000, -0.0808100],
            [-0.2263000, 1.1653200, 0.0457000],
            [0.0000000, 0.0000000, 0.9182200],
        ]
    )


def planckian_locus_xyz(
    exact: Optional[bool] = False,
) -> Callable[[unyt.unyt_quantity], tuple[float, float]]:
    if exact:

        def f(T):
            x = np.linspace(380, 780, 1001) * unyt.nm
            spectrum = Spectrum(x=x, y=blackbody(T)(x))
            spectrum /= spectrum(560 * unyt.nm)
            return Color(spectrum).xy

        return f
    else:

        def x(T):
            if T >= 1667 and T <= 4000:
                return (
                    -0.2661239e9 / T ** 3
                    - 0.2343589e6 / T ** 2
                    + 0.8776956e3 / T
                    + 0.179910
                )
            elif T > 4000 and T <= 25000:
                return (
                    -3.0258469e9 / T ** 3
                    + 2.1070379e6 / T ** 2
                    + 0.2226347e3 / T
                    + 0.240390
                )

        def y(T):
            xc = x(T)
            if T >= 1667 and T <= 2222:
                return (
                    -1.1063814 * xc ** 3
                    - 1.34811020 * xc ** 2
                    + 2.18555832 * xc
                    - 0.20219683
                )
            elif T > 2222 and T <= 4000:
                return (
                    -0.9549476 * xc ** 3
                    - 1.37418593 * xc ** 2
                    + 2.09137015 * xc
                    - 0.16748867
                )
            elif T > 4000 and T <= 25000:
                return (
                    3.0817580 * xc ** 3
                    - 5.87338670 * xc ** 2
                    + 3.75112997 * xc
                    - 0.37001483
                )

        return lambda T: (x(T=T), y(T=T))


def planckian_locus_ucs(
    exact: Optional[bool] = False,
) -> Callable[[unyt.unyt_quantity], tuple[float, float]]:
    if exact:

        def f(T):
            x = np.linspace(380, 780, 1001) * unyt.nm
            spectrum = Spectrum(x=x, y=blackbody(T)(x))
            spectrum /= spectrum(560 * unyt.nm)
            return Color(spectrum).uv

        return f
    else:

        def f(T):
            u = (0.860117757 + 1.54118254e-4 * T + 1.28641212e-7 * T ** 2) / (
                1 + 8.42420235e-4 * T + 7.08145163e-7 * T ** 2
            )

            v = (0.317398726 + 4.22806245e-5 * T + 4.20481691e-8 * T ** 2) / (
                1 - 2.89741816e-5 * T + 1.61456053e-7 * T ** 2
            )

            return u, v

        return f
