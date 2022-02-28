from materia.spectra import (
    blackbody,
    CIE_CMF_X,
    CIE_CMF_Y,
    CIE_CMF_Z,
    CIE_D,
    CIE_F,
    Color,
    Spectrum,
    StickSpectrum,
)
import numpy as np
import unittest
import unittest.mock as mock
import unyt


class TestBlackBody(unittest.TestCase):
    def test_blackbody_no_units(self):
        unyt.assert_allclose_units(
            blackbody(5000)(100 * unyt.nm),
            0.0037920102686782 * unyt.watt / unyt.nm / unyt.m ** 2,
        )
        unyt.assert_allclose_units(
            blackbody(1000)(100 * unyt.nm),
            3.89617048e-53 * unyt.watt / unyt.nm / unyt.m ** 2,
        )

    def test_blackbody_kelvin(self):
        unyt.assert_allclose_units(
            blackbody(5000 * unyt.K)(100 * unyt.nm),
            0.0037920102686782 * unyt.watt / unyt.nm / unyt.m ** 2,
        )
        unyt.assert_allclose_units(
            blackbody(1000 * unyt.K)(100 * unyt.nm),
            3.89617048e-53 * unyt.watt / unyt.nm / unyt.m ** 2,
        )


class TestStickSpectrum(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(1, 10, 10) * unyt.eV
        self.y = np.linspace(0.1, 1, 10)

    def test_instantiate(self):
        self.spec = StickSpectrum(self.x, self.y)


class TestLinearSpectrum(unittest.TestCase):
    def setUp(self):
        x = np.linspace(1, 10, 10) * unyt.nanometer
        y = 2 * np.linspace(1, 10, 10) * unyt.eV
        self.spec = Spectrum(x, y)

    def test_integrate(self):
        unyt.assert_allclose_units(self.spec.integrate(), 99 * unyt.nm * unyt.eV)
        unyt.assert_allclose_units(
            self.spec.integrate(xmax=5 * unyt.nm), 24 * unyt.nm * unyt.eV
        )
        unyt.assert_allclose_units(
            self.spec.integrate(xmin=3 * unyt.nm), 91 * unyt.nm * unyt.eV
        )
        unyt.assert_allclose_units(
            self.spec.integrate(xmin=3 * unyt.nm, xmax=5 * unyt.nm),
            16 * unyt.nm * unyt.eV,
        )
        unyt.assert_allclose_units(
            self.spec.integrate(xmin=0 * unyt.nm, xmax=100 * unyt.nm),
            99 * unyt.nm * unyt.eV,
        )
        unyt.assert_allclose_units(
            self.spec.integrate(xmin=1 * unyt.nm, xmax=1 * unyt.nm),
            0 * unyt.nm * unyt.eV,
        )

    def test_convert(self):
        x = 4 * unyt.nm
        x_ev = unyt.h * unyt.c / (4 * unyt.nm)

        spec = self.spec.convert(unyt.eV)
        self.assertEqual(spec.x.units, unyt.eV)
        unyt.assert_allclose_units(spec(x_ev), ((x / x_ev) * self.spec(x)).to(unyt.nm))

        spec = self.spec.convert(unyt.eV, jacobian=False)
        self.assertEqual(spec.x.units, unyt.eV)
        unyt.assert_allclose_units(spec(x_ev), self.spec(x))


class TestQuadraticSpectrum(unittest.TestCase):
    def setUp(self):
        x = np.linspace(1, 10, 10) * unyt.nanometer
        y = 3 * np.linspace(1, 10, 10) ** 2 * unyt.eV
        self.spec = Spectrum(x, y)

    def test_integrate(self):
        unyt.assert_allclose_units(self.spec.integrate(), 999 * unyt.nm * unyt.eV)
        unyt.assert_allclose_units(
            self.spec.integrate(xmax=5 * unyt.nm), 124 * unyt.nm * unyt.eV
        )
        unyt.assert_allclose_units(
            self.spec.integrate(xmin=3 * unyt.nm), 973 * unyt.nm * unyt.eV
        )
        unyt.assert_allclose_units(
            self.spec.integrate(xmin=3 * unyt.nm, xmax=5 * unyt.nm),
            98 * unyt.nm * unyt.eV,
        )
        unyt.assert_allclose_units(
            self.spec.integrate(xmin=0 * unyt.nm, xmax=100 * unyt.nm),
            999 * unyt.nm * unyt.eV,
        )
        unyt.assert_allclose_units(
            self.spec.integrate(xmin=1 * unyt.nm, xmax=1 * unyt.nm),
            0 * unyt.nm * unyt.eV,
        )


class TestPredefinedSpectra(unittest.TestCase):
    def test_cmfs(self):
        CIE_CMF_X()
        CIE_CMF_Y()
        CIE_CMF_Z()

        CIE_D(4000 * unyt.Kelvin)
        CIE_D(5000 * unyt.Kelvin)
        CIE_D(10000 * unyt.Kelvin)
        CIE_D(25000 * unyt.Kelvin)

        with self.assertRaises(ValueError):
            CIE_D(3999 * unyt.Kelvin)
            CIE_D(25001 * unyt.Kelvin)
        for i in range(1, 13):
            CIE_F(i)
        self.assertTrue(True)


@mock.patch(
    "materia.spectra.Color.XYZ",
    new_callable=mock.PropertyMock,
    return_value=[0.84264147, 0.50045325, 0.31863809],
)
class TestColor1(unittest.TestCase):
    def setUp(self):
        spec = Spectrum(np.array([]), np.array([]))
        self.color = Color(spec)

    def test_xyY(self, mockXYZ):
        self.assertTrue(
            np.allclose(self.color.xyY, [0.50708602, 0.30116349, 0.50045325])
        )


@mock.patch(
    "materia.spectra.Color.XYZ",
    new_callable=mock.PropertyMock,
    return_value=[0.28528629, 0.18961297, 0.64290672],
)
class TestColor2(unittest.TestCase):
    def setUp(self):
        spec = Spectrum(np.array([]), np.array([]))
        self.color = Color(spec)

    def test_xyY(self, mockXYZ):
        self.assertTrue(
            np.allclose(self.color.xyY, [0.25521986, 0.16962959, 0.18961297])
        )


# def test_timeseries_dt_uniform():
#     t = np.linspace(0, 10, 101) * mtr.s

#     test_result = mtr.TimeSeries(x=t, y=None).dt
#     check_result = 0.1 * mtr.s

#     assert test_result == check_result


# def test_timeseries_dt_nonuniform():
#     t_value = np.hstack((np.linspace(0, 10, 101), np.linspace(11, 21, 101)))
#     t = t_value * mtr.s

#     with pytest.raises(ValueError):
#         mtr.TimeSeries(x=t, y=None).dt


# def test_timeseries_T():
#     t = np.linspace(0, 10, 101) * mtr.s

#     test_result = mtr.TimeSeries(x=t, y=None).T
#     check_result = 10 * mtr.s

#     assert test_result == check_result


# def test_timeseries_T_negative_start():
#     t = np.linspace(-10, 10, 101) * mtr.s

#     test_result = mtr.TimeSeries(x=t, y=None).T
#     check_result = 20 * mtr.s

#     assert test_result == check_result
