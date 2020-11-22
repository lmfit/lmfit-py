import numpy as np
import lmfit
import pytest


tau = 2.0 * np.pi


class BaseTestForModels:
    _model = None
    _seed = 42

    @pytest.fixture(autouse=True)
    def seed(self):
        np.random.seed(self._seed)

    def check_guess_and_fit(self, noise_scale=0.0, atol=0.1, rtol=0.0,
                            offset=None, **kwargs):
        """Checks that the ``self._model()` correctly fits noisy data.

        Parameters
        -----------
        noise_scale: float, optional
            The standard deviation of Gaussian noise that is added to the test
            data.
        atol: float, optional
            Absolute tolerance for considering fit parameters close to the
            parameters test data was generated with.
        rtol: float, optional
            Relative tolerance for considering fit parameters close to the
            parameters test data was generated with.
        offset: float, optional
            This setting adds the specified offset to the test data and
            adds a `ConstantModel` to the model before attempting the fit.
        **kwargs : optional
            Keyword arguments to pass to the model function. The model fit
            parameters among these are used to generate data to try fitting
            on, the remaining ones are simply passed on to the model function,
            the guess method, and the fit method.

        Raises
        -------
        AssertionError
            Any fit parameter that is not close to the parameter used to
            generate the test data raises this error.
        """
        model = self._model()
        original_params = {name: kwargs.pop(name, np.random.rand())
                           for name in model.param_names}
        print(f"Original parameters: {original_params}")
        data = model.func(**kwargs, **original_params)
        if offset is not None:
            data += offset
        data += np.random.normal(scale=noise_scale, size=data.shape)
        guessed_params = model.guess(data, **kwargs)
        if offset is not None:
            constant_model = lmfit.models.ConstantModel()
            guessed_params += constant_model.guess(data)
            model += constant_model
        print(f"Guessed parameters: {guessed_params}")
        fit_result = model.fit(data, guessed_params, **kwargs)
        fit_values = fit_result.best_values
        for name, original_value in original_params.items():
            self._isclose(name, original_value, fit_values[name], atol, rtol)
        return fit_result

    def _isclose(self, name, actual_value, fit_value, atol, rtol):
        assert np.isclose(actual_value, fit_value, atol=atol, rtol=rtol), \
            f"wrong fit for parameter {name}: expected {actual_value}, " \
            f"fitted {fit_value}."


class TestLinearModel(BaseTestForModels):
    _model = lmfit.models.LinearModel

    def test_random_parameters(self):
        self.check_guess_and_fit(x=np.linspace(-1, 1, 300))


class TestQuadraticModel(BaseTestForModels):
    _model = lmfit.models.QuadraticModel

    def test_random_parameters(self):
        self.check_guess_and_fit(x=np.linspace(-10, 10, 300))


class TestSineModel(BaseTestForModels):
    _model = lmfit.models.SineModel

    def _isclose(self, name, actual_value, fit_value, atol, rtol):
        if name == "shift":
            # need to handle subtleties arising from subtracting two phases
            diff = abs((actual_value - fit_value + np.pi) % tau - np.pi)
            assert np.isclose(diff, 0, atol=atol, rtol=rtol), \
                f"wrong fit for parameter {name}: expected {actual_value}, " \
                f"fitted {fit_value}, phase difference {diff}."
        else:
            super()._isclose(name, actual_value, fit_value, atol, rtol)

    @pytest.mark.parametrize("shift", np.linspace(0, tau, 62))
    def test_perfect_sine(self, shift):
        self.check_guess_and_fit(
            frequency=0.3,
            amplitude=1.0,
            shift=shift,
            noise_scale=0,
            atol=0.05,
            x=np.linspace(0, 25, 100),
        )


    @pytest.mark.parametrize("shift", list(range(7)))
    def test_less_than_1_period(self, shift):
        self.check_guess_and_fit(
            frequency=0.65,
            amplitude=1.0,
            shift=shift,
            noise_scale=0.1,
            atol=0.05,
            x=np.linspace(0, tau, 1000),
        )

    @pytest.mark.parametrize("shift", list(range(7)))
    def test_typical_regime(self, shift):
        self.check_guess_and_fit(
            frequency=4.123,
            amplitude=23.5,
            shift=shift,
            noise_scale=3,
            atol=0.5,
            x=np.linspace(0, tau, 1000),
        )

    @pytest.mark.parametrize("shift", list(range(7)))
    def test_high_noise(self, shift):
        self.check_guess_and_fit(
            frequency=78.32,
            amplitude=1.0,
            shift=shift,
            noise_scale=5.0,
            atol=0.3,
            x=np.linspace(0, tau, 10000),
        )

    @pytest.mark.parametrize("offset", [0, -100, 1234.678])
    def test_with_offset(self, offset):
        self.check_guess_and_fit(
            frequency=3.23,
            amplitude=0.01,
            shift=2.1,
            noise_scale=0.01,
            offset=offset,
            atol=0.25,
            x=np.linspace(0, tau, 100),
        )
