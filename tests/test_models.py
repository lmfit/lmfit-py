import numpy as np
import lmfit
import pytest


class BaseTestForModels:
    _model = NotImplementedError
    _seed = 42

    @pytest.fixture(scope="class", autouse=True)
    def seed(self):
        np.random.seed(self._seed)

    @pytest.fixture
    def model(self):
        yield self._model()

    _orders_of_magnitude = 0.3

    def rand(self, *args, **kwargs):
        return 10.0 ** (self._orders_of_magnitude * np.random.rand(*args, *kwargs))

    @pytest.mark.parametrize("points", [1000, 1111, 9999, 2222])
    def test_guess_and_fit(self, model, points, noise=False):
        x = np.linspace(0, 1, points)
        original_params = {name: self.rand() for name in model.param_names}
        original_params["shift"] %= (2.0 * np.pi)
        original_params["shift"] %= (2.0 * np.pi)
        original_params["frequency"] += 1
        data = model.func(x, **original_params)
        if noise:
            data += self.rand() * np.random.rand(*data.shape) * original_params["amplitude"]
        guessed_params = model.guess(data, x)
        fit_result = model.fit(data, guessed_params, x=x)
        fit_values = model.post_process_fit_result(fit_result)
        for name, original_value in original_params.items():
            assert np.isclose(original_value, fit_values[name]), f"wrong fit for {name}"

class TestSineModel(BaseTestForModels):
    _model = lmfit.models.SineModel
