from benchmarks.benchmarks import MinimizerClassSuite, MinimizeSuite

mtest = MinimizeSuite()
mtest.setup()
out = mtest.time_minimize()
out = mtest.time_minimize_large()
out = mtest.time_minimize_withnan()
out = mtest.time_confinterval()

mtest = MinimizerClassSuite()
mtest.setup()
out = mtest.time_differential_evolution()
out = mtest.time_emcee()
