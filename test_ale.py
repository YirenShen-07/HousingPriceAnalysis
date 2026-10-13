"""Check ALE against an additive linear function with a known effect."""
import unittest
import numpy as np
import pandas as pd
from analyze import ale_1d
class LinearModel:
    def predict(self,X):return 2*X['income'].to_numpy()+3*X['other'].to_numpy()
class AleTests(unittest.TestCase):
    def test_linear_income_effect(self):
        X=pd.DataFrame({'income':np.linspace(0,10,1001),'other':np.linspace(10,30,1001)})
        grid,effect=ale_1d(LinearModel(),X,'income',bins=10)
        np.testing.assert_allclose(np.diff(effect),2*np.diff(grid),atol=1e-8)
if __name__=='__main__':unittest.main()
