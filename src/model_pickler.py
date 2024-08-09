import pandas as pd
from sklearn.linear_model import Ridge
import joblib
import numpy as np


def model_pickler()
    # Load data
    data = pd.read_csv('../data/data.csv')

    # Bin the LeverPosition feature
    num_bins = 9
    bin_edges = np.linspace(data['LeverPosition'].min(), data['LeverPosition'].max(), num_bins + 1)
    data['LeverPosition_bin'] = pd.cut(data['LeverPosition'], bins=bin_edges, labels=False, include_lowest=True)

    # Calculate avg_prop_torque
    data["avg_prop_torque[kN]"] = (data['StarboardPropellerTorque(Ts)[kN]'] + data['PortPropellerTorque(Tp)[kN]']) / 2

    # Define features and target
    features = ['LeverPosition_bin', 'GasTurbineShaftTorque[kNm]', 'GT_rateofrevolutions(GTn)[rpm]',
                'GasGeneratorRateofRevolutions(GGn)[rpm]', 'avg_prop_torque[kN]', 
                'HightPressure(HP)TurbineExitTemperature(T48)[C]', 'GTCompressorOutletAirTemperature(T2)[C]',
                'HPTurbineExitPressure(P48)[bar]', 'GTCompressorOutletAirPressure(P2)[bar]',
                'GTExhaustGasPressure(Pexh)[bar]', 'FuelFlow(mf)[kg/s]']
    target = 'GTTurbineDecayStateCoefficient'

    X = data[features]
    y = data[target]

    # Train the model
    model = Ridge(alpha=100)
    model.fit(X, y)

    # Save the model using joblib
    joblib.dump(model, 'ridge_model.joblib')

if __name__ == "__main__":
    model_pickler()