import streamlit as st
import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np

# Load data
data = pd.read_csv('data/data.csv')

num_bins = 9

bin_edges = np.linspace(data['LeverPosition'].min(), data['LeverPosition'].max(), num_bins + 1)

data['LeverPosition_bin'] = pd.cut(data['LeverPosition'], bins=bin_edges, labels=False, include_lowest=True)


data["avg_prop_torque[kN]"] = ((data['StarboardPropellerTorque(Ts)[kN]'] + data['PortPropellerTorque(Tp)[kN]']) / 2)

# Updated features list
features = ['LeverPosition_bin', 'GasTurbineShaftTorque[kNm]', 'GT_rateofrevolutions(GTn)[rpm]',
             'GasGeneratorRateofRevolutions(GGn)[rpm]', 'avg_prop_torque[kN]', 
             'HightPressure(HP)TurbineExitTemperature(T48)[C]', 'GTCompressorOutletAirTemperature(T2)[C]',
             'HPTurbineExitPressure(P48)[bar]', 'GTCompressorOutletAirPressure(P2)[bar]',
             'GTExhaustGasPressure(Pexh)[bar]', 'FuelFlow(mf)[kg/s]']

# Updated min and max values for sliders
slider_min_max = {
    'LeverPosition_bin': (0.0, 10.0),
    'GasTurbineShaftTorque[kNm]': (250.0, 73000.0),
    'GT_rateofrevolutions(GTn)[rpm]': (1300.0, 3600.0),
    'GasGeneratorRateofRevolutions(GGn)[rpm]': (6500.0, 10000.0),
    'avg_prop_torque[kN]': (5.0, 650.0),
    'HightPressure(HP)TurbineExitTemperature(T48)[C]': (440.0, 1120.0),
    'GTCompressorOutletAirTemperature(T2)[C]': (540.0, 780.0),
    'HPTurbineExitPressure(P48)[bar]': (1.0, 5.0),
    'GTCompressorOutletAirPressure(P2)[bar]': (5.0, 24.0),
    'GTExhaustGasPressure(Pexh)[bar]': (1.0, 1.06),
    'FuelFlow(mf)[kg/s]': (0.05, 2.0)
}

# Step sizes for sliders (floats)
slider_step = {
    'LeverPosition_bin': 1.0,
    'GasTurbineShaftTorque[kNm]': 1.0,
    'GT_rateofrevolutions(GTn)[rpm]': 1.0,
    'GasGeneratorRateofRevolutions(GGn)[rpm]': 1.0,
    'avg_prop_torque[kN]': 1.0,
    'HightPressure(HP)TurbineExitTemperature(T48)[C]': 1.0,
    'GTCompressorOutletAirTemperature(T2)[C]': 1.0,
    'HPTurbineExitPressure(P48)[bar]': 0.1,
    'GTCompressorOutletAirPressure(P2)[bar]': 0.1,
    'GTExhaustGasPressure(Pexh)[bar]': 0.0001,
    'FuelFlow(mf)[kg/s]': 0.01
}

def app():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "Feature Selection"])

    if page == "Prediction":
        st.title("Turbine Decay Coefficient Predictor")
        st.write("Input feature values to predict the target value.")

        # Input fields for prediction using sliders
        input_features = {feature: st.slider(feature, min_value=slider_min_max[feature][0], max_value=slider_min_max[feature][1], value=slider_min_max[feature][0], step=slider_step[feature]) for feature in features}
        
        if st.button("Predict"):
            if data is not None:
                try:
                    X = data[features]
                    y = data['GTTurbineDecayStateCoefficient']
                    model = Ridge(alpha=100)
                    model.fit(X, y)
                    
                    # Predict based on input
                    input_df = pd.DataFrame([input_features], columns=features)
                    prediction = model.predict(input_df)
                    st.write(f"Predicted GTTurbineDecayStateCoefficient: {prediction[0]}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.error("Data is not loaded properly.")

    elif page == "Feature Selection":
        st.title("Feature Selection Page")
        st.write("Select a feature to be the new target. Remaining features will be used to predict this new target.")

        selected_target = st.selectbox("Select New Target Feature", features)
        
        if selected_target:
            st.write("Preparing model...")
            # Show progress
            st.spinner("Training model...")
            
            # Prepare model using original data
            remaining_features = [f for f in features if f != selected_target]
            X = data[remaining_features]
            y = data[selected_target]
            model = Ridge(alpha=100)
            model.fit(X, y)
            
            # Input fields for prediction
            st.write(f"Input values for features to predict {selected_target}:")
            input_features = {feature: st.slider(feature, min_value=slider_min_max[feature][0], max_value=slider_min_max[feature][1], value=slider_min_max[feature][0], step=slider_step[feature]) for feature in remaining_features}
            
            if st.button("Predict"):
                try:
                    input_df = pd.DataFrame([input_features], columns=remaining_features)
                    prediction = model.predict(input_df)
                    st.write(f"Predicted {selected_target}: {prediction[0]}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    app()