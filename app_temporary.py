import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components  # Import Streamlit

def main():
    st.set_page_config(layout="wide")

    st.title("Time Series Prediction App")

    # Model selection
    model_option = st.selectbox("Select a Model", ["Linear Regression", "MLP", "LSTM"])

    if model_option == "Linear Regression":
        plotly_html_path = "outputs/LinearRegression.html"
    elif model_option == "MLP":
        plotly_html_path = "outputs/MLP.html"
    elif model_option == "LSTM":
        plotly_html_path = "outputs/LSTM.html"

    # Display the Plotly chart as HTML
    with open(plotly_html_path, "r") as file:
        plotly_html_content = file.read()

    st.write("Plotly Chart:")

    components.html(plotly_html_content, width=1400, height=600)


if __name__ == "__main__":
    main()
