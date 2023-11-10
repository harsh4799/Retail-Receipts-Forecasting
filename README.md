# Fetch ML Receipt Forecast Application

This application serves as a forecasting tool that predicts daily receipt counts for the year 2022 using machine learning techniques, based on receipt data from 2021.

## Data Description

The provided dataset is a time-series record of receipt scans over the course of the year 2021. It consists of two fields: `#Date` and `Receipt_Count`.


## Application Components

### Exploratory Data Analysis
Checkout : ```notebooks/Exploratory_Data_Analysis.ipynb```

### Backend: PyTorch Model
- **Data Preprocessing:** The application includes preprocessing steps where the data was normalized.
- **Training and Inference:** We have 3 models to perform training & inference on: 
    - Linear Regression Model
    - Simple MLP Model
    - LSTM Model

### Frontend: Streamlit Interface
- **Data Input:** The interface accepts data uploads in a specified format for visualization and prediction.
- **Data Visualization:** The application provides insights into the receipt count variations throughout 2021 and forecasts for 2022.

### Streamlit Application Deployment
The Streamlit application has been containerized for easy deployment. The Docker image is available and documented in `readme.md`.

## Running the Application Locally 

To launch the Streamlit application on your machine, follow these instructions after cloning the repository:

Install dependencies:


<span style="color:red">NOTE: You need a version of python > 3.8 to make this work!</span>

```shell
pip install -r requirements.txt
```

Start the Streamlit server:

```shell
streamlit run app.py
```

There is also an option to run this with a docker image, however, due to CUDA errors & bugs in the pytorch library, this may not work. Currently I've pushed a version that allows us to see interactive results with the given dataset. 

### Running with a Published Docker Image

Pull and run the published Docker image with:

```
docker pull blackrosedragon2/receipts_forcasting:latest
```
```
docker run -p 8501:8501 blackrosedragon2/receipts_forcasting:latest
```

Access the application at: `http://localhost:8501/`

## Application Usage

- The Streamlit interface will showcase visualizations for data analysis and forecast results.
- To evaluate the model with new data, upload a `daily_data.csv` file.
- Interactive visualizations available through plotly. 
- Downloadable visualizations of the model's predictions are available via the application.

## Future Development

- Introducing additional models such as ARIMA and SARIMAX, exploring and incorporating seasonal analysis and modeling.
- Implementing a hyperparameter training module to streamline the process, allowing for efficient experimentation with various models and parameter adjustments.
- Addressing CUDA errors with ```docker```
