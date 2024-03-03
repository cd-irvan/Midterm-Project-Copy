import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt
import plotly.express as px
import random
from PIL import Image
import altair as alt

def main():
    st.set_page_config(page_title="Your App Title", page_icon=":tada:", layout="wide", initial_sidebar_state="expanded", theme={"base": "dark"})
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()

Insurance_data = pd.read_csv("insurance.csv")

Insurance_data_encoded = pd.get_dummies(Insurance_data, columns=['sex', 'smoker', 'region'], drop_first=True)

st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
#app_mode = st.sidebar.selectbox('Select Page',['Introduction','Visualization','Prediction'])
pages = ['Introduction', 'Visualization', 'Prediction']
custom_css = """
<style>
div.row-widget.stRadio > div{flex-direction:row;}
div.st-bf{flex-direction:column;}
div.st-bf > div{flex-direction:row;}
label.css-19ih76x{margin-right:10px;font-size:30px;}
div.stRadio > label{font-size:16px;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Your radio button code
app_mode = st.sidebar.radio('Select Page', pages)
list_variables = Insurance_data.columns

if app_mode == 'Introduction':

    Cover_Image = Image.open("medical-bills.jpg")


    st.image(Cover_Image, width=700)

    st.title("The Cost of Medical Care")
    st.header("Understanding the factors that affect your final medical bill")

    st.markdown("                               ")

    st.markdown("##### Project Objective")
    st.markdown("Using the visualisation and analysis of sourced medical data in the United States, we aim to obtain reasonble estimates of how much medical treatments would cost based on multiple factors.")

    st.markdown("                               ")
    st.markdown("##### An Introduction")

    st.markdown(" On average, healthcare spending per capita in the United States exceeds $10,000 annually, far surpassing that of other developed nations. ")

    st.markdown("                               ")

    st.markdown(" Factors contributing to the cost of medical care include but are not limited to, the type of treatment received, the healthcare provider's location, the patient's insurance coverage, and any additional services or procedures required.")

    st.markdown("                               ")

    st.markdown(" The lack of transparency in healthcare pricing often leaves patients bewildered, making it challenging to anticipate and budget for medical expenses accurately.")

    st.markdown("                               ")

    st.markdown(" Medical billing errors are not uncommon, with studies suggesting they affect up to 80% of hospital bills, potentially leading to inflated costs for patients.")

    st.markdown("                               ")

    st.markdown(" Uninsured individuals are particularly vulnerable to exorbitant medical bills, often facing financial hardship or even bankruptcy due to the inability to cover healthcare costs.")

    num = st.number_input('No. of Rows', 5, 10)

    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))

    if head == 'Head':
      st.dataframe(Insurance_data.head(num))
    else:
      st.dataframe(Insurance_data.tail(num))

    # Display dataset based on selection
    if head == 'Head':
        st.dataframe(Insurance_data.head(num))
    else:
        st.dataframe(Insurance_data.tail(num))

    # Display description of data

    # Display descriptions for all quantitative data
    st.markdown("Descriptions for all quantitative data **(rank and streams)** by:")
    st.markdown("Count")
    st.markdown("Mean")
    st.markdown("Standard Deviation")
    st.markdown("Minimum")
    st.markdown("Quartiles")
    st.markdown("Maximum")

    st.markdown("### Description of Data")

    st.markdown("Description for the original dataset")
    
    st.dataframe(Insurance_data.describe())

    # Display missing values information
    st.markdown("### Missing Values")
    st.markdown("Null or NaN values.")
    Insurance_null = Insurance_data.isnull().sum() / len(Insurance_data) * 100
    total_miss = Insurance_null.sum().round(2)
    st.write("Percentage of total missing values:", total_miss)
    st.write(Insurance_null)

    if total_miss <= 30:
        st.success("We have a negligible amount of missing values. This helps provide us with more accurate data as the null values will not significantly affect the outcomes of our conclusions.")
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing value.")
        st.markdown("> Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")

    # Display completeness information
    st.markdown("### Completeness")
    st.markdown("The ratio of non-missing values to total records in dataset and how comprehensive the data is.")
    st.write("Total data length:", len(Insurance_data))
    non_missing = Insurance_data.notnull().sum().round(2)
    completeness = round(sum(non_missing) / len(Insurance_data), 2)
    st.write("Completeness ratio:", completeness)
    st.write(non_missing)

    if completeness >= 0.80:
        st.success("We have a completeness ratio greater than 0.85, which is good. It shows that the vast majority of the data is available for us to use and analyze.")
    else:
        st.warning("Poor data quality due to low completeness ratio (less than 0.85).")

    st.markdown("Description for the cleaned dataset")
    
    st.dataframe(Insurance_data_encoded.describe())

    # Display missing values information
    st.markdown("### Missing Values")
    st.markdown("Null or NaN values.")
    Insurance_null = Insurance_data_encoded.isnull().sum() / len(Insurance_data_encoded) * 100
    total_miss = Insurance_null.sum().round(2)
    st.write("Percentage of total missing values:", total_miss)
    st.write(Insurance_null)

    if total_miss <= 30:
        st.success("We have a negligible amount of missing values. This helps provide us with more accurate data as the null values will not significantly affect the outcomes of our conclusions.")
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing value.")
        st.markdown("> Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")

    # Display completeness information
    st.markdown("### Completeness")
    st.markdown("The ratio of non-missing values to total records in dataset and how comprehensive the data is.")
    st.write("Total data length:", len(Insurance_data_encoded))
    non_missing = Insurance_data_encoded.notnull().sum().round(2)
    completeness = round(sum(non_missing) / len(Insurance_data_encoded), 2)
    st.write("Completeness ratio:", completeness)
    st.write(non_missing)

    if completeness >= 0.80:
        st.success("We have a completeness ratio greater than 0.85, which is good. It shows that the vast majority of the data is available for us to use and analyze.")
    else:
        st.warning("Poor data quality due to low completeness ratio (less than 0.85).")

    st.markdown("##### Description of the Key Variables")
    st.markdown("- Age")
    st.markdown("This column contains the recorded ages of the patients in the dataset.")

    st.markdown("- Sex")
    st.markdown("This column contains the recorded sex of the patients in the dataset.")

    st.markdown("- BMI")
    st.markdown("This column gives the recorded Body Mass Index (BMI), calculated by dividing the weight of the patient in kilograms by the square of the height of the patient in metres, foe each patient in the dataset. ")

    st.markdown("- Children")
    st.markdown("This column indicates whether the patient is a child (represented by 1) or an adult (represented by 0).")

    st.markdown("- Smoker")
    st.markdown("This column indicates whether the patient is a smoker (represented by 1) or a non-smoker (represented by 0).")

    st.markdown("- Charges")
    st.markdown("This column records the total billed amount paid by each patient in the dataset for insurance. ")

    st.markdown("- Region")
    st.markdown("This column gives the region of the United States from where the patient receieved medical care. ")

if app_mode == "Visualization":

    st.markdown("## Visualizing the available data")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Average Insurance Charges by Region","Average BMI by Region", "Average Charges by Age", "A Scatter Plot", "A Correlation Map" ])

    width1 = st.sidebar.slider("Choose the width of the plot", 1, 25, 10)

    with tab1:

      tab1.subheader("Average Insurance Charges by Region")
      avg_charges_by_region = Insurance_data.groupby('region')['charges'].mean().sort_values()
      st.bar_chart(data=avg_charges_by_region, color=None, width=0, height=0, use_container_width=True)
      st.write(" ")
      avg_charges_by_region.plot(kind='bar', color='skyblue')
      plt.title('Average Insurance Charges by Region')
      plt.xlabel('Region')
      plt.ylabel('Average Charges')
      plt.xticks(rotation=45)
      plt.grid(axis='y', linestyle='--', alpha=0.7)
      plt.show()


    with tab2:

      # Calculate the average BMI by region

      tab2.subheader("Average BMI by Region")
      avg_bmi_by_region = Insurance_data.groupby('region')['bmi'].mean().sort_values()

      # Plotting
      st.bar_chart(data=avg_bmi_by_region, color=None, width=0, height=0, use_container_width=True)
      st.write(" ")
      plt.figure(figsize=(10, 6))
      avg_bmi_by_region.plot(kind='bar', color='lightgreen')
      plt.title('Average BMI by Region')
      plt.xlabel('Region')
      plt.ylabel('Average BMI')
      plt.xticks(rotation=45)
      plt.grid(axis='y', linestyle='--', alpha=0.7)
      plt.show()

    with tab3:

      tab3.subheader("Average Charges by Age of the patient")
      # Calculate the average charges by age
      avg_charges_by_age = Insurance_data.groupby('age')['charges'].mean().sort_values()

      # Plotting
      st.bar_chart(data=avg_charges_by_age, color=None, width=0, height=0, use_container_width=True)
      st.write(" ")
      plt.figure(figsize=(12, 6))
      avg_charges_by_age.plot(kind='bar', color='salmon')
      plt.title('Average Insurance Charges by Age')
      plt.xlabel('Age')
      plt.ylabel('Average Charges')
      plt.grid(axis='y', linestyle='--', alpha=0.7)
      plt.show()

    with tab4:

      tab4.subheader("A scatterplot")

      # Plotting
      st.scatter_chart(data=Insurance_data, x="bmi", y="charges", color=None, size=None, width=0, height=0, use_container_width=True)
      st.write(" ")
      plt.figure(figsize=(10, 6))
      plt.scatter(Insurance_data_encoded['bmi'], Insurance_data_encoded['charges'], alpha=0.5, color='dodgerblue')
      plt.title('Scatter Plot of Insurance Charges')
      plt.xlabel('bmi')
      plt.ylabel('Charges')
      plt.grid(True, which='both', linestyle='--', alpha=0.5)
      plt.show()

    with tab5:

      # Generate a heatmap
      tab5.subheader("A Correlation Map")
      fig,ax = plt.subplots(figsize=(width1, width1))
      sns.heatmap(Insurance_data_encoded.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
      plt.title('Correlation Matrix of Insurance Dataset')
      tab5.write(fig)

if app_mode == "Prediction":

    # Step 1: Define features (X) and the target variable (y)
    X = Insurance_data_encoded.drop('charges', axis=1)  # Features
    y = Insurance_data_encoded['charges']  # Target variable

    # Step 2: Split the data into training and testing sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Train a linear regression model
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 4: Predict the target variable for the testing set
    y_pred = model.predict(X_test)

    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    # Step 5: Evaluate the model
    from sklearn.metrics import mean_squared_error, r2_score

    st.subheader('Results')

    st.write("1) The Mean Absolute Error of model is:", np.round(mt.mean_absolute_error(y_test, y_pred ),2))
    st.write("2) The Mean Square Error of the model is: ", np.round(mt.mean_squared_error(y_test, y_pred),2))
    st.write("3) The R-Square score of the model is " , np.round(mt.r2_score(y_test, y_pred),2))

    st.write(" ")
    st.write("Plotting the actual final insurance costs versus the final insurance costs predicted by the model")
    st.scatter_chart(data = comparison_df, x="Actual", y="Predicted", color=None, size=None, width=0, height=0, use_container_width=True)
    st.write(" ")

    BMI = 0
    st.write("Enter your details below to estimate the cost of your final insurance bill : ")
    Age = st.number_input("Enter your age:", min_value=18, max_value=110, value=18, step=1)
    Sex = st.selectbox("Select your sex:", options=["Male", "Female"])
    Height = st.number_input("Enter your height in metres: ")
    Weight = st.number_input("Enter your weight in kilograms: ")
    if(Height != 0):
       BMI = Weight/(Height ** 2)
    Child = st.number_input("Number of Dependents:",min_value=0, max_value=100, value=0, step=1)
    Smoker = st.selectbox("Are you a smoker?", options=["Yes", "No"])
    Region = st.selectbox("Which region do you live in?", options=["Northeast", "Southeast", "Northwest", "Southwest"])

    Sex = 1 if Sex == "Male" else 0
    Smoker = 1 if Smoker == "Yes" else 0

    


    def cost_calculator(cost_caluclator):
        cost = model.predict(cost_data)
        st.write(f'Predicted final insurance bill: {cost}')

    if Region == "Northeast":
        cost_data = [[Age, BMI, Child, Sex, Smoker, 0, 0, 0]]
        cost_calculator(cost_data)
    elif Region == "Northwest":
        cost_data = [[Age, BMI, Child, Sex, Smoker, 1, 0, 0]]
        cost_calculator(cost_data)
    elif Region == "Southeast":
        cost_data = [[Age, BMI, Child, Sex, Smoker, 0, 1, 0]]
        cost_calculator(cost_data)
    elif Region == "Southwest":
        cost_data = [[Age, BMI, Child, Sex, Smoker, 0, 0, 1]]
        cost_calculator(cost_data)

if __name__=='__main__':
    main()

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;background - color: white}
     .stApp { bottom: 80px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1,

    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)
