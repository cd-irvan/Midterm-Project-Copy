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
import streamlit as st
from streamlit_image_select import image_select

def main():
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


# Set the page configuration
st.set_page_config(page_title="Your App", page_icon="🖼️")

# Define the example images for each section
EXAMPLES = {
    "Introduction": "health-insurance.jpg",
    "Visualization": "visualisation.jpg",
    "Prediction": "linear_regression.jpg"
}

# Set the initial state if it's not already set
if not st.session_state:
    st.session_state['current_section'] = "Introduction"

# Create the image selection
example_image_fp = [EXAMPLES[example] for example in EXAMPLES]
example_names = list(EXAMPLES.keys())

index_selected = image_select(
    "",
    images=example_image_fp,
    captions=example_names,
    index=example_names.index(st.session_state['current_section']),
    return_value="index"
)

# Update the current section based on selection
st.session_state['current_section'] = example_names[index_selected]

# Display the content based on the selected section
if st.session_state['current_section'] == "Introduction":
    st.header("Introduction")
    
    Cover_Image = Image.open("medical-bills.jpg")


    st.image(Cover_Image, width=700)

    st.title("The Cost of Medical Insurance")
    st.header("Understanding the factors that affect the insurance premium you pay.")

    st.markdown("                               ")

    st.markdown("##### Project Objective")
    st.markdown("Using the visualisation and analysis of sourced medical data in the United States, we aim to obtain reasonble estimates of how much medical insurance premiums would cost based on multiple factors specific to a given patient.")

    st.markdown("                               ")
    st.markdown("##### An Introduction")

    st.markdown(" On average, healthcare spending per capita in the United States exceeds $10,000 annually, far surpassing that of other developed nations. ")

    st.markdown("                               ")

    st.markdown(" Medical insurance premiums represent the amount individuals or groups pay to an insurance company for healthcare coverage. These premiums serve as a form of financial protection, providing access to medical services and helping mitigate the high costs associated with healthcare. ")

    st.markdown("                               ")

    st.markdown(" Medical insurance premiums are influenced by various qualitative factors such as age, lifestyle choices, pre-existing conditions, and even geographic location. For instance, individuals with healthier lifestyles and no pre-existing conditions may qualify for lower premiums, while those residing in areas with higher healthcare costs may face increased insurance rates.")

    st.markdown("                               ")

    st.markdown("Advancements in medical technology and treatment options can affect insurance premiums. While innovative treatments may improve patient outcomes, they can also come with higher costs, leading to adjustments in insurance pricing to account for these advancements.")

    st.markdown("                               ")

    st.markdown(" The extent of coverage desired, including deductibles, copayments, and coverage limits, directly affects insurance premiums. Comprehensive coverage with lower out-of-pocket expenses typically entails higher premiums, while higher deductibles and copayments can reduce monthly costs.")

    st.markdown("                               ")

    st.markdown(" Government regulations and healthcare policies also influence insurance premiums. Changes in legislation, such as the Affordable Care Act in the United States, can impact insurance costs by mandating coverage requirements, introducing subsidies, or altering the competitive landscape of the insurance market. Insurers also assess risk when determining premiums for a potential client, with a larger risk pool potentially resulting in lower costs for individuals.")

    st.markdown("                               ")

    st.markdown(" Most importantly, medical inflation, which outpaces general inflation rates, contributes to rising insurance premiums. As the cost of medical services, treatments, and medications increases, insurance companies adjust their premiums to accommodate these rising expenses.")

    st.markdown("                               ")

    st.markdown(" Tailoring insurance policies to individual needs and preferences can impact premiums, allowing for optimization based on specific requirements. Regularly reviewing insurance coverage and comparing options ensures individuals are receiving the best value for their premiums and adjusting to changes in their circumstances.")

    st.markdown("                               ")

    st.markdown("##### Description of the Key Variables")

    variables = {
        "Age": "This column contains the recorded ages of the insured people in the dataset.",
        "Sex": "This column contains the recorded sex of the insured people in the dataset.",
        "BMI": "This column gives the recorded Body Mass Index (BMI), calculated by dividing the weight of the patient in kilograms by the square of the height of the insured person in metres, for each patient in the dataset.",
        "Children": "This column indicates the number of dependent children the insured patient has.",
        "Smoker": "This column indicates whether the insured patient is a smoker or a non-smoker.",
        "Charges": "This column records the total billed amount paid by each patient in the dataset as their insurance premium.",
        "Region": "This column gives the region of the United States from where the insured person received medical care."
    }
    
    for variable, description in variables.items():
        st.markdown(f"- **{variable}**: {description}")




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
    st.markdown("## A Statistical Summary")

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
    completeness = Insurance_data.notnull().all(axis=1).mean()
    st.write("Completeness ratio:", completeness)

    if completeness >= 0.80:
        st.success(f"After an intial cleaning of the dataset, we have a completeness ratio of {completeness}. This shows that all of the data is available for us to use and analyze.")
    else:
        st.warning("Poor data quality due to low completeness ratio (less than 0.85).")


    
elif st.session_state['current_section'] == "Visualization":
    st.header("Visualization")
    st.markdown("## Visualizing the available data")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Average Insurance Charges by Region","Average BMI by Region", "Average Charges by Age", "Pairplot", "A Correlation Map" ])

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

      tab3.subheader("Average Charges by Age of the Patient")
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

    # with tab4:

    #   tab4.subheader("A scatterplot")

    #   # Plotting
    #   st.scatter_chart(data=Insurance_data, x="bmi", y="charges", color=None, size=None, width=0, height=0, use_container_width=True)
    #   st.write(" ")
    #   plt.figure(figsize=(10, 6))
    #   plt.scatter(Insurance_data_encoded['bmi'], Insurance_data_encoded['charges'], alpha=0.5, color='dodgerblue')
    #   plt.title('Scatter Plot of Insurance Charges')
    #   plt.xlabel('bmi')
    #   plt.ylabel('Charges')
    #   plt.grid(True, which='both', linestyle='--', alpha=0.5)
    #   plt.show()

    with tab4:
        tab4.subheader("Pairplot for All Columns")


        fig = sns.pairplot(Insurance_data)

        plt.suptitle('Pairwise Relationships Across All Columns', y=1.02)


        st.pyplot(fig)


    with tab5:

      # Generate a heatmap
      tab5.subheader("A Correlation Map")
      fig,ax = plt.subplots(figsize=(width1, width1))
      sns.heatmap(Insurance_data_encoded.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
      plt.title('Correlation Matrix of Insurance Dataset')
      tab5.write(fig)

    # Add your visualization content here
elif st.session_state['current_section'] == "Prediction":
    st.header("Prediction")
        # Add your prediction content here
    
        # Step 1: Allow users to choose independent columns
    selected_columns = st.multiselect("Choose independent columns:", Insurance_data_encoded.columns.tolist(), default=Insurance_data_encoded.columns.drop('charges').tolist())
    
    X = Insurance_data_encoded[selected_columns]  # Features
    y = Insurance_data_encoded['charges']  # Target variable
    
    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 3: Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Step 4: Predict the target variable for the testing set
    y_pred = model.predict(X_test)

    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    
    # Step 5: Evaluate the model
    st.subheader('Linear Regression Model')
    
    intercept = model.intercept_
    coefficients = model.coef_
    coefficients_with_features = dict(zip(selected_columns, coefficients))

    from sklearn.metrics import mean_squared_error, r2_score
    
    st.subheader('Results')
    
    st.write("1) The Mean Absolute Error of model is:", np.round(mt.mean_absolute_error(y_test, y_pred ),2))
    st.write("2) The Mean Square Error of the model is: ", np.round(mt.mean_squared_error(y_test, y_pred),2))
    st.write("3) The R-Square score of the model is " , np.round(mt.r2_score(y_test, y_pred),2))
    
    st.write(" ")
    st.write("Plotting the actual final insurance costs versus the final insurance costs predicted by the model")
    st.scatter_chart(data=comparison_df, x="Actual", y="Predicted", use_container_width=True)
    st.write(" ")

    st.markdown("### Coefficients of the Linear Regression Model")

    # Display coefficients
    for feature, coeff in coefficients_with_features.items():
        st.write(f"- **{feature}**: {np.round(coeff, 2)}")
    
    # Create LaTeX equation
    equation = f"y = {np.round(intercept, 2)}"
    for feature, coeff in zip(feature_names, coefficients):
        equation += f" + ({np.round(coeff, 2)}) \\times {feature}"

    st.markdown("### Equation of the Linear Regression Model for Predicting Insurance Premiums")

    Equation_Image = Image.open("Eqn1.jpg")
    st.image(Equation_Image, width=700)

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
        st.write(f'Predicted final insurance bill: {cost[0]}')

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





