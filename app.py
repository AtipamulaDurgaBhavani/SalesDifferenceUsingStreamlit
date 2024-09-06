import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Streamlit app title
st.title("Sales Difference & Feature Importance Waterfall Chart")

# File upload section
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    # Step 1: Pre-processing of data
    # Convert 'Discount' to a numeric format (remove %)
    if 'Discount' in data.columns:
        data['Discount'] = data['Discount'].str.rstrip('%').astype('float') / 100
    else:
        st.error("'Discount' column not found in the uploaded CSV.")
        st.stop()

    # Clean the 'Week' column to remove any leading/trailing spaces
    if 'Week' in data.columns:
        data['Week'] = data['Week'].str.strip()  # Ensure consistency in week labels
    else:
        st.error("'Week' column not found in the uploaded CSV.")
        st.stop()

    # Calculate total sales for Week 1 and Week 2
    week1_sales_data = data[data['Week'] == 'Week1']
    week2_sales_data = data[data['Week'] == 'Week2']

    if week1_sales_data.empty or week2_sales_data.empty:
        st.error("Week1 or Week2 data is missing from the uploaded CSV.")
        st.stop()

    total_sales_week1 = week1_sales_data['Sales'].sum()
    total_sales_week2 = week2_sales_data['Sales'].sum()

    sales_difference = total_sales_week2 - total_sales_week1

    # Compute sales difference per product (group by product and calculate the difference)
    total_sales_per_product = data.groupby('Product')['Sales'].sum().reset_index()
    total_sales_per_product['Difference'] = total_sales_per_product['Sales'].diff()
    data = data.merge(total_sales_per_product[['Product', 'Difference']], on='Product')
    data = data.dropna(subset=['Difference'])

    # Convert categorical features to dummy variables
    data = pd.get_dummies(data, columns=['IntraPortfolioCannibalization', 'PantryLoadingEffect'], drop_first=True)

    # Define features (X) and target (y)
    X = data[['Discount', 'Sales', 'IntraPortfolioCannibalization_Yes', 'PantryLoadingEffect_Yes']]
    y = data['Difference']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Compute standardized coefficients and their relative importance
    standardized_coefficients = model.coef_
    relative_importance = abs(standardized_coefficients) / sum(abs(standardized_coefficients)) * 100

    # Calculate contribution amounts based on relative importance
    contribution_amounts = (relative_importance / 100) * sales_difference

    # Prepare data for the waterfall chart
    waterfall_data = pd.DataFrame({
        'Category': ['Total Sales Week 1', 'Discount', 'Sales', 'IntraPortfolioCannibalization', 'PantryLoadingEffect', 'Total Sales Week 2'],
        'Amount': [total_sales_week1, contribution_amounts[0], contribution_amounts[1], contribution_amounts[2], contribution_amounts[3], total_sales_week2]
    })

    # Calculate positions for each bar in the waterfall chart
    cumulative_amount = [waterfall_data['Amount'][0]]  # Start with Total Sales Week 1
    for i in range(1, len(waterfall_data) - 1):
        cumulative_amount.append(cumulative_amount[-1] + waterfall_data['Amount'][i])
    cumulative_amount.append(waterfall_data['Amount'].iloc[-1])

    # Plot waterfall chart
    st.subheader("Waterfall Chart of Sales Difference and Feature Contributions")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the initial value (Total Sales Week 1)
    ax.bar(0, waterfall_data['Amount'][0], color='blue')
    ax.text(0, waterfall_data['Amount'][0] + 500, f"{waterfall_data['Amount'][0]:,.0f}", ha='center', va='bottom')

    # Plot contributions
    for i in range(1, len(waterfall_data) - 1):
        color = 'green' if waterfall_data['Amount'][i] > 0 else 'red'
        ax.bar(i, waterfall_data['Amount'][i], bottom=cumulative_amount[i - 1], color=color)
        ax.text(i, cumulative_amount[i - 1] + waterfall_data['Amount'][i] + 500, f"{waterfall_data['Amount'][i]:,.0f}", ha='center', va='bottom')

    # Plot the final value (Total Sales Week 2)
    ax.bar(len(waterfall_data) - 1, waterfall_data['Amount'].iloc[-1], color='blue')
    ax.text(len(waterfall_data) - 1, waterfall_data['Amount'].iloc[-1] + 500, f"{waterfall_data['Amount'].iloc[-1]:,.0f}", ha='center', va='bottom')

    # Customize chart appearance
    ax.set_title('Waterfall Chart of Sales Difference and Feature Contributions')
    ax.set_ylabel('Sales Amount')
    ax.set_xticks(range(len(waterfall_data)))
    ax.set_xticklabels(waterfall_data['Category'], rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.set_ylim(0, 80000)  # Set y-axis range

    # Show the chart
    st.pyplot(fig)

else:
    st.warning("Please upload a CSV file to proceed.")