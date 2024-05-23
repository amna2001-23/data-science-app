import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# Global data variable to hold the uploaded data
if 'data' not in st.session_state:
    st.session_state['data'] = None

if 'target' not in st.session_state:
    st.session_state['target'] = None

if 'inputs' not in st.session_state:
    st.session_state['inputs'] = None

def data_collection():
    st.header("Data Collection")
    data_type = st.selectbox("Select type of data source:", ["CSV", "Excel"])

    if data_type == "CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state['data'] = df
            st.success("CSV file uploaded successfully!")

    elif data_type == "Excel":
        uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            st.session_state['data'] = df
            st.success("Excel file uploaded successfully!")

def data_inspection():
    st.header("Data Inspection")
    if st.session_state['data'] is None:
        st.warning("Please upload data from the Data Collection section first.")
        return

    df = st.session_state['data']

    st.subheader("Preview the Data")
    preview_option = st.selectbox("Select number of rows to preview:", [10, 20, 30])
    st.write("#### Head")
    st.dataframe(df.head(preview_option))
    st.write("#### Tail")
    st.dataframe(df.tail(preview_option))

    st.subheader("Data Information")
    st.write(f"Shape of the Data: {df.shape}")
    st.write("Column Names:")
    st.write(df.columns.tolist())
    st.write("Data Types:")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Duplicate Values")
    st.write(df.duplicated().sum())

    st.subheader("Categorize Data Types")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    ordinal_columns = []  # This can be customized based on specific dataset characteristics

    st.write("Numeric Columns:")
    st.write(numeric_columns)
    st.write("Categorical Columns:")
    st.write(categorical_columns)
    st.write("Ordinal Columns: (customized as needed)")
    st.write(ordinal_columns)

def data_cleaning():
    st.header("Data Cleaning")

    # Check if data is uploaded
    if st.session_state.get('data') is None:
        st.warning("Please upload data from the Data Collection section first.")
        return

    df = st.session_state['data']

    # Handling Missing Values
    st.subheader("Handling Missing Values")
    columns = st.multiselect("Select columns to handle missing values:", df.columns)
    missing_value_option = st.selectbox("Select method to handle missing values:", ["Mean", "Median", "Mode", "Remove"])

    if columns:
        if missing_value_option == "Mean":
            if st.button("Fill missing values with mean"):
                for column in columns:
                    df[column].fillna(df[column].mean(), inplace=True)
                st.session_state['data'] = df
                st.success("Missing values filled with mean.")
        
        elif missing_value_option == "Median":
            if st.button("Fill missing values with median"):
                for column in columns:
                    df[column].fillna(df[column].median(), inplace=True)
                st.session_state['data'] = df
                st.success("Missing values filled with median.")
        
        elif missing_value_option == "Mode":
            if st.button("Fill missing values with mode"):
                for column in columns:
                    df[column].fillna(df[column].mode().iloc[0], inplace=True)
                st.session_state['data'] = df
                st.success("Missing values filled with mode.")
        
        elif missing_value_option == "Remove":
            if st.button("Remove rows with missing values"):
                df.dropna(subset=columns, inplace=True)
                st.session_state['data'] = df
                st.success("Rows with missing values removed.")
    else:
        st.info("Please select at least one column.")

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Remove Duplicate Values
    st.subheader("Remove Duplicate Values")
    if st.button("Drop duplicate rows"):
        df.drop_duplicates(inplace=True)
        st.session_state['data'] = df
        st.success("Duplicate rows dropped.")

    st.subheader("Duplicate Values")
    st.write(df.duplicated().sum())

    # Outlier Detection
    st.subheader("Outlier Detection")
    if st.button("Detect and Show Outliers"):
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        outliers = {}
        for col in numeric_columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        
        for col, outlier_data in outliers.items():
            if not outlier_data.empty:
                st.write(f"Outliers in {col}:")
                st.write(outlier_data)

    # Outlier Removal
    st.subheader("Outlier Removal")
    if st.button("Remove Outliers"):
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        for col in numeric_columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        st.session_state['data'] = df
        st.success("Outliers removed.")
    
        # Display box plots after removing outliers
        st.subheader("Box Plots After Removing Outliers")
        for col in numeric_columns:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            st.pyplot(fig)

    # Data Transformation
    st.subheader("Data Transformation")
    transformation_option = st.selectbox("Select transformation method:", ["Log Transformation", "Standardization", "Normalization"])

    if transformation_option == "Log Transformation":
        if st.button("Apply Log Transformation"):
            for col in df.select_dtypes(include=['number']).columns:
                df[col] = df[col].apply(lambda x: np.log(x + 1))
            st.session_state['data'] = df
            st.success("Log transformation applied.")

    elif transformation_option == "Standardization":
        if st.button("Apply Standardization"):
            scaler = StandardScaler()
            df[df.select_dtypes(include=['number']).columns] = scaler.fit_transform(df.select_dtypes(include=['number']))
            st.session_state['data'] = df
            st.success("Standardization applied.")

    elif transformation_option == "Normalization":
        if st.button("Apply Normalization"):
            scaler = MinMaxScaler()
            df[df.select_dtypes(include=['number']).columns] = scaler.fit_transform(df.select_dtypes(include=['number']))
            st.session_state['data'] = df
            st.success("Normalization applied.")

    # Encode Categorical Variables
    st.subheader("Encode Categorical Variables")
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        target_column = st.selectbox("Select target column for encoding:", categorical_columns)
        if target_column:
            if st.button("Encode Target Variable"):
                le = LabelEncoder()
                df[target_column] = le.fit_transform(df[target_column])
                st.session_state['data'] = df
                st.success(f"Target column '{target_column}' encoded.")
    else:
        st.info("No categorical columns available for encoding.")

    if st.session_state['data'] is not None:
        st.subheader("Transformed Data")
        st.write(df)

    # Select Target and Input Features
    def select_target_and_features():
        st.header("Select Target and Input Features")

        if st.session_state['data'] is None:
            st.warning("Please upload data from the Data Collection section first.")
            return

        df = st.session_state['data']
        columns = df.columns.tolist()

        st.session_state['target'] = st.selectbox("Select Target Column:", columns)
        st.session_state['inputs'] = st.multiselect("Select Input Feature Columns:", [col for col in columns if col != st.session_state['target']])

        if st.button("Set Target and Features"):
            st.success(f"Target and features set.\nTarget: {st.session_state['target']}\nFeatures: {st.session_state['inputs']}")

    select_target_and_features()

    # Download Cleaned Data
    st.subheader("Download Cleaned Data")
    download_format = st.selectbox("Select file format for download:", ["CSV", "Excel"])
    if download_format == "CSV":
        cleaned_data = df.to_csv(index=False)
        st.download_button(label="Download as CSV", data=cleaned_data, file_name="cleaned_data.csv", mime="text/csv")
    elif download_format == "Excel":
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.save()
        processed_data = output.getvalue()
        st.download_button(label="Download as Excel", data=processed_data, file_name="cleaned_data.xlsx", mime="application/vnd.ms-excel")

def data_analysis():
    st.header("Data Analysis")
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.warning("Please upload data from the Data Collection section first.")
        return

    if 'target' not in st.session_state or st.session_state['target'] is None:
        st.warning("Please select target and input features first.")
        return

    df = st.session_state['data']
    target_column = st.session_state['target']
    input_columns = st.session_state['inputs']

    st.subheader("Select Analysis Technology")
    analysis_technology = st.selectbox("Select technology:", [
        "Exploratory Data Analysis (EDA)",
        "Machine Learning",
        "Data Visualization"
    ])

    if analysis_technology == "Exploratory Data Analysis (EDA)":
        st.write("Exploratory Data Analysis (EDA)")

        # Display the first few rows of the dataframe
        st.write("First few rows of the dataset:")
        st.write(df.head())

        # Display basic statistics
        st.write("Basic statistics of the dataset:")
        st.write(df.describe())

        # Display missing values
        st.write("Missing values in the dataset:")
        st.write(df.isnull().sum())

        # Display correlation matrix
        st.write("Correlation matrix of the dataset:")
        corr = df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, ax=ax)
        st.pyplot(fig)

        # Pairplot for visualizing relationships
        st.write("Pairplot of the dataset:")
        pairplot_fig = sns.pairplot(df)
        st.pyplot(pairplot_fig)

    elif analysis_technology == "Data Visualization":
        st.write("Data Visualization")

        # Select type of plot
        plot_type = st.selectbox("Select plot type:", [
            "Bar Chart",
            "Line Chart",
            "Pie Chart",
            "Scatter Plot",
            "Histogram",
            "Heatmap"
        ])

        # Select columns for the plot based on data types
        if plot_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Heatmap"]:
            if plot_type == "Histogram":
                selected_columns = st.multiselect("Select numerical columns to plot", df.select_dtypes(include=['number']).columns.tolist())
            elif plot_type == "Heatmap":
                selected_columns = st.multiselect("Select columns to plot", df.columns.tolist())
            else:
                selected_columns = st.multiselect("Select columns to plot", df.columns.tolist(), max_selections=2)
        elif plot_type == "Pie Chart":
            selected_columns = st.selectbox("Select categorical column to plot", df.select_dtypes(include=['object', 'category']).columns.tolist())

        if st.button("Generate Plot"):
            if plot_type == "Bar Chart":
                if len(selected_columns) != 2:
                    st.error("Please select exactly 2 columns for Bar Chart.")
                else:
                    fig, ax = plt.subplots()
                    sns.barplot(x=df[selected_columns[0]], y=df[selected_columns[1]], ax=ax)
                    st.pyplot(fig)

            elif plot_type == "Line Chart":
                if len(selected_columns) != 2:
                    st.error("Please select exactly 2 columns for Line Chart.")
                else:
                    fig, ax = plt.subplots()
                    sns.lineplot(x=df[selected_columns[0]], y=df[selected_columns[1]], ax=ax)
                    st.pyplot(fig)

            elif plot_type == "Pie Chart":
                fig, ax = plt.subplots()
                df[selected_columns].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
                st.pyplot(fig)

            elif plot_type == "Scatter Plot":
                if len(selected_columns) != 2:
                    st.error("Please select exactly 2 columns for Scatter Plot.")
                else:
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=df[selected_columns[0]], y=df[selected_columns[1]], ax=ax)
                    st.pyplot(fig)

            elif plot_type == "Histogram":
                if len(selected_columns) != 1:
                    st.error("Please select exactly 1 column for Histogram.")
                else:
                    fig, ax = plt.subplots()
                    sns.histplot(df[selected_columns[0]], ax=ax)
                    st.pyplot(fig)

            elif plot_type == "Heatmap":
                if len(selected_columns) < 2:
                    st.error("Please select at least 2 columns for Heatmap.")
                else:
                    fig, ax = plt.subplots()
                    sns.heatmap(df[selected_columns].corr(), annot=True, ax=ax)
                    st.pyplot(fig)

    elif analysis_technology == "Machine Learning":
        st.write("Machine Learning Analysis")

        # Step 1: Select the target column
        target_column = st.selectbox("Select target column:", df.columns.tolist(), index=df.columns.tolist().index(st.session_state['target']))

        # Step 2: Select input features
        input_columns = st.multiselect("Select input features:", df.columns.tolist(), default=st.session_state['inputs'])

        if not target_column or not input_columns:
            st.error("Please select both target and input features to proceed.")
            return

        # Step 3: Select task type
        task_type = st.radio("Select task type:", ["Classification", "Regression"])

        # Step 4: Handle Missing Values
        X = df[input_columns]
        y = df[target_column]

        # Impute missing values in X
        imputer_X = SimpleImputer(strategy='mean')
        X = imputer_X.fit_transform(X)

        # Check for missing values in y and handle them
        if y.isnull().any():
            imputer_y = SimpleImputer(strategy='most_frequent' if y.dtype == 'O' else 'mean')
            y = imputer_y.fit_transform(y.values.reshape(-1, 1)).ravel()

        # Step 5: Data Splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if task_type == "Classification":
            # Step 6: Model Training for Classification
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            
            # Step 7: Model Evaluation for Classification
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write("Accuracy:", accuracy)
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            
            # Confusion Matrix
            st.write("Confusion Matrix:")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)
            
        else:
            # Step 6: Model Training for Regression
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            
            # Step 7: Model Evaluation for Regression
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write("Mean Squared Error:", mse)
            st.write("R-squared:", r2)
            
            # Plot actual vs predicted
            st.write("Actual vs Predicted:")
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=y_pred, ax=ax)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

    # Download Cleaned Data
    st.subheader("Download Cleaned Data")
    download_format = st.selectbox("Select file format for download:", ["CSV", "Excel"])
    if download_format == "CSV":
        cleaned_data = df.to_csv(index=False)
        st.download_button(label="Download as CSV", data=cleaned_data, file_name="cleaned_data.csv", mime="text/csv")
    elif download_format == "Excel":
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.save()
        processed_data = output.getvalue()
        st.download_button(label="Download as Excel", data=processed_data, file_name="cleaned_data.xlsx", mime="application/vnd.ms-excel")

# Streamlit app structure
st.title("Data Science App")
menu = ["Data Collection", "Data Inspection", "Data Cleaning", "Data Analysis"]
choice = st.sidebar.selectbox("Select Activity", menu)

if choice == "Data Collection":
    data_collection()
elif choice == "Data Inspection":
    data_inspection()
elif choice == "Data Cleaning":
    data_cleaning()
elif choice == "Data Analysis":
    data_analysis()
