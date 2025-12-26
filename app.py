import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import churn_model
import json

st.set_page_config(page_title="Churn Prediction", layout="wide")

@st.cache_resource
def load_resources():
    model = joblib.load('data/churn_model.pkl')
    try:
        train_df = pd.read_csv('data/customer_churn_dataset-training-master.csv')
    except Exception:
        train_df = pd.read_csv('data/customer_churn_dataset-training-master.csv')
    _, encoders, scaler, _ = churn_model.preprocess_data(train_df, is_train=True)
    return model, train_df, encoders, scaler

try:
    model, train_df, encoders, scaler = load_resources()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

st.title("Churn Prediction Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Insights", "Train", "Model Report"])

with tab1:
    st.header("Real-time Prediction")
    st.markdown("Enter customer details to predict churn probability.")
    
    col1, col2, col3 = st.columns(3)
    
    gender_opts = ["Male", "Female"]
    sub_opts = ["Basic", "Standard", "Premium"]
    contract_opts = ["Monthly", "Quarterly", "Annual"]

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender_sel = st.selectbox("Gender", gender_opts)
        tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
        usage_frequency = st.number_input("Usage Frequency (Last Month)", min_value=0, value=15)
        
    with col2:
        payment_delay = st.number_input("Payment Delay (Days)", min_value=0, value=0)
        sub_sel = st.selectbox("Subscription Type", sub_opts)
        contract_sel = st.selectbox("Contract Length", contract_opts)
        
    with col3:
        last_interaction = st.number_input("Last Interaction (Days ago)", min_value=0, value=5)

    if st.button("Predict Churn", type="primary"):
        input_data = {
            'Age': [age],
            'Gender': [gender_sel],
            'Tenure': [tenure],
            'Usage Frequency': [usage_frequency],
            'Payment Delay': [payment_delay],
            'Subscription Type': [sub_sel],
            'Contract Length': [contract_sel],
            'Last Interaction': [last_interaction]
        }
        input_df = pd.DataFrame(input_data)
        
        try:
            input_processed = churn_model.preprocess_data(input_df, is_train=False, label_encoders=encoders, scaler=scaler)
            
            prediction = model.predict(input_processed)[0]
            probability = model.predict_proba(input_processed)[0][1]
            
            st.divider()
            
            if prediction == 1:
                st.error(f"**High Churn Risk!** Probability: {probability:.1%}")
            else:
                st.success(f"**Low Churn Risk.** Probability: {probability:.1%}")
                
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.header("Dataset Insights")
    
    st.markdown("Use the filters below to explore specific segments.")
    f1, f2 = st.columns(2)
    with f1:
        contract_opts = sorted([x for x in train_df['Contract Length'].unique() if pd.notna(x)])
        sel_contract = st.multiselect("Filter by Contract", options=contract_opts, default=contract_opts)
    
    with f2:
        gender_opts = sorted([x for x in train_df['Gender'].unique() if pd.notna(x)])
        sel_gender = st.multiselect("Filter by Gender", options=gender_opts, default=gender_opts)
    
    if not sel_contract:
        sel_contract = contract_opts
    if not sel_gender:
        sel_gender = gender_opts

    filtered_df = train_df[
        (train_df['Contract Length'].isin(sel_contract)) & 
        (train_df['Gender'].isin(sel_gender))
    ]
    
    total_customers = len(filtered_df)
    churn_rate = filtered_df['Churn'].mean() if not filtered_df.empty else 0
    avg_spend = filtered_df['Total Spend'].mean() if not filtered_df.empty else 0
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Customers (Filtered)", f"{total_customers:,}")
    m2.metric("Overall Churn Rate", f"{churn_rate:.1%}")
    m3.metric("Average Total Spend", f"${avg_spend:.2f}")
    
    st.divider()
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
    else:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Churn Distribution")
            fig_pie = px.pie(filtered_df, names='Churn', title="Churn Distribution", 
                             color_discrete_sequence=['#ef553b', '#636efa'])
            fig_pie.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_b:
            st.subheader("Churn by Contract Length")
            fig_contract = px.histogram(filtered_df, x='Contract Length', color='Churn', 
                                        barmode='group', title="Churn by Contract Length",
                                        color_discrete_sequence=['#636efa', '#ef553b'])
            st.plotly_chart(fig_contract, use_container_width=True)

        st.subheader("Number of Customers per Plan")
        plan_counts = filtered_df['Subscription Type'].value_counts().reset_index()
        plan_counts.columns = ['Subscription Type', 'Count']
        fig_plans = px.bar(plan_counts, x='Subscription Type', y='Count', 
                           text='Count', title="Number of Customers per Plan",
                           color='Subscription Type',
                           color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_plans, use_container_width=True)
            
        st.divider()
        
        st.subheader("Correlation Analysis")
        st.write("How different features correlate with Churn (and each other) in this segment.")
        
        numeric_df = filtered_df.select_dtypes(include=['number'])
        if not numeric_df.empty and len(numeric_df) > 1:
            corr_matrix = numeric_df.corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                 color_continuous_scale='RdBu_r', origin='lower',
                                 title="Correlation Heatmap")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough data for correlation analysis.")
        
        st.divider()

        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("Churn Rate by Age Group")
            
            df_age = filtered_df.copy()
            df_age['Age Group'] = pd.cut(df_age['Age'], bins=[0, 25, 35, 45, 55, 65, 100], 
                                         labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
            
            age_churn = df_age.groupby('Age Group', observed=True)['Churn'].mean().reset_index()
            
            fig_age = px.bar(age_churn, x='Age Group', y='Churn', 
                             title="Churn Rate by Age Group",
                             color='Churn', color_continuous_scale='Reds')
            fig_age.layout.yaxis.tickformat = ',.0%'
            st.plotly_chart(fig_age, use_container_width=True)

        with col_d:
            st.subheader("Churn Rate by Tenure Group")
            
            df_tenure = filtered_df.copy()
            df_tenure['Tenure Group'] = pd.cut(df_tenure['Tenure'], bins=[0, 6, 12, 24, 36, 48, 60, 100], 
                                            labels=['0-6 m', '6-12 m', '1-2 y', '2-3 y', '3-4 y', '4-5 y', '5+ y'])
            
            tenure_churn = df_tenure.groupby('Tenure Group', observed=True)['Churn'].mean().reset_index()
            
            fig_tenure = px.bar(tenure_churn, x='Tenure Group', y='Churn', 
                                title="Churn Rate by Tenure",
                                color='Churn', color_continuous_scale='Reds')
            fig_tenure.layout.yaxis.tickformat = ',.0%'
            st.plotly_chart(fig_tenure, use_container_width=True)

        st.divider()
        
        st.subheader("Feature Distributions")
        feature_map_values = ['Age', 'Tenure', 'Total Spend', 'Usage Frequency', 'Support Calls']
        feature_to_plot_sel = st.selectbox("Select Feature to Visualize", feature_map_values)
        
        fig_dist = px.box(filtered_df, x='Churn', y=feature_to_plot_sel, color='Churn', 
                          title=f'{feature_to_plot_sel}',
                          color_discrete_sequence=['#636efa', '#ef553b'])
        st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    st.header("Training Details")
    
    st.subheader("Model Parameters")
    st.json(model.get_params())
    
    st.divider()
    
    st.subheader("Training Metrics")
    try:
        with open('data/metrics.json', 'r') as f:
            metrics = json.load(f)
        st.json(metrics)
    except FileNotFoundError:
        st.warning("Metrics file not found.")
        
    st.divider()
    
    st.subheader("Training Data Sample")
    st.dataframe(train_df.head(100))

with tab4:
    st.header("Model Development Report")
    
    report_content = """
### 1. Data Leakage Detection
During initial analysis, I identified that the model had "perfect" performance (~100% accuracy), which is rare in real data.

I found two main culprits:
*   **Support Calls**: Highly correlated with churn. Likely included calls made during cancellation.
*   **Total Spend**: Churning customers spent less simply because they left earlier.

### 2. Robustness Test
I performed experiments protecting against these variables:

| Scenario | Removed Variables | Accuracy | Observation |
| :--- | :--- | :--- | :--- |
| **Original** | None | ~100% | Probable Data Leakage. |
| **Stress Test** | Support Calls | ~95% | Still very strong. |
| **Production Model** | Support Calls + Total Spend | **~90%** | Realistic and robust performance. |

### 3. Final Decision
I chose the **Robust Model (90% Accuracy)** for production. It is more reliable for predicting future behavior of *new* customers.
    """
    
    st.markdown(report_content)
