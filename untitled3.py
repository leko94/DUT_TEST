
import os
import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression

# File path to the Excel file
file_path = 'Students.xlsx'

# Initialize the Dash app
app = dash.Dash(__name__)

# Load data from Sheet1
df1 = pd.read_excel(file_path, sheet_name='Sheet1')
df1.columns = ['Year', 'Actual', 'Planned']
df1['Actual'] = pd.to_numeric(df1['Actual'], errors='coerce')
df1['Planned'] = pd.to_numeric(df1['Planned'], errors='coerce')
df1 = df1.dropna()

# Create the first line chart
fig1 = px.line(
    df1,
    x='Year',
    y=['Actual', 'Planned'],
    labels={'value': 'Headcount', 'variable': 'Type'},
    title='Headcount Enrolment: Planned vs Achieved (2014-2022)',
    markers=True
)
fig1.update_layout(
    autosize=False,
    width=800,
    height=600
)

# Create the second chart with linear regression forecast
def create_linear_regression_forecast_chart(df):
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Actual'] = pd.to_numeric(df['Actual'], errors='coerce')
    df = df.dropna()
    X = df[['Year']].values.reshape(-1, 1)
    y = df['Actual'].values
    model = LinearRegression().fit(X, y)
    future_years = np.array([2030]).reshape(-1, 1)
    forecast_value = model.predict(future_years)[0]
    fig2 = px.line(
        df,
        x='Year',
        y='Actual',
        title='Linear Regression Forecast for 2030',
        labels={'Year': 'Year', 'Actual': 'Headcount'},
        markers=True
    )
    fig2.add_scatter(
        x=[2030],
        y=[forecast_value],
        mode='markers+text',
        text=['2030'],
        textposition='top right',
        marker=dict(color='red', size=10),
        name='Forecast'
    )
    return fig2

fig2 = create_linear_regression_forecast_chart(df1)

# Load data from Sheet2 and get the title from cell A1
df2 = pd.read_excel(file_path, sheet_name='Sheet2', header=1)  # Skip the header row
title2 = pd.read_excel(file_path, sheet_name='Sheet2', header=None).iloc[0, 0]  # Get the title from cell A1

# Compute the Difference
df2['Difference'] = df2['Actual'] - df2['Planned']

# Create the bar graph for Sheet2
fig3 = px.bar(
    df2,
    x='Departments',
    y=['Planned', 'Actual'],
    title=title2,
    labels={'value': 'Number of Students', 'variable': 'Type'},
    text='Difference'  # Display the difference on the bars
)
fig3.update_layout(
    autosize=False,
    width=800,
    height=600
)

# Load data from Sheet3
df3 = pd.read_excel(file_path, sheet_name='Sheet3', header=None)

# Extract title and relevant columns
title3 = df3.iloc[0, 0]  # Title from cell A1
df3.columns = ['Department', '2014', '2022', 'Ignore', 'Growth']  # Set column names manually
df3 = df3.iloc[2:]  # Skip rows before actual data
df3 = df3[['Department', '2014', '2022', 'Growth']]  # Select relevant columns

# Convert columns to numeric values and round the Growth values
df3['2014'] = pd.to_numeric(df3['2014'], errors='coerce')
df3['2022'] = pd.to_numeric(df3['2022'], errors='coerce')
df3['Growth'] = pd.to_numeric(df3['Growth'], errors='coerce').round(2)

# Melt the DataFrame for better plotting
df3_melted = df3.melt(id_vars='Department', value_vars=['2014', '2022'], var_name='Year', value_name='Value')

# Create the bar graph for Sheet3
fig4 = px.bar(
    df3_melted,
    x='Department',
    y='Value',
    color='Year',
    text=df3_melted['Department'].map(df3.set_index('Department')['Growth']),  # Add rounded Growth as text
    title=title3,
    labels={'Value': 'Number of Students', 'Year': 'Year'},
    color_discrete_map={'2014': 'blue', '2022': 'green'}
)
fig4.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig4.update_layout(
    autosize=False,
    width=800,
    height=600,
    legend_title_text='Year'
)

# Load data from Sheet4
df4 = pd.read_excel(file_path, sheet_name='Sheet4')

# Extract title and labels
title4 = '% African Students'  # Title from cell A1
x_label4 = 'Year'  # X-axis label from cell A2
y_label4 = '% African Students'  # Y-axis label from cell B2

# Set headers and extract data
df4.columns = [x_label4, y_label4]
df4 = df4.iloc[1:]  # Skip rows before actual data

# Convert '% African Students' to float after stripping '%'
df4[y_label4] = df4[y_label4].astype(str).str.rstrip('%').astype(float)

# Create the line graph for Sheet4
fig5 = px.line(
    df4,
    x=x_label4,
    y=y_label4,
    title=title4,
    labels={x_label4: 'Year', y_label4: '% African Students'},
    markers=True
)
fig5.update_layout(
    autosize=False,
    width=800,
    height=600
)

# Load data from Sheet5
df5 = pd.read_excel(file_path, sheet_name='Sheet5')

# Extract title and labels
title5 = "Percentage Female Students"  # Title from cell A1
x_label5 = 'Year'  # X-axis label from cell A2
y_label5 = 'Percentage Female Students'  # Y-axis label from cell B2

# Set headers and extract data
df5.columns = [x_label5, y_label5]
df5 = df5.iloc[1:]  # Skip rows before actual data

# Convert '% Female Students' to float after stripping '%'
df5[y_label5] = df5[y_label5].astype(str).str.rstrip('%').astype(float)

# Create the line graph for Sheet5 with a different color
fig6 = px.line(
    df5,
    x=x_label5,
    y=y_label5,
    title=title5,
    labels={x_label5: 'Year', y_label5: 'Percentage Female Students'},
    markers=True,
    line_shape='linear'
)
fig6.update_layout(
    autosize=False,
    width=800,
    height=600,
    plot_bgcolor='lightgray'  # Different color for distinction
)

# Load data from Sheet6
df6 = pd.read_excel(file_path, sheet_name='Sheet6')

# Extract title and labels
title6 = "Faculty Postgraduate Enrolment"  # Title from cell A1
x_label6 = 'Year'  # X-axis label from cell A2
y_label6 = '% Enrolment'  # Y-axis label from cell B2

# Set headers and extract data
df6.columns = [x_label6, y_label6]
df6 = df6.iloc[1:]  # Skip rows before actual data

# Convert '% Enrolment' to float after stripping '%'
df6[y_label6] = df6[y_label6].astype(str).str.rstrip('%').astype(float)

# Create the line graph
fig7 = go.Figure()

# Add the line trace
fig7.add_trace(go.Scatter(
    x=df6[x_label6],
    y=df6[y_label6],
    mode='lines',
    line=dict(color='purple'),  # Set the color to purple
    name='Enrolment Percentage'
))

# Update layout for the graph from Sheet6
fig7.update_layout(
    title=title6,
    xaxis_title=x_label6,
    yaxis_title=y_label6,
    autosize=False,
    width=800,
    height=600,
    plot_bgcolor='lightgray'  # Different color for distinction
)

# Load data from Sheet7
df7 = pd.read_excel(file_path, sheet_name='Sheet7')
df7.columns = df7.columns.str.strip()

# Create the bar chart with actual percentages on top of each bar
fig8 = px.bar(
    df7, 
    x='Department', 
    y=['UG', 'PG upto Masters', 'PG'], 
    title='Enrolment by Level',
    labels={'value': 'Percentage', 'variable': 'Enrolment Level'},
    barmode='group',
    text_auto=True  # Add text_auto=True to display values on bars
)

# Load data from Sheet8
df8 = pd.read_excel(file_path, sheet_name='Sheet8')
df8.columns = df8.columns.str.strip()

# Melt the DataFrame for better plotting
df8_melted = df8.melt(id_vars='Departments', value_vars=df8.columns[1:-1], var_name='Year', value_name='Percentage')
df8_melted['Percentage'] = df8_melted['Percentage'].astype(str).str.rstrip('%').astype(float)

# Create the line graph for Sheet8
fig9 = px.line(
    df8_melted,
    x='Year',
    y='Percentage',
    color='Departments',
    title='Postgraduate (M+D) Enrolment',
    markers=True
)
fig9.update_layout(
    autosize=False,
    width=800,
    height=600,
    legend_title_text='Departments'
)

# Create the bar graph for Departments and Difference 2014 vs 2022
fig10 = px.bar(
    df8,
    x='Departments',
    y='Difference 2014 vs 2022',
    title='Difference 2014 vs 2022 by Department',
    labels={'Departments': 'Departments', 'Difference 2014 vs 2022': 'Difference'},
    text='Difference 2014 vs 2022'
)
fig10.update_layout(
    autosize=False,
    width=800,
    height=600
)
fig10.update_traces(texttemplate='%{text:.2f}', textposition='outside')

# Load data from Sheet9
df9 = pd.read_excel(file_path, sheet_name='Sheet9')
df9.columns = df9.columns.str.strip()

# Melt the DataFrame for better plotting
df9_melted = df9.melt(id_vars='Derpatnment', value_vars=df9.columns[1:], var_name='Year', value_name='No. of Postgraduate enrolment')

# Create the bar graph for Sheet9
fig11 = px.bar(
    df9_melted,
    x='Year',
    y='No. of Postgraduate enrolment',
    color='Derpatnment',
    title='Postgraduate Enrolment - Actual Student Numbers',
    text='No. of Postgraduate enrolment',
    barmode='group'
)
fig11.update_layout(
    autosize=False,
    width=1000,  # Increased the width
    height=700,  # Increased the height
    legend_title_text='Departments'
)
fig11.update_traces(texttemplate='%{text}', textposition='outside')

# Load data from Sheet10
df10 = pd.read_excel(file_path, sheet_name='Sheet10')
df10.columns = df10.columns.str.strip()

# Melt the DataFrame for better plotting
df10_melted = df10.melt(id_vars='Department', value_vars=df10.columns[1:], var_name='Year', value_name='Percentage')
df10_melted['Percentage'] = df10_melted['Percentage'].astype(str).str.rstrip('%').astype(float)

# Create the bar graph for Sheet10
fig12 = px.bar(
    df10_melted,
    x='Year',
    y='Percentage',
    color='Department',
    title='International student Postgraduate enrolment',
    text='Percentage',
    barmode='group'
)
fig12.update_layout(
    autosize=False,
    width=1200,  # Increased the width
    height=800,  # Increased the height
    legend_title_text='Departments'
)
fig12.update_traces(texttemplate='%{text}', textposition='outside')

# Load data from Sheet11
df11 = pd.read_excel(file_path, sheet_name='Sheet11')
df11.columns = df11.columns.str.strip()

# Melt the DataFrame for better plotting
df11_melted = df11.melt(id_vars='Department', value_vars=df11.columns[1:], var_name='Year', value_name='No. of Postgraduate enrolment')

# Create the bar graph for Sheet11
fig13 = px.bar(
    df11_melted,
    x='Year',
    y='No. of Postgraduate enrolment',
    color='Department',
    title='International Students Postgraduate Enrolment - Actual Numbers',
    text='No. of Postgraduate enrolment',
    barmode='group'
)
fig13.update_layout(
    autosize=False,
    width=1200,  # Increased the width
    height=800,  # Increased the height
    legend_title_text='Departments'
)
fig13.update_traces(texttemplate='%{text}', textposition='outside')

# Define the layout of the Dash app
app.layout = html.Div(style={'textAlign': 'center'}, children=[
    html.H1("Headcount Enrolment Dashboard"),

    html.Div([
        dcc.Graph(figure=fig1)  # First chart
    ], style={'margin-bottom': '20px'}),  # Add space below the first chart

    html.Div([
        dcc.Graph(figure=fig2)  # Second chart
    ], style={'margin-bottom': '20px'}),  # Add space below the second chart

    html.Div([
        dcc.Graph(figure=fig3)  # Bar chart from Sheet2
    ], style={'margin-bottom': '20px'}),  # Add space below the third chart

    html.Div([
        dcc.Graph(figure=fig4)  # Bar chart from Sheet3
    ], style={'margin-bottom': '20px'}),  # Add space below the fourth chart

    html.Div([
        dcc.Graph(figure=fig5)  # Line chart from Sheet4
    ], style={'margin-bottom': '20px'}),  # Add space below the fifth chart

    html.Div([
        dcc.Graph(figure=fig6)  # Line chart from Sheet5
    ], style={'margin-bottom': '20px'}),  # Add space below the sixth chart

    html.Div([
        dcc.Graph(figure=fig7)  # Line chart from Sheet6
    ], style={'margin-bottom': '20px'}),  # Add space below the seventh chart

    html.Div([
        dcc.Graph(figure=fig8)  # Bar chart from Sheet7
    ], style={'margin-bottom': '20px'}),  # Add space below the eighth chart

    html.Div([
        dcc.Graph(figure=fig9)  # Line chart from Sheet8
    ], style={'margin-bottom': '20px'}),  # Add space below the ninth chart

    html.Div([
        dcc.Graph(figure=fig10)  # Bar chart for Difference 2014 vs 2022
    ], style={'margin-bottom': '20px'}),  # Add space below the tenth chart

    html.Div([
        dcc.Graph(figure=fig11)  # Bar chart from Sheet9
    ], style={'margin-bottom': '20px'}),  # Add space below the eleventh chart

    html.Div([
        dcc.Graph(figure=fig12)  # Bar chart from Sheet10
    ], style={'margin-bottom': '20px'}),  # Add space below the twelfth chart

    html.Div([
        dcc.Graph(figure=fig13)  # Bar chart from Sheet11
    ])
])

# Run the Dash app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=True, host='0.0.0.0', port=port)
