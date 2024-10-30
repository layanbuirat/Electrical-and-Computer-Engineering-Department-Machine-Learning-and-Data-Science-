#Rana Musa 1210007
#Leyan Buirat 
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import folium
import seaborn as sns

import plotly.express as px

# Define file path input
file_path = input("Please provide the file path to the dataset:e.g., C:\\Users\\hp\\Downloads\\Electric_Vehicle_Population_Data.csv) ")

# Load the dataset
data = pd.read_csv(file_path)

def document_missing_values(data):
    missing_count = data.isnull().sum()
    missing_percentage = (missing_count / len(data)) * 100
    missing_data = pd.DataFrame({'Missing Count': missing_count, 'Missing Percentage': missing_percentage})
    missing_data = missing_data[missing_data['Missing Count'] > 0]
    print("Missing Values Documentation:")
    print(missing_data)

def missing_attribute_data(attribute): 
    bool_series = data[attribute].isnull()  
    missing_count = bool_series.sum()
    filtered_data = data[bool_series]
    if missing_count > 0:
        print(f"Number of missing values in '{attribute}': {missing_count}")
        print(filtered_data)
    else:
        print(f"There are no missing values in '{attribute}'.")

def handle_missing_data():
    total_cells = data.size
    missing_cells = data.isnull().sum().sum()
    total_samples = len(data)
    samples_with_missing = data.isnull().any(axis=1).sum()
    percent_samples_with_missing = (samples_with_missing / total_samples) * 100
    
    print(f"Total cells: {total_cells}")
    print(f"Missing cells: {missing_cells}")
    print(f"Total samples: {total_samples}")
    print(f"Samples with missing data: {samples_with_missing} ({percent_samples_with_missing:.2f}%)")
    
    num_imputer = SimpleImputer(strategy='mean')
    data_imputed_num = pd.DataFrame(num_imputer.fit_transform(data.select_dtypes(include='number')))
    data_imputed_num.columns = data.select_dtypes(include='number').columns
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    data_imputed_cat = pd.DataFrame(cat_imputer.fit_transform(data.select_dtypes(include='object')))
    data_imputed_cat.columns = data.select_dtypes(include='object').columns
    
    data[data.select_dtypes(include='number').columns] = data_imputed_num
    data[data.select_dtypes(include='object').columns] = data_imputed_cat
    
    try:
        data.to_csv(file_path, index=False)
        print("Missing values handled by imputation.")
        document_missing_values(data)
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

def drop_missing_rows():
    total_samples_before = len(data)
    missing_cells_before = data.isnull().sum().sum()
    samples_with_missing_before = data.isnull().any(axis=1).sum()
    percent_samples_with_missing_before = (samples_with_missing_before / total_samples_before) * 100
    
    print(f"Before deletion:")
    print(f"Total samples: {total_samples_before}")
    print(f"Missing cells: {missing_cells_before}")
    print(f"Samples with missing data: {samples_with_missing_before} ({percent_samples_with_missing_before:.2f}%)\n")
    
    data_dropped = data.dropna()

    total_cells_after = data_dropped.size
    total_samples_after = len(data_dropped)

    print(f"After deletion:")
    print(f"Total samples: {total_samples_after}")
    print(f"Total cells: {total_cells_after}")

    data_dropped.to_csv(file_path, index=False)
    print("Rows with missing values have been dropped.")
    
    document_missing_values(data_dropped)

def one_hot_encoding():
    data = pd.read_csv(file_path)

    if 'Model' in data.columns:
        unique_Models = data['Model'].unique()
        print("Unique Models in the dataset:")
        print(unique_Models)
        
        original_Model = data[['Model']].copy()
        
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
        ohe_transformed = ohe.fit_transform(data[['Model']])
        
        data = data.drop(columns=['Model'])
        data = pd.concat([data, original_Model, ohe_transformed], axis=1)
        
        data.to_csv(file_path, index=False)
        print("One-hot encoding applied, and 'Model' column retained. Here's some of the modified data:")
        print(data.head())
    else:
        print("The 'Model' column is not found in the dataset.")


def normalize_feature(attribute):
  
    
    
    # Check if the attribute exists in the dataset
    if attribute in data.columns:
        # Save the original values of the attribute
        original_values = data[attribute].copy()
        
        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()
        
        # Reshape the data into 2D array (required for scaler)
        data_scaled = scaler.fit_transform(data[[attribute]])
        
        # Replace the original column with the normalized data
        data[attribute] = data_scaled
        
        # Save the modified DataFrame back to the original CSV file
        data.to_csv(file_path, index=False)
        
        # Print the normalized feature
        print(f"The '{attribute}' feature has been normalized using Min-Max scaling.")
       
        
        # Plot the original and normalized data
        plt.figure(figsize=(12, 6))
        
        # Plot original values
        plt.subplot(1, 2, 1)
        plt.hist(original_values, bins=30, color='blue', alpha=0.7)
        plt.title(f"Original '{attribute}' Distribution")
        plt.xlabel(attribute)
        plt.ylabel("Frequency")
        
        # Plot normalized values
        plt.subplot(1, 2, 2)
        plt.hist(data[attribute], bins=30, color='green', alpha=0.7)
        plt.title(f"Normalized '{attribute}' Distribution")
        plt.xlabel(attribute)
        plt.ylabel("Frequency")
        
        # Show the plots
        plt.tight_layout()
        plt.show()
        
    else:
        print(f"The attribute '{attribute}' is not found in the dataset.")

def descriptive_statistics(attribute):
  
    
    
    # Check if the attribute exists and is numeric
     if attribute in data.columns and pd.api.types.is_numeric_dtype(data[attribute]):
        # Calculate mean, median, and standard deviation
        mean = data[attribute].mean()
        median = data[attribute].median()
        std_dev = data[attribute].std()
        
        # Print the descriptive statistics
        print(f"Descriptive Statistics for '{attribute}':")
        print(f"Mean: {mean}")
        print(f"Median: {median}")
        print(f"Standard Deviation: {std_dev}")
     else:
        print(f"The attribute '{attribute}' is either not numeric or not found in the dataset.")






def plot_ev_distribution():
        
    
    # Count the number of EVs by County
    county_counts = data['County'].value_counts().reset_index()
    county_counts.columns = ['County', 'EV_Count']
    
    # Map county names to FIPS codes (needed for Plotly's choropleth)
   
   # Create a mapping dictionary for FIPS codes
    fips_map = {
        'Kitsap': '53035',
        'Snohomish': '53061',
        'King': '53033',
        'Thurston': '53067',
        'Yakima': '53077',
        'Skagit': '53057',
        'Chelan': '53007',
        'Stevens': '53065',
        'Kittitas': '53039',
        'Walla Walla': '53073',
        'Island': '53029',
        'Whitman': '53077',
        'Spokane': '53063',
        'Grant': '53027',
        'Clark': '53011',
        'Cowlitz': '53019',
        'Jefferson': '53035',
        'Clallam': '53011',
        'Klickitat': '53035',
        'Pierce': '53053',
        'Whatcom': '53073',
        'Grays Harbor': '53027',
        'Lewis': '53035',
        'Okanogan': '53045',
        'Pacific': '53041',
        'Franklin': '53011',
        'Skamania': '53055',
        'Pend Oreille': '53047',
        'Mason': '53043',
        'Benton': '53005',
        'San Juan': '53055',
        'Adams': '53001',
        'Douglas': '53015',
        'Macomb': '26099',
        'Lincoln': '53037',
        'Asotin': '53001',
        'Wahkiakum': '53059',
        'Polk': '41067',
        'San Diego': '06073',
        'Leavenworth': '53033',
        'Stafford': '51177',
        'Sonoma': '06097',
        'Columbia': '53009',
        'Oldham': '48237',
        'Orange': '06059',
        'District of Columbia': '11001',
        'Lee': '45063',
        'Ferry': '53013',
        'Goochland': '51077',
        'York': '51191',
        'Doña Ana': '35013',
        'Lake': '17097',
        'New London': '09009',
        'Kings': '06017',
        'Platte': '29019',
        'Collin': '48085',
        'Anne Arundel': '24003',
        'Burlington': '34005',
        'Pettis': '29163',
        'Cumberland': '34009',
        'Hamilton': '39061',
        'Los Angeles': '06037',
        'Howard': '24027',
        'Kauai': '15007',
        'Rockingham': '51079',
        'Solano': '06095',
        'Charleston': '45019',
        'Tippecanoe': '18153',
        'Montgomery': '24031',
        'New Haven': '09009',
        'Suffolk': '36103',
        'Charles': '24017',
        "Prince George's": '24033',
        'Contra Costa': '06013',
        'Norfolk': '25021',
        'Hillsborough': '12057',
        'Wake': '37183',
        'Carroll': '24025',
        'Ventura': '06111',
        'Santa Clara': '06085',
        'Monterey': '06053',
        'Albemarle': '51003',
        'Loudoun': '51107',
        'Prince George': '24033',
        'El Paso': '08141',
        'Honolulu': '15001',
        'Kent': '21097',
        'Fairfax': '51059',
        'Churchill': '32003',
        'Marion': '18105',
        'Washoe': '32031',
        'Escambia': '12033',
        'San Mateo': '06081',
        'Middlesex': '34023',
        'Richland': '45079',
        'Harford': '24025',
        'James City': '51095',
        'Pulaski': '51159',
        'Washtenaw': '26161',
        'Maui': '15009',
        'Spotsylvania': '51179',
        'Kootenai': '16055',
        "St. Mary's": '24037',
        'New York': '36061',
        'Nassau': '36059',
        'Hennepin': '27053',
        'Beaufort': '45011',
        'San Francisco': '06075',
        'Miami-Dade': '12086',
        'San Bernardino': '06071',
        'Garfield': '08045',
        'Bexar': '48029',
        'DeKalb': '13089',
        'Harnett': '37179',
        'Arlington': '51013',
        'Sarasota': '12115',
        'Riverside': '06065',
        'Ada': '16001',
        'Maricopa': '04013',
        'Multnomah': '41051',
        'Kern': '06029',
        'Virginia Beach': '51510',
        'Richmond': '51161',
        'Autauga': '01001',
        'Shelby': '01117',
        'Yolo': '06113',
        'Newport': '51031',
        'Currituck': '37051',
        'Marin': '06041',
        'El Dorado': '06017',
        'St. Louis': '29189',
        'Prince William': '51153',
        'New Castle': '10003',
        'Hoke': '37095',
        'Berkeley': '54003',
        'Hardin': '39067',
        'Arapahoe': '08005',
        'Lane': '41039',
        'Bay': '26017',
        'Nueces': '48061',
        'Hudson': '34017',
        'Duval': '12031',
        'Saratoga': '36091',
        'Gwinnett': '13057',
        'Muscogee': '13059',
        'Travis': '48453',
        'Rockdale': '13089',
        'Alameda': '06001',
        'Pinal': '04021',
        'Portsmouth': '51091',
        'Cook': '17031',
        'Sedgwick': '08059',
        'Galveston': '48167',
        'Madison': '17043',
        'Denton': '48121',
        'St. Lawrence': '36091',
        'Alexandria': '51013',
        'Houston': '48201',
        'Atlantic': '34001',
        'Sacramento': '06067',
        'Cuyahoga': '39035',
        'Fresno': '06019',
        'Christian': '29041',
        'Calvert': '24009',
        'Chesapeake': '24013',
        'Rock Island': '17093',
        'Isle of Wight': '51085',
        'Fredericksburg': '51091',
        'DuPage': '17043',
        'Williamsburg': '51095',
        'Tarrant': '48439',
        'Meade': '20089',
        'Anchorage': '02020',
        'Mercer': '42099',
        'Pitt': '37147',
        'Chesterfield': '51143',
        'Philadelphia': '42101',
        'Wichita': '20175',
        'Palm Beach': '12099',
        'St. Charles': '29241',
        'Brevard': '12009',
        'Tom Green': '48433',
        'Harris': '48201',
        'Salt Lake': '49035',
        'Williamson': '48453',
        'Laramie': '56017',
        'Bell': '48037',
        'Tooele': '49035',
        'Yuba': '06115',
        'Essex': '25009',
        'Moore': '37083',
        'Placer': '06061',
        'Plaquemines': '22093',
        'Larimer': '08069',
        'Allen': '18131',
        'Newport News': '51095',
        'Davis': '04007',
        'Otero': '35043',
        'Brown': '39017',
        'Frederick': '24021',
        'Caddo': '22017',
        'Santa Barbara': '06083',
        'Pima': '04019'
    }



    
    # Add a FIPS column to your county_counts DataFrame
    county_counts['FIPS'] = county_counts['County'].map(fips_map)

    # Create the choropleth map
    fig = px.choropleth(
        county_counts,
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        locations='FIPS',
        color='EV_Count',
        color_continuous_scale="Viridis",
        scope="usa",
        labels={'EV_Count': 'Number of EVs'},
        title='Spatial Distribution of Electric Vehicles by County'
    )

    # Update layout for better display
    fig.update_geos(fitbounds="locations", visible=False)
    fig.show()

def Model_Popularity():
   
    df = data


    # Count the number of occurrences of each model
    model_popularity = df['Model'].value_counts().reset_index()
    model_popularity.columns = ['Model', 'Count']

    # Sort by count in descending order
    model_popularity.sort_values(by='Count', ascending=False, inplace=True)

    # Visualize the top 10 most popular models with swapped axes
    plt.figure(figsize=(12, 6))
    plt.gca().set_facecolor('pink')  # Set background color to pink
    # Use a more defined color for better visibility
    sns.barplot(y='Model', x='Count', data=model_popularity.head(10), color='lightcoral', edgecolor='black')  
    plt.title('Top 10 Most Popular EV Models', fontsize=20, color='blue')  # Set title color to blue
    plt.ylabel('Number of Vehicles', fontsize=14, color='blue')  # Set y label color to blue
    plt.xlabel('Model', fontsize=14, color='blue')  # Set x label color to blue

    # Add value labels above each bar for clarity in red
    for p in plt.gca().patches:
        plt.annotate(f'{int(p.get_width())}', 
                    (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha='center', va='center', color='red', fontsize=12)

    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.show()

    # Analyze trends in model popularity over time
    trend_data = df.groupby(['Model Year', 'Model']).size().reset_index(name='Count')

    # Visualize trends for the top models
    top_models = model_popularity.head(5)['Model'].tolist()
    trend_data_top = trend_data[trend_data['Model'].isin(top_models)]

    plt.figure(figsize=(14, 7))
    plt.gca().set_facecolor('pink')  # Set background color to pink
    sns.lineplot(data=trend_data_top, x='Model Year', y='Count', hue='Model', marker='o', palette='husl')  # Use a clear color palette
    plt.title('Trends in Popularity of Top EV Models Over Years', fontsize=20, color='blue')  # Set title color to blue
    plt.xlabel('Model Year', fontsize=14, color='blue')  # Set x label color to blue
    plt.ylabel('Number of Vehicles', fontsize=14, color='blue')  # Set y label color to blue
    plt.xticks(trend_data_top['Model Year'].unique())
    plt.legend(title='Model')
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.show()

    # Count the number of occurrences by Electric Vehicle Type
    ev_type_counts = df['Electric Vehicle Type'].value_counts().reset_index()
    ev_type_counts.columns = ['Electric Vehicle Type', 'Count']

    # Visualize the distribution of EV types
    plt.figure(figsize=(8, 5))
    plt.gca().set_facecolor('pink')  # Set background color to pink
    sns.barplot(x='Count', y='Electric Vehicle Type', data=ev_type_counts, color='lightblue')  # Using a clear color
    plt.title('Distribution of Electric Vehicle Types', fontsize=20, color='blue')  # Set title color to blue
    plt.xlabel('Number of Vehicles', fontsize=14, color='blue')  # Set x label color to blue
    plt.ylabel('Electric Vehicle Type', fontsize=14, color='blue')  # Set y label color to blue

    # Add value labels above each bar in red
    for p in plt.gca().patches:
        plt.annotate(f'{int(p.get_width())}', 
                    (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha='center', va='center', color='red', fontsize=12)

    plt.tight_layout() 
    plt.show()

    # Group by Model Year and Electric Vehicle Type
    trend_by_type = df.groupby(['Model Year', 'Electric Vehicle Type']).size().reset_index(name='Count')

    plt.figure(figsize=(14, 7))
    plt.gca().set_facecolor('pink')  # Set background color to pink
    sns.lineplot(data=trend_by_type, x='Model Year', y='Count', hue='Electric Vehicle Type', marker='o', palette='husl')  # Use a clear color palette
    plt.title('Trends in Electric Vehicle Types Over Years', fontsize=20, color='blue')  # Set title color to blue
    plt.xlabel('Model Year', fontsize=14, color='blue')  # Set x label color to blue
    plt.ylabel('Number of Vehicles', fontsize=14, color='blue')  # Set y label color to blue
    plt.xticks(trend_by_type['Model Year'].unique())
    plt.legend(title='Electric Vehicle Type')
    plt.tight_layout()  
    plt.show()

def  relationship_between_numeric_features():

        
        df = data

        # Strip any whitespace from column names
        df.columns = df.columns.str.strip()

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])

        # Calculate the correlation matrix
        correlation_matrix = numeric_df.corr()

        # Visualize the correlation matrix using a heatmap
        plt.figure(figsize=(12, 8))
        plt.gca().set_facecolor('red')  # Set background color to red
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='YlGnBu', square=True, cbar_kws={"shrink": .8})  # Changed color palette for better visibility
        plt.title('Correlation Matrix of Numeric Features', fontsize=20, color='white')  # Set title color to white
        plt.xlabel('Features', fontsize=14, color='white')  # Set x label color to white
        plt.ylabel('Features', fontsize=14, color='white')  # Set y label color to white
        plt.xticks(fontsize=10, color='white')  # Set x-tick labels color to white
        plt.yticks(fontsize=10, color='white')  # Set y-tick labels color to white
        plt.show()

        # Print the correlation matrix
        print(correlation_matrix)


def Data_Exploration_Visualizations():
        
        
        df = data

        # Count the number of occurrences of each model
        model_popularity = df['Model'].value_counts().reset_index()
        model_popularity.columns = ['Model', 'Count']

        # Sort by count in descending order
        model_popularity.sort_values(by='Count', ascending=False, inplace=True)
        print(model_popularity.head(10))  # Display top 10 models

        # Set a pink background color for all plots
        sns.set_style("whitegrid")

        # Plot the distribution of electric range
        plt.figure(figsize=(10, 6))
        plt.gca().set_facecolor('pink')  # Set background color to pink
        ax = sns.histplot(df['Electric Range'], bins=30, kde=True, color='beige')  # Set the color of the plot to beige
        plt.title('Distribution of Electric Range', fontsize=20, color='blue')  # Set title color to blue
        plt.xlabel('Electric Range (miles)', fontsize=14, color='blue')  # Set x label color to blue
        plt.ylabel('Frequency', fontsize=14, color='blue')  # Set y label color to blue

        # Add text labels on the bars with red color
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', color='red', fontsize=10)  # Change color to red

        plt.show()


        # Count the number of occurrences of each model
        model_popularity = df['Model'].value_counts().reset_index()
        model_popularity.columns = ['Model', 'Count']

        # Sort by count in descending order
        model_popularity.sort_values(by='Count', ascending=False, inplace=True)
        print(model_popularity.head(10))  # Display top 10 models

        # Set a pink background color for all plots
        sns.set_style("whitegrid")

        # Count plot of Electric Vehicles by Type
        plt.figure(figsize=(12, 6))
        plt.gca().set_facecolor('pink')  # Set background color to pink

        # Create the count plot with beige bars
        sns.countplot(x='Electric Vehicle Type', data=df, color='beige')  # Set the bar color to beige

        # Title and labels with specified colors
        plt.title('Count of Electric Vehicles by Type', fontsize=20, color='blue')  # Title color set to blue
        plt.xlabel('Electric Vehicle Type', fontsize=14, color='blue')  # X-axis label color set to blue
        plt.ylabel('Count', fontsize=14, color='blue')  # Y-axis label color set to blue

        # Customize x-tick labels
        plt.xticks(rotation=0, fontsize=12, color='blue')  # X-tick labels color set to blue and no rotation

        # Optionally, you can add a grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7, color='gray')  # Add horizontal grid lines

        # Add value labels above each bar for clarity
        for p in plt.gca().patches:
            plt.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', color='black', fontsize=12)  # Change color to black

        plt.tight_layout()  # Adjust layout to avoid clipping
        plt.show()

        # Scatter plot of Electric Range vs Base MSRP
        plt.figure(figsize=(10, 6))
        plt.gca().set_facecolor('pink')  # Set background color to pink
        sns.scatterplot(x='Base MSRP', y='Electric Range', data=df, color='beige')  # Set the color of the plot to beige
        plt.title('Electric Range vs. Base MSRP', fontsize=20, color='blue')  # Set title color to blue
        plt.xlabel('Base MSRP ($)', fontsize=14, color='blue')  # Set x label color to blue
        plt.ylabel('Electric Range (miles)', fontsize=14, color='blue')  # Set y label color to blue
        plt.show()

        # Box plot of Base MSRP by Electric Vehicle Type
        plt.figure(figsize=(12, 6))
        plt.gca().set_facecolor('pink')  # Set background color to pink
        sns.boxplot(x='Electric Vehicle Type', y='Base MSRP', data=df, color='beige')  # Set the color of the plot to beige
        plt.title('Base MSRP by Electric Vehicle Type', fontsize=20, color='blue')  # Set title color to blue
        plt.xlabel('Electric Vehicle Type', fontsize=14, color='blue')  # Set x label color to blue
        plt.ylabel('Base MSRP ($)', fontsize=14, color='blue')  # Set y label color to blue
        plt.xticks(rotation=0, fontsize=12, color='blue')  # Set x-tick labels color to blue and remove rotation
        plt.show()

        # Pair plot of selected features
        sns.pairplot(df[['Base MSRP', 'Electric Range', 'Model Year']], diag_kind='kde', palette='pastel')
        plt.suptitle('Pair Plot of Selected Features', y=1.02, fontsize=20, color='blue')  # Set title color to blue
        plt.show()

        # Correlation heatmap
        plt.figure(figsize=(12, 8))
        plt.gca().set_facecolor('pink')  # Set background color to pink
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        plt.title('Correlation Heatmap', fontsize=20, color='blue')  # Set title color to blue
        plt.show()

def  Comparative_Visualization():
    

    
    # Load the dataset
    df = data

  
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['number'])

    # Compute the correlation matrix
    correlation_matrix = numeric_df.corr()

    # Plotting the correlation heatmap
    plt.figure(figsize=(12, 8))
    plt.gca().set_facecolor('pink')  # Set background color to pink
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap', fontsize=5, color='darkred')  # Set title color to dark red
    plt.xlabel('Features', fontsize=5, color='darkred')  # Set x label color to dark red
    plt.ylabel('Features', fontsize=5, color='darkred')  # Set y label color to dark red
    plt.xticks(fontsize=5, color='darkred')  # Set x-tick labels color to dark red
    plt.yticks(fontsize=5, color='darkred')  # Set y-tick labels color to dark red
    plt.show()

    # Count EVs by County
    county_counts = df['County'].value_counts().reset_index()
    county_counts.columns = ['County', 'Count']

    plt.figure(figsize=(12, 6))
    plt.gca().set_facecolor('pink')  # Set background color to pink
    sns.barplot(x='Count', y='County', data=county_counts.head(15), color='skyblue')  # Set bar color to sky blue
    plt.title('Top 15 Counties by Number of Electric Vehicles', fontsize=5, color='darkblue')  # Set title color to dark blue
    plt.xlabel('Number of Electric Vehicles', fontsize=5, color='darkblue')  # Set x label color to dark blue
    plt.ylabel('County', fontsize=5, color='darkblue')  # Set y label color to dark blue
    plt.xticks(fontsize=5, color='darkblue', rotation=90)  # Rotate x-tick labels to vertical
    plt.show()

    # Count EVs by City
    city_counts = df['City'].value_counts().reset_index()
    city_counts.columns = ['City', 'Count']

    plt.figure(figsize=(12, 6))
    plt.gca().set_facecolor('pink')  # Set background color to pink
    sns.barplot(x='Count', y='City', data=city_counts.head(15), color='lightgreen')  # Set bar color to light green
    plt.title('Top 15 Cities by Number of Electric Vehicles', fontsize=5, color='darkgreen')  # Set title color to dark green
    plt.xlabel('Number of Electric Vehicles', fontsize=5, color='darkgreen')  # Set x label color to dark green
    plt.ylabel('City', fontsize=5, color='darkgreen')  # Set y label color to dark green
    plt.xticks(fontsize=5, color='darkgreen', rotation=90)  # Rotate x-tick labels to vertical
    plt.show()

    # Count of EVs by County and Electric Vehicle Type
    county_ev_type_counts = df.groupby(['County', 'Electric Vehicle Type']).size().unstack().fillna(0)




    # Count of EVs by County and Electric Vehicle Type
    county_ev_type_counts = df.groupby(['County', 'Electric Vehicle Type']).size().unstack().fillna(0)

    # Plotting Stacked Bar Chart for County EV Distribution
    plt.figure(figsize=(12, 8))
    plt.gca().set_facecolor('pink')  # Set background color to pink
    county_ev_type_counts.plot(kind='bar', stacked=True, figsize=(12, 8), color=['red', 'blue', 'green', 'orange', 'purple'])
    plt.title('Distribution of Electric Vehicle Types by County', fontsize=14)
    plt.xlabel('County', fontsize=12)
    plt.ylabel('Number of Electric Vehicles', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title='Electric Vehicle Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()

    # Count of EVs by City and Electric Vehicle Type
    city_ev_type_counts = df.groupby(['City', 'Electric Vehicle Type']).size().unstack().fillna(0)

    # Plotting Stacked Bar Chart for City EV Distribution
    plt.figure(figsize=(12, 8))
    plt.gca().set_facecolor('pink')  # Set background color to pink
    city_ev_type_counts.plot(kind='bar', stacked=True, figsize=(12, 8), color=['red', 'blue', 'green', 'orange', 'purple'])
    plt.title('Distribution of Electric Vehicle Types by City', fontsize=14)
    plt.xlabel('City', fontsize=12)
    plt.ylabel('Number of Electric Vehicles', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title='Electric Vehicle Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()

def Temporal_Analysis():

    df = data

   
    # Convert 'Model Year' to numeric
    df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce')

    # Drop rows with NaN in 'Model Year'
    df = df.dropna(subset=['Model Year'])

    # Group by 'Model Year' to get counts of EVs
    ev_counts = df.groupby('Model Year').size().reset_index(name='Count')

    # Plot EV adoption trend over the years
    plt.figure(figsize=(12, 6))
    plt.gca().set_facecolor('lightblue')  # Set background color
    sns.lineplot(data=ev_counts, x='Model Year', y='Count', marker='o')
    plt.title('EV Adoption Trend Over Years', fontsize=20)
    plt.xlabel('Model Year', fontsize=20)
    plt.ylabel('Number of EVs', fontsize=20)
    plt.xticks(ev_counts['Model Year'].unique())
    plt.grid(True)
    plt.show()

    # Analyze model popularity over the years
    model_popularity = df.groupby(['Model Year', 'Model']).size().reset_index(name='Count')

    # Create a color palette with more distinct colors
    palette = sns.color_palette(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"])

    # Plotting model popularity with axes swapped
    plt.figure(figsize=(14, 8))
    plt.gca().set_facecolor('lightblue')  # Set background color
    sns.barplot(x='Model', y='Count', hue='Model Year', data=model_popularity, palette=palette)

    plt.title('Model Popularity Over the Years', fontsize=20)
    plt.xlabel('EV Model', fontsize=20)  # Now represents EV Model
    plt.ylabel('Number of Vehicles', fontsize=20)  # Now represents Number of Vehicles

    # Set font size for y-axis labels to 5
    plt.yticks(fontsize=5)  # Set y-ticks font size to 5

    # Customize x-axis ticks: red color and vertical labels
    plt.xticks(rotation=90, fontsize=5, color='red')  # Change color to red and set labels to vertical

    
    plt.legend(title='Model Year', fontsize=20)

    plt.tight_layout()  
    plt.show()



while True:
    print("\nEnter option:")
    print("1. Show sum and frequency of missing data for each attribute")
    print("2. Show missing rows for a specific attribute")
    print("3. Handle missing data")
    print("4. Drop rows with missing values")
    print("5. One-hot encoding")
    print("6. Normalization using MinMax scaling")
    print("7. Descriptive Statistics")
    print("8. Spatial Distribution")
    print("9. show Model Popularity")
    print("10. Investigate the relationship between every pair of numeric features")
    print("11. Data Exploration Visualizations")
    print("12. Comparative Visualization")
    print("13. Temporal Analysis")


    print("0. Exit")
    
    # Reading user input
    option = input("Choose an option: ")

    if option == '1':
        document_missing_values(data)
    elif option == '2':
        print("Enter the attribute name:")
        attribute = input("Input attribute name: ")
        missing_attribute_data(attribute)
    elif option == '3':
        handle_missing_data()
    elif option == '4':
        drop_missing_rows()
    elif option == '5':
        one_hot_encoding()
    elif option == '6':
        normalize_feature("Electric Range")
    elif option == '7':
        descriptive_statistics("Legislative District")
        print("\n")
        descriptive_statistics("Base MSRP")
        print("\n")
        descriptive_statistics("Electric Range")
        print("\n")
        descriptive_statistics("Postal Code")
        print("\n")
    elif option == '8':
        plot_ev_distribution()
        
    elif option == '9':
        Model_Popularity()
    elif option == '10':
        relationship_between_numeric_features()
    elif option == '11':
         Data_Exploration_Visualizations()
    elif option == '12':
        Comparative_Visualization()
    elif option == '13':
        Temporal_Analysis()

        
    elif option == '0':
        print("Exiting...")
        break
    else:
        print("Invalid option. Please choose again.")
