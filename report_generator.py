import sqlite3
import pandas as pd
import json
from collections import Counter

# --- CONFIGURATION ---
DB_NAME = 'feedback.db'
OUTPUT_TABLE = 'reviews_with_openai_analysis'

# --- HELPER FUNCTIONS ---

def load_data(conn):
    """Loads the analyzed data from the SQLite database."""
    try:
        df = pd.read_sql_query(f"SELECT * FROM {OUTPUT_TABLE}", conn)
        return df
    except pd.io.sql.DatabaseError:
        print(f"Error: Table {OUTPUT_TABLE} not found. Please run openai_analysis.py first.")
        return None

def calculate_overall_summary(df):
    """Calculates the count of overall sentiment categories."""
    overall_summary = df['overall_sentiment'].value_counts().reindex(['Positive', 'Negative', 'Neutral'], fill_value=0)
    return overall_summary

def extract_aspects(df):
    """Extracts all aspects and their sentiments from the JSON column."""
    aspect_sentiments = []
    
    # Safely load the JSON strings from the 'aspect_data_json' column
    for json_string in df['aspect_data_json']:
        try:
            # The 'aspect_data_json' column contains a JSON string of a list of aspects
            aspect_list = json.loads(json_string)
            if isinstance(aspect_list, list):
                for item in aspect_list:
                    if isinstance(item, dict) and 'aspect' in item and 'sentiment' in item:
                        aspect_sentiments.append((item['aspect'], item['sentiment']))
        except json.JSONDecodeError:
            continue # Skip invalid JSON entries

    return aspect_sentiments

def summarize_aspects(aspect_sentiments):
    """Summarizes aspect counts and calculates net sentiment."""
    
    # 1. Count occurrences for each (aspect, sentiment) pair
    counts = Counter(aspect_sentiments)
    
    # 2. Structure the results
    summary = {}
    
    for (aspect, sentiment), count in counts.items():
        if aspect not in summary:
            summary[aspect] = {'Total': 0, 'Positive': 0, 'Negative': 0, 'Neutral': 0}
        
        summary[aspect]['Total'] += count
        if sentiment in summary[aspect]:
            summary[aspect][sentiment] += count

    # 3. Convert to a list of dicts for DataFrame
    final_data = []
    for aspect, data in summary.items():
        # Calculate Net Sentiment (Positive - Negative)
        net_sentiment = data['Positive'] - data['Negative']
        
        # Determine Net Sentiment Category
        if net_sentiment > 0:
            net_category = 'Positive'
        elif net_sentiment < 0:
            net_category = 'Negative'
        else:
            net_category = 'Neutral'
            
        final_data.append({
            'Aspect': aspect,
            'Total Count': data['Total'],
            'Net Sentiment Score': net_sentiment,
            'Net Sentiment Category': net_category,
            'Positive': data['Positive'],
            'Negative': data['Negative'],
            'Neutral': data['Neutral'],
        })

    # Sort by Net Sentiment Score (descending)
    aspect_df = pd.DataFrame(final_data).sort_values(by='Net Sentiment Score', ascending=False).reset_index(drop=True)
    return aspect_df[['Aspect', 'Total Count', 'Net Sentiment Score', 'Net Sentiment Category']]


# --- MAIN REPORT GENERATION ---

if __name__ == "__main__":
    conn = sqlite3.connect(DB_NAME)
    df = load_data(conn)
    
    if df is None:
        conn.close()
        exit()

    # TEMPORARY CHECK LINE (This must print if the file is correctly saved and run)
    print(f"\n[CHECK] Dataframe loaded successfully with {len(df)} rows. Generating report...")
    
    # --- 1. Overall Sentiment Summary ---
    overall_summary = calculate_overall_summary(df)

    print("\n" + "="*80)
    print("               APPLE VISION PRO FEEDBACK ANALYSIS REPORT")
    print("="*80)
    print(f"Total Reviews Analyzed: {len(df)}")
    print("\n--- OVERALL SENTIMENT SUMMARY ---")
    print("This reflects the high-level, single sentiment assigned to the review as a whole.")
    print("-" * 40)
    print(overall_summary.to_string())
    print("-" * 40)

    # --- 2. Aspect-Based Sentiment Summary ---
    aspect_sentiments = extract_aspects(df)
    aspect_summary_df = summarize_aspects(aspect_sentiments)
    
    print("\n--- ASPECT-BASED SENTIMENT DEEP DIVE ---")
    print("This breaks down sentiment by specific product feature/theme (Aspect).")
    print("Net Sentiment Score = (Positive mentions - Negative mentions).")
    
    # Filter the summary to show only the most common aspects (e.g., top 10 by total count)
    top_aspects_df = aspect_summary_df.sort_values(by='Total Count', ascending=False).head(10)
    
    print("-" * 80)
    print(top_aspects_df.to_string(index=False))
    print("-" * 80)
    
    # --- 3. Key Findings and Recommendations ---
    
    print("\n--- KEY FINDINGS AND RECOMMENDATIONS ---")
    
    # Identify the strongest Positive and Negative aspects
    # Note: If top_aspects_df is empty, this could cause an error, but with 79 rows, it should be fine.
    most_positive = top_aspects_df.iloc[0]
    most_negative = top_aspects_df[top_aspects_df['Net Sentiment Score'] == top_aspects_df['Net Sentiment Score'].min()].iloc[0]

    print("\n[KEY FINDINGS]")
    print(f"1. Strongest Positive Feature: '{most_positive['Aspect']}' (Net Score: {most_positive['Net Sentiment Score']}). This feature is a major driver of positive sentiment and should be a focus in marketing.")
    print(f"2. Strongest Negative Concern: '{most_negative['Aspect']}' (Net Score: {most_negative['Net Sentiment Score']}). This represents the most critical area for immediate product improvement.")
    print(f"3. Overall Brand Perception: {overall_summary.idxmax()} (Count: {overall_summary.max()}). The product's overall reception is dominated by {overall_summary.idxmax()} sentiment.")

    print("\n[RECOMMENDED ACTIONS]")
    print(f"* **Product Team Priority:** Focus R&D on mitigating issues related to **{most_negative['Aspect']}** to reduce churn and negative word-of-mouth.")
    print(f"* **Marketing Strategy:** Immediately leverage the success of **{most_positive['Aspect']}** by prominently featuring it in all advertising and product descriptions.")
    print("* **Content Strategy:** Create dedicated customer support content or tutorials to address any common 'Neutral' or highly technical aspects to convert indifferent customers into positive advocates.")

    print("\n" + "="*80)

    conn.close()