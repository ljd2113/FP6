import sqlite3
import pandas as pd
from openai import OpenAI
import json
import time

# --- CONFIGURATION ---
# IMPORTANT: Replace the placeholder with your ACTUAL API Key in quotes
OPENAI_API_KEY = "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"
DB_NAME = 'feedback.db'
TABLE_NAME = 'reviews'
TEXT_COLUMN = 'review_text'
OUTPUT_TABLE = 'reviews_with_openai_analysis'

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# Define the System Prompt for the LLM
SYSTEM_PROMPT = (
    "You are an expert sentiment analyst specializing in product feedback. "
    "Your task is to analyze customer reviews for a new high-end AR/VR headset, "
    "Apple Vision Pro. You must extract the overall sentiment (Positive, Negative, or Neutral) "
    "and identify key product aspects mentioned, along with the customer's sentiment toward each aspect. "
    "Respond ONLY with a single JSON object that strictly adheres to the provided schema."
)

def analyze_review_openai(client, review_text):
    """
    Sends a review to OpenAI for structured aspect-based analysis using the current
    'tools' and 'json_object' parameters, fixing the Error 400.
    """
    # This is the JSON schema object, defined locally for use in the 'tools' parameter
    SCHEMA_DEFINITION = {
        "type": "object",
        "properties": {
            "overall_sentiment": {
                "type": "string",
                "enum": ["Positive", "Negative", "Neutral"],
                "description": "The overall sentiment of the review, strictly categorized as Positive, Negative, or Neutral."
            },
            "aspect_data": {
                "type": "array",
                "description": "A list of specific aspects/themes mentioned in the review and the customer's sentiment towards them.",
                "items": {
                    "type": "object",
                    "properties": {
                        "aspect": {"type": "string", "description": "The specific product feature or theme (e.g., 'Comfort', 'Display', 'Price')."},
                        "sentiment": {"type": "string", "description": "The sentiment towards this aspect (Positive, Negative, or Neutral)."}
                    },
                    "required": ["aspect", "sentiment"]
                }
            }
        },
        "required": ["overall_sentiment", "aspect_data"]
    }
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",  # Must use a model that supports JSON mode and tools
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Review to analyze: {review_text}"}
            ],
            response_format={"type": "json_object"},
            # The schema is now passed within the 'tools' parameter
            tools=[{
                "type": "function",
                "function": {
                    "name": "extract_aspect_sentiment",
                    "description": "Extract the overall sentiment and a list of specific aspects and their sentiments from a review.",
                    "parameters": SCHEMA_DEFINITION 
                }
            }],
            tool_choice={"type": "function", "function": {"name": "extract_aspect_sentiment"}},
            temperature=0
        )
        
        # Extract the JSON string from the tool_calls response:
        json_string = response.choices[0].message.tool_calls[0].function.arguments
        return json.loads(json_string)

    except Exception as e:
        print(f"API Error processing review: {e}")
        time.sleep(2) # Wait to avoid hitting rate limits on repeated errors
        return None 


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    conn = sqlite3.connect(DB_NAME)
    
    # 1. Load Data
    try:
        df = pd.read_sql_query(f"SELECT id, {TEXT_COLUMN} FROM {TABLE_NAME}", conn)
        print(f"Successfully loaded {len(df)} reviews from {TABLE_NAME}.")
    except Exception as e:
        print(f"Error loading data: {e}")
        conn.close()
        exit()

    print("\nStarting OpenAI Analysis and Aspect Extraction (This may take a few minutes)...")
    
    # 2. Process Reviews
    results = []
    for index, row in df.iterrows():
        review_text = row[TEXT_COLUMN]
        analysis_data = analyze_review_openai(client, review_text)
        
        if analysis_data:
            # Prepare data for storage
            record = {
                'review_id': row['id'],
                'overall_sentiment': analysis_data.get('overall_sentiment'),
                'aspect_data_json': json.dumps(analysis_data.get('aspect_data', [])) 
            }
            results.append(record)
        time.sleep(0.5) # Be kind to the API rate limit
    
    # 3. Save Results
    if not results:
        print("No successful analysis results to save.")
        conn.close()
        exit()

    results_df = pd.DataFrame(results)
    
    # Create the output table schema for the database
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
        review_id INTEGER PRIMARY KEY,
        overall_sentiment TEXT,
        aspect_data_json TEXT
    );
    """
    conn.execute(create_table_query)
    
    print(f"\nAnalysis complete. Saving results to new table: {OUTPUT_TABLE}...")
    
    # Save the dataframe to the new SQLite table
    try:
        results_df.to_sql(OUTPUT_TABLE, conn, if_exists='replace', index=False)
        print("Results saved successfully. You can now run report_generator.py.")
    except Exception as e:
        # This catches the final error you were seeing: list binding issue
        print(f"Error saving data to SQLite: {e}")

    conn.close()