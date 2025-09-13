#!/usr/bin/env python3
"""
CSV to JSON Converter for S&P 500 Companies Data

This script converts the S&P 500 companies CSV file to JSON format,
keeping only the specified fields:
- Symbol
- Longname
- Sector
- Industry
- Marketcap
- Ebitda
- Revenuegrowth
- Fulltimeemployees
- Weight
"""

import csv
import json
import sys
from pathlib import Path


def convert_csv_to_json(csv_file_path, output_file_path=None):
    """
    Convert S&P 500 companies CSV to JSON format with selected fields.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        output_file_path (str, optional): Path to the output JSON file.
                                        If None, uses same name as CSV with .json extension
    
    Returns:
        list: List of dictionaries containing the converted data
    """
    
    # Define the fields to keep
    fields_to_keep = [
        'Symbol',
        'Longname', 
        'Sector',
        'Industry',
        'Marketcap',
        'Ebitda',
        'Revenuegrowth',
        'Fulltimeemployees',
        'Weight'
    ]
    
    converted_data = []
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                # Create a new dictionary with only the specified fields
                filtered_row = {}
                for field in fields_to_keep:
                    if field in row:
                        value = row[field]
                        
                        # Convert numeric fields to appropriate types
                        if field in ['Marketcap', 'Ebitda']:
                            try:
                                # Remove commas, convert to integer, then divide by 1,000,000,000 to get billions
                                int_value = int(value.replace(',', '')) if value else None
                                if int_value is not None:
                                    filtered_row[field] = round(int_value / 1000000000, 3)  # Round to 3 decimal places
                                else:
                                    filtered_row[field] = None
                            except (ValueError, AttributeError):
                                filtered_row[field] = None
                        elif field == 'Fulltimeemployees':
                            try:
                                # Remove commas and convert to integer
                                filtered_row['Employees'] = int(value.replace(',', '')) if value else None
                            except (ValueError, AttributeError):
                                filtered_row[field] = None
                        elif field in ['Revenuegrowth', 'Weight']:
                            try:
                                float_value = float(value) if value else None
                                if float_value is not None and field == 'Weight':
                                    # Round weight to 5 decimal places
                                    filtered_row[field] = round(float_value, 5)
                                else:
                                    filtered_row[field] = float_value
                            except (ValueError, AttributeError):
                                filtered_row[field] = None
                        else:
                            # Keep string fields as-is
                            filtered_row[field] = value if value else None
                    else:
                        filtered_row[field] = None
                
                converted_data.append(filtered_row)
    
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    # Determine output file path
    if output_file_path is None:
        csv_path = Path(csv_file_path)
        output_file_path = csv_path.with_suffix('.json')
    
    # Write to JSON file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(converted_data, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"Successfully converted {len(converted_data)} records to JSON.")
        print(f"Output saved to: {output_file_path}")
        
    except Exception as e:
        print(f"Error writing JSON file: {e}")
        return None
    
    return converted_data


def main():
    """Main function to run the converter from command line."""
    
    if len(sys.argv) < 2:
        print("Usage: python csv_to_json_converter.py <input_csv_file> [output_json_file]")
        print("Example: python csv_to_json_converter.py sp500_companies.csv sp500_companies.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = convert_csv_to_json(input_file, output_file)
    
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
