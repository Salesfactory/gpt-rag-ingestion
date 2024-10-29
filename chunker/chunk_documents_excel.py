import logging
import time
import pandas as pd
import os
import re
import glob
import openpyxl
from utils.file_utils import get_file_extension
from utils.file_utils import get_filename



FILE_EXTENSION_DICT = [
    "xlsx",
    "csv"
    
]

##########################################################################################
# UTILITY FUNCTIONS
##########################################################################################

def cleanup_title(title):

    title = title.replace(" ", "%20")
    
    return title

        


def has_supported_file_extension(file_path: str) -> bool:
    """Checks if the given file format is supported based on its file extension.
    Args:
        file_path (str): The file path of the file whose format needs to be checked.
    Returns:
        bool: True if the format is supported, False otherwise.
    """
    file_extension = get_file_extension(file_path)
    return file_extension in FILE_EXTENSION_DICT


##########################################################################################
# DATA PROCESSING FUNCTIONS
########################################################################################## 

class DataProcessor:
    # create a constructor to initialize the attributes of an object
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        
    # function to load data from the excel file to the dataframe
    def load_data(self):
        # load the excel file to the df
        self.df = pd.read_excel(self.filepath, header = [0, 1, 2])
        
        return self.df
    
    # function to clean the dataframe
    def clean_data(self):
        # drop empty rows and columns
        self.df.dropna(axis = 0, how = 'all', inplace = True)
        self.df.dropna(axis = 1, how = 'all', inplace = True)
        
        # reset the index
        self.df = self.df.reset_index(drop = True)
        
        # drop the major section headers and reset the index
        self.df = self.df[~((self.df.iloc[:, 0].str.isupper()) & (self.df.iloc[:, 1:].isnull().all(1)))]
        self.df.reset_index(drop = True, inplace = True)
        
        return self.df

##########################################################################################
# CHUNKING FUNCTIONS
########################################################################################## 




class DataChunker:
    # create a constructor to initialize the attributes of an object
    def __init__(self, df, output_dir = 'tables', custom_folder_name = None):
        self.df = df
        self.output_dir = output_dir
        
        if custom_folder_name:
            self.output_dir = os.path.join(self.output_dir, custom_folder_name)
    
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    

    # function to add section title as context to likert scale rows
    def add_context_to_likert_scale(self, df_chunk, title):

        sentiment_start = None
        sentiment_end = None

        for i, row in df_chunk.iterrows():
            if isinstance(row[0], str):
                if sentiment_start is None:
                    # update the start sentiment to the row value
                    sentiment_start = row[0].strip() # remove leading and trailing spaces to ensure there aren't unnecessary spaces when appending

                # keep updating until the last row is found
                sentiment_end = row[0].strip() # remove leading and trailing spaces to ensure there aren't unnecessary spaces when appending 
                
        # function to append the specific sentiment based on the number
        def append_sentiment(row):
            if row[0] in [1, 2]:
                return f"{sentiment_start}_{row[0]}"
            elif row[0] in [5, 6]:
                return f"{sentiment_end}_{row[0]}"
            elif row[0] in [3, 4]:
                return f"Context Dependent_{row[0]}"
            else:
                return row[0] 
        
        df_chunk.iloc[:, 0] = df_chunk.apply(append_sentiment, axis = 1)
        return df_chunk
    
    # function to handle major categories with multiple columns
    def split_large_category(self, vertical_df, vertical_category, max_items = 5):

        anchor_column = vertical_df.columns[0]
        
        vertical_cols = vertical_df.columns[vertical_df.columns.get_level_values(0) == vertical_category]
        
        """convert vertical columns to numeric index positions"""

        vertical_col_indices = []
        
        for col in vertical_cols:
            loc_value = vertical_df.columns.get_loc(col)

            if isinstance(loc_value, slice):

                vertical_col_indices.extend(range(loc_value.start, loc_value.stop))
            else:

                vertical_col_indices.append(loc_value)
                

        num_columns = len(vertical_col_indices)
                

        if num_columns > max_items: 

            midpoint = num_columns // 2
            

            first_half_cols = vertical_col_indices[:midpoint]
            second_half_cols = vertical_col_indices[midpoint:]
            

            first_half_df = vertical_df.iloc[:, first_half_cols]
            second_half_df = vertical_df.iloc[:, second_half_cols]
            

            if anchor_column not in first_half_df.columns:
                first_half_df = pd.concat([vertical_df[[anchor_column]], first_half_df], axis = 1)
            if anchor_column not in second_half_df.columns:
                second_half_df = pd.concat([vertical_df[[anchor_column]], second_half_df], axis = 1)
            
            return [(first_half_df, f"{vertical_category}1"), (second_half_df, f"{vertical_category}2")]
        
        else:
            return [(vertical_df.iloc[:, vertical_col_indices], vertical_category)]
        
    # function to chunk the dataframe
    def chunk_table(self):

        anchor_column = self.df.columns[0]
        vertical_categories = self.df.columns.get_level_values(0).unique()
        chunk_id = 0
        chunks = []
        errors = []
        
        start_row = None
        
        
        for i, row in self.df.iterrows():

            if self.df.iloc[i, 1:].isnull().all():

                if start_row is not None: 

                    for vertical_category in vertical_categories:

                        vertical_cols = self.df.columns[self.df.columns.get_level_values(0) == vertical_category]

                        if not vertical_cols.empty:
                            vertical_cols_list = list(vertical_cols)
                            if anchor_column not in vertical_cols_list:
                                vertical_cols_list = [anchor_column] + vertical_cols_list


                            try:

                                vertical_df = self.df.loc[start_row+1: i-1, vertical_cols_list]
                                               

                                vertical_df = self.add_context_to_likert_scale(vertical_df, self.df.iloc[start_row, 0])


                                chunks_to_save = self.split_large_category(vertical_df, vertical_category)

                                for chunk_df, chunk_name in chunks_to_save:
                                    #chunk_id += 1
                                    table_title = f"{self.df.iloc[start_row, 0]}_{chunk_name}"
                                    #logging.info(f"Iterated Chunk number {chunk_id} : {chunk_df}")
                                    chunks.append(chunk_df)
                                    
                                    

                            except KeyError as e:
                                print(f"Error: {e} not found in columns.")
                                errors.append(e)

                start_row = i
        

        if start_row is not None and start_row < len(self.df) - 1:
            for vertical_category in vertical_categories:
                vertical_cols = self.df.columns[self.df.columns.get_level_values(0) == vertical_category]
                if not vertical_cols.empty:
                    vertical_cols_list = list(vertical_cols)
                    if anchor_column not in vertical_cols_list:
                        vertical_cols_list = [anchor_column] + vertical_cols_list

                    try:
                        vertical_df = self.df.loc[start_row+1:, vertical_cols_list]
                        vertical_df = self.add_context_to_likert_scale(vertical_df, self.df.iloc[start_row, 0])


                        chunks_to_save = self.split_large_category(vertical_df, vertical_category)
                        

                        for chunk_df, chunk_name in chunks_to_save:
                            table_title = f"{self.df.iloc[start_row, 0]}_{chunk_name}"
                            #chunk_id += 1
                            #Uncomment this if you want to see the Iterated Chunks
                            #logging.info(f"Iterated Chunk number {chunk_id} : {chunk_df}")
                            chunks.append(chunk_df)
                            
                            

                            
                         
                    except KeyError as e:
                        print(f"Error: {e} not found in columns.")
                        errors.append(e)

        return chunks, errors
                        
def get_chunk(content, url):

    chunk = {
        "url": url,
        "filepath": get_filename(url),
        "content": content,
        
    }
    return chunk

def chunk_document(data):

    #For testing using Localtest container, change the file_utils.py path 
    filepath = f"{data['documentUrl']}{data['documentSasToken']}"
    filepath = cleanup_title(filepath)
    url = f"{data['documentUrl']}"
    url = cleanup_title(url)
    chunks = []
    errors = []
    warnings = []
    chunk_id = 0
    exc_name = get_filename(filepath)

    #DataProcessing
    processor = DataProcessor(filepath)
    df = processor.load_data()
    df = processor.clean_data()
    
    #ChunkProcessing
    d_chunker = DataChunker(df)
    content, conterr = d_chunker.chunk_table()
    #Transform Chunks to String 
    content = [str(chunk) for chunk in content]
    errors = [str(er) for er in conterr]
    for x in content:
        chunks.append(get_chunk(content[chunk_id], url))
        chunk_id += 1    
    
    logging.info(f"Finished processing {exc_name} with {errors} errors and {warnings} warnings")


    return chunks, errors, warnings
    


    