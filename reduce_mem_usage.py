

def reduce_mem_usage(df, verbose=True):
    """
    type df: pandas.core.frame.DataFrame
    type r: pandas.core.frame.DataFrame
    """
    # This function takes df as input and return a df for which numeric dtypes have been optimized
    import numpy
    import pandas
    import re

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]

    # Memory occupied by df before dtypes transformation
    start_mem_usg = df.memory_usage(deep=True).sum() / 1024**2
    print("Memory usage of input dataframe is: {:5.2f} MB".
          format(start_mem_usg))

    for col in df.columns:

        # Extract dtypes as string and remove numbers
        col_type = re.findall("([a-z]+)", str(df[col].dtypes))[0]

        if col_type == "object":
            df[col] = df[col].astype("category")

        # use extract only root type from numerics list
        elif col_type in [re.findall("([a-z]+)", dtype)[0] for dtype in numerics]:
            # Print current column type
#             print("******************************")
#             print("Column: ", col)
#             print("dtype before: ", df[col].dtype)

            # Obtain minimum and maximum values for df[col]
            # Used later to understand which is the most suitable dtype
            has_nan = False if df[col].isnull().any() == False else True
            c_Min, c_Max = df[col].\
                                        agg(["min", "max"]).\
                                        tolist()


            # Since col_type already contains type str, no need of further transfomations
            if col_type == "int":

                # If df[col] is only positive, use np.uint classes
                if c_Min >= 0:
                    if c_Max < np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif c_Max < np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_Max < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    elif c_Max < np.iinfo(np.uint64).max:
                        df[col] = df[col].astype(np.uint64)

                else:

                  # Check the minimum int size that can store the column
                    if c_Min > np.iinfo(np.int8).min and c_Max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_Min > np.iinfo(np.int16).min and c_Max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_Min > np.iinfo(np.int32).min and c_Max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_Min > np.iinfo(np.int64).min and c_Max < np.iinfo(np.int64).max:  #not clear why not use else:
                        df[col] = df[col].astype(np.int64)

            elif col_type == "float":

                # Check the minimum float size that can store the column
                if c_Min > np.finfo(np.float16).min and c_Max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_Min > np.finfo(np.float32).min and c_Max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)


        # Print new column dtype
#         print("dtype after: ", df[col].dtype)
#         print("******************************")

    # Compute memory of the resulting df
    end_mem_usg = df.memory_usage(deep = True).sum() / 1024**2

    if verbose:
        print("Mem. usage decreased to {:5.2f} MB ({:.1f}% reduction)".
             format(end_mem_usg, 100 * (start_mem_usg - end_mem_usg) / start_mem_usg))

    return df
