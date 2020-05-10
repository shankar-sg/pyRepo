def doOneHotEncode(dataFrame,featureToEncode,dropFeature=False):
    #
    # Create a DF with only the column that needs to be encoded
    # Call pandas get dummies method to encode
    #
    df_temp = dataFrame[featureToEncode]
    df = pd.get_dummies(df_temp)
    
    #
    # Define the column names of the encoded data frame
    # Column name = "feature to encode" + "_" + value to be encoded
    # Example of encoded column name : "Sex_M" / "Sex_F" 
    #
    encodedColumns = list([featureToEncode + "_" + str(i) for i in df.columns])
    df.columns = encodedColumns
    lastColumn = len(encodedColumns) -1

    #
    # Remove the last of the encoded column - as it is can be represented by all zeros, rather than as a seperate column
    # 
    df.drop(df.columns[lastColumn],axis=1,inplace=True)
    enCodedColumns = encodedColumns.pop()
    #
    # Add the encodedColumns back to main data frame
    #
    for col in encodedColumns:
        dataFrame[col] = df[col]
        
    # Drop the column encoded if requested
    
    if dropFeature:
        dataFrame.drop(featureToEncode,axis=1,inplace=True)
    
    return dataFrame