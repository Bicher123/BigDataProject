
def vect_gender(data:str):
    col_options = {"Male": 0, "Female": 1, "Other": 2}
    if(data in col_options.keys()):
         return col_options[data]
    # else:
    #      return 3
   
def vect_relevent_experience(data:str):
    col_options = {"No relevent experience": 0, "Has relevent experience": 1}
    return col_options[data]

def vect_enrolled_university(data: str):
    col_options = {
        "no_enrollment": 0, "Part time course": 1, "Full time course": 2}
    if(data in col_options.keys()):
         return col_options[data]
    # else:
    #      return 3
    
def vect_education_level(data: str):
    col_options = {"Primary School": 0, "High School":1, "Graduate": 2, "Masters": 3, "Phd": 4}
    if(data in col_options.keys()):
         return col_options[data]
    # else:
    #      return 5

def vect_major_discipline(data: str):
    col_options = {"STEM":0, "Business Degree": 1, "Humanities": 2, "No Major": 3, "Other": 4, "Arts": 5}
    if(data in col_options.keys()):
         return col_options[data]
    # else:
    #      return 6

def vect_experience(data):
    col_options = {"<1": 0, ">20": 21}
    for i in range(20):
        col_options[str(i+1)]=i+1
    if(data in col_options.keys()):
         return col_options[data]
    # else:
    #     return 22
    
def vect_company_size(data: str):
    col_options = {"<10": 0, "10/49+": 1, "50-99": 2, "100-500": 3, "500-999":4, "1000-4999": 5, "5000-9999":6, "10000+":7}
    if(data in col_options.keys()):
         return col_options[data]
    # else:
    #     return 8
def vect_company_type(data: str):
    col_options = {"Pvt Ltd": 0, "Funded Startup": 1, "Public Sector": 2, "Early Stage Startup": 3, "Other": 4, "NGO": 5}
    if(data in col_options.keys()):
        return col_options[data]
    # else:
    #     return 6

def vect_last_new_job(data: str):
    col_options = {"never": 0,"1": 1, "2": 2, "3": 3, "4": 4, ">4": 5}
    if(data in col_options.keys()):
        return col_options[data]
    # else:
    #     return 6


def vectorise_data(df_train, df_test, df_answer, isNull):
    # df_train = df_train.drop(df_train[(df_train.isnull().sum(axis=1) >2)].index)


    df_train['city'] = df_train['city'].str[5:].values.astype('int')
    df_train['gender'] = df_train.apply(lambda row: vect_gender(row['gender']), axis=1)
    df_train['relevent_experience'] = df_train.apply(lambda row: vect_relevent_experience(row['relevent_experience']), axis=1)
    df_train['enrolled_university'] = df_train.apply(lambda row: vect_enrolled_university(row['enrolled_university']), axis=1)
    df_train['education_level'] = df_train.apply(lambda row: vect_education_level(row['education_level']), axis=1)
    df_train['major_discipline'] = df_train.apply(lambda row: vect_major_discipline(row['major_discipline']), axis=1)
    df_train['experience'] = df_train.apply(lambda row: vect_experience(row['experience']), axis=1)
    df_train['company_size'] = df_train.apply(lambda row: vect_company_size(row['company_size']), axis=1)
    df_train['company_type'] = df_train.apply(lambda row: vect_company_type(row['company_type']), axis=1)
    df_train['last_new_job'] = df_train.apply(lambda row: vect_last_new_job(row['last_new_job']), axis=1)

    df_test['city'] = df_test['city'].str[5:].values.astype('int')
    df_test['gender'] = df_test.apply(lambda row: vect_gender(row['gender']), axis=1)
    df_test['relevent_experience'] = df_test.apply(lambda row: vect_relevent_experience(row['relevent_experience']), axis=1)
    df_test['enrolled_university'] = df_test.apply(lambda row: vect_enrolled_university(row['enrolled_university']), axis=1)
    df_test['education_level'] = df_test.apply(lambda row: vect_education_level(row['education_level']), axis=1)
    df_test['major_discipline'] = df_test.apply(lambda row: vect_major_discipline(row['major_discipline']), axis=1)
    df_test['experience'] = df_test.apply(lambda row: vect_experience(row['experience']), axis=1)
    df_test['company_size'] = df_test.apply(lambda row: vect_company_size(row['company_size']), axis=1)
    df_test['company_type'] = df_test.apply(lambda row: vect_company_type(row['company_type']), axis=1)
    df_test['last_new_job'] = df_test.apply(lambda row: vect_last_new_job(row['last_new_job']), axis=1)

    if not isNull:
        df_train['gender'] = df_train['gender'].fillna(round((df_train['gender'].mean())))
        df_train['enrolled_university'] = df_train['enrolled_university'].fillna(0)
        df_train['major_discipline'] = df_train['major_discipline'].fillna(round((df_train['major_discipline'].mean())))
        df_train['company_size'] = df_train['company_size'].fillna(round((df_train['company_size'].mean())))
        df_train['company_type'] = df_train['company_type'].fillna(round((df_train['company_type'].mean())))
        df_train['experience'] = df_train['experience'].fillna(round((df_train['experience'].mean())))
        df_train['last_new_job'] = df_train['last_new_job'].fillna(round((df_train['last_new_job'].mean())))
        df_train['education_level'] = df_train['education_level'].fillna(round((df_train['education_level'].mean())))
        
        df_test['gender'] = df_test['gender'].fillna(round((df_test['gender'].mean())))
        df_test['enrolled_university'] = df_test['enrolled_university'].fillna(0)
        df_test['major_discipline'] = df_test['major_discipline'].fillna(round((df_test['major_discipline'].mean())))
        df_test['company_size'] = df_test['company_size'].fillna(round((df_test['company_size'].mean())))
        df_test['company_type'] = df_test['company_type'].fillna(round((df_test['company_type'].mean())))
        df_test['experience'] = df_test['experience'].fillna(round((df_test['experience'].mean())))
        df_test['last_new_job'] = df_test['last_new_job'].fillna(round((df_test['last_new_job'].mean())))
        df_test['education_level'] = df_test['education_level'].fillna(round((df_test['education_level'].mean())))

    return df_train, df_test, df_answer