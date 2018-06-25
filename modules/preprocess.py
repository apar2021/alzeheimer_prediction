import pickle
import pandas as pd
import numpy as np

# config
dropcolumn_merge = [
    'Date',
    'Baseline_Date',
    'Interval_\ntotal_months',
    '3year  FU',
    '3y FU and \nAD conversion',
    'Diagnosis',
    'AD_conversion',
    'Final_Diagnosis',
    'Final_Diagnosis.1',
    'Adconversion',
    'SMIreversion',
    'Dsystem',
    'Psystem',
    'BMI',
    'BMI_Category',
    'APOE',
    'mark_for_slope',
    'patient_num',
    'Sex'
]

class Imputer:
    def __init__(self):
        pass
    
    def categorize_df(self, df):
        male = df['Sex'] == 1
        female = df['Sex'] == 2

        df['sex_1'] = male
        df['sex_2'] = female

        bmi_bad = (df['BMI_Category'] < 1) | (df['BMI_Category'] > 2)
        df['bad_bmi'] = bmi_bad

        return df
    
    def get_df(self, path):
        df = pd.DataFrame.from_csv(path,index_col='hosp_id +')
        df.index.rename('hosp_id',inplace = True)
        df.columns = ['patient_num'] + [col for col in df.columns][1:]
        df = self.categorize_df(df)
        df.drop(dropcolumn_merge,axis=1,inplace=True)
        df.fillna(df.median(0),inplace=True)
        return df
        
    
    def init_df(self,patient_id):
        ret = pd.DataFrame()
        ret.Interval = np.linspace(0,3,7)
        ret['hosp_id'] = [patient_id]*7
        ret = ret.set_index('hosp_id')
        return ret
    
    def zero_five_impute(self, df, degree = 1):
        imputed_df = pd.DataFrame()
        
        index_lst = list(set(df.index))
        index_lst.sort()
        
        for idx in index_lst:
            patient_i = df.ix[idx]
            patient_i_imputed = self.init_df(idx)
            
            x = patient_i.Interval.values
            
            col_to_imputed = [ col for col in patient_i.columns if col != 'Interval']
#             print col_to_imputed
#             assert False
            
            for col in patient_i.columns:
                y = patient_i.ix[:,col].values
                z = np.polyfit(x,y,degree)
                f = np.poly1d(z)
                imputed_vals = f(np.linspace(0,3,7))
                patient_i_imputed[col] = imputed_vals
            
            
            imputed_df = pd.concat([imputed_df, patient_i_imputed],axis = 0)
            
        return imputed_df
    def get_criteria(self, df):
        df=df[df.mark_for_slope=='e']
        criteria = (df.Final_Diagnosis == 'AD').values
        return criteria
    
    def get_interval(self, df):
        df=df[df.mark_for_slope=='e']
        intervals = df.Interval.values
        return intervals
        
    def get_target(self,intervals,criteria):
        ret = []
        intervals = list(intervals)
        for time in intervals:
            threshold = time-0.5
            lst= np.linspace(0,3,7)
            if not criteria[intervals.index(time)]:
                threshold = time + 0.5
            lst = lst>threshold
            ret.append(lst)
        return ret
    
    def zipper(self, i,j):
        return (i,[0],j)
    
    def get_rnn_input(self, intervals,criteria):
        max_len = 10
        num_patient = len(intervals)
        targets = self.get_target(intervals,criteria)
        
        ret = []
        cnt = 1

        for target in targets:
            patient = []
            for t in target:
                patient.append(self.zipper(cnt,int(t)))
                cnt += 1
            for _ in range(max_len-len(patient)):
                patient.append(self.zipper(0,0))
            ret.append(patient)
        
        return ret
    
    def df_normalize(self,df,mean_std = False): # btw 0~1 for float type features
        from sklearn.preprocessing import MinMaxScaler,StandardScaler
        ret = df.copy()
        scaler = MinMaxScaler()
        exclude = [u'DM', u'HTN', u'Lipid',
           u'Heart', u'Stroke',u'sex_1', u'sex_2',"bad_bmi",]
        columns = [item for item in ret.columns if item not in exclude]
        if mean_std:
            scaler = StandardScaler()
        scaler = scaler.fit(ret[columns])
        ret[columns] = scaler.transform(ret[columns])

        return ret
    
    def get_word_vec(self, full_df):

        patient = self.zero_five_impute(full_df)
        patient = patient.drop('Interval',axis=1)
        patient = self.df_normalize(patient,mean_std=True)
        return patient.values

    def get_word_vec_text(self, full_df):
        word_vec_np = self.get_word_vec(full_df)
        wordvec_txt = ''
        for word in word_vec_np:
            for item in word:
                wordvec_txt += str(item)
                wordvec_txt += ' '
            wordvec_txt += '\n'
        return wordvec_txt

def get_target_from_csv(csv_path):
    '''
    get target dataframe from csv_path
    Arguments:
    - csv_path: csv path of longitudinal data
    Returns:
    - target : dataframe object containing patient_id, target(binary)
    '''

    merge_data = pd.DataFrame.from_csv(csv_path)
    less_than_3yr = merge_data[merge_data.Interval < 3]
    is_last = pd.Series([False for i in range(len(less_than_3yr))])
    for i in range(len(less_than_3yr)-1):    
        if less_than_3yr.iloc[i+1]['hosp_id +'] != less_than_3yr.iloc[i]['hosp_id +']:
            is_last[i] = True
            
    less_than_3yr.index = [i for i in range(len(less_than_3yr))]
    target_series = less_than_3yr[is_last].Diagnosis == 'AD'
    patient_num = [less_than_3yr.iloc[i]['hosp_id +'] for i in target_series.index]

    target_df = {'patient_id':{i:patient_num[i] for i in range(len(patient_num))},
            'target' : {i:target_series.iloc[i] for i in range(len(patient_num))}}

    return pd.DataFrame.from_dict(target_df) 

def preprocess_dataframe(df):
    def csv_preprocess(df):
        df.columns = ['patient_num']+[item for item in df.columns][1:]
    
        return df
    def category_preprocessing(patient_df):
        male = patient_df['Sex'] == 1
        female = patient_df['Sex'] == 2

        patient_df['sex_1'] = male
        patient_df['sex_2'] = female

        bmi_bad = (patient_df['BMI_Category'] < 1) | (patient_df['BMI_Category'] > 2)
        patient_df['bad_bmi'] = bmi_bad
        
        patient_df.drop(['BMI_Category'],inplace=True)
        
        return patient_df
    def fill_and_drop(df):
        df = df.drop(dropcolumn_merge,axis=1)
        df = df.fillna(df.median(0))
        return df

    df = csv_preprocess(df)
    df = category_preprocessing(df)
    df = fill_and_drop(df)
    return df

def get_3yr_feature_from_csv(csv_path,until_k = 3):
    '''
    get features imputed for state at 3 year
    Arguments:
    - csv_path : csv path of longitudinal data
    - until_k : data until `until_k` is used to impute the state at 3 year. designed for using at test time.
    Returns:
    - feature_3yr : dataframe object containing features at 3 year
    '''
    imputer = Imputer()
    patient = pd.DataFrame.from_csv(csv_path,index_col='hosp_id +')
    patient = preprocess_dataframe(patient)
    patient = imputer.zero_five_impute(patient)

    is_last = pd.Series([False for i in range(len(patient))])
    for i in range(len(patient)-1):    
        if patient.index[i+1] != patient.index[i]:
            is_last[i] = True
    patient.index = [i for i in range(len(patient))]
    feature_3yr = patient[is_last]
    return feature_3yr

if __name__ =='__main__':
    # unit test
    print get_target_from_csv('../data/merge.csv')
    print get_3yr_feature_from_csv('../data/merge.csv')