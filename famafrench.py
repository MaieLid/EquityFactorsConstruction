import pandas as pd 
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd, YearEnd
from scipy import stats
from tqdm import tqdm
import wrds

def compustat_block(upload_type, conn):

    """
        gvkey = (CHAR) = standard & poor's identifier
        datadate =
        at =
        pstkl =
        txditc =
        pstkrv =
        seq =
        pstk =

        clé pour

    """

    if upload_type =='from_wrds':
        Querry = """
        select gvkey, datadate, at, pstkl, txditc,
        pstkrv, seq, pstk
        from comp.funda
        where indfmt='INDL' 
        and datafmt='STD'
        and popsrc='D'
        and consol='C'
        and datadate >= '01/01/1959'
        """
        try:
            comp = conn.raw_sql(Querry)            
        except:
            conn = wrds.Connection()
            comp = conn.raw_sql(Querry)
           
    elif upload_type=='from_file':
        #comp = pd.read_csv('.data/comp.csv.gz', compression='gzip')
        comp = pd.read_pickle('data/comp_with_lt.pkl')
        if bool(np.isin('Unnamed: 0', comp.columns)):
            comp = comp.drop('Unnamed: 0',axis=1)
    
    comp['datadate']=pd.to_datetime(comp['datadate']) #convert datadate to date fmt
    comp['year']=comp['datadate'].dt.year

    # create preferrerd stock
    comp['ps']=np.where(comp['pstkrv'].isnull(), comp['pstkl'], comp['pstkrv'])
    comp['ps']=np.where(comp['ps'].isnull(),comp['pstk'], comp['ps'])
    comp['ps']=np.where(comp['ps'].isnull(),0,comp['ps'])

    comp['txditc']=comp['txditc'].fillna(0)

    # create book equity
    comp['be']=comp['seq']+comp['txditc']-comp['ps']
    comp['be']=np.where(comp['be']>0, comp['be'], np.nan)

    #### ajout robin 10/04/2020 #######
    # remplacer la book value of equity par total asset - liabilities si absent
    comp['be'].fillna(comp['at'] - comp['lt'], inplace=True)

    # number of years in Compustat
    comp=comp.sort_values(by=['gvkey','datadate'])
    comp['count']=comp.groupby(['gvkey']).cumcount()

    comp=comp[['gvkey','datadate','year','be','count']]
    return comp

def crsp_block(upload_type, conn):
    if upload_type =='from_wrds':
        #Market data
        Querry_crsp = """
        select a.permno, a.permco, a.date, b.shrcd, b.exchcd,
        a.ret, a.retx, a.shro$^ut, a.prc
        from crsp.msf as a
        left join crsp.msenames as b
        on a.permno=b.permno
        and b.namedt<=a.date
        and a.date<=b.nameendt
        where a.date between '01/01/1959' and '12/31/2017'
        and b.exchcd between 1 and 3
        """
        Querry_dlret = """
        select permno, dlret, dlstdt 
        from crsp.msedelist
        """
        
        try:
            crsp_m = conn.raw_sql(Querry_crsp)            
            dlret = conn.raw_sql(Querry_dlret)  
        except:# OperationalError:
            conn = wrds.Connection()
            crsp_m = conn.raw_sql(Querry_crsp)
            dlret = conn.raw_sql(Querry_dlret)  
            
    elif upload_type =='from_file':
        #crsp_m = pd.read_csv('.data/crsp_m.csv.gz', compression='gzip')
        crsp_m = pd.read_pickle('data/crsp_m_modified.pkl')
        if bool(np.isin('Unnamed: 0', crsp_m.columns)):
            crsp_m = crsp_m.drop('Unnamed: 0',axis=1)
        #dlret = pd.read_csv('.data/dlret.csv.gz', compression='gzip')
        dlret = pd.read_pickle('data/dlret.pkl')
        if bool(np.isin('Unnamed: 0', dlret.columns)):
            dlret = dlret.drop('Unnamed: 0',axis=1)

    # change variable format to int
    crsp_m[['permco','permno','shrcd','exchcd']] = crsp_m[['permco','permno','shrcd','exchcd']].astype(int)
    # Line up date to be end of month
    crsp_m['date']=pd.to_datetime(crsp_m['date'])
    crsp_m['jdate']=crsp_m['date']+MonthEnd(0)

    # add delisting return
    dlret.permno=dlret.permno.astype(int)
    dlret['dlstdt']=pd.to_datetime(dlret['dlstdt'])
    dlret['jdate']=dlret['dlstdt']+MonthEnd(0)

    crsp = pd.merge(crsp_m, dlret, how='left',on=['permno','jdate'])
    crsp['dlret']=crsp['dlret'].fillna(0)
    crsp['ret']=crsp['ret'].fillna(0)
    crsp['retadj']=(1+crsp['ret'])*(1+crsp['dlret'])-1
    crsp['me']=crsp['prc'].abs()*crsp['shrout'] # calculate market equity
    crsp = crsp.drop(['dlret','dlstdt','shrout'], axis=1)
    crsp = crsp.sort_values(by=['jdate','permco','me'])
    
    return crsp

def ccm_block(upload_type, conn):
    if upload_type =='from_wrds':
        Querry = """
        select gvkey, lpermno as permno, linktype, linkprim, 
        linkdt, linkenddt
        from crsp.ccmxpf_linktable
        where substr(linktype,1,1)='L'
        and (linkprim ='C' or linkprim='P')
        """
        try:
            ccm = conn.raw_sql(Querry)            
        except:# OperationalError:
            conn = wrds.Connection()
            ccm = conn.raw_sql(Querry)
            
    elif upload_type =='from_file':
        #ccm = pd.read_csv('.data/ccm.csv.gz', compression='gzip')
        ccm = pd.read_pickle('data/ccm.pkl')
        if bool(np.isin('Unnamed: 0', ccm.columns)):
            ccm = ccm.drop('Unnamed: 0',axis=1)
        
    ccm['linkdt']=pd.to_datetime(ccm['linkdt'])
    ccm['linkenddt']=pd.to_datetime(ccm['linkenddt'])
    # if linkenddt is missing then set to today date
    ccm['linkenddt']=ccm['linkenddt'].fillna(pd.to_datetime('today'))

    return ccm

#recup les time series de fama french
def ff_timeseries(upload_type, conn):
    # Dowload Fama-French Time series
    if upload_type =='from_wrds':
        try:
            _ff = conn.get_table(library='ff', table='factors_monthly')
        except:# OperationalError:
            conn =wrds.Connection()
            _ff = conn.get_table(library='ff', table='factors_monthly')
    elif upload_type =='from_file':
        _ff = pd.read_pickle(r'data/fama_french_ts.pkl')
        if bool(np.isin('Unnamed: 0', _ff.columns)):
            _ff = _ff.drop('Unnamed: 0',axis=1)
    
    return _ff

def merge_ccm_comp(ccm, comp):
    ccm1=pd.merge(comp[['gvkey','datadate','be', 'count']],ccm,how='left',on=['gvkey'])
    ccm1['yearend']=ccm1['datadate']+YearEnd(0)
    ccm1['jdate']=ccm1['yearend']+MonthEnd(6)

    # set link date bounds
    ccm2=ccm1[(ccm1['jdate']>=ccm1['linkdt'])&(ccm1['jdate']<=ccm1['linkenddt'])]
    ccm2=ccm2[['gvkey','permno','datadate','yearend', 'jdate','be', 'count']]

    # keep the lastest be per gvkey, permno, jdate
    ccm2 = pd.merge(ccm2, ccm2.groupby(['gvkey', 'permno', 'jdate']).datadate.max().reset_index(),
                    on =['gvkey', 'permno','jdate','datadate'], how ='inner')

    # if several gvkeys for the same permno, keep the gvkey with the highest BE value
    ccm2 = pd.merge(ccm2, ccm2.groupby(['permno', 'jdate']).be.max().reset_index(), 
             on =['permno', 'jdate', 'be'], how='inner')
    return ccm2

def marketcap_agg(crsp):
    ##Aggregate Market Cap
    # sum of me across different permno belonging to same permco a given date
    crsp_summe = crsp.groupby(['jdate','permco'])['me'].sum().reset_index()
    # largest mktcap within a permco/date
    crsp_maxme = crsp.groupby(['jdate','permco'])['me'].max().reset_index()
    # join by jdate/maxme to find the permno
    crsp1=pd.merge(crsp, crsp_maxme, how='inner', on=['jdate','permco','me'])
    # drop me column and replace with the sum of me
    crsp1=crsp1.drop(['me'], axis=1)
    # join with sum of me to get the correct market cap info
    crsp2=pd.merge(crsp1, crsp_summe, how='inner', on=['jdate','permco'])
    # sort by permno and date and also drop duplicates
    crsp2=crsp2.sort_values(by=['permno','jdate']).drop_duplicates()

    # keep December market cap
    crsp2['year']=crsp2['jdate'].dt.year
    crsp2['month']=crsp2['jdate'].dt.month

    crsp2['1+retx']=1+crsp2['retx']
    crsp2=crsp2.sort_values(by=['permno','date'])

    # lag market cap
    crsp2['lme']=crsp2.groupby(['permno'])['me'].shift(1)

    # if first permno then use me/(1+retx) to replace the missing value
    crsp2['count']=crsp2.groupby(['permno']).cumcount()
    crsp2['lme']=np.where(crsp2['count']==0, crsp2['me']/crsp2['1+retx'], crsp2['lme'])
    
    return crsp2

# function to assign momentum and size buckets
def sz_bucket_formom(row):
    if (row['ME(t-1)']==np.nan)|(row['prc(t-13)']==np.nan) :
        value=''
    elif row['ME(t-1)']<=row['sizemedn']:
        value='S'
    else:
        value='B'
    return value

def mom_bucket(row):
    if row['Prior(t-12, t-2)']<=row['Mom30']:
        value = 'D'
    elif row['Prior(t-12, t-2)'] >row['Mom70']:
        value='U'
    else: 
        value=''
    return value



# function to calculate value weighted return
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan  
    
def momentum_data_2(crsp2):
    crsp_mom = crsp2.drop(['permco', 'count', '1+retx'], axis=1)
    crsp_mom['date'] = crsp_mom.date + MonthEnd(0)
    crsp_mom = crsp_mom.set_index('date')
    mom = crsp_mom.groupby('permno').ret.rolling(window=10).apply(lambda x: (1+x).prod() -1).reset_index()
    mom['Prior(t-12, t-2)'] = mom.groupby(['permno'])['ret'].transform(lambda x:x.shift(3))

    crsp_mom['ME(t-1)'] = crsp_mom.groupby(['permno'])['me'].transform(lambda x:x.shift(2))
    crsp_mom['prc(t-13)'] = crsp_mom.groupby(['permno'])['prc'].transform(lambda x:x.shift(14))

    data_mom = pd.merge(crsp_mom.reset_index(), mom.drop('ret', axis=1), on=['date', 'permno'], how='inner')
    data_mom = data_mom.loc[(~ data_mom['prc(t-13)'].isnull()) & data_mom['shrcd'].isin([10,11])]

    nyse_mom = data_mom.loc[data_mom.exchcd ==1, ['date', 'Prior(t-12, t-2)']].groupby('date')\
    .describe(percentiles=[0.3, 0.7]).reset_index()
    nyse_mom = nyse_mom.set_index('date')['Prior(t-12, t-2)'][['30%','70%']].rename(columns={'30%':'Mom30', '70%':'Mom70'}).reset_index()

    data_mom2 = pd.merge(data_mom, nyse_mom, how='left', on=['date'])

    data_mom2['momport']=data_mom2.apply(mom_bucket, axis=1)

    return data_mom2

def momentum_data(crsp2):
    crsp_mom = crsp2.drop(['permco', 'count', '1+retx'], axis=1)
    crsp_mom['date'] = crsp_mom.date + MonthEnd(0)
    crsp_mom = crsp_mom.set_index('date')
    mom = crsp_mom.groupby('permno').ret.rolling(window=10).apply(lambda x: (1+x).prod() -1).reset_index()
    mom['Prior(t-12, t-2)'] = mom.groupby(['permno'])['ret'].transform(lambda x:x.shift(3))

    crsp_mom['ME(t-1)'] = crsp_mom.groupby(['permno'])['me'].transform(lambda x:x.shift(2))
    crsp_mom['prc(t-13)'] = crsp_mom.groupby(['permno'])['prc'].transform(lambda x:x.shift(14))

    data_mom = pd.merge(crsp_mom.reset_index(), mom.drop('ret', axis=1), on=['date', 'permno'], how='inner')
    data_mom = data_mom.loc[(~ data_mom['prc(t-13)'].isnull()) & data_mom['shrcd'].isin([10,11])]

    #Nyse Break Points
    nyse_sz =data_mom.loc[data_mom.exchcd ==1, ['date', 'ME(t-1)']].groupby('date').median()\
    .reset_index().rename(columns={'ME(t-1)':'sizemedn'})
    nyse_mom = data_mom.loc[data_mom.exchcd ==1, ['date', 'Prior(t-12, t-2)']].groupby('date')\
    .describe(percentiles=[0.3, 0.7]).reset_index()
    nyse_mom = nyse_mom.set_index('date')['Prior(t-12, t-2)'][['30%','70%']].rename(columns={'30%':'Mom30', '70%':'Mom70'}).reset_index()
    nyse_breaks = pd.merge(nyse_sz, nyse_mom, how='inner', on=['date'])
    data_mom2 = pd.merge(data_mom, nyse_breaks, how='left', on=['date'])

    # assign size portfolio and momentum portfolio
    data_mom2['szport']= data_mom2.apply(sz_bucket_formom, axis=1)
    data_mom2['momport']=data_mom2.apply(mom_bucket, axis=1)

    data_mom3 = data_mom2.loc[(data_mom2.momport !='') & (data_mom2.szport!=''), :]
    return data_mom3

def momentum_backtest(data_mom):
    # value-weigthed return
    vwret=data_mom.groupby(['date','szport','momport']).apply(wavg, 'retadj', 'lme').to_frame().reset_index().rename(columns={0: 'vwret'})
    vwret['smport']=vwret['szport']+vwret['momport']

    ff_factors_m=vwret.pivot(index='date', columns='smport', values='vwret').reset_index()
    ff_factors_m['WU']=(ff_factors_m['BU']+ff_factors_m['SU'])/2
    ff_factors_m['WD']=(ff_factors_m['BD']+ff_factors_m['SD'])/2
    ff_factors_m['WUMD'] = ff_factors_m['WU']-ff_factors_m['WD']
    return ff_factors_m

def wght_mom(data_mom):
    sumwt = data_mom.groupby(['date','szport','momport']).lme.sum().rename('sumwt').reset_index()
    data_mom4 = pd.merge(data_mom, sumwt, on =['date','szport','momport'], how='inner')
    data_mom4['wght'] = data_mom4['lme'].div(data_mom4['sumwt'])
    data_mom4['sbport']=data_mom4['szport']+data_mom4['momport']

    # Create dataframe for momentum strategy weights 
    wght_mom = data_mom4.loc[:, ['date', 'permno', 'sbport', 'wght', 'retadj']]
    wght_mom = wght_mom.drop_duplicates()
    wght_mom['wght'] = wght_mom.wght/2
    wght_mom.loc[wght_mom.sbport.isin(['BD', 'SD']), 'wght'] = - wght_mom.loc[wght_mom.sbport.isin(['BD', 'SD']),'wght']
    wght_mom['wght*(1+sign(wght)ret)'] =wght_mom['wght'].mul(1+np.sign(wght_mom['wght'])*wght_mom['retadj'])
    wght_mom['wght*ret'] =wght_mom['wght'].mul(wght_mom['retadj'])
    return wght_mom


    
    
def size_value_backtest(ccm4):
    # value-weigthed return
    vwret=ccm4.groupby(['jdate','szport','bmport']).apply(wavg, 'retadj','wt').to_frame().reset_index().rename(columns={0: 'vwret'})
    vwret['sbport']=vwret['szport']+vwret['bmport']

    # tranpose
    ff_factors_sv=vwret.pivot(index='jdate', columns='sbport', values='vwret').reset_index()

    # create HML factors
    ff_factors_sv['WH'] = ff_factors_sv[[i for i in ff_factors_sv.columns if i in ['BH','SH']]].mean(1)    
    ff_factors_sv['WL']= ff_factors_sv[[i for i in ff_factors_sv.columns if i in ['BL','SL']]].mean(1)
    ff_factors_sv['WHML'] = ff_factors_sv['WH'] - ff_factors_sv['WL']

    #create SMB factor
    ff_factors_sv['WB']= ff_factors_sv[[i for i in ff_factors_sv.columns if i in ['BL','BM','BH']]].mean(1)
    ff_factors_sv['WS']= ff_factors_sv[[i for i in ff_factors_sv.columns if i in ['SL','SM','SH']]].mean(1)
    ff_factors_sv['WSMB'] = ff_factors_sv['WS']-ff_factors_sv['WB']
    ff_factors_sv=ff_factors_sv.rename(columns={'jdate':'date'})
     
    """
    ancienne version
    ff_factors_sv['WH']=(ff_factors_sv['BH']+ff_factors_sv['SH'])/2
    ff_factors_sv['WL']=(ff_factors_sv['BL']+ff_factors_sv['SL'])/2
    ff_factors_sv['WHML'] = ff_factors_sv['WH']-ff_factors_sv['WL']

    ff_factors_sv['WB']=(ff_factors_sv['BL']+ff_factors_sv['BM']+ff_factors_sv['BH'])/3
    ff_factors_sv['WS']=(ff_factors_sv['SL']+ff_factors_sv['SM']+ff_factors_sv['SH'])/3
    ff_factors_sv['WSMB'] = ff_factors_sv['WS']-ff_factors_sv['WB']
    ff_factors_sv=ff_factors_sv.rename(columns={'jdate':'date'})
    """
    return ff_factors_sv

def plot_comparizon_figure(ffcomp):
    _ffcomp70 = ffcomp[ffcomp['date']>='01/01/1970']
    _ffcomp70 = _ffcomp70.set_index('date')
    cols =['smb', 'WSMB', 'hml', 'WHML', 'umd', 'WUMD']
    assert all(np.isin(cols, _ffcomp70.columns))
    
    print(stats.pearsonr(_ffcomp70['smb'], _ffcomp70['WSMB']))
    print(stats.pearsonr(_ffcomp70['hml'], _ffcomp70['WHML']))
    print(stats.pearsonr(_ffcomp70['umd'], _ffcomp70['WUMD']))
    
    plt.figure(figsize=(14, 5))
    ax =plt.subplot(131)
    (100*(_ffcomp70.loc[(_ffcomp70.index >='01/07/1999')& (_ffcomp70.index <'01/01/2012'), ['smb', 'WSMB']]+1)\
     .cumprod().rename(columns={'smb':'FF', 'WSMB':'wrds'})).plot(grid =True, ax =ax)
    plt.title('Size Factor Small minus Big')

    ax =plt.subplot(132)
    (100*(_ffcomp70.loc[(_ffcomp70.index>='01/07/1999')& (_ffcomp70.index <'01/01/2012'), ['hml', 'WHML']]+1)\
     .cumprod().rename(columns={'hml':'FF', 'WHML':'wrds'})).plot(grid =True, ax =ax)
    plt.title('Value Factor High minus Low')

    ax =plt.subplot(133)
    (100*(_ffcomp70.loc[(_ffcomp70.index>='01/07/1999')& (_ffcomp70.index <'01/01/2012'), ['umd', 'WUMD']]+1)\
     .cumprod().rename(columns={'umd':'FF', 'WUMD':'wrds'})).plot(grid =True, ax =ax)
    plt.title('Momentum Factor High minus Low')

    plt.tight_layout()
    plt.show()
    
def read_prepare_data(path):
    if 'csv' in path:
        if 'gz' in path:
            trades_year = pd.read_csv(path, compression='gzip')
        else:
            trades_year = pd.read_csv(path)
    elif 'pkl' in path:
        if 'gz' in path:
            trades_year = pd.read_pickle(path, compression= 'gzip')
        else:
            trades_year = pd.read_pickle(path)
       
    assert all(np.isin(['permno', 'odtOrderDate'], trades_year.columns))
    trades_year['odtOrderDate'] = pd.to_datetime(trades_year.odtOrderDate, format='%Y-%m-%d')
    trades_year['month'] = trades_year.set_index('odtOrderDate').index.month
    trades_year['permno'] =trades_year.permno.apply(lambda x: str(int(x)))
    
    return trades_year
    
#fonction rajouté par robin au 04/05/2020 pour préparer les data avant d'être triées
def prepare_ccm_jun_nyse_crsp3(crsp2, ccm2):
    ### December Market Capitalization
    decme = crsp2[crsp2['month'] == 12]
    decme = decme[['permno', 'date', 'jdate', 'me', 'year']].rename(columns={'me': 'dec_me'})

    ### July to June dates
    crsp2['ffdate'] = crsp2['jdate'] + MonthEnd(-6)
    crsp2['ffyear'] = crsp2['ffdate'].dt.year
    crsp2['ffmonth'] = crsp2['ffdate'].dt.month
    # cumret by stock
    crsp2['cumretx'] = crsp2.groupby(['permno', 'ffyear'])['1+retx'].cumprod()
    # lag cumret
    crsp2['lcumretx'] = crsp2.groupby(['permno'])['cumretx'].shift(1)

    # baseline me
    mebase = crsp2[crsp2['ffmonth'] == 1][['permno', 'ffyear', 'lme']].rename(columns={'lme': 'mebase'})

    # merge result back together
    crsp3 = pd.merge(crsp2, mebase, how='left', on=['permno', 'ffyear'])
    crsp3['wt'] = np.where(crsp3['ffmonth'] == 1, crsp3['lme'], crsp3['mebase'] * crsp3['lcumretx'])

    decme['year'] = decme['year'] + 1
    decme = decme[['permno', 'year', 'dec_me']]

    # Info as of June
    crsp3_jun = crsp3[crsp3['month'] == 6]

    crsp_jun = pd.merge(crsp3_jun, decme, how='inner', on=['permno', 'year'])
    crsp_jun = crsp_jun[
        ['permno', 'date', 'jdate', 'shrcd', 'exchcd', 'retadj', 'me', 'wt', 'cumretx', 'mebase', 'lme', 'dec_me']]
    crsp_jun = crsp_jun.sort_values(by=['permno', 'jdate']).drop_duplicates()

    # link comp and crsp
    ccm_jun = pd.merge(crsp_jun, ccm2, how='inner', on=['permno', 'jdate'])
    ccm_jun['beme'] = ccm_jun['be'] * 1000 / ccm_jun['dec_me']

    # keep the lastest be per gvkey, permno, jdate
    ccm2 = pd.merge(ccm2, ccm2.groupby(['gvkey', 'permno', 'jdate']).datadate.max().reset_index(),
                    on=['gvkey', 'permno', 'jdate', 'datadate'], how='inner')

    # if several gvkeys for the same permno, keep the gvkey with the highest BE value
    ccm2 = pd.merge(ccm2, ccm2.groupby(['permno', 'jdate']).be.max().reset_index(),
                    on=['permno', 'jdate', 'be'], how='inner')
    # link comp and crsp
    ccm_jun = pd.merge(crsp_jun, ccm2, how='inner', on=['permno', 'jdate'])
    ccm_jun['beme'] = ccm_jun['be'] * 1000 / ccm_jun['dec_me']

    # select NYSE stocks for bucket breakdown
    # exchcd = 1 and positive beme and positive me and shrcd in (10,11) and at least 2 years in comp
    nyse = ccm_jun[(ccm_jun['exchcd'] == 1) & (ccm_jun['beme'] > 0) & (ccm_jun['me'] > 0) & \
                   (ccm_jun['count'] >= 1) & ((ccm_jun['shrcd'] == 10) | (ccm_jun['shrcd'] == 11))]

    return ccm_jun, nyse, crsp3

################################ fonctions pour faire le tri indépendant et séquentiel #######################

#fonction pour assigner un flag de portefeuille value
def bm_bucket_vector(data, additinnal_condition):

    data['bmport'] = ''
    
    if additinnal_condition:
        data.loc[(data['me']>0) & (data['count']>=1) & (data['beme'] != 0) & (data['bm30'] > data['beme']), 'bmport'] = 'L'
        data.loc[(data['me']>0) & (data['count']>=1) &(data['beme'] != 0) & (data['bm70'] >= data['beme']) & (data['beme'] > data['bm30']), 'bmport'] = 'M'
        data.loc[(data['me']>0) & (data['count']>=1) &(data['beme'] != 0) & (data['beme'] > data['bm70']), 'bmport'] = 'H'
    else:
        data.loc[(data['beme'] != 0) & (data['bm30'] > data['beme']), 'bmport'] = 'L'
        data.loc[(data['beme'] != 0) & (data['bm70'] >= data['beme']) & (data['beme'] > data['bm30']), 'bmport'] = 'M'
        data.loc[(data['beme'] != 0) & (data['beme'] > data['bm70']), 'bmport'] = 'H'
    
    return data['bmport']

#fonction pour assigner un flag de portefeuille size
def sz_bucket_vector(data, number_port, additionnal_conditions):

    data['szport'] = ''    
        
    if additionnal_conditions:
        if number_port == 3:
            data.loc[(data['count']>=1) & (data['beme'] != 0) & (data['me'] > 0) & (data['sz30'] > data['me']), 'szport'] = 'S'
            data.loc[(data['count']>=1) & (data['beme'] != 0) & (data['me'] > 0) & (data['sz70'] >= data['me']) & (data['me'] > data['sz30']), 'szport'] = 'M'
            data.loc[(data['count']>=1) & (data['beme'] != 0) & (data['me'] > 0) & (data['me'] > data['sz70']), 'szport'] = 'B'
        elif number_port == 2:
            data.loc[(data['count']>=1) & (data['beme'] != 0) & (data['me'] > 0) & (data['me'] > data['sizemedn']), 'szport'] = 'B'
            data.loc[(data['count']>=1) & (data['beme'] != 0) & (data['me'] > 0) & (data['me'] <= data['sizemedn']), 'szport'] = 'S'    
    else:        
        if number_port == 3:    
            data.loc[(data['me'] > 0) & (data['sz30'] > data['me']), 'szport'] = 'S'
            data.loc[(data['me'] > 0) & (data['sz70'] >= data['me']) & (data['me'] > data['sz30']), 'szport'] = 'M'
            data.loc[(data['me'] > 0) & (data['me'] > data['sz70']), 'szport'] = 'B'
        elif number_port ==2:
            data.loc[(data['me'] > 0) & (data['me'] > data['sizemedn']),'szport'] = 'B'
            data.loc[(data['me'] > 0) & (data['me'] <= data['sizemedn']),'szport'] = 'S'
    
    return data['szport']

#fonction à appeler pour réaliser des portefeuilles triés 
def sort_different_ways(ccm_jun_input,
                        nyse_input, 
                        crsp3,
                        is_sequential, 
                        post_or_pre, 
                        control_variable, 
                        priced_variable,
                        breakpoints_universe,# ['NYSE','all_exch'] => pour décider 
                        number_port_me,
                        is_inverse=False): #number of sub portfolio when sorting by size    
    
    #dans la manière de constuire, certains portefeuilles sont similaires, après ils ne représentent 
    #plus la même chose, puisque d'un côté on fait High minus Low et de l'autre on fait 
    #Small minus Big 
        
    if is_sequential: 
        if breakpoints_universe == 'NYSE':
            data_breakpoints = nyse_input.set_index('jdate')
        else:
            data_breakpoints = ccm_jun_input.set_index('jdate')
            
        if (post_or_pre == 'post' and priced_variable == 'beme') or (post_or_pre == 'pre' and priced_variable == 'me'):            
            
            beme_percentile = data_breakpoints.groupby(['jdate'])['beme'].describe(percentiles=[0.3, 0.7])[['30%','70%']].rename(columns={'30%':'bm30', '70%':'bm70'})                            
            data_breakpoints[['bm30','bm70']] = beme_percentile
            
            data_breakpoints['bmport'] = bm_bucket_vector(data_breakpoints, False)
            
            if number_port_me == 2: #trier le facteur size en 2 = la médiane
                nyse_median = data_breakpoints.groupby(['jdate', 'bmport']).median().reset_index().rename(columns={'me':'sizemedn'})[['jdate','bmport','sizemedn']]                                
            elif number_port_me == 3: #trier le facteur size en 3
                nyse_median = data_breakpoints.groupby(['jdate', 'bmport']).describe(percentiles=[0.3, 0.7])['me'].reset_index()[['jdate','30%','bmport','70%']].rename(columns={'30%':'sz30', '70%':'sz70'})                                
                
            #on applique les breakpoints à l'univers entier
            ccm1_jun = ccm_jun_input.merge(beme_percentile, on=['jdate'])
            ccm1_jun['bmport'] = bm_bucket_vector(ccm1_jun, False)            
            ccm1_jun = ccm1_jun.merge(nyse_median, on=['jdate','bmport'])
            ccm1_jun['szport'] = sz_bucket_vector(ccm1_jun, number_port_me, True)                                
        
            #pour retourner l'output breaks
            breaks = nyse_median.merge(beme_percentile,on='jdate')
            
        elif (post_or_pre == 'pre' and priced_variable == 'beme') or (post_or_pre == 'post' and priced_variable == 'me'):
            #on trie d'abords sur me puis sur beme 
            
            if number_port_me == 2:
                me_mediane = data_breakpoints.groupby(['jdate']).median().reset_index().rename(columns={'me':'sizemedn'})[['jdate','sizemedn']]
            elif number_port_me == 3: #cas du portefeuille 3x3
                me_mediane = data_breakpoints.groupby(['jdate']).describe(percentiles=[0.3, 0.7])['me'].reset_index()[['jdate','30%','70%']].rename(columns={'30%':'sz30', '70%':'sz70'})

            data_breakpoints = data_breakpoints.merge(me_mediane, on=['jdate'])
            
            #assign to portfolio size 
            data_breakpoints['szport'] = sz_bucket_vector(data_breakpoints, number_port_me, False)

            nyse_percentiles = data_breakpoints.groupby(['jdate', 'szport'])['beme'].describe(percentiles=[0.3, 0.7]).reset_index()[['jdate', 'szport', '30%','70%']].rename(columns={'30%':'bm30', '70%':'bm70'})

            #on applique les breakpoints à l'univers entier
            ccm1_jun = ccm_jun_input.merge(me_mediane, on=['jdate'])
            ccm1_jun['szport'] = sz_bucket_vector(ccm1_jun, number_port_me, False)                
            ccm1_jun = ccm1_jun.merge(nyse_percentiles, on=['jdate','szport'])
            ccm1_jun['bmport'] = bm_bucket_vector(ccm1_jun, True)
            
            #pour retourner l'output breaks
            breaks = me_mediane.merge(nyse_percentiles,on='jdate')
        else: 
            return 'Vérifie tes inputs ! ' 
    else:         
        ############## METHODE CLASSIQUE SANS TRI SEQUENTIEL #########
        
        #read directly from the breaks previously calculated to save time
        if breakpoints_universe == 'NYSE':
            breaks_bm = pd.read_pickle(r'data/NYSE_bm_23.pkl').rename(columns={'30%':'bm30', '70%':'bm70'})
            if number_port_me == 2:
                breaks_sz = pd.read_pickle(r'data/NYSE_sz_23.pkl').rename(columns={'30%':'sz30', '70%':'sz70'})
            else:
                breaks_sz = pd.read_pickle(r'data/NYSE_sz_33.pkl').rename(columns={'30%':'sz30', '70%':'sz70'})
        else: # all names
            breaks_bm = pd.read_pickle(r'data/all_exch_bm_23.pkl').rename(columns={'30%':'bm30', '70%':'bm70'})
            if number_port_me == 2:
                breaks_sz = pd.read_pickle(r'data/all_exch_sz_23.pkl').rename(columns={'30%':'sz30', '70%':'sz70'})
            else:
                breaks_sz = pd.read_pickle(r'data/all_exch_sz_33.pkl').rename(columns={'30%':'sz30', '70%':'sz70'})
        
        if is_inverse:
            if breakpoints_universe == 'NYSE':
                breaks_bm = pd.read_pickle(r'data/NYSE_bm_23_inverse.pkl').rename(columns={'30%': 'bm30', '70%': 'bm70'})
            else:
                breaks_bm = pd.read_pickle(r'data/all_exch_bm_23_inverse.pkl').rename(columns={'30%': 'bm30', '70%': 'bm70'})
        
        #merge les deux dataframes finalement 
        #nyse_breaks = pd.merge(nyse_sz, nyse_bm, how='inner', on=['jdate'])
        breaks = pd.merge(breaks_sz, breaks_bm, how='inner', on=['jdate'])

        #ccm1_jun = pd.merge(ccm_jun_input, nyse_breaks, how='left', on=['jdate'])        
        ccm1_jun = pd.merge(ccm_jun_input, breaks, how='left', on=['jdate'])        
        
        # assign size portfolio
        ccm1_jun['szport'] = sz_bucket_vector(ccm1_jun, number_port_me, True)

        # assign book-to-market portfolio
        ccm1_jun['bmport'] = bm_bucket_vector(ccm1_jun, True)
    
    # create positivebmeme and nonmissport variable
    ccm1_jun['posbm']=np.where((ccm1_jun['beme']>0)&(ccm1_jun['me']>0)&(ccm1_jun['count']>=1), 1, 0) #pas sûr que ça serve à grand chose 
    ccm1_jun['nonmissport']=np.where((ccm1_jun['bmport']!=''), 1, 0)

    # store portfolio assignment as of June
    june=ccm1_jun[['permno','jdate', 'bmport','szport','posbm','nonmissport','beme','me']]
    june['ffyear']=june['jdate'].dt.year

    # merge back with monthly records
    crsp3 = crsp3[['date','permno','shrcd','exchcd','retadj','wt','cumretx','ffyear','jdate']]
    ccm3 = pd.merge(crsp3, june.drop('jdate', axis=1), how='left', on=['permno','ffyear'])
    
    # keeping only records that meet the criteria
    ccm4=ccm3[(ccm3['wt']>0)& (ccm3['posbm']==1) & (ccm3['nonmissport']==1) & 
              ((ccm3['shrcd']==10) | (ccm3['shrcd']==11))]
    
    
    breaks['year'] = breaks['jdate'].dt.year
    breaks = breaks.set_index('year').drop('jdate',axis=1)

    return ccm4, breaks

def bm_bucket(row):
    if 0<=row['beme']<=row['bm30']:
        value = 'L'
    elif row['bm30'] <row['beme'] <=row['bm70']:
        value='M'
    elif row['beme'] >row['bm70']:
        value='H'
    else: 
        value=''
    return value

# function to assign sz and bm bucket
def sz_bucket_forvalue(row):
    if row['me']==np.nan:
        value=''
    elif row['me']<=row['sizemedn']:
        value='S'
    else:
        value='B'
    return value

    
print('fama french loaded')
    
